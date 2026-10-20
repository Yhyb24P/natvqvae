"""Prepare, train and evaluate the declared WT-column experiment."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from natvqvae.aligned_generator import AlignedGenerator, ALPHABET, UNKNOWN, GAP, MASK, legacy_implementation
from natvqvae.intake import load_manifest, source_root, verify_entry, sha256_file
from natvqvae.sequence import wt_sequence, WT_SHA256

RUN = Path(__file__).parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def resolve_inputs(source):
    wanted = {'allowed_samples', 'family_roles', 'core_split'}
    result = {}
    for e in load_manifest()['entries']:
        if e['name'] in wanted:
            checked = verify_entry(e, source)
            check(checked.ok, checked.detail)
            result[e['name']] = checked.path
    check(set(result) == wanted, 'Missing declared input')
    return result


def prepare(source, plan):
    directory = RUN / 'data'
    check(not directory.exists(), 'Data already prepared; preserve existing run')
    paths = resolve_inputs(source)
    # This input contains only training and validation payloads.
    samples = pd.read_parquet(paths['allowed_samples']).sort_values('sequence_id')
    metadata = pd.read_parquet(paths['family_roles'], columns=['sequence_id', 'training_role_v2', 'component_v2'])
    core = json.loads(paths['core_split'].read_text())
    check(len(samples) == 1871 and samples.sequence_id.is_unique, 'Allowed membership')
    for role, count in [('training',1644), ('validation',227)]:
        selected = samples[samples.training_role_v2 == role]
        check(len(selected) == count and set(selected.sequence_id) == set(core['splits'][role]), 'Split identity')
    joined = samples.merge(metadata, on='sequence_id', suffixes=('', '_reference'), validate='one_to_one')
    check((joined.training_role_v2 == joined.training_role_v2_reference).all() and
          (joined.component_v2 == joined.component_v2_reference).all(), 'Role/component differs')
    check(not samples.source_annotation_conflict.any(), 'Annotation conflict')
    check(not set(samples[samples.training_role_v2=='training'].component_v2) &
          set(samples[samples.training_role_v2=='validation'].component_v2), 'Component leakage')
    legacy_implementation(source)
    from nad_qvae.design.sequence_corpus import align_to_wt
    directory.mkdir()
    rows = []
    for row in samples.itertuples():
        import hashlib
        check(hashlib.sha256(row.sequence.encode()).hexdigest() == row.sequence_id, 'Sequence identity')
        aligned, mapping, insertions = align_to_wt(wt_sequence(), row.sequence)
        check(len(aligned)==341 and aligned == ''.join(row.sequence[i] if i >= 0 else '-' for i in mapping),
              'Aligned source correspondence')
        observed = [i for i in mapping if i >= 0]
        check(set(observed).isdisjoint(insertions) and sorted(observed+insertions)==list(range(len(row.sequence))),
              'Source residues lost or duplicated')
        rows.append({'sequence_id':row.sequence_id, 'role':row.training_role_v2, 'component':row.component_v2,
                     'aligned_sequence':aligned, 'wt_to_source_index':mapping, 'insertions':insertions,
                     'source_length':len(row.sequence)})
    (directory/'aligned_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    condition_relative = 'experiments/run_reports/natv_sequence_design_dev_v1/condition/wt_condition.pt'
    receipt = json.loads((ROOT/'discovery/natv_model_inheritance/source_checks.json').read_text())
    condition_path = source/condition_relative
    check(sha256_file(condition_path)==receipt['inputs'][condition_relative]['sha256'], 'WT condition differs')
    cached = torch.load(condition_path, map_location='cpu', weights_only=True)
    check(cached['wt_sequence']==wt_sequence() and cached['wt_embeddings'].shape==(341,1280), 'WT identity/shape')
    # Copy only the fixed WT embedding, not the old checkpoints or corpus.
    torch.save(cached['wt_embeddings'],directory/'wt_embeddings.pt')
    train = [r for r in rows if r['role']=='training']
    counts = Counter(r['component'] for r in train)
    weights = [1/len(counts)/counts[r['component']] for r in train]
    np.save(directory/'training_weights.npy',np.array(weights))
    write_json(directory/'input_checks.json', {
        'source_root':str(source),'WT_sha256':WT_SHA256,'role_counts':dict(Counter(r['role'] for r in rows)),
        'component_counts':{role:len({r['component'] for r in rows if r['role']==role}) for role in ['training','validation']},
        'inserted_residues_retained':sum(len(r['insertions']) for r in rows),
        'gap_columns':sum(r['aligned_sequence'].count('-') for r in rows),
        'inputs':{k:{'path':str(v),'sha256':sha256_file(v)} for k,v in paths.items()},
        'condition_source':{'path':str(condition_path),'sha256':sha256_file(condition_path)},
        'outputs':{p.name:sha256_file(p) for p in directory.iterdir() if p.is_file()}})
    print('Prepared 1644 training / 227 validation WT-column inputs',flush=True)


def load_data():
    directory=RUN/'data'
    receipt=json.loads((directory/'input_checks.json').read_text())
    for name, expected in receipt['outputs'].items():
        check(sha256_file(directory/name)==expected,'Prepared data differs: '+name)
    rows=[json.loads(line) for line in (directory/'aligned_samples.jsonl').read_text().splitlines()]
    groups={role:[r for r in rows if r['role']==role] for role in ['training','validation']}
    tensors={role:torch.tensor([[ALPHABET.index(a) for a in r['aligned_sequence']] for r in group])
             for role,group in groups.items()}
    embedding=torch.load(directory/'wt_embeddings.pt',map_location='cpu',weights_only=True)
    return groups,tensors,embedding,np.load(directory/'training_weights.npy')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


@torch.no_grad()
def validation(model, targets, rows, epoch):
    model.eval()
    records=[]
    generator=torch.Generator(device='cpu').manual_seed(91000)
    # Reset RNG so validation Gumbel draws do not perturb training streams.
    with torch.random.fork_rng(devices=[targets.device.index] if targets.is_cuda else []):
        torch.manual_seed(92000)
        for start in range(0,len(targets),64):
            target=targets[start:start+64]
            mask=(torch.rand(target.shape,generator=generator).to(target.device)<.15)&(target<GAP)
            logits,kl,_=model(target.masked_fill(mask,MASK))
            known=target!=UNKNOWN
            per=torch.nn.functional.cross_entropy(logits.transpose(1,2),target.masked_fill(~known,-100),ignore_index=-100,reduction='none')
            ce=(per*known).sum(1)/known.sum(1)
            bound=ce+kl/known.sum(1)
            for row,c,k,b in zip(rows[start:start+64],ce.tolist(),kl.tolist(),bound.tolist()):
                records.append({'component':row['component'],'ce':c,'kl':k,'negative_elbo':b})
    return float(pd.DataFrame(records).groupby('component').negative_elbo.mean().mean())


def train(source,plan):
    groups,tensors,embedding,weights=load_data()
    device=torch.device('cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>8*1024**3 else 'cpu')
    tensors={role:value.to(device) for role,value in tensors.items()}
    directory=RUN/'training';check(not directory.exists(),'Refusing to overwrite training')
    directory.mkdir()
    config=plan['training'];steps=math.ceil(len(groups['training'])/config['batch_size'])
    identity={'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'plan_sha256':sha256_file(RUN/'plan.json'),'runner_sha256':sha256_file(Path(__file__)),
              'model_wrapper_sha256':sha256_file(ROOT/'src/natvqvae/aligned_generator.py'),
              'input_checks_sha256':sha256_file(RUN/'data/input_checks.json'),
              'source_receipt_sha256':sha256_file(ROOT/'discovery/natv_model_inheritance/source_checks.json'),
              'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
              'torch':str(torch.__version__),'threads':torch.get_num_threads()}
    write_json(directory/'identity.json',identity)
    summaries=[]
    for seed in plan['seeds']:
        models={};optimizers={};best={arm:float('inf') for arm in plan['arms']};stale=dict.fromkeys(plan['arms'],0)
        best_epochs={};elapsed=dict.fromkeys(plan['arms'],0.);rng_states={};logs={};started=time.monotonic()
        for arm in plan['arms']:
            seed_all(seed)
            models[arm]=AlignedGenerator(source,plan['model'],embedding,arm).to(device)
            optimizers[arm]=torch.optim.AdamW(models[arm].parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            rng_states[arm]=(torch.get_rng_state(),torch.cuda.get_rng_state() if device.type=='cuda' else None)
            attempt=directory/f'{arm}_seed{seed}';attempt.mkdir()
            logs[arm]=(attempt/'training_log.jsonl').open('x')
            write_json(attempt/'manifest.json',{'arm':arm,'seed':seed,'parameters':sum(p.numel() for p in models[arm].parameters()),
                        'training_sequences':1644,'validation_sequences':227,'steps_per_epoch':steps,**identity})
        check(all(torch.equal(a,b) for a,b in zip(models['categorical_vae'].parameters(),models['no_latent'].parameters())),
              'Paired initialization differs')
        for epoch in range(1,config['max_epochs']+1):
            # Re-create the same data stream for both arms, independent of torch dropout/Gumbel draws.
            epoch_seed=seed*100000+epoch
            for arm in plan['arms']:
                tick=time.monotonic();torch.set_rng_state(rng_states[arm][0])
                if device.type=='cuda':torch.cuda.set_rng_state(rng_states[arm][1])
                model=models[arm];model.train()
                rng=np.random.default_rng(epoch_seed);losses=[];kls=[];beta=min(1.,epoch/config['kl_warmup_epochs'])
                for step in range(steps):
                    indices=rng.choice(len(weights),config['batch_size'],p=weights)
                    target=tensors['training'][indices]
                    mask=torch.tensor(rng.random(target.shape)<config['mask_fraction'],device=device)&(target<GAP)
                    logits,kl,_=model(target.masked_fill(mask,MASK),config['temperature'])
                    loss,ce,regularizer=model.objective(logits,target,kl,beta)
                    check(torch.isfinite(loss).item(),'Nonfinite training objective')
                    optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();losses.append(float(loss.detach()));kls.append(float(regularizer.detach()))
                score=validation(model,tensors['validation'],groups['validation'],epoch)
                if score<best[arm]-1e-5:
                    best[arm]=score;stale[arm]=0;best_epochs[arm]=epoch
                    torch.save({'state_dict':model.state_dict(),'arm':arm,'seed':seed,'epoch':epoch,'model_config':plan['model'],
                                'identity':identity},directory/f'{arm}_seed{seed}/model_best.pt')
                else:stale[arm]+=1
                rng_states[arm]=(torch.get_rng_state(),torch.cuda.get_rng_state() if device.type=='cuda' else None);elapsed[arm]+=time.monotonic()-tick
                record={'epoch':epoch,'optimizer_steps':epoch*steps,'training_loss':float(np.mean(losses)),
                        'kl_per_column':float(np.mean(kls)),'validation_negative_elbo':score,
                        'beta':beta,'seconds':time.monotonic()-tick}
                logs[arm].write(json.dumps(record)+'\n');logs[arm].flush()
                if epoch==1 or epoch%5==0:print(json.dumps({'arm':arm,'seed':seed,**record}),flush=True)
            if epoch>=config['min_epochs'] and all(stale[a]>=config['patience'] for a in plan['arms']):break
        for arm in plan['arms']:
            logs[arm].close();path=directory/f'{arm}_seed{seed}/model_best.pt'
            summary={'arm':arm,'seed':seed,'epochs':epoch,'optimizer_steps':epoch*steps,'best_epoch':best_epochs[arm],
                     'best_validation_negative_elbo':best[arm],'seconds':elapsed[arm],'checkpoint':str(path.relative_to(RUN)),
                     'sha256':sha256_file(path)}
            summaries.append(summary);write_json(path.parent/'summary.json',summary)
        print(f'Paired seed {seed} finished in {time.monotonic()-started:.1f}s',flush=True)
    write_json(directory/'checkpoint_freeze.json',{'identity':identity,'runs':summaries})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train','evaluate'])
    parser.add_argument('--root',type=Path);args=parser.parse_args()
    plan=json.loads((RUN/'plan.json').read_text());source=(args.root or source_root(load_manifest())).resolve()
    torch.set_num_threads(plan['compute']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.stage=='prepare':prepare(source,plan)
    elif args.stage=='train':train(source,plan)
    else:
        from evaluate import evaluate
        evaluate(source,plan)


if __name__=='__main__':main()
