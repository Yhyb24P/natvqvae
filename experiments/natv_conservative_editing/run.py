"""Generate matched conservative substitutions and sourced-event repair trajectories."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse
from collections import defaultdict
import gzip,hashlib,json,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from natvqvae.intake import sha256_file
from natvqvae.sequence import wt_sequence
from natvqvae.conservative_editing import scaffold,current_positions,constrained_letters,edit_distance,legacy_scripts
from natvqvae.conditional_backend import ConditionalBackend

RUN=Path(__file__).parent

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def group_plan(plan,events):
    result=[];wt=wt_sequence();protected={p-1 for p in plan['protected_WT_positions_1based']}
    for seed in plan['seeds']:
        rng=np.random.default_rng(seed+20261021)
        def add(mode,budget,group,event=None,source_member=None):
            base=scaffold(event);amount=budget-(event['length'] if event else 0)
            eligible=[p for p in range(341) if p not in protected and base.wt_to_current[p] is not None]
            local=[]
            if event:
                near=[p for p in eligible if abs(p-(event['anchor']-1))<=12]
                local=list(map(int,rng.choice(near,min((amount+1)//2,len(near)),replace=False)))
            rest=[p for p in eligible if p not in local]
            sites=sorted(local+list(map(int,rng.choice(rest,amount-len(local),replace=False))))
            stages=[sites]+[sites[:len(sites)//2] if i%2==0 else sites[len(sites)//2:] for i in range(2*plan['update_sweeps'])]
            # Every selected site is non-WT throughout; random numbers are paired across arms.
            draws=rng.random((plan['draws_per_shared_mask'],len(stages),len(sites))).tolist()
            result.append({'group_id':f'{mode}_s{seed}_b{budget}_g{group}','mode':mode,'seed':seed,'budget':budget,
                           'event':event,'event_source_member':source_member,'WT_positions_0based':sites,
                           'update_WT_positions_0based':stages,'uniforms':draws,'prior_seed':int(rng.integers(1,2**31-1))})
        for budget in plan['substitution_budgets']:
            for group in range(plan['shared_masks_per_substitution_budget']):add('substitution',budget,group)
        for budget in plan['indel_total_edit_budgets']:
            for group,event in enumerate(events):
                by_component=defaultdict(list)
                for supporter in event['current_train_support']:by_component[supporter['component']].append(supporter)
                component=rng.choice(sorted(by_component));donors=by_component[component];donor=donors[int(rng.integers(len(donors)))]
                selected={k:event[k] for k in ['type','anchor','length']};selected['sequence']=donor['inserted_or_deleted_sequence']
                add('sourced_indel_repair',budget,group,selected,donor['sequence_id'])
    return result


def freeze_plan(plan):
    path=RUN/'sampling_groups.json'
    if path.exists():raise ValueError('Sampling plan already frozen')
    events=json.loads((RUN/'inputs/current_event_support.json').read_text())['retained']
    groups=group_plan(plan,events);write(path,groups)
    code=[Path(__file__),RUN/'evaluate.py',ROOT/'src/natvqvae/family_screen.py',ROOT/'src/natvqvae/conservative_editing.py',ROOT/'src/natvqvae/conditional_backend.py']
    files=code+[ROOT/'inputs/manifest.json',ROOT/'contracts/objective.yaml',RUN/'plan.json',RUN/'sampling_groups.json']+sorted((RUN/'inputs').glob('*'))
    write(RUN/'run_identity.json',{'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':{str(p.relative_to(ROOT)):sha256_file(p) for p in files if p.is_file()},
                                 'groups':len(groups),'attempts':len(groups)*plan['draws_per_shared_mask']*len(plan['arms'])})
    return groups


def verify_identity():
    identity=json.loads((RUN/'run_identity.json').read_text())
    for path,expected in identity['files'].items():
        if sha256_file(ROOT/path)!=expected:raise ValueError('Run identity changed: '+path)
    return identity


def run_group(backend,arm,g,plan,destination,edit_script,apply_script):
    count=plan['draws_per_shared_mask'];base=scaffold(g['event']);current=[base.sequence]*count
    model=backend.model(arm,g['seed']);z=None
    if arm=='QVAE':
        generator=torch.Generator(device='cuda').manual_seed(g['prior_seed'])
        z=model.sample_prior(count,generator=generator,device='cuda')
    latent_hashes=['' if z is None else hashlib.sha256(z[i].cpu().numpy().tobytes()).hexdigest() for i in range(count)]
    traces=[];diagnostics=[];start=time.monotonic()
    for stage,wt_sites in enumerate(g['update_WT_positions_0based']):
        positions=current_positions(base,wt_sites);masks=[positions]*count
        encoded,evolution,receipts,unique = yield (current,masks,destination/f'stage{stage}')
        logits,full=backend.logits(arm,model,encoded,evolution,z)
        if arm=='QVAE' and stage==len(g['update_WT_positions_0based'])-1:
            with torch.no_grad():
                disabled=model(encoded,encoded['target_mask'],evolution=evolution,disable_latent=True)['logits']
            diagnostics.append({'stage':stage,'latent_disabled_logit_max_difference':float((logits-disabled).abs().max()),
                                'interpretation':'mechanism_probe_not_generation_gain'})
        for i in range(count):
            before=current[i]
            letters=constrained_letters(logits[i,positions].detach().float().cpu().numpy(),
                        [wt_sequence()[p] for p in wt_sites],g['uniforms'][i][stage][:len(positions)],plan['temperature'])
            new=list(before)
            for pos,letter in zip(positions,letters):new[pos]=letter
            current[i]=''.join(new)
            traces.append({'arm':arm,'group_id':g['group_id'],'draw':i,'stage':stage,'WT_positions_0based':wt_sites,
                           'current_positions_0based':positions,'before':before,'after':current[i],
                           'masked_query':''.join('X' if j in positions else aa for j,aa in enumerate(before)),
                           'uniform_draws':g['uniforms'][i][stage][:len(positions)],'latent_sha256':latent_hashes[i],
                           'retrieval':receipts[i],'fresh_current_context':True,'unique_batch_contexts':unique})
        del encoded,evolution,logits,full
    rows=[];scripts=[]
    for draw,sequence in enumerate(current):
        candidate=f'{arm}_{g["group_id"]}_d{draw:02d}';distance=edit_distance(sequence,wt_sequence())
        similarity=1-distance/max(len(sequence),341)
        actual=[p for p,index in enumerate(base.wt_to_current) if index is not None and sequence[index]!=wt_sequence()[p]]
        operations=edit_script(wt_sequence(),sequence)
        if apply_script(wt_sequence(),operations)!=sequence:raise ValueError('Explicit edit reconstruction failed')
        valid=(set(actual)==set(g['WT_positions_0based']) and similarity>=plan['candidate_identity_min'] and distance>0)
        rows.append({'candidate_id':candidate,'arm':arm,'group_id':g['group_id'],'seed':g['seed'],'mode':g['mode'],
                     'edit_budget':g['budget'],'draw':draw,'sequence':sequence,'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),
                     'length':len(sequence),'Levenshtein_distance':distance,'WT_global_identity':similarity,
                     'substitutions':len(actual),'WT_substitution_positions':json.dumps([p+1 for p in actual]),
                     'event':json.dumps(g['event']),'event_source_member':g['event_source_member'] or '',
                     'latent_sha256':latent_hashes[draw],'constraint_status':'PASS' if valid else 'FAIL',
                     'function_status':'UNKNOWN','indel_tolerance':'UNKNOWN' if g['event'] else 'not_applicable','error':''})
        scripts.append({'candidate_id':candidate,'operations':operations,'reconstructs_exactly':True,
                        'alignment':'legacy_global_BLOSUM62; Levenshtein_distance_computed_separately',
                        'construction_event':g['event'],'WT_to_scaffold_current':base.wt_to_current})
    with gzip.open(destination/'trajectory.jsonl.gz','wt') as handle:
        for row in traces:handle.write(json.dumps(row)+'\n')
    with gzip.open(destination/'edit_scripts.jsonl.gz','wt') as handle:
        for row in scripts:handle.write(json.dumps(row)+'\n')
    write(destination/'latent_diagnostics.json',diagnostics)
    write(destination/'results.json',{'rows':rows,'seconds':time.monotonic()-start,
               'artifacts':{p.name:sha256_file(p) for p in destination.glob('*.gz')}})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    plan=json.loads((RUN/'plan.json').read_text());torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.resume:
        verify_identity();groups=json.loads((RUN/'sampling_groups.json').read_text())
    else:groups=freeze_plan(plan)
    source=Path(json.loads((RUN/'inputs/source_checks.json').read_text())['source_root'])
    if not torch.cuda.is_available():raise RuntimeError('This run requires an available GPU for frozen ESM-2')
    backend=ConditionalBackend(source,RUN);edit_script,apply_script=legacy_scripts(source)
    generation=RUN/'generation';generation.mkdir(exist_ok=args.resume)
    rows=[];start=time.monotonic();completed=0
    try:
        buckets=defaultdict(list)
        for group in groups:
            length=len(scaffold(group['event']).sequence)
            for arm in plan['arms']:buckets[(arm,group['seed'],length)].append(group)
        for (arm,seed,length),bucket in buckets.items():
            for offset in range(0,len(bucket),plan['execution']['groups_per_batch']):
                pending=[]
                for group in bucket[offset:offset+plan['execution']['groups_per_batch']]:
                    directory=generation/arm/group['group_id'];directory.parent.mkdir(exist_ok=True)
                    if directory.exists():
                        if not args.resume or not (directory/'results.json').exists():raise ValueError('Incomplete group requires inspection: '+str(directory))
                        record=json.loads((directory/'results.json').read_text())
                        for name,expected in record.get('artifacts',{}).items():
                            if sha256_file(directory/name)!=expected:raise ValueError('Group artifact changed')
                        rows.extend(record['rows']);completed+=1
                    else:
                        directory.mkdir()
                        job=run_group(backend,arm,group,plan,directory,edit_script,apply_script)
                        pending.append((job,next(job)))
                batch_folder=generation/arm/f'batch_s{seed}_l{length}_{offset}'
                stage=0
                while pending:
                    seqs=[];masks=[];sizes=[]
                    for job,(sequences,positions,folder) in pending:
                        sizes.append(len(sequences));seqs.extend(sequences);masks.extend(positions)
                    encoded,evolution,receipts,unique=backend.features(seqs,masks,batch_folder/f'stage{stage}',
                                             with_encoder=arm!='profile',with_retrieval=arm!='ESM2')
                    following=[];begin=0
                    for (job,request),size in zip(pending,sizes):
                        end=begin+size
                        enc=None if encoded is None else {k:v[begin:end] if isinstance(v,torch.Tensor) else v for k,v in encoded.items()}
                        response=(enc,evolution[begin:end],receipts[begin:end],unique)
                        try:following.append((job,job.send(response)))
                        except StopIteration as done:rows.extend(done.value);completed+=1
                        begin=end
                    pending=following;stage+=1
                    del encoded,evolution
                write(RUN/'run_status.json',{'status':'RUNNING','completed_groups':completed,'attempts':len(rows),
                          'elapsed_seconds':time.monotonic()-start,'last_arm':arm})
                print(json.dumps({'completed_groups':completed,'groups':len(groups)*len(plan['arms']),
                                  'attempts':len(rows),'arm':arm,'elapsed_seconds':time.monotonic()-start}),flush=True)
        frame=pd.DataFrame(rows);frame.to_csv(RUN/'raw_candidates.csv',index=False)
        if len(frame)!=json.loads((RUN/'run_identity.json').read_text())['attempts']:raise ValueError('Attempt denominator differs')
        write(RUN/'generation_receipt.json',{'loaded_models':backend.loaded,'imported_modules':backend.module_hashes,
                  'unique_actual_PLM_sequence_encodings':backend.forward_sequences,'retrieval_queries':backend.retrieval_sequences,
                  'stage_timing':backend.timing,'PLM_trainable_parameters':0,'elapsed_seconds':time.monotonic()-start,
                  'raw_candidates_sha256':sha256_file(RUN/'raw_candidates.csv')})
        write(RUN/'run_status.json',{'status':'GENERATION_COMPLETED','attempts':len(rows),'constraint_counts':frame.constraint_status.value_counts().to_dict()})
    except BaseException as exc:
        write(RUN/'run_status.json',{'status':'FAILED','completed_groups':completed,'attempts_preserved':len(rows),'error':repr(exc)})
        raise

if __name__=='__main__':main()
