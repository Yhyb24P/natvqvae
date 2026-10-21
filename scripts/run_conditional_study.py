"""Fresh paired NatV fits and conditional generation; all outputs remain in this repo."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['HF_HUB_OFFLINE']='1'
from functools import lru_cache
import argparse,gzip,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from natvqvae.conditional_generation import ConditionalModel,FrozenEncoder,AA,sample_letters
from natvqvae.sequence import wt_sequence
from natvqvae.intake import sha256_file


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def status(run,phase,**extra):
    write(run/'status.json',{'phase':phase,**extra});print(json.dumps({'phase':phase,**extra}),flush=True)


def setup(run):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    plan=json.loads((run/'plan.json').read_text());manifest=json.loads((ROOT/'data/manifest.json').read_text())
    source=Path(manifest['source_root_default']);backbone=source/'nad_models/esm2'
    files=[ROOT/'data/processed'/n for n in ['training.fasta','validation.fasta','samples.csv']]
    files += [backbone/n for n in ['config.json','model.safetensors','special_tokens_map.json','tokenizer_config.json','vocab.txt']]
    files += [Path(__file__),ROOT/'src/natvqvae/conditional_generation.py',run/'plan.json']
    status(run,'VERIFYING_INPUTS')
    hashes={str(p.resolve()):sha256_file(p) for p in files}
    role_entry=next(e for e in manifest['entries'] if e['name']=='family_roles')
    if sha256_file(source/role_entry['path'])!=role_entry['sha256']:raise ValueError('Source roles changed')
    formal=pd.read_parquet(source/role_entry['path'],columns=['sequence_id','training_role_v2','component_v2'])
    rows=pd.read_csv(ROOT/'data/processed/samples.csv');rows=rows[rows.training_role_v2.isin(['training','validation'])].copy()
    if rows.sequence_id.duplicated().any():raise ValueError('Duplicate sequence ids')
    for role,count in [('training',1644),('validation',227)]:
        frame=rows[rows.training_role_v2==role]
        if len(frame)!=count or set(frame.sequence_id)!=set(formal[formal.training_role_v2==role].sequence_id):raise ValueError('Role mismatch')
        # FASTA equality must include actual letters, not only names.
        from Bio import SeqIO
        fasta={r.id:str(r.seq) for r in SeqIO.parse(ROOT/f'data/processed/{role}.fasta','fasta')}
        if fasta!=frame.set_index('sequence_id').sequence.to_dict():raise ValueError('FASTA differs')
    expected=formal.set_index('sequence_id').component_v2
    if not rows.component_v2.eq(rows.sequence_id.map(expected)).all():raise ValueError('Component mapping differs')
    if set(rows[rows.training_role_v2=='training'].component_v2)&set(rows[rows.training_role_v2=='validation'].component_v2):raise ValueError('Component leakage')
    for r in rows.itertuples():
        if hashlib.sha256(r.sequence.encode()).hexdigest()!=r.sequence_id or not set(r.sequence)<=set(AA):raise ValueError('Sequence identity differs')
    write(run/'inputs/identity.json',{'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'sha256':hashes,'source_root':str(source),'NatV_checkpoint_loaded':False,'backbone':str(backbone),
            'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':torch.cuda.get_device_name(0)}})
    return plan,rows.sort_values('sequence_id').reset_index(drop=True),backbone


def features(run,plan,rows,backbone):
    cache=ROOT/'data/interim'/run.name/'features';cache.mkdir(parents=True,exist_ok=False)
    specs=[];rng=np.random.default_rng(plan['features']['seed'])
    for row in rows.itertuples():
        for panel,fraction in enumerate(plan['features']['mask_fractions']):
            count=max(1,round(len(row.sequence)*fraction));pos=sorted(map(int,rng.choice(len(row.sequence),count,replace=False)))
            specs.append({'id':f'{row.sequence_id}_p{panel}','sequence_id':row.sequence_id,'role':row.training_role_v2,
                         'component':row.component_v2,'sequence':row.sequence,'panel':panel,'positions':pos})
    write(run/'inputs/masks.json',specs)
    encoder=FrozenEncoder(backbone);batch_size=plan['features']['batch_size'];receipts=[];started=time.monotonic()
    status(run,'PREPARING_MASKED_FEATURES',completed=0,total=len(specs))
    for begin in range(0,len(specs),batch_size):
        batch=specs[begin:begin+batch_size];seqs=[q['sequence'] for q in batch]
        h,b,v,m=encoder(seqs,[q['positions'] for q in batch])
        if begin==0:
            checks={}
            for arm in plan['arms']:
                torch.manual_seed(plan['seeds'][0]);probe=ConditionalModel(arm,**plan['model']).cuda()
                optimizer=torch.optim.AdamW(probe.parameters(),lr=1e-4)
                for step in range(2):
                    optimizer.zero_grad()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,kl=probe(h[:2].clone(),b[:2].clone(),v[:2].clone(),m[:2].clone())
                        loss=logits.square().mean()+.01*kl
                    loss.backward();optimizer.step()
                if probe.global_map.weight.grad.abs().sum()==0 or probe.global_projection.weight.grad.abs().sum()==0:raise ValueError('Disconnected global branch')
                probe.eval();z=probe.sample_prior(2,torch.Generator(device='cuda').manual_seed(1),'cuda') if arm=='QVAE' else None
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output,_=probe(h[:2],b[:2],v[:2],m[:2],z=z)
                if not torch.isfinite(output).all():raise ValueError('Nonfinite GPU inference')
                checks[arm]={'active_parameters':sum(p.numel() for p in probe.parameters()),'global_branch_gradient_nonzero':True,'actual_masked_backbone_forward_finite':True}
                del probe,optimizer
            write(run/'gpu_checks.json',checks)
        for i,q in enumerate(batch):
            n=len(q['sequence']);path=cache/(q['id']+'.pt')
            labels=torch.tensor([AA.index(a) for a in q['sequence']],dtype=torch.long)
            torch.save({'features':h[i,:n].cpu().clone(),'baseline':b[i,:n].cpu().clone(),'mask':m[i,:n].cpu().clone(),'labels':labels},path)
            receipts.append({'id':q['id'],'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path)})
        if begin%128==0 or begin+batch_size>=len(specs):status(run,'PREPARING_MASKED_FEATURES',completed=min(begin+batch_size,len(specs)),total=len(specs),seconds=time.monotonic()-started)
    write(run/'inputs/feature_manifest.json',receipts)
    del encoder;torch.cuda.empty_cache()
    return specs,{r['id']:ROOT/r['path'] for r in receipts}


@lru_cache(maxsize=6000)
def cached_feature(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def batch_features(specs,paths):
    saved=[cached_feature(paths[q['id']]) for q in specs]
    n=len(saved);width=max(len(s['labels']) for s in saved)
    h=torch.zeros(n,width,1280,dtype=torch.bfloat16);b=torch.zeros(n,width,20,dtype=torch.bfloat16)
    valid=torch.zeros(n,width,dtype=torch.bool);mask=valid.clone();labels=torch.zeros(n,width,dtype=torch.long)
    for i,s in enumerate(saved):
        size=len(s['labels']);h[i,:size]=s['features'];b[i,:size]=s['baseline'];mask[i,:size]=s['mask'];labels[i,:size]=s['labels'];valid[i,:size]=True
    return [x.cuda() for x in [h,b,valid,mask,labels]]


def sequence_ce(logits,labels,mask):
    ce=F.cross_entropy(logits.transpose(1,2),labels,reduction='none')
    return (ce*mask).sum(1)/mask.sum(1)


def fixed_prior(specs,seed,model):
    codes=[]
    for q in specs:
        value=int(hashlib.sha256(f'{seed}:{q["id"]}'.encode()).hexdigest()[:16],16)
        codes.append(np.random.default_rng(value).integers(model.states,size=model.factors))
    return F.one_hot(torch.tensor(np.stack(codes),device='cuda'),model.states).float()


@torch.no_grad()
def validate(model,queries,paths,seed):
    model.eval();records=[]
    for begin in range(0,len(queries),64):
        batch=queries[begin:begin+64];h,b,v,m,labels=batch_features(batch,paths)
        z=fixed_prior(batch,seed,model) if model.arm=='QVAE' else None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,kl=model(h,b,v,m,z=z);ce=sequence_ce(logits,labels,m)
        posterior=None
        if model.arm=='QVAE':
            with torch.autocast('cuda',dtype=torch.bfloat16):posterior=sequence_ce(model(h,b,v,m,z='posterior')[0],labels,m)
        for i,q in enumerate(batch):records.append({'component':q['component'],'CE':float(ce[i]),'posterior_CE':float(posterior[i]) if posterior is not None else float(ce[i])})
    frame=pd.DataFrame(records)
    return {'prior_CE':float(frame.groupby('component').CE.mean().mean()),'posterior_CE':float(frame.groupby('component').posterior_CE.mean().mean())}


def train(run,plan,specs,paths):
    cfg=plan['training'];models={};optimizers={};initial={};best={};history=[]
    for seed in plan['seeds']:
        for arm in plan['arms']:
            torch.manual_seed(seed);model=ConditionalModel(arm,**plan['model']).cuda()
            key=f'{arm}/{seed}';models[key]=model;optimizers[key]=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            initial[key]=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            best[key]=float('inf')
        if initial[f'QVAE/{seed}']!=initial[f'deterministic/{seed}']:raise ValueError('Unmatched initial state')
    counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}
    if len(set(counts.values()))!=1:raise ValueError('Unequal active parameter counts')
    write(run/'initialization.json',{'state_hashes':initial,'parameters':counts,'legacy_NatV_checkpoint_loaded':False})
    train_rows=[q for q in specs if q['role']=='training'];validation=[q for q in specs if q['role']=='validation']
    components=pd.DataFrame(train_rows).drop_duplicates('sequence_id').component.value_counts().to_dict()
    mean_weight=np.mean([1/components[q['component']] for q in train_rows])
    steps=0;started=time.monotonic()
    for epoch in range(cfg['epochs']):
        panel=epoch%3;queries=[q for q in train_rows if q['panel']==panel]
        order=np.random.default_rng(plan['features']['seed']+epoch).permutation(len(queries));queries=[queries[i] for i in order]
        losses={k:[] for k in models}
        status(run,'TRAINING',epoch=epoch+1,total_epochs=cfg['epochs'],completed_batches=0)
        for begin in range(0,len(queries),cfg['batch_size']):
            batch=queries[begin:begin+cfg['batch_size']];h,b,v,m,labels=batch_features(batch,paths)
            weights=torch.tensor([1/components[q['component']]/mean_weight for q in batch],device='cuda')
            for key,model in models.items():
                model.train();optimizer=optimizers[key];optimizer.zero_grad(set_to_none=True)
                seed=int(key.split('/')[1]);torch.manual_seed(seed+steps)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,kl=model(h,b,v,m);ce=(sequence_ce(logits,labels,m)*weights).mean()
                coefficient=cfg['KL_coefficient']*min(1,(steps+1)/cfg['KL_warmup_steps'])
                loss=ce+coefficient*kl
                if not torch.isfinite(loss):raise ValueError('Nonfinite objective: '+key)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient: '+key)
                optimizer.step();losses[key].append([float(ce.detach()),float(kl.detach())])
            steps+=1
            if begin%(cfg['batch_size']*10)==0:status(run,'TRAINING',epoch=epoch+1,total_epochs=cfg['epochs'],completed_batches=begin//cfg['batch_size']+1,seconds=time.monotonic()-started)
        for key,model in models.items():
            seed=int(key.split('/')[1]);metrics=validate(model,validation,paths,seed)
            record={'key':key,'epoch':epoch+1,'training_CE':float(np.mean(losses[key],axis=0)[0]),'training_KL':float(np.mean(losses[key],axis=0)[1]),**metrics};history.append(record)
            if metrics['prior_CE']<best[key]:
                best[key]=metrics['prior_CE'];path=run/'checkpoints'/key/'best.pt';path.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'state_dict':model.state_dict(),'arm':model.arm,'seed':seed,'epoch':epoch+1,'validation':metrics,'model_config':plan['model']},path)
        write(run/'training_history.json',history)
        print(json.dumps({'epoch':epoch+1,'validation':{k:best[k] for k in models}}),flush=True)
    receipt={}
    for key in models:
        path=run/'checkpoints'/key/'best.pt';receipt[key]={'path':str(path.relative_to(run)),'sha256':sha256_file(path),'selected_prior_CE':best[key]}
    write(run/'checkpoint_manifest.json',receipt)
    del models,optimizers;torch.cuda.empty_cache();return receipt


def generate(run,plan,backbone,checkpoints):
    cfg=plan['generation'];wt=wt_sequence();rng=np.random.default_rng(plan['features']['seed']+1234);groups=[]
    for seed in plan['seeds']:
        for region,(low,high) in cfg['regions_1based'].items():
            for size in cfg['mask_sizes'][region]:
                for repeat in range(cfg['shared_masks_per_condition']):
                    sites=sorted(map(int,rng.choice(np.arange(low-1,high),size,replace=False)))
                    groups.append({'id':f'{region}_s{seed}_m{size}_r{repeat}','seed':seed,'region':region,'sites':sites,
                         'prior_seed':int(rng.integers(2**31)),'uniforms':rng.random((cfg['draws_per_mask'],cfg['update_steps'],size)).tolist()})
    write(run/'generation_groups.json',groups)
    encoder=FrozenEncoder(backbone);rows=[];ledger=[];prior_records=[];started=time.monotonic()
    with gzip.open(run/'trajectories.jsonl.gz','wt') as trace:
        for seed in plan['seeds']:
            for arm in plan['arms']:
                key=f'{arm}/{seed}';entry=checkpoints[key];path=run/entry['path']
                if sha256_file(path)!=entry['sha256']:raise ValueError('Checkpoint changed')
                saved=torch.load(path,map_location='cpu',weights_only=True);model=ConditionalModel(arm,**plan['model']).cuda().eval();model.load_state_dict(saved['state_dict'],strict=True)
                selected=[g for g in groups if g['seed']==seed]
                for begin in range(0,len(selected),cfg['groups_per_batch']):
                    batch=selected[begin:begin+cfg['groups_per_batch']];n=cfg['draws_per_mask'];parents=[wt]*(len(batch)*n)
                    zparts=[]
                    if arm=='QVAE':
                        for g in batch:
                            generator=torch.Generator(device='cuda').manual_seed(g['prior_seed']);zparts.append(model.sample_prior(n,generator,'cuda'))
                        z=torch.cat(zparts)
                    else:z=None
                    if z is not None:
                        for i,g in enumerate(batch):prior_records.append({'group':g['id'],'codes':z[i*n:(i+1)*n].argmax(-1).cpu().tolist()})
                        write(run/'prior_samples.json',prior_records)
                    prior_sha=['' if z is None else hashlib.sha256(z[i].cpu().numpy().tobytes()).hexdigest() for i in range(len(parents))]
                    for stage in range(cfg['update_steps']):
                        masks=[]
                        for g in batch:
                            sites=g['sites'];pos=sites if stage==0 else sites[:len(sites)//2] if stage%2 else sites[len(sites)//2:];masks.extend([pos]*n)
                        # Deduplicate identical masked inputs within this actual batch only.
                        unique=[];lookup={};inverse=[]
                        for seq,mask in zip(parents,masks):
                            query=(seq,tuple(mask))
                            if query not in lookup:lookup[query]=len(unique);unique.append(query)
                            inverse.append(lookup[query])
                        h,b,v,m=encoder([s for s,p in unique],[p for s,p in unique]);idx=torch.tensor(inverse,device='cuda')
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(h[idx],b[idx],v[idx],m[idx],z=z)
                        for i,(parent,pos) in enumerate(zip(parents,masks)):
                            g=batch[i//n];draw=i%n;u=np.asarray(g['uniforms'][draw][stage][:len(pos)])
                            letters=sample_letters(logits[i,pos].cpu().numpy(),u,cfg['temperature']);chars=list(parent)
                            for p,a in zip(pos,letters):chars[p]=str(a)
                            after=''.join(chars);trace.write(json.dumps({'arm':arm,'group':g['id'],'draw':draw,'stage':stage,'positions_0based':pos,'before':parent,'after':after,'uniforms':u.tolist(),'prior_sha256':prior_sha[i]})+'\n');parents[i]=after
                        del h,b,v,m,logits
                    for i,sequence in enumerate(parents):
                        g=batch[i//n];edits=[{'position':p+1,'from':a,'to':b} for p,(a,b) in enumerate(zip(wt,sequence)) if a!=b]
                        # With substitutions, Levenshtein may be smaller than Hamming; compute independently.
                        from Bio.Align import PairwiseAligner
                        aligner=PairwiseAligner();aligner.mode='global';aligner.match_score=0;aligner.mismatch_score=-1;aligner.open_gap_score=-1;aligner.extend_gap_score=-1
                        distance=int(round(-aligner.score(wt,sequence)));identity=1-distance/341
                        reason='WT_reference' if not edits else 'outside_design_space' if identity<cfg['identity_min'] else ''
                        candidate=f'{arm}_{g["id"]}_d{i%n:02d}'
                        rows.append({'candidate_id':candidate,'arm':arm,'seed':seed,'group':g['id'],'region':g['region'],'mask_size':len(g['sites']),
                            'draw':i%n,'sequence':sequence,'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),
                            'actual_substitutions':len(edits),'edit_distance':distance,'WT_identity':identity,'reason':reason,'prior_sha256':prior_sha[i],
                            'function':'UNKNOWN','edits':json.dumps(edits)})
                    ledger.append({'arm':arm,'groups':[g['id'] for g in batch],'attempts':len(parents)})
                    pd.DataFrame(rows).to_csv(run/'candidates.csv',index=False)
                    status(run,'GENERATING',completed_attempts=len(rows),total=cfg['attempts'],seconds=time.monotonic()-started)
                del model;torch.cuda.empty_cache()
    if len(rows)!=cfg['attempts']:raise ValueError('Attempt denominator differs')
    write(run/'generation_ledger.json',ledger)
    (run/'candidates.fasta').write_text(''.join(f'>{r["candidate_id"]}\n{r["sequence"]}\n' for r in rows))
    summarize(run,plan,pd.DataFrame(rows))


def summarize(run,plan,frame):
    frame['duplicate_within_arm']=frame.duplicated(['arm','sequence_sha256']);frame.to_csv(run/'candidates.csv',index=False)
    frame.groupby(['arm','region','mask_size','edit_distance','reason'],dropna=False).size().rename('attempts').reset_index().to_csv(run/'distance_coverage.csv',index=False)
    good=frame[frame.reason.eq('')];sets={arm:set(good[good.arm==arm].sequence_sha256) for arm in plan['arms']};shared=set.intersection(*sets.values())
    write(run/'coverage_summary.json',{'attempts':len(frame),'by_arm':{arm:{'attempts':int((frame.arm==arm).sum()),'unique_nonWT_in_space':len(sets[arm]),'shared':len(shared),'model_specific':len(sets[arm]-shared),
             'duplicate_attempts':int(frame[frame.arm==arm].duplicate_within_arm.sum()),'reasons':frame[frame.arm==arm].reason.value_counts().to_dict()} for arm in sets},'function':'UNKNOWN'})
    # Fixed descriptive selection across realized region/distance cells; no score ranking.
    representatives=[]
    for (arm,region),cell in good.groupby(['arm','region']):
        unique=cell.drop_duplicates('sequence_sha256').sort_values(['edit_distance','sequence_sha256'])
        for fraction in [0.,.5,1.]:
            row=unique.iloc[round(fraction*(len(unique)-1))].to_dict();row['selection']='observed_region_distance_min_median_max; deterministic_hash_ties'
            row['structure_question']='candidate_vs_WT_full_length_fold_and_domain_organization; mapped_NMNAT_and_Nudix_environment_changes';representatives.append(row)
    if representatives:pd.DataFrame(representatives).drop_duplicates('candidate_id').to_csv(run/'structure_representatives.csv',index=False)
    else:pd.DataFrame(columns=['candidate_id','selection','structure_question']).to_csv(run/'structure_representatives.csv',index=False)
    write(run/'artifact_hashes.json',{p.name:sha256_file(p) for p in run.iterdir() if p.is_file() and p.name not in {'status.json','artifact_hashes.json','output.log'}})
    status(run,'COMPLETED',attempts=len(frame))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();run=args.run.resolve()
    if not run.is_relative_to(ROOT/'experiments'):raise ValueError('Run directory must be inside experiments/')
    if (run/'status.json').exists():raise ValueError('Existing run must be inspected, not overwritten')
    try:
        plan,rows,backbone=setup(run);specs,paths=features(run,plan,rows,backbone)
        receipt=train(run,plan,specs,paths);generate(run,plan,backbone,receipt)
    except BaseException as exc:
        status(run,'FAILED',error=repr(exc));raise

if __name__=='__main__':main()
