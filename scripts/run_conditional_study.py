"""Fresh paired NatV fits and conditional generation; all outputs remain in this repo."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['HF_HUB_OFFLINE']='1'
from functools import lru_cache
import argparse,gzip,hashlib,json,subprocess,sys,time,traceback,signal
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
    source=Path(os.environ.get('NATV_SOURCE_ROOT',manifest['source_root_default']));backbone=source/'nad_models/esm2'
    files=[ROOT/'data/processed'/n for n in ['training.fasta','validation.fasta','samples.csv']]
    files += [backbone/n for n in ['config.json','model.safetensors','special_tokens_map.json','tokenizer_config.json','vocab.txt']]
    files += [Path(__file__),ROOT/'src/natvqvae/conditional_generation.py',run/'plan.json']
    status(run,'VERIFYING_INPUTS')
    hashes={str(p.resolve()):sha256_file(p) for p in files}
    previous=json.loads((ROOT/plan['reuse_features_from']/'inputs/identity.json').read_text())
    for path in files[:-3]:
        if hashes[str(path.resolve())]!=previous['sha256'][str(path.resolve())]:
            raise ValueError('Input differs from verified feature source: '+str(path))
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


def reuse_features(run,plan,rows):
    origin=ROOT/plan['reuse_features_from'];raw=ROOT/'data/results'/origin.name/'raw'
    snapshot=json.loads((raw/'result_manifest.json').read_text())
    specs=json.loads((raw/'masks.json').read_text())
    receipts=json.loads((raw/'feature_manifest.json').read_text())
    if sha256_file(raw/'feature_manifest.json')!=snapshot['inputs/feature_manifest.json'] or sha256_file(raw/'masks.json')!=snapshot['inputs/masks.json']:
        raise ValueError('Historical feature manifest or masks changed')
    if len(specs)!=len(rows)*3 or len({q['id'] for q in specs})!=len(specs):raise ValueError('Incomplete masks')
    rng=np.random.default_rng(plan['features']['seed']);expected=[]
    for row in rows.itertuples():
        for panel,fraction in enumerate(plan['features']['mask_fractions']):
            positions=sorted(map(int,rng.choice(len(row.sequence),max(1,round(len(row.sequence)*fraction)),replace=False)))
            expected.append({'id':f'{row.sequence_id}_p{panel}','sequence_id':row.sequence_id,'role':row.training_role_v2,
                'component':row.component_v2,'sequence':row.sequence,'panel':panel,'positions':positions})
    if expected!=specs:raise ValueError('Masks do not match current plan/population')
    paths={r['id']:ROOT/r['path'] for r in receipts}
    if set(paths)!={q['id'] for q in specs}:raise ValueError('Feature population differs')
    for i,r in enumerate(receipts):
        path=paths[r['id']]
        if not path.resolve().is_relative_to((ROOT/'data/interim'/origin.name).resolve()) or sha256_file(path)!=r['sha256']:
            raise ValueError('Feature identity differs: '+r['id'])
        if i%1000==0:status(run,'VERIFYING_CACHED_FEATURES',completed=i,total=len(receipts))
    destination=ROOT/'data/results'/run.name/'raw';destination.mkdir(parents=True,exist_ok=True)
    write(destination/'masks.json',specs);write(destination/'feature_manifest.json',receipts)
    write(run/'inputs/cache_identity.json',{'origin':str(origin.relative_to(ROOT)),
        'masks_sha256':sha256_file(raw/'masks.json'),'feature_manifest_sha256':sha256_file(raw/'feature_manifest.json'),
        'verified_feature_files':len(receipts),'cache_reused_without_copy':True})
    return specs,paths


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
    model.eval();records=[];probabilities=[]
    for begin in range(0,len(queries),64):
        batch=queries[begin:begin+64];h,b,v,m,labels=batch_features(batch,paths)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            context=model.encode_context(h,v)
            if model.arm=='QVAE':
                z=fixed_prior(batch,seed,model);prior=model.decode(context,b,v,z.flatten(1))
                scores=model.posterior_scores(context,v,m,labels);prob=scores.float().softmax(-1)
                mean_logits=model.decode(context,b,v,prob.to(context.dtype).flatten(1))
                uniforms=[]
                for query in batch:
                    code=int(hashlib.sha256(f'posterior:{seed}:{query["id"]}'.encode()).hexdigest()[:16],16)
                    uniforms.append(np.random.default_rng(code).random(model.factors))
                u=torch.tensor(np.array(uniforms),device='cuda')
                codes=(u[...,None]>=prob.cumsum(-1)).sum(-1).clamp_max(model.states-1)
                sample_logits=model.decode(context,b,v,F.one_hot(codes,model.states).to(context.dtype).flatten(1))
                ce=sequence_ce(prior,labels,m);mean_ce=sequence_ce(mean_logits,labels,m);sample_ce=sequence_ce(sample_logits,labels,m)
                kl=(prob*(prob.clamp_min(1e-12).log()+np.log(model.states))).sum(-1).mean(-1)
                probabilities.extend(prob.cpu().numpy())
            else:
                pooled=(context*v[...,None]).sum(1)/v.sum(1,keepdim=True)
                prior=model.decode(context,b,v,F.gelu(model.global_map(pooled)))
                ce=sequence_ce(prior,labels,m);mean_ce=sample_ce=ce;kl=torch.zeros_like(ce)
        for i,q in enumerate(batch):
            records.append({'id':q['id'],'component':q['component'],'panel':q['panel'],'CE':float(ce[i]),
                'posterior_CE':float(mean_ce[i]),'posterior_sample_CE':float(sample_ce[i]),'KL':float(kl[i])})
    frame=pd.DataFrame(records)
    def balanced(part,column):return float(part.groupby('component')[column].mean().mean())
    metrics={'prior_CE':balanced(frame,'CE'),'posterior_CE':balanced(frame,'posterior_CE'),
        'posterior_sample_CE':balanced(frame,'posterior_sample_CE'),'validation_KL':balanced(frame,'KL'),
        'panels':{str(panel):{'prior_CE':balanced(part,'CE'),'posterior_CE':balanced(part,'posterior_CE'),
            'posterior_sample_CE':balanced(part,'posterior_sample_CE')} for panel,part in frame.groupby('panel')}}
    if probabilities:
        probs=np.array(probabilities);aggregate=probs.mean(0)
        entropy=lambda p:-(p*np.log(np.maximum(p,1e-12))).sum(-1)
        conditional=entropy(probs).mean(0);marginal=entropy(aggregate)
        metrics['posterior_factors']={'conditional_entropy':conditional.tolist(),'aggregate_entropy':marginal.tolist(),
            'aggregate_minus_conditional_entropy':(marginal-conditional).tolist(),
            'argmax_category_counts':[[int((probs.argmax(-1)[:,f]==k).sum()) for k in range(model.states)] for f in range(model.factors)],
            'weighting':'unweighted_validation_inputs; descriptive_not_conditional_mutual_information'}
    return metrics,records


@torch.no_grad()
def latent_sensitivity(model,queries,paths,seed):
    model.eval();queries=queries[:24];h,b,v,m,labels=batch_features(queries,paths)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        context=model.encode_context(h,v);outputs=[]
        for draw in range(8):
            generator=torch.Generator(device='cuda').manual_seed(seed+10000+draw)
            z=model.sample_prior(len(queries),generator,'cuda')
            outputs.append(model.decode(context,b,v,z.flatten(1)).float().softmax(-1)[m])
    probabilities=torch.stack(outputs);reference=probabilities[0]
    tv=(probabilities[1:]-reference).abs().sum(-1)/2
    disagreement=(probabilities[1:].argmax(-1)!=reference.argmax(-1)).float()
    return {'inputs':[q['id'] for q in queries],'prior_draws':8,'prior_seeds':[seed+10000+i for i in range(8)],
        'masked_positions':int(m.sum()),'mean_TV_to_first_draw':float(tv.mean()),'max_TV_to_first_draw':float(tv.max()),
        'argmax_disagreement_to_first_draw':float(disagreement.mean()),'sampling':'prediction_probabilities_only_no_sequence_generation'}


def train(run,plan,specs,paths):
    cfg=plan['training'];models={};optimizers={};initial={};best={};history=[];stale={};stopped={};receipt={}
    model_root=ROOT/'data/models'/run.name/'checkpoints';raw=ROOT/'data/results'/run.name/'raw'
    for seed in plan['seeds']:
        for arm in plan['arms']:
            torch.manual_seed(seed);model=ConditionalModel(arm,**plan['model']).cuda()
            key=f'{arm}/{seed}';models[key]=model;optimizers[key]=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            shared={k:v for k,v in model.state_dict().items() if not k.startswith(('target_embedding.','posterior_encoder.'))}
            initial[key]=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in shared.values())).hexdigest()
            best[key]=float('inf');stale[key]=0
        if initial[f'QVAE/{seed}']!=initial[f'deterministic/{seed}']:raise ValueError('Shared initialization differs')
    write(run/'initialization.json',{'shared_state_hashes':initial,'parameters':{k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
        'legacy_NatV_checkpoint_loaded':False,'posterior_only_parameters_in_QVAE':True})
    train_rows=[q for q in specs if q['role']=='training'];validation=[q for q in specs if q['role']=='validation']
    counts=pd.DataFrame(train_rows).drop_duplicates('sequence_id').component.value_counts().to_dict()
    mean_weight=np.mean([1/counts[q['component']] for q in train_rows]);steps=0;started=time.monotonic()
    for epoch in range(cfg['epochs']):
        active=[key for key in models if key not in stopped]
        if not active:break
        panel=epoch%3;queries=[q for q in train_rows if q['panel']==panel]
        order=np.random.default_rng(plan['features']['seed']+epoch).permutation(len(queries));queries=[queries[i] for i in order]
        losses={k:np.zeros(3) for k in active}
        status(run,'TRAINING',epoch=epoch+1,total_epochs=cfg['epochs'],active_fits=active)
        for begin in range(0,len(queries),cfg['batch_size']):
            batch=queries[begin:begin+cfg['batch_size']];h,b,v,m,labels=batch_features(batch,paths)
            weights=torch.tensor([1/counts[q['component']]/mean_weight for q in batch],device='cuda')
            for key in active:
                model=models[key];model.train();optimizer=optimizers[key];optimizer.zero_grad(set_to_none=True)
                seed=int(key.split('/')[1]);torch.manual_seed(seed+steps)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,kl=model(h,b,v,m,labels=labels if model.arm=='QVAE' else None)
                    ce=(sequence_ce(logits,labels,m)*weights).mean()
                    coefficient=cfg['KL_coefficient']*min(1,(steps+1)/cfg['KL_warmup_steps'])
                    loss=ce+coefficient*kl
                if not torch.isfinite(loss):raise ValueError('Nonfinite objective: '+key)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient: '+key)
                optimizer.step();losses[key]+=np.array([float(ce.detach())*len(batch),float(kl.detach())*len(batch),len(batch)])
            steps+=1
        for key in active:
            model=models[key];seed=int(key.split('/')[1]);metrics,records=validate(model,validation,paths,seed)
            if not np.isfinite([metrics['prior_CE'],metrics['posterior_CE'],metrics['posterior_sample_CE']]).all():
                raise ValueError('Nonfinite validation: '+key)
            with gzip.open(raw/'validation_records.jsonl.gz','at') as handle:
                for record in records:handle.write(json.dumps({'key':key,'epoch':epoch+1,**record})+'\n')
            record={'key':key,'epoch':epoch+1,'training_panel':panel,'training_CE':float(losses[key][0]/losses[key][2]),
                'training_KL':float(losses[key][1]/losses[key][2]),'KL_coefficient':coefficient,**metrics};history.append(record)
            if metrics['prior_CE']<best[key]:
                best[key]=metrics['prior_CE'];stale[key]=0;path=model_root/key/'best.pt';path.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'state_dict':model.state_dict(),'arm':model.arm,'seed':seed,'epoch':epoch+1,
                    'validation':metrics,'model_config':plan['model']},path)
                receipt[key]={'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path),'selected_prior_CE':best[key],'epoch':epoch+1}
                write(run/'checkpoint_manifest.json',receipt)
            else:stale[key]+=1
            if stale[key]>=cfg['patience']:stopped[key]={'epoch':epoch+1,'reason':'early_stopping'}
            elif epoch+1==cfg['epochs']:stopped[key]={'epoch':epoch+1,'reason':'max_epochs'}
            write(run/'training_history.json',history)
        write(run/'fit_status.json',{'stopped':stopped,'best_CE':best,'stale_epochs':stale})
        print(json.dumps({'epoch':epoch+1,'validation':{k:history[-len(active)+i]['prior_CE'] for i,k in enumerate(active)},'seconds':time.monotonic()-started}),flush=True)
    diagnostics={}
    for key,model in models.items():
        if model.arm!='QVAE':continue
        saved=torch.load(ROOT/receipt[key]['path'],map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict'])
        diagnostics[key]={'checkpoint_sha256':receipt[key]['sha256'],**latent_sensitivity(model,validation,paths,int(key.split('/')[1]))}
    write(run/'latent_sensitivity.json',diagnostics)
    del models,optimizers;cached_feature.cache_clear();torch.cuda.empty_cache()
    return receipt


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
                key=f'{arm}/{seed}';entry=checkpoints[key];path=ROOT/entry['path']
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


class BudgetExpired(Exception):
    pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    if not run.is_relative_to(ROOT/'experiments'):raise ValueError('Run directory must be inside experiments/')
    if (run/'status.json').exists():raise ValueError('Existing run must be inspected, not overwritten')
    plan=json.loads((run/'plan.json').read_text())
    if plan['task']!='target_conditioned_posterior_training':raise ValueError('Use the recorded Git revision for historical runs')
    raw=ROOT/'data/results'/run.name/'raw';raw.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();outcome='FAILED'
    def deadline(signum,frame):raise BudgetExpired('Authorized wall-clock budget reached')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,plan['budget']['wall_seconds'])
    try:
        plan,rows,backbone=setup(run);specs,paths=reuse_features(run,plan,rows)
        train(run,plan,specs,paths);outcome='COMPLETED'
    except BudgetExpired as exc:
        outcome='BUDGET_EXHAUSTED';write(raw/'termination.json',{'reason':str(exc)})
    except BaseException as exc:
        write(raw/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        elapsed=time.monotonic()-started
        write(run/'cost.json',{'wall_seconds':elapsed,'wall_budget_seconds':plan['budget']['wall_seconds'],
            'single_gpu_allocation_seconds_upper_bound':elapsed,'gpu':'NVIDIA GeForce RTX 5090 D',
            'attempted_fits':6,'attempt':1,'outcome':outcome})
        write(run/'result_manifest.json',{'raw_files':{str(p.relative_to(ROOT)):sha256_file(p) for p in raw.iterdir() if p.is_file()},
            'source_files':{str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),ROOT/'src/natvqvae/conditional_generation.py',run/'plan.json']}})
        status(run,outcome,wall_seconds=elapsed)

if __name__=='__main__':main()
