"""条件生成的同源联合统计、真实候选和研究交付。"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .conditional_data import ROOT, REGIONS, SIZES, stable_seed, write_json, status
from .conditional_generation import (AA, ConditionalModel, FrozenEncoder, categorical_samples,
                                     categorical_log_prob, target_tensors, sample_letters)
from .intake import sha256_file
from .sequence import wt_sequence


def mixture_statistics(probabilities, use=None):
    """K×T×A 条件概率的共享联合与同边缘独立分布（小字母表可穷举）。"""
    p = np.asarray(probabilities, dtype=np.float64)
    if use is not None: p = p[:, use]
    joint = p[:, 0]
    for i in range(1, p.shape[1]):
        joint = (joint[..., None]*p[:, i].reshape((len(p),)+(1,)*i+(p.shape[-1],)))
    shared = joint.mean(0)
    independent = p.mean(0)[0]
    for marginal in p.mean(0)[1:]: independent = independent[..., None]*marginal
    return shared, independent


def _domain_mapping(reference, sequence):
    from Bio.Align import PairwiseAligner, substitution_matrices
    aligner = PairwiseAligner(mode='global', substitution_matrix=substitution_matrices.load('BLOSUM62'))
    aligner.open_gap_score=-10; aligner.extend_gap_score=-.5
    alignments = aligner.align(reference, sequence)
    try: count=len(alignments)
    except OverflowError: count=129
    if count>128:
        return np.full(len(reference),-1,dtype=np.int16), {'optimal_alignments':'>128','reliable_positions':0}
    maps=[]
    for alignment in alignments:
        mapping=np.full(len(reference),-1,dtype=np.int16)
        for i,j in alignment.indices.T:
            if i>=0 and j>=0: mapping[i]=j
        maps.append(mapping)
    result=maps[0].copy()
    if len(maps)>1:
        for mapping in maps[1:]:result[mapping!=result]=-1
    return result, {'optimal_alignments':count,'reliable_positions':int((result>=0).sum())}


def homology_mapping(rows):
    wt=wt_sequence(); mapping=np.full((len(rows),341),-1,dtype=np.int16); annotations=[]
    for i,row in enumerate(rows.itertuples()):
        intervals={'NMNAT':(int(row.nmnat_cdd_from)-1,int(row.nmnat_cdd_to)),
            'Nudix':(int(row.nudix_cdd_from)-1,int(row.nudix_cdd_to)),
            'linker':(int(row.nmnat_cdd_to),int(row.nudix_cdd_from)-1)}
        for region,(low,high) in intervals.items():
            a,b=REGIONS[region]; local,meta=_domain_mapping(wt[a-1:b],row.sequence[low:high])
            mapped=local>=0;local[mapped]+=low;mapping[i,a-1:b]=local
            annotations.append({'row':i,'region':region,**meta})
    return mapping,annotations


def build_statistical_sets(run,plan,rows):
    raw=ROOT/'data/results'/run.name/'raw';out=raw/'statistical_sets.json'
    if out.exists():
        return json.loads(out.read_text()), np.load(raw/'homology_mapping.npz')['positions']
    started=time.monotonic();mapping,annotations=homology_mapping(rows)
    chars=np.full(mapping.shape,-1,dtype=np.int16)
    for i,row in enumerate(rows.itertuples()):
        valid=mapping[i]>=0
        chars[i,valid]=np.array([AA.index(row.sequence[j]) for j in mapping[i,valid]],dtype=np.int16)
    train=np.flatnonzero(rows.training_role_v2.eq('training'))
    part=rows.iloc[train];counts=part.component_v2.value_counts()
    weights=np.array([1/(len(counts)*counts[c]) for c in part.component_v2])
    groups=part.component_v2.to_numpy();codes=chars[train]
    coverage=((codes>=0)*weights[:,None]).sum(0)
    eligible=[];events={};frequencies={}
    for col in range(341):
        if coverage[col]<plan['statistics']['position_coverage']:continue
        valid=codes[:,col]>=0
        frequency=np.bincount(codes[valid,col],weights=weights[valid],minlength=20)/coverage[col]
        order=np.argsort(-frequency,kind='stable')
        if frequency[order[1]]<=0:continue
        event=codes[:,col]==order[1]
        support=len(set(groups[event]))
        if support<plan['statistics']['event_min_training_components']:continue
        eligible.append(col);events[col]=event;frequencies[col]={'alphabet':[AA[order[0]],AA[order[1]],'OTHER'],
            'event_AA':AA[order[1]],'coverage':float(coverage[col]),'event_components':support}
    pair_candidates=[]
    for a,b in itertools.combinations(eligible,2):
        joint_event=events[a]&events[b]
        support=len(set(groups[joint_event]))
        if support<plan['statistics']['event_min_training_components']:continue
        valid=(codes[:,a]>=0)&(codes[:,b]>=0);w=weights*valid
        if w.sum()<=0:continue
        labels=events[a].astype(int)*2+events[b].astype(int)
        table=np.bincount(labels,weights=w,minlength=4).reshape(2,2);table/=table.sum()
        independent=table.sum(1)[:,None]*table.sum(0)[None,:]
        use=table>0;mi=float((table[use]*np.log(table[use]/independent[use])).sum())
        pair_candidates.append({'columns':[a+1,b+1],'selection_MI':mi,'event_components':support})
    pair_candidates.sort(key=lambda x:(-x['selection_MI'],x['columns']))
    pairs=pair_candidates[:plan['statistics']['max_pairs']]
    triples=set()
    for x,y in itertools.combinations(pairs,2):
        union=tuple(sorted(set(x['columns'])|set(y['columns'])))
        if len(union)==3:triples.add(union)
    triple_candidates=[]
    for cols in sorted(triples):
        indices=[c-1 for c in cols];event=np.logical_and.reduce([events[c] for c in indices])
        support=len(set(groups[event]))
        if support<plan['statistics']['event_min_training_components']:continue
        valid=(codes[:,indices]>=0).all(1); w=weights*valid
        states=sum(events[c].astype(int)*(2**j) for j,c in enumerate(indices))
        table=np.bincount(states,weights=w,minlength=8);table/=table.sum()
        marg=[sum(table[s] for s in range(8) if s&(1<<j)) for j in range(3)]
        independent=np.array([np.prod([marg[j] if s&(1<<j) else 1-marg[j] for j in range(3)]) for s in range(8)])
        use=table>0;tc=float((table[use]*np.log(table[use]/independent[use])).sum())
        triple_candidates.append({'columns':list(cols),'selection_total_correlation':tc,'event_components':support})
    triple_candidates.sort(key=lambda x:(-x['selection_total_correlation'],x['columns']))
    selected=[]
    for kind,values in [('pair',pairs),('triple',triple_candidates[:plan['statistics']['max_triples']])]:
        for i,item in enumerate(values):
            selected.append({'id':f'{kind}_{i+1:03d}','kind':kind,**item,
                'positions':[{'WT_position':c,**frequencies[c-1]} for c in item['columns']]})
    description={'selection_population':'1644_training_sequences_only','mapping':'CDD-domain-wise global BLOSUM62 open=-10 extend=-0.5; intersect <=128 tied optima; >128 unknown',
        'position_coverage_threshold':plan['statistics']['position_coverage'],
        'event_min_training_components':plan['statistics']['event_min_training_components'],
        'eligible_positions':len(eligible),'sets':selected,'selection_seconds':time.monotonic()-started,
        'limitations':'training support selects hypotheses; does not establish validation precision or physical epistasis',
        'population_sha256':sha256_file(raw/'population.csv')}
    np.savez_compressed(raw/'homology_mapping.npz',positions=mapping,AA_codes=chars)
    pd.DataFrame(annotations).to_csv(raw/'alignment_uncertainty.csv.gz',index=False)
    write_json(out,description)
    write_json(run/'statistical_set_summary.json',{'eligible_positions':len(eligible),'pairs':len(pairs),
        'triples':sum(s['kind']=='triple' for s in selected),'sets_sha256':sha256_file(out),
        'mapping_sha256':sha256_file(raw/'homology_mapping.npz'),'selection_seconds':description['selection_seconds']})
    return description,mapping


def prepare_statistic_features(run,plan,data):
    raw=ROOT/'data/results'/run.name/'raw';cache=ROOT/'data/interim'/run.name
    sets,mapping=build_statistical_sets(run,plan,data.rows)
    queries=[];offset=0
    for item in sets['sets']:
        cols=np.array(item['columns'])-1
        for row_index,row in data.rows.iterrows():
            if row.training_role_v2!='validation':continue
            sites=mapping[row_index,cols]
            if (sites<0).any():continue
            region='global'
            for name in ['NMNAT','Nudix','linker']:
                low,high=REGIONS[name]
                if all(low<=c<=high for c in item['columns']):region=name;break
            intervals={'global':(int(row.nmnat_cdd_from)-1,int(row.nudix_cdd_to)),
                'NMNAT':(int(row.nmnat_cdd_from)-1,int(row.nmnat_cdd_to)),
                'Nudix':(int(row.nudix_cdd_from)-1,int(row.nudix_cdd_to)),
                'linker':(int(row.nmnat_cdd_to),int(row.nudix_cdd_from)-1)}
            low,high=intervals[region];size=4 if region=='linker' else 8
            if not all(low<=p<high for p in sites) or len(set(sites))!=len(sites):continue
            rng=np.random.default_rng(stable_seed(plan['statistics']['seed'],item['id'],row.sequence_id))
            rest=np.setdiff1d(np.arange(low,high),sites)
            fill=rng.choice(rest,min(size-len(sites),len(rest)),replace=False)
            mask=sorted(map(int,np.r_[sites,fill]))
            queries.append({'row':int(row_index),'offset':offset,'length':int(row.length),
                'panel':item['id'],'statistic_id':item['id'],'region':region,'size_rule':'statistic_subset_in_task_mask',
                'positions':mask,'subset_positions':list(map(int,sites)),'weight':1.,
                'feature_file':str((cache/'statistical_features.npy').relative_to(ROOT))})
            offset+=int(row.length)
    path=raw/'statistical_queries.json';receipt_path=raw/'statistical_feature_manifest.json'
    if path.exists() and json.loads(path.read_text())!=queries:raise ValueError('Statistical queries changed')
    write_json(path,queries)
    if not queries:return queries
    cache_path=cache/'statistical_features.npy'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if sha256_file(cache_path)!=receipt['sha256']:raise ValueError('Statistical features changed')
        return queries
    # 统计输入保持完整M，S以外的目标位点仍然遮蔽。
    data_identity=json.loads((run/'inputs/data_identity.json').read_text())
    backbone=Path(data_identity['source_root'])/'nad_models/esm2'
    if sha256_file(backbone/'model.safetensors')!=plan['backbone_sha256']['model.safetensors']:
        raise ValueError('Frozen ESM changed before statistical queries')
    mmap=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.uint16,shape=(offset,1280))
    encoder=FrozenEncoder(backbone);started=time.monotonic()
    for start in range(0,len(queries),plan['feature_batch_size']):
        batch=queries[start:start+plan['feature_batch_size']]
        h,_,_=encoder([data.rows.iloc[q['row']].sequence for q in batch],[q['positions'] for q in batch])
        values=h.to(torch.bfloat16).cpu().view(torch.uint16).numpy()
        for i,q in enumerate(batch):mmap[q['offset']:q['offset']+q['length']]=values[i,:q['length']]
        if start%(plan['feature_batch_size']*64)==0:
            status(run,'PREPARING_STATISTICAL_FEATURES',completed=start,total=len(queries),seconds=time.monotonic()-started)
    mmap.flush();del encoder;torch.cuda.empty_cache()
    write_json(receipt_path,{'sha256':sha256_file(cache_path),'queries_sha256':sha256_file(path),
        'queries':len(queries),'path':str(cache_path.relative_to(ROOT)),'seconds':time.monotonic()-started})
    return queries


def _load_model(entry,plan):
    path=ROOT/entry['path']
    if sha256_file(path)!=entry['sha256']:raise ValueError('Formal checkpoint identity differs')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    config=saved.get('model_config',entry.get('model_config',plan['model']))
    model=ConditionalModel(saved.get('arm',entry['arm']),**config).cuda().eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    return model


def generate_candidates(run,plan,data,formal_manifest):
    raw=ROOT/'data/results'/run.name/'raw';generation=raw/'generation';generation.mkdir(parents=True,exist_ok=True)
    cfg=plan['generation'];wt=wt_sequence();groups=[]
    for region,(low,high) in cfg['regions_1based'].items():
        for size in cfg['mask_sizes'][region]:
            for repeat in range(cfg['masks_per_condition']):
                gid=f'{region}_m{size:02d}_r{repeat:02d}'
                rng=np.random.default_rng(stable_seed(cfg['seed'],gid,'positions'))
                groups.append({'id':gid,'region':region,'mask_size':size,'positions':sorted(map(int,rng.choice(np.arange(low-1,high),size,replace=False)))})
    write_json(raw/'generation_masks.json',groups)
    cache_path=ROOT/'data/interim'/run.name/'WT_generation_features.pt';cache_manifest=raw/'WT_generation_feature_manifest.json'
    if cache_path.exists() and cache_manifest.exists():
        if sha256_file(cache_path)!=json.loads(cache_manifest.read_text())['sha256']:raise ValueError('WT feature cache changed')
        features=torch.load(cache_path,map_location='cpu',weights_only=True)
    else:
        identity=json.loads((run/'inputs/data_identity.json').read_text());backbone=Path(identity['source_root'])/'nad_models/esm2'
        if sha256_file(backbone/'model.safetensors')!=plan['backbone_sha256']['model.safetensors']:raise ValueError('Frozen model changed')
        encoder=FrozenEncoder(backbone);parts=[]
        for start in range(0,len(groups),plan['feature_batch_size']):
            batch=groups[start:start+plan['feature_batch_size']]
            h,_,_=encoder([wt]*len(batch),[g['positions'] for g in batch]);parts.append(h.to(torch.bfloat16).cpu())
        features=torch.cat(parts);torch.save(features,cache_path)
        write_json(cache_manifest,{'sha256':sha256_file(cache_path),'inputs_sha256':sha256_file(raw/'generation_masks.json'),
            'WT_sha256':hashlib.sha256(wt.encode()).hexdigest()})
        del encoder;torch.cuda.empty_cache()
    from Bio.Align import PairwiseAligner
    aligner=PairwiseAligner(mode='global',match_score=0,mismatch_score=-1,open_gap_score=-1,extend_gap_score=-1)
    distance_cache={};total=0;started=time.monotonic();all_parts=[];failures=[]
    for key,entry in sorted(formal_manifest.items()):
        arm=entry.get('arm',key.split('/')[0]);seed=int(entry.get('seed',key.split('/')[1]))
        part_path=generation/f'{arm}_{seed}.parquet';receipt_path=generation/f'{arm}_{seed}_receipt.json'
        if part_path.exists() and receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if receipt['checkpoint_sha256']!=entry['sha256'] or sha256_file(part_path)!=receipt['sha256']:raise ValueError('Generation result changed')
            all_parts.append(pd.read_parquet(part_path));total+=receipt['attempts'];continue
        model=_load_model({**entry,'arm':arm},plan);rows=[];priors={}
        for index,group in enumerate(groups):
            positions=group['positions'];n=cfg['draws_per_mask'];uniform_seed=stable_seed(cfg['seed'],seed,group['id'],'residues')
            prior_seed=stable_seed(cfg['seed'],seed,group['id'],'latent')
            uniforms=np.random.default_rng(uniform_seed).random((n,len(positions)))
            try:
                h=features[index:index+1].cuda();valid=torch.ones(1,341,dtype=torch.bool,device='cuda');mask=torch.zeros_like(valid);mask[:,positions]=True
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    context=model.encode_context(h,valid,mask)
                    labels=torch.zeros(1,341,dtype=torch.long,device='cuda')
                    targets,_,_,_=target_tensors(context,mask,labels)
                    if arm=='QVAE':
                        p=model.prior(context,valid,mask)
                        codes=categorical_samples(p,n,torch.Generator(device='cuda').manual_seed(prior_seed))
                        logits=model.decode_codes(targets,codes)[:,0].float().cpu().numpy()
                        logp=categorical_log_prob(p,codes)[:,0].cpu().numpy()
                        codes_array=codes[:,0].cpu().numpy();priors[group['id']]=p.float().cpu().numpy()[0]
                    else:
                        latent=model.deterministic_context(context,valid,mask)[None]
                        values=model.decode(targets,latent)[0,0].float().cpu().numpy()
                        logits=np.broadcast_to(values,(n,)+values.shape);codes_array=None;logp=None
                letters=sample_letters(logits,uniforms,cfg['temperature'])
                for draw in range(n):
                    chars=list(wt)
                    for position,letter in zip(positions,letters[draw]):chars[position]=str(letter)
                    sequence=''.join(chars);sha=hashlib.sha256(sequence.encode()).hexdigest()
                    edits=[{'position':i+1,'from':a,'to':b} for i,(a,b) in enumerate(zip(wt,sequence)) if a!=b]
                    if sha not in distance_cache:distance_cache[sha]=int(round(-aligner.score(wt,sequence)))
                    count=len(edits)
                    rows.append({'candidate_id':f'{arm}_s{seed}_{group["id"]}_d{draw:02d}','arm':arm,'seed':seed,'group':group['id'],
                        'region':group['region'],'mask_size':group['mask_size'],'draw':draw,'sequence':sequence,'sequence_sha256':sha,
                        'Hamming':count,'Levenshtein':distance_cache[sha],'in_design_space':bool(1<=count<=34),
                        'reason':'WT_reference' if count==0 else 'outside_design_space' if count>34 else '',
                        'z_codes':json.dumps(codes_array[draw].tolist()) if codes_array is not None else '',
                        'log_prior_z':float(logp[draw]) if logp is not None else None,'prior_seed':prior_seed if arm=='QVAE' else None,
                        'uniform_seed':uniform_seed,'edits':json.dumps(edits),'structure':'UNKNOWN','function':'UNKNOWN'})
            except Exception as exc:
                failure={'arm':arm,'seed':seed,'group':group['id'],'exception':repr(exc),'attempts':n};failures.append(failure)
                for draw in range(n):rows.append({'candidate_id':f'{arm}_s{seed}_{group["id"]}_d{draw:02d}','arm':arm,'seed':seed,'group':group['id'],
                    'region':group['region'],'mask_size':group['mask_size'],'draw':draw,'sequence':'','sequence_sha256':'','Hamming':None,
                    'Levenshtein':None,'in_design_space':False,'reason':'generation_failure','uniform_seed':uniform_seed,'edits':'[]','structure':'UNKNOWN','function':'UNKNOWN'})
                write_json(raw/'generation_failures.json',failures)
            if (index+1)%32==0:
                status(run,'GENERATING_SINGLE_PASS',completed=total+len(rows),total=2*cfg['attempts_per_arm'],seconds=time.monotonic()-started)
        frame=pd.DataFrame(rows);frame.to_parquet(part_path,index=False)
        if priors:np.savez_compressed(generation/f'{arm}_{seed}_conditional_prior_logits.npz',**priors)
        write_json(receipt_path,{'sha256':sha256_file(part_path),'checkpoint_sha256':entry['sha256'],'attempts':len(frame),
            'groups':len(groups),'prior_conditioning':'masked_WT_and_M_only','posterior_used':False})
        all_parts.append(frame);total+=len(frame);del model;torch.cuda.empty_cache()
    result=pd.concat(all_parts,ignore_index=True)
    if not result.groupby('arm').size().eq(cfg['attempts_per_arm']).all() or len(result)!=2*cfg['attempts_per_arm']:
        raise ValueError('Attempt budget differs')
    nonempty=result.sequence.ne('')
    result['duplicate_within_arm']=False
    result.loc[nonempty,'duplicate_within_arm']=result[nonempty].duplicated(['arm','sequence_sha256'])
    result.to_parquet(raw/'candidates.parquet',index=False)
    with gzip.open(raw/'candidates.fasta.gz','wt') as handle:
        handle.write('>KVP40_WT_reference\n'+wt+'\n')
        for row in result[nonempty].itertuples():handle.write(f'>{row.candidate_id}\n{row.sequence}\n')
    write_json(run/'generation_summary.json',{'attempts':len(result),'by_arm':{arm:{'attempts':len(part),
        'nonWT_in_space':int(part.in_design_space.sum()),'unique_nonWT_in_space':int(part[part.in_design_space].sequence_sha256.nunique()),
        'duplicate_attempts':int(part.duplicate_within_arm.sum()),'reasons':part.reason.value_counts().to_dict(),
        'Hamming_min':int(part.Hamming.min()) if part.Hamming.notna().any() else None,
        'Hamming_max':int(part.Hamming.max()) if part.Hamming.notna().any() else None}
        for arm,part in result.groupby('arm')},'seconds':time.monotonic()-started,'failures':failures,
        'max_realizable_Hamming':32,'supplementary_distance':'Levenshtein_not_substitution_count'})
    return result


def family_profile_scores(run,plan,frame):
    raw=ROOT/'data/results'/run.name/'raw';path=raw/'family_profile_scores.parquet'
    if path.exists():return pd.read_parquet(path)
    import pyhmmer
    from pyhmmer import easel,plan7
    manifest=json.loads((ROOT/'data/manifest.json').read_text());entries={e['name']:e for e in manifest['entries']}
    source=Path(json.loads((run/'inputs/data_identity.json').read_text())['source_root'])
    unique=frame[frame.sequence.ne('')][['sequence_sha256','sequence']].drop_duplicates('sequence_sha256')
    alphabet=easel.Alphabet.amino();sequences=easel.DigitalSequenceBlock(alphabet)
    for row in unique.itertuples():sequences.append(easel.TextSequence(name=row.sequence_sha256.encode(),sequence=row.sequence).digitize(alphabet))
    wtsha=hashlib.sha256(wt_sequence().encode()).hexdigest()
    if wtsha not in set(unique.sequence_sha256):sequences.append(easel.TextSequence(name=wtsha.encode(),sequence=wt_sequence()).digitize(alphabet))
    scores={sha:{'sequence_sha256':sha} for sha in [s.name.decode() if isinstance(s.name,bytes) else s.name for s in sequences]}
    identities={}
    for region,name in [('NMNAT','hmm_pf01467'),('Nudix','hmm_pf00293')]:
        entry=entries[name];profile=source/entry['path']
        if sha256_file(profile)!=entry['sha256']:raise ValueError('HMM source changed')
        identities[name]=entry['sha256']
        with plan7.HMMFile(profile) as handle:hmm=handle.read()
        for hits in pyhmmer.hmmsearch(hmm,sequences,cpus=8,E=100.,domE=100.):
            for hit in hits:
                sha=hit.name.decode() if isinstance(hit.name,bytes) else hit.name
                domains=list(hit.domains)
                if not domains:continue
                domain=max(domains,key=lambda d:d.score)
                scores[sha].update({f'{region}_HMM_score':float(hit.score),f'{region}_domain_score':float(domain.score),
                    f'{region}_env_from':int(domain.env_from),f'{region}_env_to':int(domain.env_to),f'{region}_Evalue':float(hit.evalue)})
        for record in scores.values():
            record.setdefault(f'{region}_HMM_score',np.nan)
    result=pd.DataFrame(scores.values())
    result['ordered_domain_hits']=(result.NMNAT_env_to<result.Nudix_env_from) if {'NMNAT_env_to','Nudix_env_from'}<=set(result) else False
    result.to_parquet(path,index=False)
    write_json(run/'profile_evaluation.json',{'profile_sha256':identities,'unique_sequences_including_WT':len(result),
        'scores_sha256':sha256_file(path),'interpretation':'family profile compatibility, not independent activity evidence; no universal engineering rejection threshold'})
    return result


def _chemical_class(aa):
    if aa in 'DE':return 'acidic'
    if aa in 'KR':return 'basic'
    if aa=='H':return 'histidine'
    if aa in 'NQST':return 'polar'
    if aa in 'AVILMFWY':return 'hydrophobic'
    return aa


def annotate_candidates(frame,statistical_sets):
    nmnat={14,17,135,138};nudix={232,236,247,248,251,252,307,308}
    annotations=[]
    for row in frame.itertuples():
        edits=json.loads(row.edits);positions={e['position'] for e in edits}
        n=sorted(positions&nmnat);u=sorted(positions&nudix)
        events=[f'{e["position"]}:{_chemical_class(e["from"])}->{_chemical_class(e["to"])}' for e in edits if e['position'] in nudix]
        supported=[]
        for item in statistical_sets['sets']:
            if row.sequence and all(row.sequence[p['WT_position']-1]==p['event_AA'] for p in item['positions']):supported.append(item['id'])
        annotations.append({'NMNAT_anchor_changes':json.dumps(n),'Nudix_anchor_changes':json.dumps(u),
            'Nudix_chemical_events':json.dumps(events),'training_supported_events':json.dumps(supported),
            'NMNAT_substitutions':sum(4<=p<=181 for p in positions),'Nudix_substitutions':sum(204<=p<=338 for p in positions),
            'linker_substitutions':sum(182<=p<=203 for p in positions)})
    return pd.concat([frame.reset_index(drop=True),pd.DataFrame(annotations)],axis=1)


def candidate_supply_summary(run,frame):
    raw=ROOT/'data/results'/run.name/'raw'
    frame.groupby(['arm','region','mask_size','Hamming','reason'],dropna=False).size().rename('attempts').reset_index().to_csv(run/'distance_coverage.csv',index=False)
    matched=[];loss=[]
    good=frame[frame.in_design_space]
    for (seed,group,distance),part in good.groupby(['seed','group','Hamming']):
        cells={arm:part[part.arm==arm].sort_values(['sequence_sha256','candidate_id']) for arm in ['QVAE','deterministic']}
        size=min(map(len,cells.values()))
        for arm,cell in cells.items():
            matched.extend(cell.head(size).index.tolist())
            loss.append({'seed':int(seed),'group':group,'Hamming':int(distance),'arm':arm,'original':len(cell),'retained':size})
    paired=frame.loc[matched].copy();paired[['candidate_id','arm','seed','group','Hamming']].to_parquet(raw/'matched_attempts.parquet',index=False)
    pd.DataFrame(loss).to_csv(run/'matching_coverage.csv',index=False)
    def summarize(part):
        return {'attempts':len(part),'unique_sequences':int(part.sequence_sha256.nunique()),
            'mean_Hamming':float(part.Hamming.mean()) if len(part) else None,
            'mean_NMNAT_HMM':float(part.NMNAT_HMM_score.mean()) if len(part) and part.NMNAT_HMM_score.notna().any() else None,
            'mean_Nudix_HMM':float(part.Nudix_HMM_score.mean()) if len(part) and part.Nudix_HMM_score.notna().any() else None,
            'ordered_domain_hit_fraction':float(part.ordered_domain_hits.mean()) if len(part) else None,
            'NMNAT_profile_scored':int(part.NMNAT_HMM_score.notna().sum()),'Nudix_profile_scored':int(part.Nudix_HMM_score.notna().sum())}
    report={'raw_in_space':{a:summarize(g) for a,g in good.groupby('arm')},
        'matched_same_seed_mask_and_Hamming':{a:summarize(g) for a,g in paired.groupby('arm')},
        'attempt_denominators':frame.groupby('arm').size().to_dict(),
        'interpretation':'Raw efficiency and matched comparison are separate; profile scores are family evidence, not activity or rejection gates'}
    write_json(run/'candidate_supply_summary.json',report)
    return report


def select_representatives(run,plan,frame):
    good=frame[frame.in_design_space].drop_duplicates(['arm','sequence_sha256']).copy()
    selected=[];wt=wt_sequence()
    for arm,pool in good.groupby('arm'):
        chosen=set();covered=set();entries=[]
        risky=pool[pool.Nudix_anchor_changes.ne('[]')&pool.NMNAT_anchor_changes.eq('[]')].copy()
        for _ in range(plan['candidates']['max_Nudix_exploratory_per_arm']):
            options=[]
            for idx,row in risky.iterrows():
                if idx in chosen:continue
                tags=set(json.loads(row.Nudix_chemical_events));gain=len(tags-covered)
                if not gain:continue
                options.append((-gain,int(row.NMNAT_substitutions),int(row.Hamming),row.sequence_sha256,idx,tags))
            if not options:break
            options.sort(key=lambda x:x[:4]);_,_,_,_,idx,tags=options[0]
            row=pool.loc[idx].to_dict();row.update(selection_class='Nudix_exploratory',paired_to='',
                selection_reason='新增可检验的Nudix锚点化学变化；NMNAT四个对应锚点未改，家族低支持不作淘汰',
                research_question='检查Nudix对应配体/金属邻域与NMNAT结构基础；水解方向和净积累均UNKNOWN')
            chosen.add(idx);covered|=tags;entries.append(row)
            controls=pool[~pool.index.isin(chosen)&pool.Nudix_anchor_changes.eq('[]')&pool.NMNAT_anchor_changes.eq('[]')].copy()
            controls=controls[(controls.region==row['region'])&(controls.mask_size==row['mask_size'])&(abs(controls.Hamming-row['Hamming'])<=2)]
            if len(controls):
                controls['mask_mismatch']=(controls.group!=row['group']).astype(int)
                controls['seed_mismatch']=(controls.seed!=row['seed']).astype(int)
                controls['distance_difference']=abs(controls.Hamming-row['Hamming'])
                control=controls.sort_values(['mask_mismatch','seed_mismatch','distance_difference','sequence_sha256']).iloc[0]
                record=pool.loc[control.name].to_dict();record.update(selection_class='matched_sequence_reference',paired_to=row['candidate_id'],
                    selection_reason='相同区域与掩码规模、相近替换数的无锚点改变参照；是否同一mask另列',
                    research_question='与配对Nudix探索序列比较；不是已知功能阴性对照')
                chosen.add(control.name);entries.append(record)
        # 普通代表要求可解释家族证据；该条件不约束上面的探索候选。
        ordinary=pool[pool.ordered_domain_hits&pool.NMNAT_HMM_score.notna()&pool.Nudix_HMM_score.notna()]
        covered_sets=set();covered_bins=set();region_order=['global','NMNAT','Nudix','linker']
        while len(entries)<plan['candidates']['max_per_arm']:
            added=False
            for region in region_order:
                options=[]
                for idx,row in ordinary[ordinary.region==region].iterrows():
                    if idx in chosen:continue
                    events=set(json.loads(row.training_supported_events))
                    distance_bin='1-5' if row.Hamming<=5 else '6-15' if row.Hamming<=15 else '16-32'
                    tags={(region,distance_bin),('NMNAT_anchor',row.NMNAT_anchor_changes),('Nudix_anchor',row.Nudix_anchor_changes)}
                    gain=len(events-covered_sets)+len(tags-covered_bins)
                    if not gain:continue
                    # Hamming差异不作主目标，家族证据和未覆盖问题优先。
                    support=float(row.NMNAT_HMM_score)+float(row.Nudix_HMM_score)
                    options.append((-gain,-support,row.sequence_sha256,idx,events,tags))
                if not options:continue
                options.sort(key=lambda x:x[:3]);_,_,_,idx,events,tags=options[0]
                row=pool.loc[idx].to_dict();row.update(selection_class='family_background_and_combination',paired_to='',
                    selection_reason='补充未覆盖的区域/替换强度/天然组合或锚点变化问题，并保留有序双域profile证据',
                    research_question='核对该组合的全长结构基础和相应催化环境；天然支持不等于工程收益')
                chosen.add(idx);covered_sets|=events;covered_bins|=tags;entries.append(row);added=True
                if len(entries)>=plan['candidates']['max_per_arm']:break
            if not added:break
        for number,row in enumerate(entries[:plan['candidates']['max_per_arm']],1):
            row['representative_id']=f'{arm}_R{number:02d}';selected.append(row)
    result=pd.DataFrame(selected) if selected else pd.DataFrame(columns=['arm','representative_id','candidate_id','sequence'])
    result.to_csv(run/'representative_candidates.csv',index=False)
    with (run/'representative_candidates.fasta').open('w') as handle:
        handle.write('>KVP40_WT_reference\n'+wt+'\n')
        for row in result.itertuples():handle.write(f'>{row.representative_id} {row.candidate_id}\n{row.sequence}\n')
    lines=['# KVP40 代表候选','', '全部为计算提出的序列；结构、活性及NAD⁺净积累均为UNKNOWN。完整序列见 `representative_candidates.fasta`，逐条编辑及选择依据见CSV。','',
        '| 候选 | 区域 | 替换数 | 类别 | Nudix锚点变化 | 配对参照目标 |','|---|---|---:|---|---|---|']
    for row in result.itertuples():lines.append(f'| {row.representative_id} | {row.region} | {int(row.Hamming)} | {row.selection_class} | {row.Nudix_anchor_changes} | {row.paired_to} |')
    (run/'CANDIDATES.md').write_text('\n'.join(lines)+'\n')
    return result


@torch.no_grad()
def statistic_predictions(run,plan,data,formal_manifest,queries):
    raw=ROOT/'data/results'/run.name/'raw';destination=raw/'statistical_predictions';destination.mkdir(parents=True,exist_ok=True)
    sets=json.loads((raw/'statistical_sets.json').read_text());lookup={s['id']:s for s in sets['sets']}
    decision=json.loads((run/'evaluation_sample_decision.json').read_text())
    repeats=int(plan['calibration']['likelihood_repetitions'])
    if not queries:return {}
    n=len(queries);receipts={};batch_size=16
    for key,entry in sorted(formal_manifest.items()):
        arm=entry.get('arm',key.split('/')[0]);seed=int(entry.get('seed',key.split('/')[1]));name=f'{arm}_{seed}'
        count=max(2,int(decision['arms']['QVAE']['evaluation_sample_count']))  # 同预算的实际tuple采样；确定性概率本身可精确计算
        path=destination/f'{name}.npz';receipt_path=destination/f'{name}.json'
        if path.exists() and receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if sha256_file(path)!=receipt['sha256'] or entry['sha256']!=receipt['checkpoint_sha256'] or receipt['count']!=count or receipt['repeats']!=repeats:raise ValueError('Statistic prediction changed')
            receipts[key]=receipt;continue
        model=_load_model({**entry,'arm':arm},plan)
        shared=np.zeros((repeats,n,27),dtype=np.float32);independent=np.zeros_like(shared);sampled=np.zeros_like(shared)
        log_shared=np.zeros((repeats,n),dtype=np.float64);log_ind=np.zeros_like(log_shared)
        margins=np.zeros((repeats,n,3,3),dtype=np.float32)
        started=time.monotonic()
        # 同一统计集相邻排列，batch不会混合不同阶数和字母表。
        for sid,grouped in itertools.groupby(enumerate(queries),key=lambda x:x[1]['statistic_id']):
            indexed=list(grouped);item=lookup[sid];order=len(item['columns'])
            alphabets=[p['alphabet'][:2] for p in item['positions']]
            codes_by_position=[[AA.index(a) for a in alphabet] for alphabet in alphabets]
            for begin in range(0,len(indexed),batch_size):
                selected=indexed[begin:begin+batch_size];indices=[i for i,q in selected];batch=[q for i,q in selected]
                h,_,v,m,y=data.batch(batch,need_full=False)
                with torch.autocast('cuda',dtype=torch.bfloat16):context=model.encode_context(h,v,m)
                sites=torch.tensor([q['subset_positions'] for q in batch],device='cuda')
                targets=context.gather(1,sites[...,None].expand(-1,-1,context.shape[-1]))
                labels=y.gather(1,sites)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prior=model.prior(context,v,m) if arm=='QVAE' else None
                    deterministic=model.deterministic_context(context,v,m)[None] if arm=='deterministic' else None
                for repeat in range(repeats):
                    generator=torch.Generator(device='cuda').manual_seed(stable_seed(plan['statistics']['seed'],seed,sid,begin,repeat,'latent'))
                    residue_generator=torch.Generator(device='cuda').manual_seed(stable_seed(plan['statistics']['seed'],seed,sid,begin,repeat,'residues'))
                    codes=categorical_samples(prior,count,generator) if prior is not None else None
                    probs_sum=torch.zeros(len(batch),order,20,device='cuda',dtype=torch.float64)
                    joint_sum=torch.zeros((len(batch),)+(3,)*order,device='cuda',dtype=torch.float64)
                    tuple_counts=torch.zeros(len(batch),3**order,device='cuda',dtype=torch.float64)
                    true_joint_sum=torch.zeros(len(batch),device='cuda',dtype=torch.float64)
                    for start in range(0,count,64):
                        width=min(64,count-start)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            logits=model.decode_codes(targets,codes[start:start+width]) if codes is not None else model.decode(targets,deterministic).expand(width,-1,-1,-1)
                        prob=logits.float().softmax(-1).double();probs_sum+=prob.sum(0)
                        true=prob.gather(-1,labels[None,...,None].expand(width,-1,-1,1)).squeeze(-1)
                        true_joint_sum+=true.prod(-1).sum(0)
                        reduced=[]
                        for pos,aa_codes in enumerate(codes_by_position):
                            first=prob[:,:,pos,aa_codes[0]];second=prob[:,:,pos,aa_codes[1]]
                            reduced.append(torch.stack((first,second,(1-first-second).clamp_min(0)),-1))
                        if order==2:joint_sum+=torch.einsum('kbi,kbj->bij',reduced[0],reduced[1])
                        else:joint_sum+=torch.einsum('kbi,kbj,kbl->bijl',*reduced)
                        uniforms=torch.rand(width,len(batch),order,device='cuda',generator=residue_generator,dtype=torch.float64)
                        letters=(uniforms[...,None]>=prob.cumsum(-1)).sum(-1).clamp_max(19)
                        states=torch.zeros(width,len(batch),dtype=torch.long,device='cuda')
                        for pos,aa_codes in enumerate(codes_by_position):
                            category=torch.where(letters[:,:,pos]==aa_codes[0],0,torch.where(letters[:,:,pos]==aa_codes[1],1,2))
                            states=states*3+category
                        for b in range(len(batch)):tuple_counts[b]+=torch.bincount(states[:,b],minlength=3**order)
                    mean_prob=probs_sum/count
                    reduced_marg=[]
                    for pos,aa_codes in enumerate(codes_by_position):
                        a=mean_prob[:,pos,aa_codes[0]];b=mean_prob[:,pos,aa_codes[1]]
                        reduced_marg.append(torch.stack((a,b,(1-a-b).clamp_min(0)),-1))
                    independent_prob=reduced_marg[0]
                    for marginal in reduced_marg[1:]:independent_prob=independent_prob[...,None]*marginal.reshape((len(batch),)+(1,)*(independent_prob.ndim-1)+(3,))
                    shared[repeat,indices,:3**order]=(joint_sum/count).flatten(1).cpu().numpy()
                    independent[repeat,indices,:3**order]=independent_prob.flatten(1).cpu().numpy()
                    sampled[repeat,indices,:3**order]=(tuple_counts/count).cpu().numpy()
                    margins[repeat,indices,:order,:]=torch.stack(reduced_marg,1).cpu().numpy()
                    log_shared[repeat,indices]=(true_joint_sum/count).clamp_min(1e-300).log().cpu().numpy()
                    log_ind[repeat,indices]=mean_prob.gather(-1,labels[...,None]).squeeze(-1).clamp_min(1e-300).log().sum(-1).cpu().numpy()
        np.savez_compressed(path,shared=shared,independent=independent,sampled=sampled,
            log_shared=log_shared,log_independent=log_ind,marginals=margins)
        receipt={'sha256':sha256_file(path),'checkpoint_sha256':entry['sha256'],'count':count,'repeats':repeats,
            'queries_sha256':sha256_file(raw/'statistical_queries.json'),'evaluation_decision_sha256':sha256_file(run/'evaluation_sample_decision.json'),'seconds':time.monotonic()-started,
            'target_probabilities':'canonical_20_AA','frequency_alphabet':'two training-major AAs plus OTHER',
            'actual_sampled_tuples':count*repeats*n,'conditioning':'full_task_mask_M_remains_hidden; only_S_scored'}
        write_json(receipt_path,receipt);receipts[key]=receipt;del model;torch.cuda.empty_cache()
        status(run,'EVALUATING_STATISTICAL_SUBSETS',completed_models=len(receipts),total_models=len(formal_manifest))
    write_json(run/'statistical_prediction_manifest.json',receipts)
    return receipts


def _interval(value, draws):
    draws=np.asarray(draws,dtype=float)
    draws=draws[np.isfinite(draws)]
    return {'estimate':float(value),'CI95':np.quantile(draws,[.025,.975]).tolist() if len(draws) else None,
            'bootstrap_finite':len(draws)}


def _component_means(values, query_components, components):
    values=np.asarray(values)
    return np.stack([values[...,query_components==g,:].mean(-2) if np.any(query_components==g)
                     else np.zeros(values.shape[:-2]+(values.shape[-1],)) for g in components],axis=-2)


def summarize_statistical_predictions(run,plan,data,formal_manifest,queries):
    """同背景的联合计分与家族混合统计分开；同时重采样组分和正式种子。"""
    raw=ROOT/'data/results'/run.name/'raw'
    sets=json.loads((raw/'statistical_sets.json').read_text())['sets']
    components=np.array(sorted(data.rows.loc[data.rows.training_role_v2.eq('validation'),'component_v2'].unique()))
    seeds=list(plan['formal_seeds']);nboot=plan['statistics']['bootstrap_repeats']
    rng=np.random.default_rng(plan['statistics']['seed'])
    component_draws=rng.multinomial(len(components),np.full(len(components),1/len(components)),size=nboot)
    seed_draws=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),size=nboot)/len(seeds)
    prediction={}
    for arm in plan['arms']:
        for seed in seeds:
            with np.load(raw/'statistical_predictions'/f'{arm}_{seed}.npz') as source:
                prediction[arm,seed]={key:source[key] for key in source.files}
    results=[];boot_by_kind={'pair':[],'triple':[]}
    row_components=data.rows.component_v2.to_numpy()
    for item in sets:
        indices=np.array([i for i,q in enumerate(queries) if q['statistic_id']==item['id']],dtype=int)
        if not len(indices):continue
        batch=[queries[i] for i in indices];order=len(item['columns']);cells=3**order
        query_components=row_components[[q['row'] for q in batch]]
        present=np.array([np.any(query_components==c) for c in components])
        effective_draws=component_draws*present[None]
        totals=effective_draws.sum(-1)
        background_weights=effective_draws/np.maximum(1,totals[:,None])
        base_weights=present/present.sum()
        observed=np.zeros((len(indices),cells))
        joint_event=[]
        for i,q in enumerate(batch):
            seq=data.rows.iloc[q['row']].sequence;state=0;event=True
            for pos,definition in zip(q['subset_positions'],item['positions']):
                aa=seq[pos];alphabet=definition['alphabet'];state=state*3+(alphabet.index(aa) if aa in alphabet[:2] else 2)
                event=event and aa==definition['event_AA']
            observed[i,state]=1;joint_event.append(event)
        obs_components=_component_means(observed,query_components,components)
        observed_distribution=base_weights@obs_components
        sampled_observed=background_weights@obs_components
        probabilities={};logs={};repeated_logs={}
        for arm in plan['arms']:
            variants=['shared','independent'] if arm=='QVAE' else ['shared']
            for variant in variants:
                label=f'{arm}_{variant}'
                values=np.stack([prediction[arm,seed][variant][:,indices,:cells].mean(0) for seed in seeds])
                probabilities[label]=_component_means(values,query_components,components)
                logkey='log_shared' if variant=='shared' else 'log_independent'
                repeated_logs[label]=np.stack([prediction[arm,seed][logkey][:,indices] for seed in seeds])
                logs[label]=_component_means(repeated_logs[label].mean(1)[...,None]/order,query_components,components)[...,0]
        estimates={};draw_values={};mc={}
        for label,p in probabilities.items():
            aggregate=np.einsum('g,sgc->c',base_weights,p)/len(seeds)
            expected=np.einsum('bs,bg,sgc->bc',seed_draws,background_weights,p)
            tv=float(.5*np.abs(aggregate-observed_distribution).sum())
            tv_draw=.5*np.abs(expected-sampled_observed).sum(-1)
            score=float(np.einsum('g,sg->',base_weights,logs[label])/len(seeds))
            score_draw=np.einsum('bs,bg,sg->b',seed_draws,background_weights,logs[label])
            estimates[label]={'true_log_probability_per_scored_residue':_interval(score,score_draw),
                'background_mixture_TV':_interval(tv,tv_draw)}
            draw_values[label]=(score_draw,tv_draw)
            repeat_scores=[]
            for r in range(repeated_logs[label].shape[1]):
                groups=_component_means(repeated_logs[label][:,r,:,None]/order,query_components,components)[...,0]
                repeat_scores.append(float(np.einsum('g,sg->',base_weights,groups)/len(seeds)))
            mc[label]=np.asarray(repeat_scores)
        sampled_frequencies={}
        for arm in plan['arms']:
            values=np.stack([prediction[arm,seed]['sampled'][:,indices,:cells].mean(0) for seed in seeds])
            grouped=_component_means(values,query_components,components)
            aggregate_frequency=np.einsum('g,sgc->c',base_weights,grouped)/len(seeds)
            boot_frequency=np.einsum('bs,bg,sgc->bc',seed_draws,background_weights,grouped)
            sampled_frequencies[arm]={'TV_to_observed_background_mixture':_interval(
                .5*np.abs(aggregate_frequency-observed_distribution).sum(),
                .5*np.abs(boot_frequency-sampled_observed).sum(-1)),
                'draws_per_background_per_seed':int(prediction[arm,seeds[0]]['sampled'].shape[0])*int(json.loads((raw/'statistical_predictions'/f'{arm}_{seeds[0]}.json').read_text())['count'])}
        shared='QVAE_shared';comparisons={};kind_draws={}
        for baseline in ['QVAE_independent','deterministic_shared']:
            score_difference=estimates[shared]['true_log_probability_per_scored_residue']['estimate']-estimates[baseline]['true_log_probability_per_scored_residue']['estimate']
            score_draw=draw_values[shared][0]-draw_values[baseline][0]
            tv_improvement=estimates[baseline]['background_mixture_TV']['estimate']-estimates[shared]['background_mixture_TV']['estimate']
            tv_draw=draw_values[baseline][1]-draw_values[shared][1]
            repeat_differences=mc[shared]-mc[baseline]
            mc_se=float(repeat_differences.std(ddof=1)/np.sqrt(len(repeat_differences))) if len(repeat_differences)>1 else None
            comparisons[baseline]={'log_probability_gain_per_residue':_interval(score_difference,score_draw),
                'TV_improvement':_interval(tv_improvement,tv_draw),'MC_SE_log_gain':mc_se,
                'numerically_resolved':bool(mc_se is not None and abs(score_difference)>2*mc_se)}
            kind_draws[baseline]=(score_difference,score_draw,tv_improvement,tv_draw)
        event_components=len(set(query_components[np.asarray(joint_event,dtype=bool)]))
        support={'validation_sequences':len(batch),'validation_components':int(present.sum()),
            'joint_event_components':event_components,'nonzero_coarse_cells':int((observed.sum(0)>0).sum()),
            'coarse_cells':cells,'minimum_joint_event_support_met':event_components>=5,
            'precision':'descriptive/underpowered' if event_components<5 else 'inspect_intervals_and_MC; support_alone_is_not_precision'}
        results.append({'id':item['id'],'kind':item['kind'],'WT_positions':item['columns'],'support':support,
            'scores':estimates,'sampled_frequencies':sampled_frequencies,'comparisons':comparisons})
        boot_by_kind[item['kind']].append(kind_draws)
    aggregate={}
    for kind,items in boot_by_kind.items():
        if not items:continue
        aggregate[kind]={}
        for baseline in ['QVAE_independent','deterministic_shared']:
            parts=[it[baseline] for it in items]
            aggregate[kind][baseline]={'sets':len(parts),
                'mean_log_probability_gain_per_residue':_interval(np.mean([p[0] for p in parts]),np.mean([p[1] for p in parts],axis=0)),
                'mean_TV_improvement':_interval(np.mean([p[2] for p in parts]),np.mean([p[3] for p in parts],axis=0))}
    report={'aggregate':aggregate,'sets':results,'bootstrap':{'repetitions':nboot,
        'units':'paired validation homology components and paired formal training seeds; entire component background stays together'},
        'likelihood':'prior Monte Carlo on scored subset S with all context mask M hidden; canonical 20 AA',
        'family_statistics':'coarse training-chosen alphabet; identical validation background mixture, not empirical fixed-context conditional truth',
        'interpretation':'TV and joint log probability answer different questions; neither establishes physical epistasis or enzyme activity'}
    write_json(run/'joint_statistics_summary.json',report)
    return report


def conditional_prior_diagnostics(run):
    raw=ROOT/'data/results'/run.name/'raw';records=[]
    for path in sorted((raw/'generation').glob('*_conditional_prior_logits.npz')):
        with np.load(path) as source:
            logits=np.stack([source[key] for key in source.files]).astype(np.float64)
        logits-=logits.max(-1,keepdims=True);p=np.exp(logits);p/=p.sum(-1,keepdims=True)
        mean=p.mean(0);entropy=-(p*np.log(np.maximum(p,1e-300))).sum(-1)
        records.append({'source':str(path.relative_to(ROOT)),'sha256':sha256_file(path),
            'mean_entropy_per_factor':entropy.mean(0).tolist(),
            'effective_states_per_factor':np.exp(entropy.mean(0)).tolist(),
            'context_mean_KL_to_aggregate_per_factor':(p*np.log(np.maximum(p,1e-300)/np.maximum(mean,1e-300))).sum(-1).mean(0).tolist(),
            'interpretation':'prior variation is a diagnostic, not proof of learned natural dependencies'})
    write_json(run/'conditional_prior_diagnostics.json',records)
    return records


def finish_study(run,plan,data,formal_manifest):
    """正式训练之后完成真实联合评价、单轮候选及可校验研究交付。"""
    from .conditional_evaluation import evaluate_formal_models
    raw=ROOT/'data/results'/run.name/'raw'
    started=time.monotonic();export_training_curves(run,plan)
    queries=prepare_statistic_features(run,plan,data)
    full=evaluate_formal_models(run,plan,data,formal_manifest)
    statistic_predictions(run,plan,data,formal_manifest,queries)
    statistical=summarize_statistical_predictions(run,plan,data,formal_manifest,queries)
    write_json(run/'formal_evaluation_summary.json',full)
    candidates=generate_candidates(run,plan,data,formal_manifest)
    profiles=family_profile_scores(run,plan,candidates)
    sets=json.loads((raw/'statistical_sets.json').read_text())
    merged=candidates.merge(profiles,on='sequence_sha256',how='left',validate='many_to_one')
    merged['ordered_domain_hits']=merged.ordered_domain_hits.fillna(False).astype(bool)
    frame=annotate_candidates(merged,sets)
    frame.to_parquet(raw/'annotated_candidates.parquet',index=False)
    supply=candidate_supply_summary(run,frame)
    representatives=select_representatives(run,plan,frame)
    conditional_prior_diagnostics(run)
    write_json(raw/'finish_cost.json',{'seconds':time.monotonic()-started})
    _write_report(run,plan,statistical,supply,representatives)
    checks={}
    for directory in [raw,ROOT/'data/models'/run.name]:
        for path in sorted(directory.rglob('*')):
            if path.is_file():checks[str(path.relative_to(ROOT))]={'sha256':sha256_file(path),'bytes':path.stat().st_size}
    write_json(run/'result_checksums.json',checks)
    return {'raw_files':len(checks),'representatives':len(representatives),'seconds':time.monotonic()-started}


def _write_report(run,plan,statistics,supply,representatives):
    raw=ROOT/'data/results'/run.name/'raw'
    cost=json.loads((raw/'cost.json').read_text())
    lines=['# NatV 条件 QVAE：联合建模与候选序列供给','',
        '采用完整序列后验、学习型条件先验和共享16×8 Categorical潜变量；正式生成仅从条件先验采样。研究定义与预算见 `PLAN.md`，输入身份见 `inputs/`，完整产物身份见 `result_checksums.json`。','',
        f'完成 {cost["attempted_fits"]} 次拟合，拟合计算累计 {cost["fit_seconds"]/3600:.2f} 小时。原始逐输入记录、失败、权重与完整候选位于 `data/results/natv_conditional_qvae/raw/` 和 `data/models/natv_conditional_qvae/`。','',
        '训练与验证曲线见 `training_curves.png`，27种验证条件分图见 `validation_panel_curves.png`，计量口径见 `curve_definitions.json`。QVAE训练后验重构CE与验证条件边缘NLL不是同一概率量。','',
        '全目标条件联合概率、重要性采样ESS及估计不确定性见 `formal_evaluation_summary.json`。以下统计在保持完整任务掩码M的条件下，仅计分训练预选位置S。正差值表示QVAE改善；区间包含零或差异与MC误差相当时，判为证据不足。','',
        '| 位置集合 | 对照 | 联合log概率增益（nats/残基，95%区间） | 混合背景TV改善（95%区间） |',
        '|---|---|---|---|']
    def formatted(item):
        ci=item['CI95']
        return f'{item["estimate"]:.5f} [{ci[0]:.5f}, {ci[1]:.5f}]' if ci else f'{item["estimate"]:.5f} [UNKNOWN]'
    for kind,arms in statistics['aggregate'].items():
        for baseline,values in arms.items():lines.append(f'| {kind} | {baseline} | {formatted(values["mean_log_probability_gain_per_residue"])} | {formatted(values["mean_TV_improvement"])} |')
    weak=sum(not item['support']['minimum_joint_event_support_met'] for item in statistics['sets'])
    lines.extend(['',f'{len(statistics["sets"])} 个统计集合中，{weak} 个在验证组分上的预定位点联合事件支持少于5个组分。逐集合区间与MC误差见 `joint_statistics_summary.json`；不能将跨背景的组合频率解释为固定上下文的真实条件分布。','',
        '| 模型 | 尝试数 | 范围内非WT尝试 | 唯一非WT序列 | 代表序列 |','|---|---:|---:|---:|---:|'])
    generation=json.loads((run/'generation_summary.json').read_text())
    for arm,item in generation['by_arm'].items():
        lines.append(f'| {arm} | {item["attempts"]} | {item["nonWT_in_space"]} | {item["unique_nonWT_in_space"]} | {int((representatives.arm==arm).sum()) if len(representatives) else 0} |')
    lines.extend(['','原始生成效率与同种子、同掩码、同Hamming替换数匹配后的比较分别见 `candidate_supply_summary.json`；匹配损失见 `matching_coverage.csv`。WT作为单独参照，不计入非WT产出；本轮最大掩码32，不能生成34次替换。','',
        '代表序列、真实替换及后续研究问题见 `CANDIDATES.md`、`representative_candidates.csv` 和 `.fasta`。家族HMM相容性只作为序列证据；Nudix探索候选允许较弱天然支持。结构、催化环境、合成与水解活性以及NAD⁺净积累仍为UNKNOWN。','',
        '联合概率改善与可靠共变统计改善同时成立才支持联合建模收益；候选数量、KL非零或改变z产生差异均不能单独支持该结论。该计算研究提供后续可检验序列，不证明工程性能改善。',''])
    (run/'REPORT.md').write_text('\n'.join(lines))


def export_training_curves(run,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw=ROOT/'data/results'/run.name/'raw';histories=[]
    for path in sorted((raw/'curves').glob('formal_*.json')):
        histories.extend(json.loads(path.read_text())['history'])
    if not histories:return {}
    colors={'QVAE':'#2465b0','deterministic':'#d37420'}
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    metrics=['validation','train_reconstruction_per_residue','train_kl_per_factor']
    titles=['Validation conditional joint NLL','Training reconstruction CE (mixed tasks)','Training categorical KL per factor']
    for axis,metric,title in zip(axes,metrics,titles):
        for arm in plan['arms']:
            if metric=='train_kl_per_factor' and arm!='QVAE':continue
            groups=[]
            for seed in plan['formal_seeds']:
                group=[h for h in histories if h['arm']==arm and h['seed']==seed]
                xs=[];ys=[]
                for h in group:
                    value=h.get(metric)
                    if metric=='validation':value=value.get('score') if value else None
                    if value is not None and np.isfinite(value):xs.append(h['epoch']);ys.append(value)
                if xs:
                    axis.plot(xs,ys,color=colors[arm],alpha=.22,linewidth=.8)
                    groups.append(dict(zip(xs,ys)))
            if groups:
                common=sorted(set.intersection(*(set(g) for g in groups)))
                if common:
                    values=np.array([[g[x] for x in common] for g in groups])
                    axis.plot(common,values.mean(0),color=colors[arm],label=arm)
                    axis.fill_between(common,values.min(0),values.max(0),color=colors[arm],alpha=.12)
        axis.set(title=title,xlabel='Epoch',ylabel='nats / residue' if metric!='train_kl_per_factor' else 'nats / factor')
        axis.grid(alpha=.2);axis.legend()
    fig.suptitle('Thin lines: individual seeds; thick line / range: common observed epochs',fontsize=10)
    fig.tight_layout();fig.savefig(run/'training_curves.png',dpi=180);fig.savefig(run/'training_curves.svg');plt.close(fig)
    panels=sorted({p for h in histories if h.get('validation') for p in h['validation'].get('panels',{})})
    fig,axes=plt.subplots(7,4,figsize=(16,21));flat=axes.ravel()
    for axis,panel in zip(flat,panels):
        for arm in plan['arms']:
            for seed in plan['formal_seeds']:
                points=[(h['epoch'],h['validation']['panels'][panel]['score']) for h in histories
                    if h['arm']==arm and h['seed']==seed and h.get('validation')
                    and panel in h['validation'].get('panels',{}) and h['validation']['panels'][panel]['score'] is not None]
                if points:axis.plot(*zip(*points),color=colors[arm],alpha=.5,linewidth=.8)
        axis.set(title=panel,xlabel='Epoch',ylabel='nats / residue');axis.grid(alpha=.2)
    for axis in flat[len(panels):]:axis.set_visible(False)
    fig.tight_layout();fig.savefig(run/'validation_panel_curves.png',dpi=130);plt.close(fig)
    note={'formal_seeds':plan['formal_seeds'],'training':'component-weighted reconstruction; QVAE uses posterior, deterministic has no latent',
        'validation':'component-balanced fixed panel mixture, QVAE conditional marginal NLL estimate; deterministic exact',
        'bands':'min-max across formal seeds only at common observed epochs, not confidence intervals',
        'training_panels':'training batches mix per-sequence panels; per-panel training history was not recorded',
        'calibration':'excluded: original calibration scalar reconstruction and KL records have a known batch scaling error; weights and validation unaffected'}
    write_json(run/'curve_definitions.json',note)
    return note
