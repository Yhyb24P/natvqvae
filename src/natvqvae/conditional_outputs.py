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
