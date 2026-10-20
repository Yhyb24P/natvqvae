"""Complete raw prior draws, frozen distribution comparison and local HMM annotation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from run import RUN, ROOT, check, load_data, write_json
from natvqvae.aligned_generator import AlignedGenerator, ALPHABET
from natvqvae.intake import load_manifest, verify_entry, sha256_file
from natvqvae.sequence import wt_sequence


def kernel(a,b):
    result=np.empty((len(a),len(b)),dtype=np.float64)
    for start in range(0,len(a),32):
        distance=(a[start:start+32,None,:]!=b[None,:,:]).mean(-1)
        result[start:start+32]=np.exp(-distance/.1)
    return result


def screen_domains(source,directory,frame,protocol):
    import pyhmmer
    fasta=directory/'raw_ungapped.fasta'
    fasta.write_text(''.join(f'>{r.candidate_id}\n{r.sequence}\n' for r in frame.itertuples() if r.sequence))
    thresholds=protocol['HMM_screen'];all_hits=[];inputs={}
    alphabet=pyhmmer.easel.Alphabet.amino()
    sequences=pyhmmer.easel.DigitalSequenceBlock(alphabet,[
        pyhmmer.easel.TextSequence(name=r.candidate_id,sequence=r.sequence).digitize(alphabet)
        for r in frame.itertuples() if r.sequence])
    for key,domain in [('hmm_pf01467','NMNAT'),('hmm_pf00293','Nudix')]:
        entry=next(e for e in load_manifest()['entries'] if e['name']==key)
        verified=verify_entry(entry,source);check(verified.ok,verified.detail)
        inputs[key]={'path':str(verified.path),'sha256':sha256_file(verified.path),'pyhmmer':pyhmmer.__version__}
        with pyhmmer.plan7.HMMFile(str(verified.path)) as handle:profile=handle.read()
        top=next(iter(pyhmmer.hmmsearch([profile],sequences,cpus=4,E=.001,domE=.001)))
        for hit in top:
            for dom in hit.domains:
                alignment=dom.alignment
                coverage=(alignment.hmm_to-alignment.hmm_from+1)/profile.M
                name=hit.name.decode() if isinstance(hit.name,bytes) else hit.name
                all_hits.append({'candidate_id':name,'domain':domain,'independent_evalue':float(dom.i_evalue),
                                 'hmm_coverage':coverage,'sequence_from':int(alignment.target_from),
                                 'sequence_to':int(alignment.target_to),
                                 'qualifies':dom.i_evalue<=thresholds['independent_domain_evalue_max'] and
                                             coverage>=thresholds['HMM_coverage_min']})
    hits=pd.DataFrame(all_hits,columns=['candidate_id','domain','independent_evalue','hmm_coverage','sequence_from','sequence_to','qualifies'])
    hits.to_csv(directory/'domain_hits.csv',index=False)
    ordered=set();any_dual=set()
    for candidate,group in hits[hits.qualifies].groupby('candidate_id'):
        first=group[group.domain=='NMNAT'];second=group[group.domain=='Nudix']
        if len(first) and len(second):
            any_dual.add(candidate)
            if any(a.sequence_to<b.sequence_from for a in first.itertuples() for b in second.itertuples()):ordered.add(candidate)
    frame['HMM_double_domain']=frame.candidate_id.isin(any_dual)
    frame['HMM_ordered_double_domain']=frame.candidate_id.isin(ordered)
    return inputs


def evaluate(source,plan):
    directory=RUN/'evaluation';check(not directory.exists(),'Evaluation already exists; preserve frozen output')
    freeze=json.loads((RUN/'training/checkpoint_freeze.json').read_text());identity=freeze['identity']
    check(sha256_file(RUN/'plan.json')==identity['plan_sha256'],'Plan changed after training')
    check(sha256_file(RUN/'run.py')==identity['runner_sha256'],'Training implementation changed')
    check(sha256_file(ROOT/'src/natvqvae/aligned_generator.py')==identity['model_wrapper_sha256'],'Model changed')
    check(sha256_file(RUN/'data/input_checks.json')==identity['input_checks_sha256'],'Data identity changed')
    groups,tensors,embedding,weights=load_data();protocol=json.loads((RUN/'evaluation_protocol.json').read_text())
    directory.mkdir();start=time.monotonic()
    write_json(directory/'identity.json',{'training_identity':identity,'checkpoint_freeze_sha256':sha256_file(RUN/'training/checkpoint_freeze.json'),
               'evaluation_protocol_sha256':sha256_file(RUN/'evaluation_protocol.json'),'evaluation_code_sha256':sha256_file(Path(__file__)),
               'attempts_per_arm_seed':plan['evaluation']['generation_attempts_per_arm_seed']})
    device=torch.device(identity['device']);attempts=plan['evaluation']['generation_attempts_per_arm_seed']
    reference=tensors['validation'].numpy();train=tensors['training'].numpy()
    reference_counts=Counter(r['component'] for r in groups['validation'])
    reference_weights=np.array([1/len(reference_counts)/reference_counts[r['component']] for r in groups['validation']])
    reference_term=float(reference_weights@kernel(reference,reference)@reference_weights)
    source_hashes={r['sequence_id'] for rows in groups.values() for r in rows}
    source_aligned={r['aligned_sequence'] for rows in groups.values() for r in rows}
    results=[];records=[];features={};within_terms={};cross_terms={}
    for run in freeze['runs']:
        arm,seed=run['arm'],run['seed'];path=RUN/run['checkpoint'];check(sha256_file(path)==run['sha256'],'Checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=AlignedGenerator(source,plan['model'],embedding,arm).to(device)
        model.load_state_dict(saved['state_dict'],strict=True);model.eval()
        latent_rng=torch.Generator(device=device).manual_seed(seed+protocol['latent_seed_offset'])
        aa_rng=torch.Generator(device=device).manual_seed(seed+protocol['AA_seed_offset'])
        generated=[];latents=[];entropy=[]
        with torch.no_grad():
            for offset in range(0,attempts,64):
                count=min(64,attempts-offset)
                if arm=='categorical_vae':
                    latent=model.network.sample_prior(count,latent_rng);logits=model.network.decode(latent)
                    latents.extend([''.join(map(str,row)) for row in latent.argmax(-1).cpu().tolist()])
                else:
                    logits=model.prior_logits(count,latent_rng);latents.extend(['']*count)
                check(torch.isfinite(logits).all().item(),'Nonfinite prior logits')
                prob=logits.softmax(-1);cdf=prob.cumsum(-1);cdf[...,-1]=1.
                uniform=torch.rand(prob.shape[:2],generator=aa_rng,device=device)
                draws=(uniform[...,None]>cdf).sum(-1)
                generated.append(draws.cpu().numpy());entropy.extend((-(prob*prob.clamp_min(1e-12).log()).sum(-1).mean(-1)).cpu().tolist())
        generated=np.concatenate(generated);features[(arm,seed)]=generated
        kgg=kernel(generated,generated);cross=kernel(generated,reference).mean(0)
        within_terms[(arm,seed)]=float(kgg.mean());cross_terms[(arm,seed)]=cross
        mmd=float(kgg.mean()-2*cross@reference_weights+reference_term)
        distance_train=1+np.log(np.maximum(kernel(generated,train),1e-300))*.1
        # distance_train is identity fraction; largest value is the nearest match.
        nearest_train=distance_train.max(1)
        aligned=[''.join(ALPHABET[int(t)] for t in row) for row in generated]
        hashes=[]
        for i,(sequence,latent,ent,identity_train) in enumerate(zip(aligned,latents,entropy,nearest_train)):
            raw=sequence.replace('-','');h=hashlib.sha256(raw.encode()).hexdigest();hashes.append(h)
            records.append({'candidate_id':f'{arm}_s{seed}_{i+1:04d}','arm':arm,'seed':seed,'attempt':i+1,
               'aligned_sequence':sequence,'sequence':raw,'sequence_sha256':h,'ungapped_length':len(raw),
               'gap_count':sequence.count('-'),'WT_aligned_hamming':sum(a!=b for a,b in zip(sequence,wt_sequence())),
               'exact_source_sequence_match_training_validation':h in source_hashes,'exact_aligned_source_match':sequence in source_aligned,
               'nearest_training_aligned_identity':float(identity_train),'mean_conditional_entropy':ent,'prior_latent_states':latent})
        frequencies=np.stack([(generated==state).mean(0) for state in range(21)],axis=-1)
        pairwise_hamming=float((1-(frequencies**2).sum(-1)).mean()*attempts/(attempts-1))
        result={'arm':arm,'seed':seed,'attempts':attempts,'unique_aligned':len(set(aligned)),'unique_ungapped':len(set(hashes)),
                'mmd_squared':mmd,'pairwise_aligned_hamming_fraction':pairwise_hamming,
                'median_ungapped_length':float(np.median((generated!=20).sum(1))),
                'mean_nearest_training_aligned_identity':float(nearest_train.mean())}
        results.append(result);print(json.dumps(result),flush=True)
        del model
    frame=pd.DataFrame(records)
    hmm_inputs=screen_domains(source,directory,frame,protocol)
    frame.to_csv(directory/'raw_candidates.csv',index=False)
    for result in results:
        subset=frame[(frame.arm==result['arm'])&(frame.seed==result['seed'])]
        result.update(HMM_ordered_double_domain=int(subset.HMM_ordered_double_domain.sum()),
                      exact_training_validation_sequence_matches=int(subset.exact_source_sequence_match_training_validation.sum()),
                      gap_free=int((subset.gap_count==0).sum()))
    pd.DataFrame(results).to_csv(directory/'per_seed_summary.csv',index=False)
    # In the paired MMD difference the reference/reference kernel cancels exactly.
    component_names=sorted(reference_counts);component_cross={}
    for seed in plan['seeds']:
        diff=cross_terms[('no_latent',seed)]-cross_terms[('categorical_vae',seed)]
        component_cross[seed]=np.array([np.mean([diff[i] for i,r in enumerate(groups['validation']) if r['component']==comp])
                                        for comp in component_names])
    rng=np.random.default_rng(protocol['bootstrap_seed']);boot=[];paired=[]
    for seed in plan['seeds']:
        paired.append(within_terms[('categorical_vae',seed)]-within_terms[('no_latent',seed)]+2*component_cross[seed].mean())
    for _ in range(plan['evaluation']['bootstrap_replicates']):
        sampled_seeds=rng.choice(plan['seeds'],len(plan['seeds']),replace=True)
        comps=rng.integers(0,len(component_names),size=len(component_names))
        boot.append(np.mean([within_terms[('categorical_vae',int(seed))]-within_terms[('no_latent',int(seed))]
                            +2*component_cross[int(seed)][comps].mean() for seed in sampled_seeds]))
    summary={'status':'COMPLETED','total_attempts':len(frame),'unique_aligned':int(frame.aligned_sequence.nunique()),
             'per_seed':results,'paired_VAE_minus_control_MMD_squared':paired,'paired_mean_difference':float(np.mean(paired)),
             'conditional_development_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
             'HMM_inputs':hmm_inputs,'generation_and_evaluation_seconds':time.monotonic()-start,
             'scope':'historically_exposed_validation; checkpoint_selected; three_training_seeds; not_independent_confirmation',
             'novelty_scope':'training_validation_only; test_payload_not_read','activity':'UNKNOWN'}
    write_json(directory/'results.json',summary)
    write_json(directory/'artifact_hashes.json',{p.name:sha256_file(p) for p in directory.iterdir() if p.is_file()})
    print(json.dumps({k:v for k,v in summary.items() if k not in ['per_seed','HMM_inputs']}),flush=True)
