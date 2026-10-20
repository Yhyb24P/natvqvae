"""Evaluate all declared proposal attempts without introducing a selector-derived judge."""
from pathlib import Path
import gzip,json,sys,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from natvqvae.family_screen import screen_sequences
from natvqvae.intake import sha256_file
from natvqvae.sequence import wt_sequence

RUN=Path(__file__).parent

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    from run import verify_identity
    identity=verify_identity();plan=json.loads((RUN/'plan.json').read_text())
    frame=pd.read_csv(RUN/'raw_candidates.csv',keep_default_na=False)
    if len(frame)!=identity['attempts'] or not frame.candidate_id.is_unique:raise ValueError('Attempt denominator or IDs differ')
    out=RUN/'evaluation';out.mkdir(exist_ok=False)
    source=Path(json.loads((RUN/'inputs/source_checks.json').read_text())['source_root'])
    # HMM is applied to all proposals, not only to preselected unique sequences.
    hits,ordered,inputs=screen_sequences(frame,source,plan['evaluation']['HMM_independent_evalue_max'],plan['evaluation']['HMM_coverage_min'])
    hits.to_csv(out/'HMM_domains.csv',index=False)
    frame['ordered_double_domain']=frame.candidate_id.isin(ordered)
    frame['eligible']=frame.constraint_status.eq('PASS')
    train=pd.read_csv(ROOT/'discovery/natv_data_readiness/samples.csv')
    training_hashes=set(train[train.training_role_v2=='training'].sequence_id)
    validation_hashes=set(train[train.training_role_v2=='validation'].sequence_id)
    frame['exact_training_match']=frame.sequence_sha256.isin(training_hashes)
    frame['exact_validation_match']=frame.sequence_sha256.isin(validation_hashes)
    frame.to_csv(out/'candidate_evaluation.csv',index=False)
    (out/'candidates.fasta').write_text(''.join(f'>{r.candidate_id}\n{r.sequence}\n' for r in frame.itertuples()))
    groups=[]
    for (arm,group_id),group in frame.groupby(['arm','group_id'],sort=False):
        count=len(group);first=group.iloc[0];eligible=group[group.eligible]
        unique=eligible.sequence_sha256.nunique()
        letters=np.array([list(s) for s in group.sequence])
        pairwise=(letters[:,None,:]!=letters[None,:,:]).sum(-1)
        diversity=float(pairwise.sum()/(count*(count-1))/int(first.substitutions)) if count>1 else 0.
        groups.append({'arm':arm,'group_id':group_id,'seed':int(first.seed),'mode':first['mode'],'edit_budget':int(first.edit_budget),
                       'attempts':count,'HMM_qualified':int(group.ordered_double_domain.sum()),'valid':int(group.constraint_status.eq('PASS').sum()),
                       'unique_all':int(group.sequence_sha256.nunique()),'constraint_unique':int(unique),
                       'constraint_unique_fraction':float(unique/count),'within_mask_diversity':diversity})
    grouped=pd.DataFrame(groups);grouped.to_csv(out/'shared_mask_summary.csv',index=False)
    per_seed=grouped.groupby(['arm','seed','mode','edit_budget']).agg(attempts=('attempts','sum'),
                constraint_unique_fraction=('constraint_unique_fraction','mean'),within_mask_diversity=('within_mask_diversity','mean'),
                qualified=('HMM_qualified','sum'),valid=('valid','sum'),unique_within_masks=('unique_all','sum')).reset_index()
    per_seed.to_csv(out/'per_seed_summary.csv',index=False)
    # Resampling is conditional on sampling masks/events and historical fitted seeds, not biological replication.
    rng=np.random.default_rng(plan['evaluation']['bootstrap_seed']);contrasts=[]
    for (mode,budget),cell in grouped.groupby(['mode','edit_budget']):
        paired=cell.pivot(index=['seed','group_id'],columns='arm',values='constraint_unique_fraction')
        difference=paired['QVAE']-paired['SE'];by_seed={int(seed):part.to_numpy() for seed,part in difference.groupby(level='seed')}
        boot=[];seeds=sorted(by_seed)
        for _ in range(plan['evaluation']['bootstrap_replicates']):
            sampled=rng.choice(seeds,len(seeds),replace=True)
            boot.append(np.mean([rng.choice(by_seed[int(seed)],len(by_seed[int(seed)]),replace=True).mean() for seed in sampled]))
        contrasts.append({'mode':mode,'edit_budget':int(budget),'QVAE_minus_SE':float(difference.mean()),
                         'conditional_95_interval':np.quantile(boot,[.025,.975]).tolist(),
                         'saturated_all_arms':bool(cell.constraint_unique_fraction.eq(1).all())})
    diagnostics=[]
    for path in (RUN/'generation/QVAE').glob('*/latent_diagnostics.json'):
        diagnostics.extend(json.loads(path.read_text()))
    write(out/'results.json',{'status':'COMPLETED','attempts':len(frame),'constraint_counts':frame.constraint_status.value_counts().to_dict(),
             'primary':plan['primary'],'by_arm':frame.groupby('arm').agg(attempts=('candidate_id','size'),unique=('sequence_sha256','nunique'),
                  HMM_qualified=('ordered_double_domain','sum'),exact_training=('exact_training_match','sum'),
                  exact_validation=('exact_validation_match','sum'),minimum_identity=('WT_global_identity','min')).to_dict('index'),
             'contrasts':contrasts,'HMM_inputs':inputs,'latent_probe_count':len(diagnostics),
             'latent_probe_max_logit_difference':max((d['latent_disabled_logit_max_difference'] for d in diagnostics),default=None),
             'scope':'conditional_development_comparison_not_indel_tolerance_or_catalytic_activity',
             'function':'UNKNOWN','source_test_payloads_used':False})
    write(out/'artifact_hashes.json',{p.name:sha256_file(p) for p in out.iterdir() if p.is_file()})
    write(RUN/'run_status.json',{'status':'COMPLETED','attempts':len(frame),'results_sha256':sha256_file(out/'results.json')})
    print((out/'results.json').read_text(),flush=True)

if __name__=='__main__':main()
