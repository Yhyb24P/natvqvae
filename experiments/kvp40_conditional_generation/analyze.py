"""Audit actual trajectories and summarize paired generation coverage."""
from pathlib import Path
import gzip,hashlib,json,sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from natvqvae.sequence import wt_sequence
from natvqvae.intake import sha256_file
from natvqvae.conditional_generation import AA
RUN=Path(__file__).parent


def main():
    frame=pd.read_csv(RUN/'candidates.csv',keep_default_na=False);plan=json.loads((RUN/'plan.json').read_text());wt=wt_sequence()
    assert len(frame)==4608 and frame.candidate_id.is_unique
    groups={g['id']:g for g in json.loads((RUN/'generation_groups.json').read_text())}
    prior={r['group']:r['codes'] for r in json.loads((RUN/'prior_samples.json').read_text())}
    last={};reversions=0;steps=0
    with gzip.open(RUN/'trajectories.jsonl.gz','rt') as handle:
        for line in handle:
            r=json.loads(line);key=(r['arm'],r['group'],r['draw']);before,stage=last.get(key,(wt,-1))
            assert before==r['before'] and r['stage']==stage+1
            sites=r['positions_0based'];allowed=groups[r['group']]['sites']
            assert set(sites)<=set(allowed)
            assert len(r['after'])==341 and set(r['after'])<=set(AA)
            assert all(a==b for p,(a,b) in enumerate(zip(before,r['after'])) if p not in sites)
            if r['arm']=='QVAE':
                codes=prior[r['group']][r['draw']];z=np.eye(8,dtype=np.float32)[codes]
                assert hashlib.sha256(z.tobytes()).hexdigest()==r['prior_sha256']
            else:assert not r['prior_sha256']
            reversions+=sum(before[p]!=wt[p] and r['after'][p]==wt[p] for p in sites)
            last[key]=(r['after'],r['stage']);steps+=1
    for row in frame.itertuples():
        assert last[(row.arm,row.group,row.draw)]==(row.sequence,4)
        chars=list(wt)
        for edit in json.loads(row.edits):
            p=edit['position']-1;assert chars[p]==edit['from'];chars[p]=edit['to']
        assert ''.join(chars)==row.sequence
        assert sum(a!=b for a,b in zip(wt,row.sequence))==row.actual_substitutions
        assert hashlib.sha256(row.sequence.encode()).hexdigest()==row.sequence_sha256
        assert row.edit_distance<=row.actual_substitutions
        assert abs(row.WT_identity-(1-row.edit_distance/341))<1e-12
    assert steps==23040 and len(last)==4608
    checkpoints=json.loads((RUN/'checkpoint_manifest.json').read_text());initial=json.loads((RUN/'initialization.json').read_text())
    for seed in plan['seeds']:assert initial['state_hashes'][f'QVAE/{seed}']==initial['state_hashes'][f'deterministic/{seed}']
    for key,entry in checkpoints.items():assert sha256_file(RUN/entry['path'])==entry['sha256']
    history=pd.DataFrame(json.loads((RUN/'training_history.json').read_text()));assert len(history)==180 and history.groupby('key').epoch.nunique().eq(30).all()
    # Denominators remain all draws, even for out-of-space sequences and WT.
    summaries=[]
    for (arm,gid),cell in frame.groupby(['arm','group']):
        seq=np.array([list(s) for s in cell.sequence]);pair=(seq[:,None]!=seq[None,:]).sum(-1)
        summaries.append({'arm':arm,'group':gid,'region':cell.iloc[0].region,'mask_size':int(cell.iloc[0].mask_size),'attempts':len(cell),
          'unique_all':cell.sequence_sha256.nunique(),'unique_nonWT_in_space':cell[cell.reason.eq('')].sequence_sha256.nunique(),
          'pairwise_Hamming_all_attempts':float(pair.sum()/(len(cell)*(len(cell)-1))),
          'WT_retained_in_allowed_sites':int((len(groups[gid]['sites'])-cell.actual_substitutions).sum())})
    pd.DataFrame(summaries).to_csv(RUN/'combination_coverage.csv',index=False)
    overlap=[]
    for (region,size),cell in frame.groupby(['region','mask_size']):
        good=cell[cell.reason.eq('')];q=set(good[good.arm=='QVAE'].sequence_sha256);d=set(good[good.arm=='deterministic'].sequence_sha256)
        overlap.append({'region':region,'mask_size':int(size),'QVAE_unique':len(q),'deterministic_unique':len(d),'shared':len(q&d),'QVAE_only':len(q-d),'deterministic_only':len(d-q)})
    pd.DataFrame(overlap).to_csv(RUN/'model_coverage_by_condition.csv',index=False)
    anchors=pd.read_csv(ROOT/'data/processed/anchor_mapping_audit.csv');anchors=anchors.set_index('map_wt_position')
    reps=pd.read_csv(RUN/'structure_representatives.csv',keep_default_na=False);basis=[]
    unique=reps.drop_duplicates('sequence_sha256')
    (RUN/'structure_targets.fasta').write_text('>KVP40_WT\n'+wt+'\n'+''.join(f'>{row.sequence_sha256}\n{row.sequence}\n' for row in unique.itertuples()))
    reps[['candidate_id','sequence_sha256','region','edit_distance']].to_csv(RUN/'structure_target_mapping.csv',index=False)
    for row in reps.itertuples():
        edits=json.loads(row.edits);changed={int(e['position']) for e in edits};mapped=[]
        for pos in sorted(changed & set(anchors.index)):
            a=anchors.loc[pos];mapped.append({'WT_position':pos,'WT_residue':a.wt_actual,'NMNAT_or_Nudix':a.domain,
              '2QJO_construct':int(a['2QJO_construct_pos']),'2QJT_construct':int(a['2QJT_construct_pos']),'2R5W_construct':int(a['2R5W_construct_pos'])})
        counts={name:sum(low<=p<=high for p in changed) for name,(low,high) in plan['generation']['regions_1based'].items() if name!='global'}
        question={'NMNAT':'NMNAT domain fold and correctly mapped catalytic-anchor surroundings',
          'Nudix':'Nudix fold and mapped metal/product-binding surroundings',
          'linker':'relative domain organization and linker geometry',
          'global':'both domain folds, NMNAT environment and Nudix surroundings'}[row.region]
        basis.append({'candidate_id':row.candidate_id,'sequence':row.sequence,'edits':edits,'changed_known_anchors':mapped,
          'edited_residues_per_domain':counts,'structure_question':question,'reference':'WT plus homolog experimental 2QJO/2QJT/2R5W; 2QJT and 2R5W share a sequence',
          'candidate_structure':'UNKNOWN','catalytic_function':'UNKNOWN'})
    (RUN/'structure_evaluation_basis.json').write_text(json.dumps({'source_hashes':{str(p.relative_to(ROOT)):sha256_file(p) for p in [ROOT/'data/processed/anchor_mapping_audit.csv',ROOT/'docs/NATV_PAIRING_AUDIT.md']},'candidates':basis},ensure_ascii=False,indent=2)+'\n')
    (RUN/'audit.json').write_text(json.dumps({'attempts':len(frame),'verified_trajectory_steps':steps,'exact_sequence_reconstructions':len(frame),'WT_reversion_events':reversions,
      'fresh_initialization_and_checkpoint_hashes_verified':True,'same_parameters':initial['parameters'],'training_records':len(history),
      'analysis_code_sha256':sha256_file(Path(__file__))},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm,color in [('QVAE','#b2453d'),('deterministic','#286a96')]:
        counts=frame[frame.arm==arm].edit_distance.value_counts().sort_index();axes[0].plot(counts.index,counts.values,label=arm,color=color)
        for index,region in enumerate(['global','NMNAT','Nudix','linker']):
            cell=frame[(frame.arm==arm)&(frame.region==region)].edit_distance
            offset=-.12 if arm=='QVAE' else .12
            axes[1].errorbar(index+offset,cell.median(),yerr=[[cell.median()-cell.min()],[cell.max()-cell.median()]],fmt='o',color=color,label=arm if index==0 else None,capsize=3)
    axes[0].axvline(34.5,color='grey',linestyle='--');axes[0].set(xlabel='Actual edit distance',ylabel='All attempts');axes[0].legend()
    axes[1].set(xticks=range(4),xticklabels=['global','NMNAT','Nudix','linker'],ylabel='Actual edit distance',title='Median and observed range');axes[1].legend()
    fig.tight_layout();fig.savefig(RUN/'actual_coverage.png',dpi=180);plt.close(fig)
    print((RUN/'audit.json').read_text())

if __name__=='__main__':main()
