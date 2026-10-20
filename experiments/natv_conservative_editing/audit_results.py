"""Check exported attempts, trajectories, coordinates and exact edit reconstruction."""
from pathlib import Path
import gzip,json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from natvqvae.conservative_editing import legacy_scripts,scaffold,edit_distance
from natvqvae.sequence import wt_sequence
from natvqvae.intake import sha256_file
RUN=Path(__file__).parent
from run import verify_identity
verify_identity()
plan=json.loads((RUN/'plan.json').read_text());groups={g['group_id']:g for g in json.loads((RUN/'sampling_groups.json').read_text())}
frame=pd.read_csv(RUN/'raw_candidates.csv',keep_default_na=False).set_index('candidate_id');assert len(frame)==2976 and frame.index.is_unique
source=Path(json.loads((RUN/'inputs/source_checks.json').read_text())['source_root']);_,apply_script=legacy_scripts(source)
scripts=0
with gzip.open(RUN/'edit_scripts.jsonl.gz','rt') as f:
 for line in f:
  j=json.loads(line);assert apply_script(wt_sequence(),j['operations'])==frame.loc[j['candidate_id'],'sequence'];scripts+=1
last={};traces=0
with gzip.open(RUN/'trajectory.jsonl.gz','rt') as f:
 for line in f:
  j=json.loads(line);key=(j['arm'],j['group_id'],j['draw']);g=groups[j['group_id']];base=scaffold(g['event'])
  before,stage=last.get(key,(base.sequence,-1));assert j['stage']==stage+1 and j['before']==before
  sites=j['current_positions_0based'];assert j['masked_query']==''.join('X' if p in sites else aa for p,aa in enumerate(before))
  assert all(a==b for p,(a,b) in enumerate(zip(before,j['after'])) if p not in sites)
  assert len(before)==len(j['after']) and j['fresh_current_context']
  last[key]=(j['after'],j['stage']);traces+=1
for (arm,gid,draw),(sequence,stage) in last.items():
 candidate=f'{arm}_{gid}_d{draw:02d}';row=frame.loc[candidate];g=groups[gid];base=scaffold(g['event'])
 assert stage==4 and sequence==row.sequence
 actual=[p for p,index in enumerate(base.wt_to_current) if index is not None and sequence[index]!=wt_sequence()[p]]
 assert actual==g['WT_positions_0based']
 for anchor in plan['protected_WT_positions_1based']:assert sequence[base.wt_to_current[anchor-1]]==wt_sequence()[anchor-1]
 assert row.Levenshtein_distance==edit_distance(sequence,wt_sequence())
 assert row.WT_global_identity>=.9 and row.constraint_status=='PASS'
assert scripts==2976 and traces==14880 and len(last)==2976
(RUN/'result_checks.json').write_text(json.dumps({'attempts':len(frame),'exact_script_reconstructions':scripts,'verified_trajectory_steps':traces,
 'protected_anchors_and_exact_selected_sites_verified':True,'protocol_hashes_verified':True,
 'artifacts':{name:sha256_file(RUN/name) for name in ['raw_candidates.csv','trajectory.jsonl.gz','edit_scripts.jsonl.gz']},
 'audit_code_sha256':sha256_file(Path(__file__))},indent=2)+'\n')
print('2976 attempts, 14880 steps and all edit reconstructions verified')
