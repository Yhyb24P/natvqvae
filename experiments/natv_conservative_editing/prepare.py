"""Pin existing assets and recalculate current-train support for the sourced short events."""
from pathlib import Path
import json,hashlib,subprocess,sys,os
from collections import defaultdict,Counter
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from natvqvae.intake import load_manifest,source_root,verify_entry,sha256_file
from natvqvae.sequence import wt_sequence
from natvqvae.conservative_editing import normalized_events,legacy_scripts,scaffold

RUN=Path(__file__).parent

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    source=source_root(load_manifest()).resolve();out=RUN/'inputs';pinned={}
    def pin(path,expected=None):
        path=Path(path);digest=sha256_file(path)
        if expected and digest!=expected:raise ValueError('Input differs: '+str(path))
        pinned[str(path)]={'sha256':digest,'bytes':path.stat().st_size};return path
    receipt=json.loads((ROOT/'discovery/natv_model_inheritance/source_checks.json').read_text())
    for relative,item in receipt['inputs'].items():
        if relative.startswith('src/') or relative.endswith('generation/checkpoint_freeze.json'):
            pin(source/relative,item['sha256'])
    for relative in ['src/nad_qvae/design/g0_benchmark.py','src/nad_qvae/design/se_full.py',
                     'src/nad_qvae/design/vnext_edits.py','src/nad_qvae/design/vnext_evaluation_contracts.py',
                     'src/nad_qvae/design/h2_generation_v1.py']:
        pin(source/relative)
    paths={}
    for entry in load_manifest()['entries']:
        if entry['name'] in ['allowed_samples','core_split','masked_input_plan','train_sequences']:
            r=verify_entry(entry,source)
            if not r.ok:raise ValueError(r.detail)
            paths[entry['name']]=pin(r.path,entry['sha256'])
    samples=pd.read_parquet(paths['allowed_samples']);train=samples[samples.training_role_v2=='training'].sort_values('sequence_id')
    core=json.loads(paths['core_split'].read_text());plan=json.loads(paths['masked_input_plan'].read_text())
    if len(train)!=1644 or train.component_v2.nunique()!=47 or set(train.sequence_id)!=set(core['splits']['training']):raise ValueError('Training identity')
    if set(plan['retrieval_database_ids'])!=set(train.sequence_id):raise ValueError('Retrieval role identity')
    src=json.loads((out/'g1_source.json').read_text())
    pin(out/'g1_event_qualification.json',src['sha256'])
    pin(out/'g1b_event_qualification_source.py',src['qualification_source']['sha256'])
    definitions=json.loads((out/'g1_event_qualification.json').read_text())['qualified_candidates']['rows']
    keys={(e['type'],e['anchor_wt_1based'],e['length']) for e in definitions if e['length']<=3}
    declared=defaultdict(list);perturbed=defaultdict(set);wt=wt_sequence()
    for row in train.itertuples():
        for scheme,sink in [(None,declared),({'matrix':'BLOSUM45','open_gap':-12.,'extend_gap':-.8},perturbed)]:
            seen=set()
            for event in normalized_events(wt,row.sequence,scheme):
                key=(event['type'],event['anchor'],event['length'])
                if key in keys and key not in seen:
                    seen.add(key)
                    if scheme is None:sink[key].append({'sequence_id':row.sequence_id,'component':row.component_v2,'inserted_or_deleted_sequence':event['sequence']})
                    else:sink[key].add(row.component_v2)
    retained=[];all_events=[]
    for key in sorted(keys):
        supporters=declared[key];components={r['component'] for r in supporters};shared=components&perturbed[key]
        event={'type':key[0],'anchor':key[1],'length':key[2], 'current_train_support':supporters,
               'supporting_components':len(components),'shared_perturbation_components':len(shared)}
        event['admitted']=len(components)>=3 and len(shared)>=2
        if event['admitted']:
            strings=Counter(r['inserted_or_deleted_sequence'] for r in supporters)
            event['sequences']=[{'sequence':s,'support':n} for s,n in sorted(strings.items())]
            retained.append(event)
        all_events.append(event)
    write(out/'current_event_support.json',{'definitions':all_events,'retained':retained,'support_semantics':'alignment_correspondence_not_activity_or_indel_rate'})
    # Bind local ESM-2 files to the previously audited model identity.
    historical=json.loads((source/'experiments/run_reports/natv_vnext_v5_6_wp0/actual_esm_mask_input_probe.json').read_text())
    for name,expected in historical['model_files'].items():pin(source/'nad_models/esm2'/name,expected)
    # Database files are used read-only and its exported records must equal core train exactly.
    db=source/'data/processed/releases/natv_formal_training_data_v1/retrieval_db/train'
    for path in sorted(db.parent.glob('train*')):
        if path.is_file():pin(path)
    export=out/'retrieval_database.fasta';binary=Path(sys.executable).with_name('mmseqs')
    subprocess.run([str(binary),'convert2fasta',str(db),str(export),'-v','0'],check=True)
    records={};current=None
    for line in export.read_text().splitlines():
        if line.startswith('>'):current=line[1:].split()[0];records[current]=''
        else:records[current]+=line.strip()
    if records!=dict(zip(train.sequence_id,train.sequence)):raise ValueError('MMseqs database does not equal formal train')
    edit_script,apply_script=legacy_scripts(source)
    for event in retained:
        example={k:event[k] for k in ['type','anchor','length']};example['sequence']=event['sequences'][0]['sequence']
        changed=scaffold(example).sequence
        if apply_script(wt,edit_script(wt,changed))!=changed:raise ValueError('Legacy reconstruction failed')
    for name in ['src/natvqvae/conservative_editing.py','experiments/natv_conservative_editing/prepare.py']:pin(ROOT/name)
    write(out/'source_checks.json',{'source_root':str(source),'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
          'mmseqs_version':subprocess.check_output([str(binary),'version'],text=True).strip(),'files':pinned,
          'training_sequences':len(train),'retrieval_records_verified':len(records),'retained_short_events':len(retained),
          'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    print(json.dumps({'retained_events':[{k:e[k] for k in ['type','anchor','length','supporting_components','shared_perturbation_components']} for e in retained]}),flush=True)

if __name__=='__main__':main()
