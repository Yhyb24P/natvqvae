"""Export actual trace/script records without copying MMseqs caches into Git."""
from pathlib import Path
import gzip,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from natvqvae.intake import sha256_file
RUN=Path(__file__).parent


def main():
    status=json.loads((RUN/'run_status.json').read_text())
    if status['status']!='COMPLETED':raise ValueError('Wait for a complete evaluated run')
    groups=json.loads((RUN/'sampling_groups.json').read_text());plan=json.loads((RUN/'plan.json').read_text())
    ledger=[];counts={'trajectory':0,'edit_scripts':0};outputs={}
    for kind in counts:
        out=RUN/f'{kind}.jsonl.gz'
        if out.exists():raise ValueError('Export already exists; retain frozen archive')
        with gzip.open(out,'wt') as destination:
            for group in groups:
                for arm in plan['arms']:
                    folder=RUN/'generation'/arm/group['group_id'];path=folder/f'{kind}.jsonl.gz'
                    record=json.loads((folder/'results.json').read_text())
                    expected=record['artifacts'][path.name]
                    if sha256_file(path)!=expected:raise ValueError('Group archive differs: '+str(path))
                    with gzip.open(path,'rt') as handle:
                        for line in handle:destination.write(line);counts[kind]+=1
                    if kind=='trajectory':
                        ledger.append({'arm':arm,'group_id':group['group_id'],'seconds':record['seconds'],
                                      'results_sha256':sha256_file(folder/'results.json'),'artifacts':record['artifacts']})
        outputs[out.name]={'sha256':sha256_file(out),'bytes':out.stat().st_size,'records':counts[kind]}
    (RUN/'group_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    outputs['group_ledger.json']={'sha256':sha256_file(RUN/'group_ledger.json')}
    (RUN/'export_checks.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(outputs,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
