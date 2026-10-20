"""Within-model context counterfactuals; auxiliary mechanism probes, not a method ranking."""
from pathlib import Path
import gzip,hashlib,json,sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from natvqvae.conditional_backend import ConditionalBackend
from natvqvae.conservative_editing import scaffold,AA
from natvqvae.intake import sha256_file
RUN=Path(__file__).parent


def main():
    status=json.loads((RUN/'run_status.json').read_text())
    if status['status']!='COMPLETED':raise ValueError('Run the frozen comparison before auxiliary probes')
    groups=json.loads((RUN/'sampling_groups.json').read_text());plan=json.loads((RUN/'plan.json').read_text())
    source=Path(json.loads((RUN/'inputs/source_checks.json').read_text())['source_root'])
    out=RUN/'context_diagnostic';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    backend=ConditionalBackend(source,RUN);selected={}
    for group in groups:selected.setdefault((group['mode'],group['budget']),group)
    records=[];inputs=[]
    for group in selected.values():
        base=scaffold(group['event'])
        for arm in ['ESM2','SE','QVAE']:
            path=RUN/'generation'/arm/group['group_id']/'trajectory.jsonl.gz'
            trace=[json.loads(line) for line in gzip.open(path,'rt')]
            row=[r for r in trace if r['draw']==0][-1]
            actual=row['before'];counterfactual=base.sequence;positions=row['current_positions_0based']
            model=backend.model(arm,group['seed']);z=None
            if arm=='QVAE':
                rng=torch.Generator(device='cuda').manual_seed(group['prior_seed'])
                prior=model.sample_prior(plan['draws_per_shared_mask'],generator=rng,device='cuda')[0]
                if hashlib.sha256(prior.cpu().numpy().tobytes()).hexdigest()!=row['latent_sha256']:raise ValueError('Prior mismatch')
                z=prior[None].expand(2,-1,-1)
            encoded,evolution,receipts,_=backend.features([actual,counterfactual],[positions,positions],out/f'{arm}_{group["group_id"]}',with_retrieval=arm!='ESM2')
            logits,_=backend.logits(arm,model,encoded,evolution,z)
            logits=logits[:,positions].float();probability=logits.softmax(-1).clamp_min(1e-12)
            divergence=(probability[0]*(probability[0].log()-probability[1].log())).sum(-1).mean()
            record={'arm':arm,'group_id':group['group_id'],'mode':group['mode'],'budget':group['budget'],
                    'seed':group['seed'],'visible_changed_residues':sum(a!=b for i,(a,b) in enumerate(zip(actual,counterfactual)) if i not in positions),
                    'masked_sites':len(positions),'target_logit_max_difference':float((logits[0]-logits[1]).abs().max()),
                    'target_probability_L1_mean':float((probability[0]-probability[1]).abs().sum(-1).mean()),
                    'target_KL_actual_vs_counterfactual':float(divergence)}
            records.append(record);inputs.append({'group_id':group['group_id'],'arm':arm,'actual':actual,
                  'WT_scaffold_counterfactual':counterfactual,'positions':positions,'fixed_latent_sha256':row['latent_sha256'],
                  'retrieval_receipts':receipts,'source_trajectory_sha256':sha256_file(path)})
    pd.DataFrame(records).to_csv(out/'context_response.csv',index=False)
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (out/'identity.json').write_text(json.dumps({'code_sha256':sha256_file(Path(__file__)),
         'selection':'first_seed_and_first_mask_or_first_event_per_mode_budget; own_trajectory_draw0_last_stage',
         'scope':'within_model_context_response_only; parents_differ_between_models; not_a_primary_generation_judge',
         'rows':len(records),'artifacts':{p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}},indent=2)+'\n')
    print(pd.DataFrame(records).groupby('arm')[['visible_changed_residues','target_probability_L1_mean']].mean().to_string(),flush=True)

if __name__=='__main__':main()
