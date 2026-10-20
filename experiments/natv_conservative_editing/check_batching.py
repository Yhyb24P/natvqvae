"""Measure real input equivalence and throughput before restarting generation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,gzip,time
from pathlib import Path
import numpy as np
import torch
RUN=Path(__file__).resolve().parent;sys.path.insert(0,str(RUN.parents[1]/'src'))
from natvqvae.conditional_backend import ConditionalBackend
from natvqvae.conservative_editing import AA

torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
source=Path(json.loads((RUN/'inputs/source_checks.json').read_text())['source_root'])
b=ConditionalBackend(source,RUN);folder=RUN/'batching_check';folder.mkdir(exist_ok=False)
seq=[];pos=[]
paths=sorted((RUN/'interrupted_slow_run/generation/SE').glob('substitution*/trajectory.jsonl.gz'))[:8]
for path in paths:
 with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
 for row in rows:
  if row['stage']==1:seq.append(row['before']);pos.append(row['current_positions_0based'])
assert len(seq)==64
start=time.monotonic();small,e1,r1,u1=b.features(seq[:8],pos[:8],folder/'small');torch.cuda.synchronize();t1=time.monotonic()-start
start=time.monotonic();large,e2,r2,u2=b.features(seq,pos,folder/'large');torch.cuda.synchronize();t2=time.monotonic()-start
np.testing.assert_array_equal(e1.cpu().numpy(),e2[:8].cpu().numpy())
assert r1==r2[:8]
model=b.model('SE',20261081)
l1,_=b.logits('SE',model,small,e1,None);l2,_=b.logits('SE',model,large,e2,None)
mask=small['target_mask'];delta=(l1-l2[:8]).abs()[mask]
result={'queries':64,'reference_queries':8,'reference_seconds':t1,'batch_seconds':t2,'throughput_speedup':8*t1/t2,
        'retrieval_features_bitwise_equal':True,'retrieval_receipts_equal':True,'masked_SE_logits_max_abs_difference':float(delta.max()),
        'masked_SE_logits_mean_abs_difference':float(delta.mean()),'timing':b.timing,'max_allocated_GB':torch.cuda.max_memory_allocated()/1e9,
        'interpretation':'batching can change bfloat16 arithmetic; restarted run has a distinct numerical implementation'}
(folder/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
assert float(delta.max())<0.25
