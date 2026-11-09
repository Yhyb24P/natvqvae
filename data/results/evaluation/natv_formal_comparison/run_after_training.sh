set -euo pipefail
cd /home/yhshy/git_files/natvqvae
PY=/home/yhshy/miniconda3/envs/nad_data/bin/python
LOG=data/results/evaluation/natv_formal_comparison/queue.log
TRAIN=data/results/training/natv_fragment_transfer_deterministic
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date -Is) BEGIN"
for condition in random self external; do
  for seed in 9201 9202 9203; do
    PYTHONPATH=src "$PY" - "$TRAIN/${condition}_seed${seed}/training_summary.json" "$seed" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['status']=='COMPLETED', sys.argv[1]
assert d.get('arm')=='deterministic', (sys.argv[1], d.get('arm'))
assert d['seed']==int(sys.argv[2]) and d['steps']==2704, sys.argv[1]
PY
  done
done
echo "$(date -Is) training verified"
PYTHONPATH=src "$PY" -m natvqvae.cli evaluate-batch --config config/transfer.yaml \
  --runs config/fragment_evaluation_batch_validation.yaml --split validation --device cuda
echo "$(date -Is) deterministic validation complete"
ESM_TEST=data/results/evaluation/esm2_natv_test_gpu_bf16
if [ ! -f "$ESM_TEST/evaluation.json" ]; then
  PYTHONPATH=src "$PY" -m natvqvae.cli esm-evaluate --config config/transfer.yaml \
    --sources config/sources.yaml --target config/target.yaml --output "$ESM_TEST" \
    --split test --test-data data/results/preprocessing/natv_test_recovered --device cuda
fi
"$PY" - "$ESM_TEST" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]) / 'evaluation.json'
assert p.is_file(), 'missing GPU ESM test baseline'
d=json.load(open(p)); assert d['status']=='COMPLETED' and d['split']=='test', d.get('status')
identity=json.load(open(Path(sys.argv[1]) / 'identity.json'))
assert 'bfloat16' in identity['precision'], identity['precision']
PY
mkdir -p data/results/evaluation/natv_six_cell
echo "$(date -Is) GPU ESM test baseline verified"
PYTHONPATH=src "$PY" -m natvqvae.cli evaluate-batch --config config/transfer.yaml \
  --runs config/fragment_evaluation_batch_test.yaml --split test \
  --test-data data/results/preprocessing/natv_test_recovered --device cuda
echo "$(date -Is) test eval complete"
for split in validation test; do
  PYTHONPATH=src "$PY" -m natvqvae.cli six-cell-compare \
    --config config/fragment_six_cell_comparison.yaml \
    --output "data/results/evaluation/natv_six_cell/${split}" --split "$split"
  echo "$(date -Is) six-cell ${split} complete"
done
LATENT=data/results/evaluation/natv_six_cell/test_latent_precision
if [ ! -f "$LATENT/summary.json" ]; then
  PYTHONPATH=src "$PY" -m natvqvae.cli latent-precision \
    --comparison-dir data/results/evaluation/natv_six_cell/test/QVAE --output "$LATENT"
fi
echo "$(date -Is) test latent precision complete"
sha256sum config/transfer.yaml config/fragment_six_cell_comparison.yaml \
  src/natvqvae/fragment_comparison.py src/natvqvae/evaluate.py src/natvqvae/train.py \
  src/natvqvae/cli.py > data/results/evaluation/natv_formal_comparison/code_config_sha256.txt
echo "$(date -Is) DONE"
