set -euo pipefail
cd /home/yhshy/git_files/natvqvae
PY=/home/yhshy/miniconda3/envs/nad_data/bin/python
ROOT=data/results/generation/natv_six_cell
LOG=data/results/evaluation/natv_formal_comparison/generation.log
TEST=data/results/preprocessing/natv_test_recovered
BG=data/results/preprocessing/natv_generation_backgrounds/backgrounds.json
mkdir -p "$ROOT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

batch_complete() {
  local json=$1 checkpoint=$2 distribution=$3
  [ -f "$json" ] || return 1
  "$PY" -c '
import hashlib, json, sys
from pathlib import Path
manifest, checkpoint, distribution, source = sys.argv[1:5]
record = json.loads(Path(manifest).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert record["status"] == "COMPLETED"
assert record["backgrounds"] == 45
assert record["attempts"] == 45 * 6 * 64
assert record["distribution"] == distribution
assert record["checkpoint_sha256"] == digest(checkpoint)
assert record["backgrounds_manifest_sha256"] == digest(source)
' "$json" "$checkpoint" "$distribution" "$BG"
}

echo "$(date -Is) GENERATION BEGIN"
for condition in random self external; do
  for seed in 9201 9202 9203; do
    for distribution in joint site_marginal; do
      run="QVAE_${condition}_seed${seed}_${distribution}"
      checkpoint="data/weights/qvae/fragment_${condition}_seed${seed}/best.pt"
      if batch_complete "$ROOT/$run/batch_generation.json" "$checkpoint" "$distribution"; then
        echo "$(date -Is) batch already complete $run"
        continue
      fi
      PYTHONPATH=src "$PY" -m natvqvae.cli generate \
        --config config/fragment_generation.yaml --sources config/sources.yaml \
        --target config/target.yaml --checkpoint "$checkpoint" --output "$ROOT/$run" \
        --device cuda --draws 64 --distribution "$distribution" \
        --backgrounds-manifest "$BG" --test-data "$TEST"
      batch_complete "$ROOT/$run/batch_generation.json" "$checkpoint" "$distribution"
      echo "$(date -Is) generation complete $run"
    done
    run="deterministic_${condition}_seed${seed}_factorized"
    checkpoint="data/weights/qvae/${condition}_seed${seed}/best.pt"
    if batch_complete "$ROOT/$run/batch_generation.json" "$checkpoint" factorized; then
      echo "$(date -Is) batch already complete $run"
      continue
    fi
    PYTHONPATH=src "$PY" -m natvqvae.cli generate \
      --config config/fragment_generation.yaml --sources config/sources.yaml \
      --target config/target.yaml --checkpoint "$checkpoint" --output "$ROOT/$run" \
      --device cuda --draws 64 \
      --backgrounds-manifest "$BG" --test-data "$TEST"
    batch_complete "$ROOT/$run/batch_generation.json" "$checkpoint" factorized
    echo "$(date -Is) generation complete $run"
  done
done
if [ ! -f data/results/evaluation/natv_six_cell_generation/summary.json ]; then
  PYTHONPATH=src "$PY" -m natvqvae.cli generation-compare \
    --config config/fragment_generation_comparison.yaml \
    --output data/results/evaluation/natv_six_cell_generation
fi
echo "$(date -Is) GENERATION DONE"
