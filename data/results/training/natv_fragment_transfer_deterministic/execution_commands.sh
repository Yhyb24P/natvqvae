set -u
cd /home/yhshy/git_files/natvqvae
PY=/home/yhshy/miniconda3/envs/nad_data/bin/python
ROOT=data/results/training/natv_fragment_transfer_deterministic
mkdir -p "$ROOT/logs"
DRIVER="$ROOT/logs/driver.log"
run() {
  local name="$1"; shift
  echo "$(date -Is) CMD $*" >> "$DRIVER"
  local start=$(date +%s)
  if "$@" >> "$ROOT/logs/$name.log" 2>&1; then
    echo "$(date -Is) EXIT 0 $name seconds=$(( $(date +%s) - start ))" >> "$DRIVER"
  else
    local code=$?
    echo "$(date -Is) EXIT $code $name seconds=$(( $(date +%s) - start ))" >> "$DRIVER"
    exit "$code"
  fi
}
echo "$(date -Is) DRIVER start root=$ROOT" >> "$DRIVER"
for c in external self; do
  for s in 9101 9102 9103; do
    run "${c}_pretraining_seed$s" "$PY" -m natvqvae.cli aux-train \
      --config "config/fragment_aux_${c}_deterministic.yaml" --seed "$s" --device cuda \
      --output "$ROOT/${c}_pretraining_seed$s"
  done
done
for s in 9201 9202 9203; do
  run "random_seed$s" "$PY" -m natvqvae.cli train --config config/transfer.yaml \
    --seed "$s" --device cuda --arm deterministic --output "$ROOT/random_seed$s"
done
for c in self external; do
  for s in 9201 9202 9203; do
    a=91; d=92
    seed_a=91${s#92}
    ckpt=$("$PY" -c "import json;print(json.load(open('$ROOT/${c}_pretraining_seed${seed_a}/training_summary.json'))['arms']['deterministic']['checkpoints']['last']['path'])")
    run "${c}_seed$s" "$PY" -m natvqvae.cli train --config config/transfer.yaml \
      --init-checkpoint "$ckpt" --seed "$s" --device cuda --arm deterministic \
      --output "$ROOT/${c}_seed$s"
  done
done
echo "$(date -Is) DRIVER complete" >> "$DRIVER"
