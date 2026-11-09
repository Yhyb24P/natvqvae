set -eu
cd /home/yhshy/git_files/natvqvae
PY=/home/yhshy/miniconda3/envs/nad_data/bin/python
OUT=data/results/evaluation/natv_fragment_transfer_deterministic/validation
mkdir -p "$OUT/logs"
for condition in random self external; do
  for seed in 9201 9202 9203; do
    run="${condition}_seed${seed}"
    "$PY" -m natvqvae.cli evaluate --config config/transfer.yaml \
      --checkpoint "data/weights/qvae/${run}/best.pt" \
      --output "$OUT/${run}" --split validation --device cuda \
      > "$OUT/logs/${run}.log" 2>&1
  done
done
