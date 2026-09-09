#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-logs/lora/lr_sweep/der_dinov2_block12_lora_dense_adapter_r4_a8_lr_sweep_jamesbond_20260827_231603}"
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
lrs=(1e-6 1e-7)

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"
for lr in "${lrs[@]}"; do
  lr_name="${lr//-/_}"
  variant="dinov2_lora_block12_dense_adapter_r4_a8_lr_${lr_name}"
  for seed in 1 2 3 4 5; do
    run_name="der_${variant}_Jamesbond_seed_${seed}"
    run_dir="$RUN_ROOT/lr_${lr_name}/Jamesbond/seed_${seed}"
    log_file="$run_dir/train_stdout.log"
    mkdir -p "$run_dir"
    if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm der --variant "$variant" --game Jamesbond --seed "$seed"; then
      echo "Skipping completed run: ${variant} Jamesbond seed ${seed}"
      continue
    fi
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
      experiment=der/dinov2_lora_atari100k environment.task=Jamesbond \
      algorithm.dinov2_output_block=12 algorithm.lr=1e-4 \
      algorithm.encoder_lr="$lr" algorithm.adapter_lr=1e-4 \
      algorithm.lora_rank=4 algorithm.lora_alpha=8 \
      trainer.seed="$seed" trainer.devices=[0] "run_name=$run_name" \
      "hydra.run.dir=$run_dir" "checkpoint.save_dir=$run_dir/checkpoints" \
      checkpoint.enabled=false >"$log_file" 2>&1
    exit_code="$?"
    set -e
    "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm der --variant "$variant" --game Jamesbond --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
    "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"
    [[ "$exit_code" -eq 0 ]] || exit "$exit_code"
  done
done
