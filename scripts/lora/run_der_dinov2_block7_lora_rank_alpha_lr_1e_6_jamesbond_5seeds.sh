#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/lora/new_lora/der_dinov2_block7_lora_rank_alpha_lr_1e_6_jamesbond_${STAMP}}"
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
combos=("1 2" "2 4" "8 16" "16 32")
seeds=(1 2 3 4 5)

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

for combo in "${combos[@]}"; do
  read -r rank alpha <<<"$combo"
  variant="dinov2_lora_block7_r${rank}_a${alpha}_lr_1e_6"
  for seed in "${seeds[@]}"; do
    run_name="der_${variant}_Jamesbond_seed_${seed}"
    run_dir="$RUN_ROOT/r${rank}_a${alpha}/Jamesbond/seed_${seed}"
    log_file="$run_dir/train_stdout.log"
    mkdir -p "$run_dir"

    if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm der --variant "$variant" --game Jamesbond --seed "$seed"; then
      echo "Skipping completed run: ${variant} Jamesbond seed ${seed}"
      continue
    fi

    echo "Starting DER DINOv2 LoRA block7 rank=${rank} alpha=${alpha} lr=1e-6 Jamesbond seed ${seed} on GPU ${GPU}"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
      experiment=der/dinov2_lora_atari100k \
      environment.task=Jamesbond \
      algorithm.dinov2_output_block=7 \
      algorithm.encoder_lr=1e-6 \
      algorithm.adapter_lr=1e-6 \
      algorithm.lora_rank="$rank" \
      algorithm.lora_alpha="$alpha" \
      trainer.seed="$seed" \
      trainer.devices=[0] \
      "run_name=$run_name" \
      "hydra.run.dir=$run_dir" \
      "checkpoint.save_dir=$run_dir/checkpoints" \
      checkpoint.enabled=false \
      >"$log_file" 2>&1
    exit_code="$?"
    set -e

    "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm der --variant "$variant" --game Jamesbond --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
    "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

    if [[ "$exit_code" -ne 0 ]]; then
      echo "Run failed: rank=${rank} alpha=${alpha} seed=${seed}. See ${log_file}"
      exit "$exit_code"
    fi
  done
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
