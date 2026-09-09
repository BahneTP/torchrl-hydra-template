#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_SUFFIX="${RUN_SUFFIX:-3games}"
RUN_ROOT="${RUN_ROOT:-logs/lora/new_lora/der_dinov2_block7_lora_r4_a8_lr_1e_6_${RUN_SUFFIX}_${STAMP}}"
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
VARIANT="dinov2_lora_block7_r4_a8_lr_1e_6"
read -r -a games <<<"${GAMES:-Assault BankHeist RoadRunner}"
seeds=(1 2 3 4 5)

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

for game in "${games[@]}"; do
  for seed in "${seeds[@]}"; do
    run_name="der_${VARIANT}_${game}_seed_${seed}"
    run_dir="$RUN_ROOT/${game}/seed_${seed}"
    log_file="$run_dir/train_stdout.log"
    mkdir -p "$run_dir"

    if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm der --variant "$VARIANT" --game "$game" --seed "$seed"; then
      echo "Skipping completed run: ${VARIANT} ${game} seed ${seed}"
      continue
    fi

    echo "Starting DER DINOv2 LoRA block7 rank=4 alpha=8 lr=1e-6 ${game} seed ${seed} on GPU ${GPU}"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
      experiment=der/dinov2_lora_atari100k \
      environment.task="$game" \
      algorithm.dinov2_output_block=7 \
      algorithm.encoder_lr=1e-6 \
      algorithm.adapter_lr=1e-6 \
      algorithm.lora_rank=4 \
      algorithm.lora_alpha=8 \
      trainer.seed="$seed" \
      trainer.devices=[0] \
      "run_name=$run_name" \
      "hydra.run.dir=$run_dir" \
      "checkpoint.save_dir=$run_dir/checkpoints" \
      checkpoint.enabled=false \
      >"$log_file" 2>&1
    exit_code="$?"
    set -e

    "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm der --variant "$VARIANT" --game "$game" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
    "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

    if [[ "$exit_code" -ne 0 ]]; then
      echo "Run failed: ${VARIANT} ${game} seed ${seed}. See ${log_file}"
      exit "$exit_code"
    fi
  done
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
