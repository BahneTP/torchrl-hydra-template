#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/bbf-testing/lora/lora_rank_alpha_sweep_all_lr_1e_4_with_shrink_perturb_jamesbond_seed1_${STAMP}}"
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
game="Jamesbond"
seed=1
combinations=("1 2" "2 4" "8 16" "16 32")

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

for combination in "${combinations[@]}"; do
  read -r rank alpha <<<"$combination"
  variant="resnet18_layer2_lora_r${rank}_a${alpha}_all_lr_1e_4_with_shrink_perturb"
  run_name="bbf_${variant}_${game}_seed_${seed}"
  run_dir="$RUN_ROOT/r${rank}_a${alpha}/${game}/seed_${seed}"
  log_file="$run_dir/train_stdout.log"
  mkdir -p "$run_dir"

  echo "Starting BBF LoRA rank=${rank} alpha=${alpha} all_lr=1e-4 ${game} seed ${seed} on GPU ${GPU}"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
    experiment=bbf/resnet18_layer2_lora_atari100k \
    environment.task="$game" \
    evaluation.final_num_episodes=100 \
    algorithm.lr=1e-4 \
    algorithm.encoder_lr=1e-4 \
    algorithm.adapter_lr=1e-4 \
    algorithm.lora_rank="$rank" \
    algorithm.lora_alpha="$alpha" \
    algorithm.reset_interval=40000 \
    trainer.seed="$seed" \
    trainer.devices=[0] \
    "run_name=$run_name" \
    "hydra.run.dir=$run_dir" \
    "checkpoint.save_dir=$run_dir/checkpoints" \
    checkpoint.enabled=false \
    >"$log_file" 2>&1
  exit_code="$?"
  set -e

  "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm bbf --variant "$variant" --game "$game" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
  "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "Run failed: rank=${rank} alpha=${alpha}. See ${log_file}"
    exit "$exit_code"
  fi
done
