#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/bbf-testing/lora_rank_alpha_sweep_lr_1e_8_with_shrink_perturb_jamesbond_seed1_${STAMP}}"
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
ALGORITHM="bbf"

if [[ -z "$PYTHON" ]]; then
  PYTHON=".venv/bin/python"
fi

game="Jamesbond"
seed=1
lora_lr="1e-8"
combos=("1 2" "2 4" "8 16" "16 32")

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

for combo in "${combos[@]}"; do
  read -r rank alpha <<<"$combo"
  VARIANT="resnet18_layer2_lora_r${rank}_a${alpha}_lr_1e_8_with_shrink_perturb"
  run_name="${ALGORITHM}_${VARIANT}_${game}_seed_${seed}"
  run_dir="$RUN_ROOT/r${rank}_a${alpha}/${game}/seed_${seed}"
  log_file="$run_dir/train_stdout.log"
  mkdir -p "$run_dir"

  echo "Starting BBF ResNet18 layer2 LoRA rank=${rank} alpha=${alpha} lr=${lora_lr} with shrink-and-perturb ${game} seed ${seed} on GPU ${GPU}"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
    experiment=bbf/resnet18_layer2_lora_atari100k \
    environment.task="$game" \
    evaluation.final_num_episodes=100 \
    algorithm.lr=1e-4 \
    algorithm.encoder_lr="$lora_lr" \
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

  "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$VARIANT" --game "$game" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
  "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "Run failed: rank=${rank} alpha=${alpha} lora_lr=${lora_lr} ${game} seed ${seed} (exit_code=${exit_code}). See ${log_file}"
    exit "$exit_code"
  fi
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
