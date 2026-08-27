#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/bbf-testing/lora_lr_sweep_with_shrink_perturb_jamesbond_seed1_${STAMP}}"
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
ALGORITHM="bbf"

if [[ -z "$PYTHON" ]]; then
  PYTHON=".venv/bin/python"
fi

game="Jamesbond"
seed=1
lrs=(1e-5 1e-6 1e-7 1e-8)

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

for lora_lr in "${lrs[@]}"; do
  lr_name="${lora_lr//-/_}"
  VARIANT="resnet18_layer2_lora_lr_${lr_name}_with_shrink_perturb"
  run_name="${ALGORITHM}_${VARIANT}_${game}_seed_${seed}"
  run_dir="$RUN_ROOT/lora_lr_${lr_name}/${game}/seed_${seed}"
  log_file="$run_dir/train_stdout.log"
  mkdir -p "$run_dir"

  echo "Starting BBF ResNet18 layer2 LoRA lr=${lora_lr} with shrink-and-perturb ${game} seed ${seed} on GPU ${GPU}"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
    experiment=bbf/resnet18_layer2_lora_atari100k \
    environment.task="$game" \
    evaluation.final_num_episodes=100 \
    algorithm.lr=1e-4 \
    algorithm.encoder_lr="$lora_lr" \
    algorithm.adapter_lr=1e-4 \
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
    echo "Run failed: ${game} seed ${seed} lora_lr=${lora_lr} (exit_code=${exit_code}). See ${log_file}"
    exit "$exit_code"
  fi
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
