#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/bbf-testing/linear/linear_conv_probe_lr_1e_4_without_shrink_perturb_jamesbond_seed1_${STAMP}}"
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
ALGORITHM="bbf"
VARIANT="resnet18_layer2_linear_conv_probe_lr_1e_4_without_shrink_perturb"
game="Jamesbond"
seed=1

if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

run_name="${ALGORITHM}_${VARIANT}_${game}_seed_${seed}"
run_dir="$RUN_ROOT/$game/seed_${seed}"
log_file="$run_dir/train_stdout.log"
mkdir -p "$run_dir"

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

echo "Starting BBF ResNet18 layer2 linear+conv probe_lr=1e-4 without shrink-and-perturb ${game} seed ${seed} on GPU ${GPU}"
set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
  experiment=bbf/resnet18_layer2_linear_conv_atari100k \
  environment.task="$game" \
  evaluation.final_num_episodes=100 \
  algorithm.lr=1e-4 \
  algorithm.adapter_lr=1e-4 \
  algorithm.probe_lr=1e-4 \
  algorithm.reset_interval=0 \
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
  echo "Run failed: ${game} seed ${seed} (exit_code=${exit_code}). See ${log_file}"
  exit "$exit_code"
fi

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
