#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
RUN_ROOT="${RUN_ROOT:-logs/plain_algorithms/der/atari100k_20260810_181403}"
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
ALGORITHM="der"
VARIANT="plain"
GAME="BankHeist"
SEED="6"

if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

"$PYTHON" scripts/lib/results.py init \
  --results "$RESULTS_CSV" \
  --summary "$SUMMARY_CSV"

run_name="${ALGORITHM}_${VARIANT}_${GAME}_seed_${SEED}"
run_dir="$RUN_ROOT/${GAME}/seed_${SEED}"
log_file="$run_dir/train_stdout.log"
mkdir -p "$run_dir"

if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$VARIANT" --game "$GAME" --seed "$SEED"; then
  echo "Skipping completed run: ${GAME} seed ${SEED}"
  exit 0
fi

echo "Starting DER ${GAME} seed ${SEED} on GPU ${GPU}"
set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
  experiment=der/atari100k \
  environment.task="$GAME" \
  algorithm.eval_noise=true \
  evaluation.final_num_episodes=100 \
  trainer.seed="$SEED" \
  trainer.devices=[0] \
  "run_name=$run_name" \
  "hydra.run.dir=$run_dir" \
  "checkpoint.save_dir=$run_dir/checkpoints" \
  checkpoint.enabled=false \
  >"$log_file" 2>&1
exit_code="$?"
set -e

"$PYTHON" scripts/lib/results.py append \
  --results "$RESULTS_CSV" \
  --algorithm "$ALGORITHM" \
  --variant "$VARIANT" \
  --game "$GAME" \
  --seed "$SEED" \
  --exit-code "$exit_code" \
  --run-name "$run_name" \
  --log-file "$log_file"

"$PYTHON" scripts/lib/results.py summarize \
  --results "$RESULTS_CSV" \
  --summary "$SUMMARY_CSV"

if [[ "$exit_code" -ne 0 ]]; then
  echo "Run failed: ${GAME} seed ${SEED} (exit_code=${exit_code}). See ${log_file}"
  exit "$exit_code"
fi

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
