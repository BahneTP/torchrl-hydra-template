#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/plain_algorithms/der/bankheist_${STAMP}}"
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
GAME="BankHeist"
ALGORITHM="der"
VARIANT="plain_google_noisy_life_loss"
seeds=(1 2 3 4 5)

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

for seed in "${seeds[@]}"; do
  run_name="${ALGORITHM}_${VARIANT}_${GAME}_seed_${seed}"
  run_dir="$RUN_ROOT/seed_${seed}"
  log_file="$run_dir/train_stdout.log"
  mkdir -p "$run_dir"

  if "$PYTHON" scripts/lib/results.py completed \
    --results "$RESULTS_CSV" \
    --algorithm "$ALGORITHM" \
    --variant "$VARIANT" \
    --game "$GAME" \
    --seed "$seed"; then
    echo "Skipping completed run: ${GAME} seed ${seed}"
    continue
  fi

  echo "Starting pure DER ${GAME} seed ${seed} on GPU ${GPU}"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
    experiment=der/atari100k \
    environment.task="$GAME" \
    trainer.seed="$seed" \
    trainer.devices=[0] \
    evaluation.final_num_episodes=100 \
    "run_name=$run_name" \
    "hydra.run.dir=$run_dir" \
    checkpoint.enabled=false \
    >"$log_file" 2>&1
  exit_code="$?"
  set -e

  "$PYTHON" scripts/lib/results.py append \
    --results "$RESULTS_CSV" \
    --algorithm "$ALGORITHM" \
    --variant "$VARIANT" \
    --game "$GAME" \
    --seed "$seed" \
    --exit-code "$exit_code" \
    --run-name "$run_name" \
    --log-file "$log_file"

  "$PYTHON" scripts/lib/results.py summarize \
    --results "$RESULTS_CSV" \
    --summary "$SUMMARY_CSV"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "Run failed: ${GAME} seed ${seed} (exit_code=${exit_code}). See ${log_file}"
    exit "$exit_code"
  fi
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
