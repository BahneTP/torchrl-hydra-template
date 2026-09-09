#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "$RUN_ROOT" ]]; then
  latest="$(find logs/bbf_tl_final/linear_3games_5seeds_* -maxdepth 0 -type d 2>/dev/null | sort | tail -n 1 || true)"
  RUN_ROOT="${latest:-logs/bbf_tl_final/linear_3games_5seeds_${STAMP}}"
fi
RESULTS="$RUN_ROOT/results.csv"
SUMMARY="$RUN_ROOT/summary.csv"
variant="resnet18_layer2_linear_all_lr_1e_4"

"$PYTHON" scripts/lib/results.py init --results "$RESULTS" --summary "$SUMMARY"
for game in Assault BankHeist RoadRunner; do
  for seed in 1 2 3 4 5; do
    run_name="bbf_${variant}_${game}_seed_${seed}"
    run_dir="$RUN_ROOT/$game/seed_${seed}"
    log_file="$run_dir/train_stdout.log"
    mkdir -p "$run_dir"
    if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS" --algorithm bbf --variant "$variant" --game "$game" --seed "$seed"; then
      echo "Skipping completed run: ${game} seed ${seed}"
      continue
    fi
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
      experiment=bbf/resnet18_layer2_linear_atari100k environment.task="$game" \
      algorithm.lr=1e-4 algorithm.encoder_lr=1e-4 \
      algorithm.adapter_lr=1e-4 algorithm.probe_lr=1e-4 \
      algorithm.reset_interval=40000 trainer.seed="$seed" trainer.devices=[0] \
      "run_name=$run_name" "hydra.run.dir=$run_dir" \
      "checkpoint.save_dir=$run_dir/checkpoints" checkpoint.enabled=false \
      >"$log_file" 2>&1
    exit_code="$?"
    set -e
    "$PYTHON" scripts/lib/results.py append --results "$RESULTS" --algorithm bbf --variant "$variant" --game "$game" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
    "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS" --summary "$SUMMARY"
    [[ "$exit_code" -eq 0 ]] || exit "$exit_code"
  done
done
