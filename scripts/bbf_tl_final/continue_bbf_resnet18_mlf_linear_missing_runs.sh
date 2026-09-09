#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-logs/bbf_tl_final/bbf_resnet18_mlf_linear_all_3games_20260821_234445}"
RESULTS="$RUN_ROOT/results.csv"
SUMMARY="$RUN_ROOT/summary.csv"
VARIANT="resnet18_mlf_linear_all"
jobs=("BankHeist 4" "BankHeist 5" "RoadRunner 1" "RoadRunner 2" "RoadRunner 3" "RoadRunner 4" "RoadRunner 5")

"$PYTHON" scripts/lib/results.py init --results "$RESULTS" --summary "$SUMMARY"
for job in "${jobs[@]}"; do
  read -r game seed <<<"$job"
  run_name="bbf_${VARIANT}_${game}_seed_${seed}"
  run_dir="$RUN_ROOT/$game/seed_${seed}"
  log_file="$run_dir/train_stdout.log"
  mkdir -p "$run_dir"
  if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS" --algorithm bbf --variant "$VARIANT" --game "$game" --seed "$seed"; then
    echo "Skipping completed run: ${game} seed ${seed}"
    continue
  fi
  echo "Starting missing BBF MLF linear run ${game} seed ${seed} on GPU ${GPU}"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
    experiment=bbf/resnet18_mlf_linear_atari100k environment.task="$game" \
    evaluation.final_num_episodes=100 \
    algorithm.lr=1e-4 algorithm.adapter_lr=1e-4 \
    algorithm.resnet18_mix_layers=[1,2,3,4] algorithm.reset_interval=40000 \
    trainer.seed="$seed" trainer.devices=[0] \
    "run_name=$run_name" "hydra.run.dir=$run_dir" \
    "checkpoint.save_dir=$run_dir/checkpoints" checkpoint.enabled=false \
    >"$log_file" 2>&1
  exit_code="$?"
  set -e
  "$PYTHON" scripts/lib/results.py append --results "$RESULTS" --algorithm bbf --variant "$VARIANT" --game "$game" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
  "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS" --summary "$SUMMARY"
  [[ "$exit_code" -eq 0 ]] || exit "$exit_code"
done

echo "Wrote $RESULTS"
echo "Wrote $SUMMARY"
