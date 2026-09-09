#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-logs/bbf_tl_final/mlf_attentive_lr_1e_4_assault_${STAMP}}"
RESULTS="$RUN_ROOT/results.csv"
SUMMARY="$RUN_ROOT/summary.csv"
VARIANT="resnet18_mlf_attentive_self_attention_all_lr_1e_4"
GAME="Assault"

"$PYTHON" scripts/lib/results.py init --results "$RESULTS" --summary "$SUMMARY"
for seed in 1 2 3 4 5; do
  run_name="bbf_${VARIANT}_${GAME}_seed_${seed}"
  run_dir="$RUN_ROOT/$GAME/seed_${seed}"
  log_file="$run_dir/train_stdout.log"
  mkdir -p "$run_dir"
  if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS" --algorithm bbf --variant "$VARIANT" --game "$GAME" --seed "$seed"; then
    echo "Skipping completed run: ${GAME} seed ${seed}"
    continue
  fi
  echo "Starting BBF MLF attentive lr=1e-4 ${GAME} seed ${seed} on GPU ${GPU}"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
    experiment=bbf/resnet18_mlf_attentive_atari100k environment.task="$GAME" \
    evaluation.final_num_episodes=100 \
    algorithm.lr=1e-4 algorithm.encoder_lr=1e-4 algorithm.adapter_lr=1e-4 algorithm.probe_lr=1e-4 \
    algorithm.resnet18_mix_layers=[1,2,3,4] algorithm.attentive_probe_type=self_attention \
    algorithm.reset_interval=40000 trainer.seed="$seed" trainer.devices=[0] \
    "run_name=$run_name" "hydra.run.dir=$run_dir" \
    "checkpoint.save_dir=$run_dir/checkpoints" checkpoint.enabled=false \
    >"$log_file" 2>&1
  exit_code="$?"
  set -e
  "$PYTHON" scripts/lib/results.py append --results "$RESULTS" --algorithm bbf --variant "$VARIANT" --game "$GAME" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
  "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS" --summary "$SUMMARY"
  [[ "$exit_code" -eq 0 ]] || exit "$exit_code"
done
