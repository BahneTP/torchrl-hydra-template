#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "$RUN_ROOT" ]]; then
  latest_run_root="$(find logs/mixedlayerfusion/der_resnet18_dinov2_mlf_self_attention_probe_lr_1e_6_jamesbond_* -maxdepth 0 -type d 2>/dev/null | sort | tail -n 1 || true)"
  RUN_ROOT="${latest_run_root:-logs/mixedlayerfusion/der_resnet18_dinov2_mlf_self_attention_probe_lr_1e_6_jamesbond_${STAMP}}"
fi
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
ALGORITHM="der"
PROBE_TYPE="self_attention"
PROBE_LR="1e-6"
GAME="Jamesbond"

if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

seeds=(1 2 3 4 5)
resnet_cases=(
  "all|[1,2,3,4]"
)
dino_cases=(
  "all|[1,2,3,4,5,6,7,8,9,10,11,12]"
)

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

for case_spec in "${resnet_cases[@]}"; do
  case_name="${case_spec%%|*}"
  layers="${case_spec#*|}"
  variant="resnet18_mlf_${PROBE_TYPE}_${case_name}_probe_lr_1e_6"
  for seed in "${seeds[@]}"; do
    run_name="${ALGORITHM}_${variant}_${GAME}_seed_${seed}"
    run_dir="$RUN_ROOT/resnet18/${case_name}/${GAME}/seed_${seed}"
    log_file="$run_dir/train_stdout.log"
    mkdir -p "$run_dir"

    if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$GAME" --seed "$seed"; then
      echo "Skipping completed run: ${variant} ${GAME} seed ${seed}"
      continue
    fi

    echo "Starting DER ResNet18 MLF ${PROBE_TYPE} ${case_name} probe_lr=${PROBE_LR} ${GAME} seed ${seed} on GPU ${GPU}"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
      experiment=der/resnet18_attentive_atari100k \
      environment.task="$GAME" \
      algorithm.transfer_layer_mix=true \
      algorithm.resnet18_mix_layers="$layers" \
      algorithm.attentive_probe_type="$PROBE_TYPE" \
      algorithm.probe_lr="$PROBE_LR" \
      trainer.seed="$seed" \
      trainer.devices=[0] \
      "run_name=$run_name" \
      "hydra.run.dir=$run_dir" \
      "checkpoint.save_dir=$run_dir/checkpoints" \
      checkpoint.enabled=false \
      >"$log_file" 2>&1
    exit_code="$?"
    set -e

    "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$GAME" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
    "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

    if [[ "$exit_code" -ne 0 ]]; then
      echo "Run failed: ${variant} ${GAME} seed ${seed} (exit_code=${exit_code}). See ${log_file}"
      exit "$exit_code"
    fi
  done
done

for case_spec in "${dino_cases[@]}"; do
  case_name="${case_spec%%|*}"
  blocks="${case_spec#*|}"
  variant="dinov2_mlf_${PROBE_TYPE}_${case_name}_probe_lr_1e_6"
  for seed in "${seeds[@]}"; do
    run_name="${ALGORITHM}_${variant}_${GAME}_seed_${seed}"
    run_dir="$RUN_ROOT/dinov2/${case_name}/${GAME}/seed_${seed}"
    log_file="$run_dir/train_stdout.log"
    mkdir -p "$run_dir"

    if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$GAME" --seed "$seed"; then
      echo "Skipping completed run: ${variant} ${GAME} seed ${seed}"
      continue
    fi

    echo "Starting DER DINOv2 MLF ${PROBE_TYPE} ${case_name} probe_lr=${PROBE_LR} ${GAME} seed ${seed} on GPU ${GPU}"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
      experiment=der/dinov2_attentive_atari100k \
      environment.task="$GAME" \
      algorithm.transfer_layer_mix=true \
      algorithm.dinov2_mix_blocks="$blocks" \
      algorithm.attentive_probe_type="$PROBE_TYPE" \
      algorithm.probe_lr="$PROBE_LR" \
      trainer.seed="$seed" \
      trainer.devices=[0] \
      "run_name=$run_name" \
      "hydra.run.dir=$run_dir" \
      "checkpoint.save_dir=$run_dir/checkpoints" \
      checkpoint.enabled=false \
      >"$log_file" 2>&1
    exit_code="$?"
    set -e

    "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$GAME" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
    "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

    if [[ "$exit_code" -ne 0 ]]; then
      echo "Run failed: ${variant} ${GAME} seed ${seed} (exit_code=${exit_code}). See ${log_file}"
      exit "$exit_code"
    fi
  done
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
