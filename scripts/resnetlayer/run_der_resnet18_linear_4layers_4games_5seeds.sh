#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "$RUN_ROOT" ]]; then
  latest_run_root="$(find logs/resnetlayer/der_resnet18_linear_* -maxdepth 0 -type d 2>/dev/null | sort | tail -n 1 || true)"
  RUN_ROOT="${latest_run_root:-logs/resnetlayer/der_resnet18_linear_${STAMP}}"
fi
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
ALGORITHM="der"
PROBE="linear"
LAYERS="${LAYERS:-1 2 3 4}"

if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

games=(Jamesbond)
seeds=(1 2 3 4 5)

"$PYTHON" scripts/lib/results.py init \
  --results "$RESULTS_CSV" \
  --summary "$SUMMARY_CSV"

for layer in $LAYERS; do
  variant="resnet18_${PROBE}_layer${layer}"
  resnet_variant="resnet_layer${layer}"
  for game in "${games[@]}"; do
    for seed in "${seeds[@]}"; do
      run_name="${ALGORITHM}_${variant}_${game}_seed_${seed}"
      run_dir="$RUN_ROOT/layer_${layer}/${game}/seed_${seed}"
      log_file="$run_dir/train_stdout.log"
      mkdir -p "$run_dir"

      if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$game" --seed "$seed"; then
        echo "Skipping completed run: ${variant} ${game} seed ${seed}"
        continue
      fi

      echo "Starting DER ResNet18 ${PROBE} layer ${layer} ${game} seed ${seed} on GPU ${GPU}"
      set +e
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
        experiment=der/resnet18_linear_atari100k \
        environment.task="$game" \
        algorithm.resnet18_variant="$resnet_variant" \
        trainer.seed="$seed" \
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
        --variant "$variant" \
        --game "$game" \
        --seed "$seed" \
        --exit-code "$exit_code" \
        --run-name "$run_name" \
        --log-file "$log_file"

      "$PYTHON" scripts/lib/results.py summarize \
        --results "$RESULTS_CSV" \
        --summary "$SUMMARY_CSV"

      if [[ "$exit_code" -ne 0 ]]; then
        echo "Run failed: layer ${layer} ${game} seed ${seed} (exit_code=${exit_code}). See ${log_file}"
        exit "$exit_code"
      fi
    done
  done
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
