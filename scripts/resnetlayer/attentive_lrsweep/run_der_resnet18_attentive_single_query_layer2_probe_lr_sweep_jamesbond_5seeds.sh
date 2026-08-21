#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "$RUN_ROOT" ]]; then
  latest_run_root="$(find logs/resnetlayer/attentive_lrsweep/der_resnet18_attentive_single_query_layer2_probe_lr_sweep_jamesbond_* -maxdepth 0 -type d 2>/dev/null | sort | tail -n 1 || true)"
  RUN_ROOT="${latest_run_root:-logs/resnetlayer/attentive_lrsweep/der_resnet18_attentive_single_query_layer2_probe_lr_sweep_jamesbond_${STAMP}}"
fi
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
ALGORITHM="der"
PROBE_TYPE="single_query"
PROBE_LRS="${PROBE_LRS:-1e-3 1e-4 1e-5 1e-6 1e-7}"

if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

games=(Jamesbond)
seeds=(1 2 3 4 5)

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

for probe_lr in $PROBE_LRS; do
  lr_name="${probe_lr//./p}"
  lr_name="${lr_name//-/_}"
  variant="resnet18_attentive_${PROBE_TYPE}_layer2_probe_lr_${lr_name}"
  for game in "${games[@]}"; do
    for seed in "${seeds[@]}"; do
      run_name="${ALGORITHM}_${variant}_${game}_seed_${seed}"
      run_dir="$RUN_ROOT/probe_lr_${lr_name}/${game}/seed_${seed}"
      log_file="$run_dir/train_stdout.log"
      mkdir -p "$run_dir"

      if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$game" --seed "$seed"; then
        echo "Skipping completed run: ${variant} ${game} seed ${seed}"
        continue
      fi

      echo "Starting DER ResNet18 attentive ${PROBE_TYPE} layer2 probe_lr=${probe_lr} ${game} seed ${seed} on GPU ${GPU}"
      set +e
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
        experiment=der/resnet18_attentive_atari100k \
        environment.task="$game" \
        algorithm.resnet18_variant=resnet_layer2 \
        algorithm.attentive_probe_type="$PROBE_TYPE" \
        algorithm.probe_lr="$probe_lr" \
        trainer.seed="$seed" \
        trainer.devices=[0] \
        "run_name=$run_name" \
        "hydra.run.dir=$run_dir" \
        "checkpoint.save_dir=$run_dir/checkpoints" \
        checkpoint.enabled=false \
        >"$log_file" 2>&1
      exit_code="$?"
      set -e

      "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$game" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
      "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

      if [[ "$exit_code" -ne 0 ]]; then
        echo "Run failed: probe_lr ${probe_lr} ${game} seed ${seed} (exit_code=${exit_code}). See ${log_file}"
        exit "$exit_code"
      fi
    done
  done
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
