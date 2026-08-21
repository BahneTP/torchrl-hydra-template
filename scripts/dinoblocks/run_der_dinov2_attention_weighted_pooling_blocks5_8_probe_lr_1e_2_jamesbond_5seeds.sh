#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "$RUN_ROOT" ]]; then
  latest_run_root="$(find logs/dinoblocks/der_dinov2_attention_weighted_pooling_blocks5_8_probe_lr_1e_2_jamesbond_* -maxdepth 0 -type d 2>/dev/null | sort | tail -n 1 || true)"
  RUN_ROOT="${latest_run_root:-logs/dinoblocks/der_dinov2_attention_weighted_pooling_blocks5_8_probe_lr_1e_2_jamesbond_${STAMP}}"
fi
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
ALGORITHM="der"
PROBE_TYPE="attention_weighted_pooling"
PROBE_LR="1e-2"
BLOCKS="${BLOCKS:-5 6 7 8}"

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

for block in $BLOCKS; do
  variant="dinov2_attentive_${PROBE_TYPE}_block${block}_probe_lr_1e_2"
  for game in "${games[@]}"; do
    for seed in "${seeds[@]}"; do
      run_name="${ALGORITHM}_${variant}_${game}_seed_${seed}"
      run_dir="$RUN_ROOT/block_${block}/${game}/seed_${seed}"
      log_file="$run_dir/train_stdout.log"
      mkdir -p "$run_dir"

      if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$game" --seed "$seed"; then
        echo "Skipping completed run: ${variant} ${game} seed ${seed}"
        continue
      fi

      echo "Starting DER DINOv2 attentive ${PROBE_TYPE} block ${block} probe_lr=${PROBE_LR} ${game} seed ${seed} on GPU ${GPU}"
      set +e
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
        experiment=der/dinov2_attentive_atari100k \
        environment.task="$game" \
        algorithm.dinov2_output_block="$block" \
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

      "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$game" --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
      "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

      if [[ "$exit_code" -ne 0 ]]; then
        echo "Run failed: block ${block} ${game} seed ${seed} (exit_code=${exit_code}). See ${log_file}"
        exit "$exit_code"
      fi
    done
  done
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
