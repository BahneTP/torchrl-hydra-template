#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "$RUN_ROOT" ]]; then
  latest_run_root="$(find logs/lora/der_resnet18_layer4_* -maxdepth 0 -type d 2>/dev/null | sort | tail -n 1 || true)"
  RUN_ROOT="${latest_run_root:-logs/lora/der_resnet18_layer4_${STAMP}}"
fi
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
ALGORITHM="der"

if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

combos=("1 2" "2 4" "4 8" "8 16" "16 32")
games=(Jamesbond)
seeds=(1 2 3 4 5)

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"

for combo in "${combos[@]}"; do
  read -r rank alpha <<<"$combo"
  variant="resnet18_lora_layer4_r${rank}_a${alpha}"
  for game in "${games[@]}"; do
    for seed in "${seeds[@]}"; do
      run_name="${ALGORITHM}_${variant}_${game}_seed_${seed}"
      run_dir="$RUN_ROOT/r${rank}_a${alpha}/${game}/seed_${seed}"
      log_file="$run_dir/train_stdout.log"
      mkdir -p "$run_dir"

      if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm "$ALGORITHM" --variant "$variant" --game "$game" --seed "$seed"; then
        echo "Skipping completed run: ${variant} ${game} seed ${seed}"
        continue
      fi

      echo "Starting DER ResNet18 LoRA layer4 rank=${rank} alpha=${alpha} ${game} seed ${seed} on GPU ${GPU}"
      set +e
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
        experiment=der/resnet18_lora_atari100k \
        environment.task="$game" \
        algorithm.resnet18_variant=resnet_layer4 \
        algorithm.encoder_lr=1e-4 \
        algorithm.adapter_lr=1e-4 \
        algorithm.lora_rank="$rank" \
        algorithm.lora_alpha="$alpha" \
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
        echo "Run failed: rank ${rank} alpha ${alpha} ${game} seed ${seed} (exit_code=${exit_code}). See ${log_file}"
        exit "$exit_code"
      fi
    done
  done
done

echo "Wrote $RESULTS_CSV"
echo "Wrote $SUMMARY_CSV"
