#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "$RUN_ROOT" ]]; then
  latest_run_root="$(find logs/lora/new_lora \
    -maxdepth 1 -type d \
    -name 'der_resnet18_layer4_lora_dense_adapter_rank_alpha_lr_1e_9_jamesbond_*' \
    2>/dev/null | sort | tail -n 1 || true)"
  RUN_ROOT="${latest_run_root:-logs/lora/new_lora/der_resnet18_layer4_lora_dense_adapter_rank_alpha_lr_1e_9_jamesbond_${STAMP}}"
fi
RESULTS_CSV="$RUN_ROOT/results.csv"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
combos=("1 2" "2 4" "8 16" "16 32")

"$PYTHON" scripts/lib/results.py init --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"
for combo in "${combos[@]}"; do
  read -r rank alpha <<<"$combo"
  variant="resnet18_lora_layer4_dense_adapter_r${rank}_a${alpha}_lr_1e_9"
  for seed in 1 2 3 4 5; do
    run_name="der_${variant}_Jamesbond_seed_${seed}"
    run_dir="$RUN_ROOT/r${rank}_a${alpha}/Jamesbond/seed_${seed}"
    log_file="$run_dir/train_stdout.log"
    mkdir -p "$run_dir"
    if "$PYTHON" scripts/lib/results.py completed --results "$RESULTS_CSV" --algorithm der --variant "$variant" --game Jamesbond --seed "$seed"; then
      echo "Skipping completed run: ${variant} Jamesbond seed ${seed}"
      continue
    fi
    echo "Starting DER ResNet18 layer4 LoRA rank=${rank} alpha=${alpha} encoder_lr=1e-9 Jamesbond seed ${seed} on GPU ${GPU}"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/train.py \
      experiment=der/resnet18_lora_atari100k environment.task=Jamesbond \
      algorithm.resnet18_variant=resnet_layer4 algorithm.lr=1e-4 \
      algorithm.encoder_lr=1e-9 algorithm.adapter_lr=1e-4 \
      algorithm.lora_rank="$rank" algorithm.lora_alpha="$alpha" \
      trainer.seed="$seed" trainer.devices=[0] "run_name=$run_name" \
      "hydra.run.dir=$run_dir" "checkpoint.save_dir=$run_dir/checkpoints" \
      checkpoint.enabled=false >"$log_file" 2>&1
    exit_code="$?"
    set -e
    "$PYTHON" scripts/lib/results.py append --results "$RESULTS_CSV" --algorithm der --variant "$variant" --game Jamesbond --seed "$seed" --exit-code "$exit_code" --run-name "$run_name" --log-file "$log_file"
    "$PYTHON" scripts/lib/results.py summarize --results "$RESULTS_CSV" --summary "$SUMMARY_CSV"
    [[ "$exit_code" -eq 0 ]] || exit "$exit_code"
  done
done
