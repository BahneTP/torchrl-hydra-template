#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-}"
PROBE_LR="${PROBE_LR:-1e-2}"

run_existing_sweep() {
  local probe_type="$1"
  local script="$2"
  local pattern="logs/resnetlayer/attentive_lrsweep/der_resnet18_attentive_${probe_type}_layer2_probe_lr_sweep_jamesbond_*"
  local run_root

  run_root="$(find $pattern -maxdepth 0 -type d 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$run_root" ]]; then
    echo "No existing run folder found for ${probe_type} (${pattern})"
    exit 1
  fi

  echo "Running ${probe_type} probe_lr=${PROBE_LR} into ${run_root}"
  GPU="$GPU" PYTHON="$PYTHON" PROBE_LRS="$PROBE_LR" RUN_ROOT="$run_root" "$script"
}

run_existing_sweep \
  self_attention \
  ./scripts/resnetlayer/attentive_lrsweep/run_der_resnet18_attentive_self_attention_layer2_probe_lr_sweep_jamesbond_5seeds.sh

run_existing_sweep \
  single_query \
  ./scripts/resnetlayer/attentive_lrsweep/run_der_resnet18_attentive_single_query_layer2_probe_lr_sweep_jamesbond_5seeds.sh

run_existing_sweep \
  attention_weighted_pooling \
  ./scripts/resnetlayer/attentive_lrsweep/run_der_resnet18_attentive_attention_weighted_pooling_layer2_probe_lr_sweep_jamesbond_5seeds.sh
