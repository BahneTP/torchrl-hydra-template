#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export GAMES="RoadRunner"
export RUN_SUFFIX="roadrunner"
exec scripts/lora/run_der_resnet18_layer2_lora_r4_a8_lr_1e_8_3games_5seeds_new_lora.sh
