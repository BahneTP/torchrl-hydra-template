#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"

for seed in 1 2 3 4 5; do
  uv run python src/train.py \
    experiment=der/resnet18_linear_sign_torchrl_atari100k \
    environment.task=Jamesbond \
    trainer.seed="${seed}" \
    trainer.accelerator=gpu \
    trainer.devices="[${gpu}]"
done
