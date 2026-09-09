# Plots erstellen

Die PDFs in `torchrl-hydra-template/plots/test/` und
`torchrl-hydra-template/plots/validation/` werden jeweils von einem eigenen
Python-Skript in `plots/test/scripts/` bzw. `plots/validation/scripts/`
erzeugt.

## Voraussetzungen

```bash
cd torchrl-hydra-template
uv sync
```

## Test-Plots

```bash
cd torchrl-hydra-template
uv run python plots/test/scripts/<script>.py
```

verfügbare Skripte in `plots/test/scripts/`:

- `encoder_learning_curves.py`
- `encoder_boxplots.py`
- `bbf_encoder_boxplots.py`
- `bbf_encoder_learning_curves_three_games.py`
- `der_encoder_learning_curves_three_games.py`

Die erzeugten PDFs landen direkt in `plots/test/`.

## Validation-Plots

```bash
cd torchrl-hydra-template
uv run python plots/validation/scripts/<script>.py
```

verfügbare Skripte in `plots/validation/scripts/`:

- `attention_methods.py`
- `bbf_lora_resnet18_layer2.py`
- `bbf_variants.py`
- `der_variants.py`
- `dino_blocks.py`
- `dinov2_mixed_layer_fusion.py`
- `dinov2_mixed_layer_weights.py`
- `full_finetuning.py`
- `lora_dinov2_block7.py`
- `lora_resnet18_layer2.py`
- `mixed_layer_fusion.py`
- `resnet18_mixed_layer_weights.py`
- `resnet_layer.py`

Die erzeugten PDFs landen direkt in `plots/validation/`.

## Hinweise

- Die Skripte lesen die zugrunde liegenden Trainingsläufe aus `logs/` und
  cachen W&B-Historien unter `plots/cache/wandb_history/`. Mit dem Flag
  `--refresh` wird der Cache übersprungen und neu von W&B geladen:

  ```bash
  uv run python plots/test/scripts/encoder_learning_curves.py --refresh
  ```

- Jedes Skript ist eigenständig ausführbar und erzeugt nur die PDF(s), die
  in seinem Dateinamen genannt sind.
