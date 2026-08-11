# Figure Tags

W&B project: `LatentLab/transfer-learning-rl`

These tags select finished runs for `openrlbenchmark` figure generation. The
tagging scripts are dry-run by default; pass `--apply` to update W&B.

## Plain DER

| Tag | Contents | Games | Seeds | Source |
|---|---|---|---|---|
| `fig-plain-der-atari4-v1` | Plain DER baseline | Assault, BankHeist, Jamesbond, RoadRunner | 1-5 | `logs/plain_algorithms/der/atari100k_20260810_181403` |

Script:

```bash
python scripts/tagging/tag_plain_der_figures.py --no-descriptive-tags --apply
```

## ResNet Layer

| Tag | Contents | Games | Seeds | Source |
|---|---|---|---|---|
| `fig-resnetlayer-jamesbond-v1-resnet18-linear-layer1` | ResNet18 linear layer 1 | Jamesbond | 1-5 | `logs/resnetlayer/der_resnet18_linear_20260808_152354` |
| `fig-resnetlayer-jamesbond-v1-resnet18-linear-layer2` | ResNet18 linear layer 2 | Jamesbond | 1-5 | `logs/resnetlayer/der_resnet18_linear_20260808_152354` |
| `fig-resnetlayer-jamesbond-v1-resnet18-linear-layer3` | ResNet18 linear layer 3 | Jamesbond | 1-5 | `logs/resnetlayer/der_resnet18_linear_20260808_152354` |
| `fig-resnetlayer-jamesbond-v1-resnet18-linear-layer4` | ResNet18 linear layer 4 | Jamesbond | 1-5 | `logs/resnetlayer/der_resnet18_linear_20260808_152354` |
| `fig-resnetlayer-jamesbond-v1-resnet18-attentive-layer1` | ResNet18 attentive layer 1 | Jamesbond | 1-5 | `logs/resnetlayer/der_resnet18_attentive_20260808_152406` |
| `fig-resnetlayer-jamesbond-v1-resnet18-attentive-layer2` | ResNet18 attentive layer 2 | Jamesbond | 1-5 | `logs/resnetlayer/der_resnet18_attentive_20260808_152406` |
| `fig-resnetlayer-jamesbond-v1-resnet18-attentive-layer3` | ResNet18 attentive layer 3 | Jamesbond | 1-5 | `logs/resnetlayer/der_resnet18_attentive_20260808_152406` |
| `fig-resnetlayer-jamesbond-v1-resnet18-attentive-layer4` | ResNet18 attentive layer 4 | Jamesbond | 1-5 | `logs/resnetlayer/der_resnet18_attentive_20260808_152406` |
| `fig-resnetlayer-atari4-v1-resnet18-linear-layer2` | ResNet18 linear layer 2 | Assault, BankHeist, Jamesbond, RoadRunner | 1-5 | `logs/resnetlayer/der_resnet18_linear_20260808_152354`, `logs/resnetlayer/der_resnet18_linear_layer2_3games_20260810_195914` |

Script:

```bash
python scripts/tagging/tag_resnetlayer_figures.py --no-descriptive-tags --apply
```

## Mixed Layer Fusion

| Tag | Contents | Games | Seeds | Source |
|---|---|---|---|---|
| `fig-mlf-jamesbond-v1-resnet18-mlf-linear-all` | ResNet18 MLF linear all | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_resnet18_mlf_linear_20260808_155703` |
| `fig-mlf-jamesbond-v1-resnet18-mlf-linear-layer1-4` | ResNet18 MLF linear layer1+4 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_resnet18_mlf_linear_20260808_155703` |
| `fig-mlf-jamesbond-v1-resnet18-mlf-linear-layer2-3` | ResNet18 MLF linear layer2+3 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_resnet18_mlf_linear_20260808_155703` |
| `fig-mlf-jamesbond-v1-resnet18-mlf-attentive-all` | ResNet18 MLF attentive all | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_resnet18_mlf_attentive_20260808_155718` |
| `fig-mlf-jamesbond-v1-resnet18-mlf-attentive-layer1-4` | ResNet18 MLF attentive layer1+4 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_resnet18_mlf_attentive_20260808_155718` |
| `fig-mlf-jamesbond-v1-resnet18-mlf-attentive-layer2-3` | ResNet18 MLF attentive layer2+3 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_resnet18_mlf_attentive_20260808_155718` |
| `fig-mlf-jamesbond-v1-dinov2-mlf-linear-all` | DINOv2 MLF linear all | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_dinov2_mlf_linear_20260808_155743` |
| `fig-mlf-jamesbond-v1-dinov2-mlf-linear-first5` | DINOv2 MLF linear first5 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_dinov2_mlf_linear_20260808_155743` |
| `fig-mlf-jamesbond-v1-dinov2-mlf-linear-last5` | DINOv2 MLF linear last5 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_dinov2_mlf_linear_20260808_155743` |
| `fig-mlf-jamesbond-v1-dinov2-mlf-linear-block3-7-11` | DINOv2 MLF linear blocks 3,7,11 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_dinov2_mlf_linear_20260808_155743` |
| `fig-mlf-jamesbond-v1-dinov2-mlf-attentive-all` | DINOv2 MLF attentive all | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_dinov2_mlf_attentive_20260808_155758` |
| `fig-mlf-jamesbond-v1-dinov2-mlf-attentive-first5` | DINOv2 MLF attentive first5 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_dinov2_mlf_attentive_20260808_155758` |
| `fig-mlf-jamesbond-v1-dinov2-mlf-attentive-last5` | DINOv2 MLF attentive last5 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_dinov2_mlf_attentive_20260808_155758` |
| `fig-mlf-jamesbond-v1-dinov2-mlf-attentive-block3-7-11` | DINOv2 MLF attentive blocks 3,7,11 | Jamesbond | 1-5 | `logs/mixedlayerfusion/der_dinov2_mlf_attentive_20260808_155758` |
| `fig-mlf-atari4-v1-resnet18-linear-all` | ResNet18 MLF linear all | Assault, BankHeist, Jamesbond, RoadRunner | Assault/BankHeist/Jamesbond 1-5; RoadRunner 1-3 currently | `logs/mixedlayerfusion/der_resnet18_mlf_linear_20260808_155703`, `logs/mixedlayerfusion/der_resnet18_mlf_linear_all_3games_20260811_004533` |

Script:

```bash
python scripts/tagging/tag_mixedlayerfusion_figures.py --no-descriptive-tags --apply
```

## Full Finetuning

| Tag | Contents | Games | Seeds | Source |
|---|---|---|---|---|
| `fig-fullfinetuning-jamesbond-v1-resnet18-layer2-lr-sweep` | ResNet18 full finetuning layer 2 LR sweep | Jamesbond | lr 1e-4/1e-5/1e-6: 1-5; lr 1e-7: 1-2 currently | `logs/fullfinetuning/der_resnet18_layer2_20260811_025944` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-layer4-lr-sweep` | ResNet18 full finetuning layer 4 LR sweep | Jamesbond | lr 1e-4/1e-5/1e-6/1e-7/1e-8: 1-5 | `logs/fullfinetuning/der_resnet18_layer4_20260809_001602` |
| `fig-fullfinetuning-jamesbond-v1-dinov2-block12-lr-sweep` | DINOv2 full finetuning block 12 LR sweep | Jamesbond | lr 1e-4/1e-5/1e-6: 1-5; lr 1e-7: 1-2 currently | `logs/fullfinetuning/der_dinov2_block12_20260809_001637` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-full-layer2-lr-1e-4` | ResNet18 full layer 2, lr 1e-4 | Jamesbond | 1-5 | `logs/fullfinetuning/der_resnet18_layer2_20260811_025944` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-full-layer2-lr-1e-5` | ResNet18 full layer 2, lr 1e-5 | Jamesbond | 1-5 | `logs/fullfinetuning/der_resnet18_layer2_20260811_025944` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-full-layer2-lr-1e-6` | ResNet18 full layer 2, lr 1e-6 | Jamesbond | 1-5 | `logs/fullfinetuning/der_resnet18_layer2_20260811_025944` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-full-layer2-lr-1e-7` | ResNet18 full layer 2, lr 1e-7 | Jamesbond | 1-2 currently | `logs/fullfinetuning/der_resnet18_layer2_20260811_025944` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-full-layer4-lr-1e-4` | ResNet18 full layer 4, lr 1e-4 | Jamesbond | 1-5 | `logs/fullfinetuning/der_resnet18_layer4_20260809_001602` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-full-layer4-lr-1e-5` | ResNet18 full layer 4, lr 1e-5 | Jamesbond | 1-5 | `logs/fullfinetuning/der_resnet18_layer4_20260809_001602` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-full-layer4-lr-1e-6` | ResNet18 full layer 4, lr 1e-6 | Jamesbond | 1-5 | `logs/fullfinetuning/der_resnet18_layer4_20260809_001602` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-full-layer4-lr-1e-7` | ResNet18 full layer 4, lr 1e-7 | Jamesbond | 1-5 | `logs/fullfinetuning/der_resnet18_layer4_20260809_001602` |
| `fig-fullfinetuning-jamesbond-v1-resnet18-full-layer4-lr-1e-8` | ResNet18 full layer 4, lr 1e-8 | Jamesbond | 1-5 | `logs/fullfinetuning/der_resnet18_layer4_20260809_001602` |
| `fig-fullfinetuning-jamesbond-v1-dinov2-full-block12-lr-1e-4` | DINOv2 full block 12, lr 1e-4 | Jamesbond | 1-5 | `logs/fullfinetuning/der_dinov2_block12_20260809_001637` |
| `fig-fullfinetuning-jamesbond-v1-dinov2-full-block12-lr-1e-5` | DINOv2 full block 12, lr 1e-5 | Jamesbond | 1-5 | `logs/fullfinetuning/der_dinov2_block12_20260809_001637` |
| `fig-fullfinetuning-jamesbond-v1-dinov2-full-block12-lr-1e-6` | DINOv2 full block 12, lr 1e-6 | Jamesbond | 1-5 | `logs/fullfinetuning/der_dinov2_block12_20260809_001637` |
| `fig-fullfinetuning-jamesbond-v1-dinov2-full-block12-lr-1e-7` | DINOv2 full block 12, lr 1e-7 | Jamesbond | 1-2 currently | `logs/fullfinetuning/der_dinov2_block12_20260809_001637` |

Script:

```bash
python scripts/tagging/tag_fullfinetuning_figures.py --no-descriptive-tags --apply
```

## LoRA

| Tag | Contents | Games | Seeds | Source |
|---|---|---|---|---|
| `fig-lora-jamesbond-v1-resnet18-layer2-rank-sweep` | ResNet18 LoRA layer 2 rank sweep | Jamesbond | r1/a2, r2/a4: 1-5; r4/a8: 1-2 currently | `logs/lora/der_resnet18_layer2_20260811_030006` |
| `fig-lora-jamesbond-v1-resnet18-layer4-rank-sweep` | ResNet18 LoRA layer 4 rank sweep | Jamesbond | r1/a2, r2/a4, r4/a8, r8/a16, r16/a32: 1-5 | `logs/lora/der_resnet18_layer4_20260809_001801` |
| `fig-lora-jamesbond-v1-dinov2-block12-rank-sweep` | DINOv2 LoRA block 12 rank sweep | Jamesbond | r1/a2, r2/a4: 1-5; r4/a8: 1-4 currently | `logs/lora/der_dinov2_block12_20260809_095640` |
| `fig-lora-jamesbond-v1-resnet18-lora-layer2-r1-a2` | ResNet18 LoRA layer 2 r1/a2 | Jamesbond | 1-5 | `logs/lora/der_resnet18_layer2_20260811_030006` |
| `fig-lora-jamesbond-v1-resnet18-lora-layer2-r2-a4` | ResNet18 LoRA layer 2 r2/a4 | Jamesbond | 1-5 | `logs/lora/der_resnet18_layer2_20260811_030006` |
| `fig-lora-jamesbond-v1-resnet18-lora-layer2-r4-a8` | ResNet18 LoRA layer 2 r4/a8 | Jamesbond | 1-2 currently | `logs/lora/der_resnet18_layer2_20260811_030006` |
| `fig-lora-jamesbond-v1-resnet18-lora-layer4-r1-a2` | ResNet18 LoRA layer 4 r1/a2 | Jamesbond | 1-5 | `logs/lora/der_resnet18_layer4_20260809_001801` |
| `fig-lora-jamesbond-v1-resnet18-lora-layer4-r2-a4` | ResNet18 LoRA layer 4 r2/a4 | Jamesbond | 1-5 | `logs/lora/der_resnet18_layer4_20260809_001801` |
| `fig-lora-jamesbond-v1-resnet18-lora-layer4-r4-a8` | ResNet18 LoRA layer 4 r4/a8 | Jamesbond | 1-5 | `logs/lora/der_resnet18_layer4_20260809_001801` |
| `fig-lora-jamesbond-v1-resnet18-lora-layer4-r8-a16` | ResNet18 LoRA layer 4 r8/a16 | Jamesbond | 1-5 | `logs/lora/der_resnet18_layer4_20260809_001801` |
| `fig-lora-jamesbond-v1-resnet18-lora-layer4-r16-a32` | ResNet18 LoRA layer 4 r16/a32 | Jamesbond | 1-5 | `logs/lora/der_resnet18_layer4_20260809_001801` |
| `fig-lora-jamesbond-v1-dinov2-lora-block12-r1-a2` | DINOv2 LoRA block 12 r1/a2 | Jamesbond | 1-5 | `logs/lora/der_dinov2_block12_20260809_095640` |
| `fig-lora-jamesbond-v1-dinov2-lora-block12-r2-a4` | DINOv2 LoRA block 12 r2/a4 | Jamesbond | 1-5 | `logs/lora/der_dinov2_block12_20260809_095640` |
| `fig-lora-jamesbond-v1-dinov2-lora-block12-r4-a8` | DINOv2 LoRA block 12 r4/a8 | Jamesbond | 1-4 currently | `logs/lora/der_dinov2_block12_20260809_095640` |

Script:

```bash
python scripts/tagging/tag_lora_figures.py --no-descriptive-tags --apply
```
