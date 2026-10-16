# Implementation Plan — CTU

## Paper
Zheng et al., TBIOM 2024, DOI 10.1109/TBIOM.2024.3361677.  
Slug: `cascade-transformer-reasoning-embedded-by-uncertainty-for-occluded-person-re`

## Method
ResNet-50 (last stride 1) → 1×1 reduce → cascade of **P=2** Uncertainty Embedded Transformer layers with Cascade Attention between them. Losses: `Lsample + Lhd + Lcls + Lun` per layer + batch-hard triplet on final GAP.

## Figure inventory
| Paper | Ours |
|-------|------|
| Holistic Market metrics | `market_metrics.svg` |
| CMC ranks | `cmc.svg` |
| Train dynamics | `train_loss.svg` |

## Data
Market-1501 local (~0.14 GiB) → full train. Duke / Occluded-Duke / partial: missing.

## Size gate
Market &lt; 5 GiB → full train after smoke.
