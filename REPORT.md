# REPORT — CTU

**Paper:** Cascade Transformer Reasoning Embedded by Uncertainty for Occluded Person Re-Identification
**DOI:** 10.1109/TBIOM.2024.3361677

## Summary

| Split | mAP | Rank-1 | Rank-5 | Rank-10 |
|-------|-----|--------|--------|---------|
| Market-1501 (ours) | 69.23 | 80.23 | 94.77 | 96.50 |
| Market-1501 (paper CTU) | 88.1 | — | — | — |

Duke / Occluded-Duke / partial sets not run (unavailable); see `DEVIATIONS.md`.

## Figures

![loss](outputs/figures/train_loss.svg)

![metrics](outputs/figures/market_metrics.svg)

![cmc](outputs/figures/cmc.svg)

## Reproduce

```bash
bash scripts/setup_env.sh
bash scripts/train_full.sh
bash scripts/eval.sh
bash scripts/report.sh
```

