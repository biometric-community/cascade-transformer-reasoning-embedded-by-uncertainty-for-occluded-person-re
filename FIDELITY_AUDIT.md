# FIDELITY_AUDIT — CTU

Exactly **5** passes; stop after Pass 5.

| Pass | Focus | Confidence | Notes |
|------|-------|------------|-------|
| 1 | Architecture | 80% | ResNet-50 last-stride-1, P=2 UT, CA mask, CT |
| 2 | Losses | 85% | Lsample, Lcls, Lun (β=log4), Lhd, Ltri (Eq. 13–20) |
| 3 | Data/eval | 82% | Market CMC/mAP; Duke/occ blocked (D1) |
| 4 | Hyperparams | 78% | PK 16×4, 256×128, RE; N=4 vs 20 (D2); epochs 60 (D3) |
| 5 | Packaging | 90% | scripts, REPORT, deviations, real Market metrics only |

**Weighted ~83%.** Stop.

## Measured Market-1501 (full train)

| Checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 |
|------------|-----|--------|--------|---------|
| best (ep60) | 69.23 | 80.23 | 94.77 | 96.50 |
| paper CTU (Market mAP) | 88.1 | — | — | — |
