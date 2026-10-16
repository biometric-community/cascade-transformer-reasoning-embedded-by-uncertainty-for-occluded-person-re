# CTU — Cascade Transformer Reasoning Embedded by Uncertainty

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of **CTU** (Zheng et al., TBIOM 2024): ResNet-50 with uncertainty-embedded transformer layers and cascade attention for occluded / holistic person ReID.

**Repository:** https://github.com/biometric-community/cascade-transformer-reasoning-embedded-by-uncertainty-for-occluded-person-re

**Paper:** [10.1109/TBIOM.2024.3361677](https://doi.org/10.1109/TBIOM.2024.3361677)

## Setup / Train / Eval

```bash
bash scripts/setup_env.sh
bash scripts/train.sh configs/smoke.yaml   # smoke
bash scripts/train_full.sh                 # Market-1501 full (<5 GiB gate)
bash scripts/eval.sh
bash scripts/report.sh
```

Data: `projects/datasets/market1501` (not shipped). See `DEVIATIONS.md` for missing Duke / Occluded-Duke.
