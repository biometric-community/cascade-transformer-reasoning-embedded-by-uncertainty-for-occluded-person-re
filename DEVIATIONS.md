# DEVIATIONS — CTU

| ID | Severity | Description |
|----|----------|-------------|
| D1 | High | DukeMTMC / Occluded-Duke / Partial-* unavailable → **Market-1501 only** |
| D2 | Medium | Query samples **N=4** (default) vs paper **N=20** for GPU/runtime |
| D3 | Medium | Epochs **60** (LR cosine after warmup); paper omits an explicit epoch count in the excerpt |
| D4 | Low | Final embedding uses GAP of last UT mean output (paper: GAP of feature feeding last UT) |
| D5 | Low | Adam 3.5e-4 default ReID schedule (paper lists batch/PK/aug but not LR in excerpt) |
