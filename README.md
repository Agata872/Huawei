# RoomSense: cross-room occupancy from WiFi CSI

Huawei 2026 France Tech Arena, Future Network track. The task is to predict the
headcount (0–4) from 600 CSI frames (1 Tx × 3 Rx × 1024 subcarriers) in rooms
that never appear in the labeled data. See
[FutureNetwork-starting-kit/problem_statement.pdf](FutureNetwork-starting-kit/problem_statement.pdf).

## Layout

| Path | Contents |
| --- | --- |
| `common/` | Shared code: data index and cache, ONNX-exportable preprocessing, leave-one-room-out (LORO) evaluation, ONNX export and checks |
| `feature_model/` | Route A: hand-crafted, scale-invariant window features + LightGBM or logistic regression |
| `deep_model/` | Route B: a compact CNN on the time × subcarrier dynamic CSI |
| `data/` | Downloaded data (`extracted/`) and caches (`cache/`); git-ignored |
| `runs/` | LORO results (JSON), logs and checkpoints; git-ignored |
| `submissions/` | Exported `.onnx` files; `*.onnx` is git-ignored |

## Setup

```bash
source .venv/bin/activate        # Python 3.14
# rebuilding the env: torch must come from the cu126 index (T4 GPUs, driver 535)
pip install -r FutureNetwork-starting-kit/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt
```

Data goes in `data/extracted/{labeled,unlabeled}` (see
`FutureNetwork-starting-kit/data_downloads.md`). Then build the pooled cache
once, which takes about 10 s and 7 GB:

```bash
python -m common.build_cache
```

The cache drops the always-zero subcarriers (0–5, 1019–1023) and averages
complex CSI over groups of 8 subcarriers, giving (12000, 3, 126, 2) per trace.
The exported models apply the same pooling to raw input, using
`common.preprocess.pool_subcarriers`.

## Evaluation protocol

Each LORO fold holds out every labeled trace of one room (a1, b1, c1, e1, g1,
unknown) and scores that room's non-overlapping 600-frame windows (20 per
trace). Windows of one trace are never split between training and evaluation.
Labeled headcounts 2–4 exist only in a1 and c1, so those two folds matter most.
The summary also prints the score of always predicting 1 person (0.375 pooled)
as a reference.

## Route A: features

```bash
python -m feature_model.extract                          # ~3 min, features for all windows
python -m feature_model.train --model logreg             # LORO; also --model lgbm
python -m feature_model.train --model lgbm --exclude ati_,sfamp_   # ablate feature groups
python -m feature_model.export --model logreg            # -> submissions/feature_logreg.onnx
```

## Route B: CNN

```bash
python -m deep_model.train --name cnn_base               # LORO, folds spread over all GPUs
python -m deep_model.train --folds a1,c1 --steps 1000    # quick check on the key folds
python -m deep_model.train --final --name cnn_base       # all rooms -> runs/deep_model/cnn_base.pt
python -m deep_model.export runs/deep_model/cnn_base.pt  # -> submissions/deep_cnn_base.onnx
```

`python -m deep_model.train --help` lists the model and augmentation switches.

## Using the unlabeled data

The public test set is exactly class-balanced (constant predictions score
0.20 for every class), and its rooms appear in the unlabeled data: per-room
calibration moved the public score of the same CNN from 0.32 to 0.34.
`common.calibrate` fits, for each unlabeled room with enough traces, a logit
bias that makes the model predict every headcount 20% of the time; inside the
ONNX graph a window's room is recognised from its static-channel fingerprint.
`common.gate` labels windows whose dynamics are pure noise as 0 people.

```bash
python -m common.calibrate runs/deep_model/<loro-run>.json        # offline check on LORO logits
python -m common.calibrate --export runs/deep_model/cnn_base.pt    # -> submissions/deep_cnn_base_cal.onnx
python -m common.gate runs/deep_model/cnn_base.pt                  # -> submissions/deep_cnn_base_gate.onnx
```

Dropped after testing: domain-adversarial training (DANN, CDAN and a
class-conditional room adversary), confidence-based pseudo-labeling, and
labeling whole unlabeled recordings by balanced assignment (public 0.25).

## Export checks

`common.export` checks every exported file the way the platform will run it:
the input/output names and shapes, the 100 MB limit, agreement with PyTorch on
real windows, and the mean CPU time per sample with 4 threads (limit 0.7 s).
