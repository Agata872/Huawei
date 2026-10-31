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
| `temporal_model/` | Route C: the Reference P13 network (public 0.50), precomputed block maps, leave-one-room-out training, multi-seed retraining, ensembles, per-room calibration and leaderboard probes |
| `analysis/` | Per-room checks of how CSI features and P13 behave across headcounts and rooms (CPU only) |
| `colab/` | Notebooks that run the GPU experiments on Google Colab |
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

## Route C: P13 temporal model

`temporal_model/model.py` reproduces the Reference P13 network exactly (same
logits, same parameter names); its checkpoints live in the untracked
`Reference/` folder. P13 only sees block means over 10 frames × 8
subcarriers, so `temporal_model.maps` stores those per trace (about 200 MB
for the labeled split instead of 14 GB of raw CSI) and training runs on them.

```bash
python -m temporal_model.maps --check      # block maps, checked against the raw-input features
python -m temporal_model.loro              # rooms / single recordings held out, 3 seeds (GPU)
python -m temporal_model.train --p11b <P11B checkpoint.pt> --seeds 1,2,3,4,5   # P13 recipe, new seeds (GPU)
python -m temporal_model.export --checkpoint <P13> runs/temporal_model/p13_seed*.pt   # ensemble -> submissions/
python -m temporal_model.ehunam                                       # external captures -> block maps (~25 GB streamed)
python -m temporal_model.train ... --exclude-rooms a1 --ehunam-share 0.5 --name p13_noa1_ehunam
```

On Colab, `colab/loro.ipynb`, `colab/train_p13.ipynb` and
`colab/train_ehunam.ipynb` do this, including the data download; the last two
need the P11B checkpoint uploaded by hand.

External data: [EHUNAM](https://doi.org/10.1038/s41597-025-06238-4) (de Armas
et al., Scientific Data 2025, CC BY 4.0) has 802.11ac CSI over 80 MHz from
Raspberry Pi monitors in six rooms with 0–4 people (179 usable captures of
60 s). `temporal_model.ehunam` turns each capture recorded by receivers 1–3
into a 3-antenna trace on the competition grid: competition frame timing,
per-frame strongest-path phase referencing (Nexmon phase is random between
frames), the k ≥ −64 sign flip of that hardware undone, the level of the
labeled rooms, and the time-varying part shrunk 10–25x to the size it has in
the competition data.

Public leaderboard probes of P13 (0.50), which need no labels:

| Submission | Public | Reading |
| --- | --- | --- |
| P13 with predicted 2/3/4 merged into 2 / 3 / 4 (`export --merge-to`) | 0.37 / 0.41 / 0.42 | the 2–4 split is informative on the public rooms; true 2-person windows are the least often called a crowd |
| per-room calibration, 0/1/crowd or all 5 classes (`temporal_model.calibrate`) | 0.50 / 0.50 | no effect |
| windows matched to any unlabeled room set to 0 (`calibrate --probe 0`) | 0.41 | part of the public set comes from unlabeled rooms |
| windows matched to v1 set to 0 (`--probe 0 --probe-rooms v1`) | 0.49 | hardly any of it from v1 |
| P13 recipe with 5 new seeds, mean logits (`temporal_model.train`, `export`) | 0.46 | P13 is on the lucky side of the seed spread |
| the same 5 seeds + P13 | 0.47 | ensembling does not help on public |

A new seed agrees with P13 on about 90% of the unlabeled windows
(`analysis.seed_agreement`), and the 6-model ensemble on 93%.

Room matching works on recordings left out of the fingerprint bank (h1, i1,
u1, d1, j1: 100% of windows; v1: 89%); the five `unknown` unlabeled traces
all match u1.

## Analyses

```bash
python -m analysis.headcount_separability   # do features order headcounts the same way in a1 and c1?
python -m analysis.unlabeled_rooms          # P13 predictions and feature ranges in the unlabeled rooms
python -m analysis.synthetic_check          # do windows built by adding one-person dynamics look like real crowds?
python -m analysis.sharp_doppler            # the same question for SHARP-style Doppler traces
python -m analysis.motion_states            # and for motion states found by K-means, then an HMM, on every room
```

All run on CPU and write to `runs/analysis/`. So far: no feature orders
1–4 people the same way in a1 and c1, and features fitted in one of them
transfer at chance; 0 vs 1 transfers moderately (AUC ≈ 0.7 across rooms).
Doppler traces as in SHARP (Meneghello et al., TMC 2023) do no better: in c1
the motion power and the Doppler profile are the same for 1–4 people, while in
a1 the motion power below 10 Hz falls by ~20 dB from 1 to 4 people (the
normalised profile then looks wider only because the noise floor shows). SHARP's per-antenna phase sanitization has little to remove here,
since the raw CSI phase moves by only ~0.002 rad between frames.
Motion states found without labels on every room (`analysis.motion_states`:
K-means on 50 ms blocks, then an HMM) are shared by all rooms and readable as
noise, bursts, slow and one-sided motion, but none of their window shares or
switching rates orders 1–4 people the same way in a1 and c1, and with g1 held
out they separate 0 from 1 person no better than the existing features (0.68).
In the unlabeled rooms P13 rarely predicts 3 or 4 people. Synthetic
multi-person windows (`temporal_model.synthesis`) do not look like real ones
to P13 or to a feature classifier: in real recordings people move less when
more of them are present, which adding one-person dynamics cannot reproduce.

## Using the unlabeled data

The public test set is exactly class-balanced (constant predictions score
0.20 for every class), and some of its rooms appear in the unlabeled data:
per-room calibration moved the public score of the same CNN from 0.32 to 0.34
(for P13 it changed nothing, see Route C).
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
