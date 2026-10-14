"""Compute window features for every cached trace and store them for fast CV.

    python -m feature_model.extract [--stride 150] [--device cuda]

Writes data/cache/features/<split>.npz. Windows start every `stride` frames,
so the non-overlapping evaluation windows (start % 600 == 0) are included.
"""

import argparse

import numpy as np
import torch

from common.data import CACHE_ROOT, SPLITS, WINDOW, load_traces, window
from feature_model.features import FeatureExtractor

FEATURE_DIR = CACHE_ROOT / "features"


def feature_path(split):
    return FEATURE_DIR / f"{split}.npz"


@torch.no_grad()
def extract_split(split, extractor, stride, device, batch=128):
    rows, csis, atis = [], [], []
    for t in load_traces(split):
        for start in range(0, t.num_frames - WINDOW + 1, stride):
            csi, ati = window(t, start)
            csis.append(csi)
            atis.append(ati)
            rows.append((t.ident, t.layout, t.room, t.label, start))
    feats = []
    for i in range(0, len(rows), batch):
        h = torch.from_numpy(np.stack(csis[i:i + batch])).to(device)
        a = torch.from_numpy(np.stack(atis[i:i + batch])).to(device)
        feats.append(extractor(h, a).cpu().numpy())
    X = np.concatenate(feats).astype(np.float32)
    ident, layout, room, label, start = zip(*rows)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(feature_path(split), X=X, names=np.array(extractor.names), ident=np.array(ident),
             layout=np.array(layout), room=np.array(room), label=np.array(label),
             start=np.array(start))
    bad = ~np.isfinite(X).all(axis=1)
    print(f"{split}: {X.shape[0]} windows x {X.shape[1]} features, non-finite rows: {bad.sum()}")


def load_features(split):
    d = np.load(feature_path(split))
    return {k: d[k] for k in d.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    extractor = FeatureExtractor().to(args.device).eval()
    for split in SPLITS:
        extract_split(split, extractor, args.stride, args.device)


if __name__ == "__main__":
    main()
