"""Dataset index, raw trace loading and the pooled-CSI cache shared by both routes.

The cache stores, for every trace, the CSI after `preprocess.pool_subcarriers`
(complex values averaged over groups of 8 subcarriers) so that training never
has to decompress the raw 1024-subcarrier arrays again. Build it once with:

    python -m common.build_cache
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import blosc2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ROOMSENSE_DATA", REPO / "data" / "extracted"))
CACHE_ROOT = Path(os.environ.get("ROOMSENSE_CACHE", REPO / "data" / "cache"))
RUNS_ROOT = REPO / "runs"
SPLITS = ("labeled", "unlabeled")
NUM_CLASSES = 5
WINDOW = 600  # frames per validation/test sample


@dataclass
class Trace:
    ident: str
    split: str
    layout: str
    room: str
    label: int  # -1 for unlabeled traces
    csi: np.ndarray  # (T, 3, K, 2) float32, pooled complex CSI (real, imag)
    ati: np.ndarray  # (T,) float32, microseconds, last value 0

    @property
    def num_frames(self):
        return len(self.ati)


def read_index(split):
    with (DATA_ROOT / split / "samples.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["split"] = split
        row["label"] = -1 if row["headcount"] == "unlabeled" else int(row["headcount"])
    return rows


def load_raw(split, ident):
    """Raw arrays exactly as distributed: CSI (T, 1, 3, 1024) complex64, ATI (T,)."""
    folder = DATA_ROOT / split
    csi = blosc2.load_array(str(folder / f"CSI_{ident}.b2"))
    ati = blosc2.load_array(str(folder / f"ATI_{ident}.b2"))
    return csi, ati


def raw_to_frames(csi):
    """Complex64 (T, 1, 3, 1024) -> float32 (T, 1, 3, 1024, 2), the submission input layout."""
    return np.ascontiguousarray(csi).view(np.float32).reshape(*csi.shape, 2)


def cache_paths(split, ident):
    folder = CACHE_ROOT / split
    return folder / f"{ident}_csi.npy", folder / f"{ident}_ati.npy"


def load_traces(split, mmap=False):
    """All traces of a split from the pooled cache, in samples.csv order."""
    traces = []
    for row in read_index(split):
        csi_path, ati_path = cache_paths(split, row["identifier"])
        if not csi_path.exists():
            raise FileNotFoundError(f"{csi_path} missing; run `python -m common.build_cache`")
        traces.append(Trace(
            ident=row["identifier"], split=split, layout=row["layout"], room=row["room"],
            label=row["label"],
            csi=np.load(csi_path, mmap_mode="r" if mmap else None),
            ati=np.load(ati_path),
        ))
    return traces


def window(trace, start, length=WINDOW):
    """One evaluation-style sample: `length` frames, last interval padded with 0."""
    csi = trace.csi[start:start + length]
    ati = trace.ati[start:start + length].copy()
    ati[-1] = 0
    return csi, ati
