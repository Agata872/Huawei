"""WiMANS (Huang et al., ECCV 2024) as external pretraining data.

WiMANS has 3 s samples with 0-5 people in three rooms (classroom, meeting
room, empty room) at 2.4 and 5 GHz, recorded with Intel 5300 NICs: about
1000 packets/s of 3 Tx x 3 Rx x 30 subcarriers. The organisers allow external
CSI data for pretraining. Samples with 0-4 people are converted to our input
layout:

  * time: for each 5 ms grid point take the nearest packet -> 600 frames,
    with the real packet spacing kept as the intervals (ATI analogue);
  * phase: Intel 5300 adds a random linear phase across subcarriers to every
    packet (packet detection delay). It is estimated from adjacent-subcarrier
    products summed over Rx antennas and removed together with the common
    phase, which keeps the phase differences between antennas. Our routers
    do not need this (raw phase std <= 0.07 rad);
  * antennas: every Tx antenna gives an independent 1 x 3 Rx stream;
  * subcarriers: kept at 30 here; `to_groups` interpolates to our 126 groups.

    python -m common.wimans                 # data/external/wimans/mat -> data/external/wimans/cache
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.data import REPO, WINDOW

ROOT = REPO / "data" / "external" / "wimans"
CACHE = ROOT / "cache"
GRID_US = 5000.0
MAX_USERS = 4
NUM_SC = 30


def read_annotation(root=ROOT):
    d = pd.read_csv(root / "annotation.csv", dtype=str, encoding="utf-8-sig")
    d["users"] = d["number_of_users"].astype(int)
    acts = [c for c in d.columns if c.endswith("_activity")]
    d["static"] = d[acts].apply(lambda r: all(pd.isna(v) or v == "nothing" for v in r), axis=1)
    return d[d["users"] <= MAX_USERS].reset_index(drop=True)


def load_mat(path):
    """-> timestamps (P,) us, csi (P, 3 tx, 3 rx, 30) complex64; packets with other shapes dropped."""
    import scipy.io as sio

    trace = sio.loadmat(path)["trace"]
    ts, csi = [], []
    for i in range(trace.shape[0]):
        e = trace[i, 0]
        c = e["csi"][0, 0]
        if c.shape == (3, 3, NUM_SC):
            ts.append(int(e["timestamp_low"][0, 0].ravel()[0]) & 0xFFFFFFFF)
            csi.append(c)
    ts = np.array(ts, dtype=np.int64)
    ts = np.concatenate([[0], np.cumsum(np.diff(ts) % (1 << 32))])  # 32-bit microsecond counter
    return ts.astype(np.float64), np.stack(csi).astype(np.complex64)


def sanitize_phase(csi):
    """Remove each packet's linear phase across subcarriers and its common phase (per Tx)."""
    k = np.arange(NUM_SC)
    slope = np.angle((csi[..., 1:] * np.conj(csi[..., :-1])).sum(axis=(-2, -1)))      # (P, tx)
    rot = np.exp(-1j * slope[..., None, None] * k)
    common = np.angle((csi * rot).sum(axis=(-2, -1)))                                  # (P, tx)
    return csi * rot * np.exp(-1j * common)[..., None, None]


def convert_one(path):
    """-> csi (3 tx, 600, 3 rx, 30, 2) float16, intervals (600,) float32."""
    ts, csi = load_mat(path)
    csi = sanitize_phase(csi)
    grid = np.arange(WINDOW) * GRID_US
    idx = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
    idx = np.where(np.abs(ts[idx - 1] - grid) <= np.abs(ts[idx] - grid), idx - 1, idx)
    x = csi[idx].transpose(1, 0, 2, 3)                                                # (tx, 600, rx, 30)
    out = np.stack([x.real, x.imag], axis=-1).astype(np.float16)
    ati = np.diff(ts[idx]).astype(np.float32)
    return out, np.append(ati, 0.0).astype(np.float32)


def _convert(args):
    i, path = args
    try:
        return i, *convert_one(path)
    except Exception as e:  # corrupt or truncated file
        print(f"skip {path.name}: {e}", flush=True)
        return i, None, None


def build_cache(root=ROOT, workers=16):
    ann = read_annotation(root)
    paths = [root / "mat" / f"{lab}.mat" for lab in ann["label"]]
    ok = np.array([p.exists() for p in paths])
    ann, paths = ann[ok].reset_index(drop=True), [p for p, o in zip(paths, ok) if o]
    CACHE.mkdir(parents=True, exist_ok=True)
    csi = np.lib.format.open_memmap(CACHE / "csi.npy", mode="w+", dtype=np.float16,
                                    shape=(len(paths), 3, WINDOW, 3, NUM_SC, 2))
    ati = np.zeros((len(paths), WINDOW), dtype=np.float32)
    good = np.zeros(len(paths), dtype=bool)
    with mp.Pool(workers) as pool:
        for n, (i, x, a) in enumerate(pool.imap_unordered(_convert, enumerate(paths), chunksize=8)):
            if x is not None:
                csi[i], ati[i], good[i] = x, a, True
            if n % 1000 == 0:
                print(f"{n}/{len(paths)}", flush=True)
    csi.flush()
    np.save(CACHE / "ati.npy", ati)
    ann["ok"] = good
    ann[["label", "environment", "wifi_band", "users", "static", "ok"]].to_csv(CACHE / "meta.csv", index=False)
    print(f"cached {good.sum()} / {len(paths)} samples -> {CACHE}")


def load_cache():
    """csi (N, 3, 600, 3, 30, 2) float16 memmap, ati (N, 600), meta DataFrame (rows with ok=False are empty)."""
    meta = pd.read_csv(CACHE / "meta.csv")
    return np.load(CACHE / "csi.npy", mmap_mode="r"), np.load(CACHE / "ati.npy"), meta


def interp_matrix(n_out, n_in=NUM_SC):
    """(n_out, n_in) linear-interpolation weights mapping 30 subcarriers onto n_out groups."""
    pos = np.linspace(0, n_in - 1, n_out)
    lo = np.floor(pos).astype(int).clip(max=n_in - 2)
    w = pos - lo
    m = np.zeros((n_out, n_in), dtype=np.float32)
    m[np.arange(n_out), lo] = 1 - w
    m[np.arange(n_out), lo + 1] = w
    return torch.from_numpy(m)


def to_groups(x, matrix):
    """(..., 30, 2) -> (..., n_out, 2) by linear interpolation along subcarriers."""
    return torch.einsum("ok,...kc->...oc", matrix, x)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    build_cache(workers=args.workers)


if __name__ == "__main__":
    main()
