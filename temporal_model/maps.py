"""Precomputed block maps: train the temporal model without touching raw CSI.

`model.pooled_maps` averages over 10-frame x 8-subcarrier blocks before any
window-level step, so a 12000-frame trace reduces to 1200 blocks of log
amplitude (3 x 128) and phase increments (4 x 128), about 4.4 MB instead of
295 MB. A window starting at a multiple of 10 frames is then a slice of 60
blocks, and `model.features_from_maps` on it equals `model.input_features` on
the raw window.

    python -m temporal_model.maps [--split labeled] [--check]

Writes data/cache/maps/<split>/<identifier>.npz (a, ph, ati); CPU, ~2 min for
the labeled split. --check compares a few windows with the raw-input path.
"""

import argparse
import math

import numpy as np
import torch

from common.data import CACHE_ROOT, WINDOW, load_raw, raw_to_frames, read_index
from temporal_model.model import features_from_maps, input_features, pooled_maps

MAP_ROOT = CACHE_ROOT / "maps"
BLOCK = 10  # frames per block
WINDOW_BLOCKS = WINDOW // BLOCK  # 60


def map_path(split, ident):
    return MAP_ROOT / split / f"{ident}.npz"


@torch.no_grad()
def trace_maps(csi, chunk=600):
    """Raw CSI (T, 1, 3, 1024) complex64 -> a (T/10, 3, 128), ph (T/10, 4, 128) float32."""
    frames = torch.from_numpy(raw_to_frames(csi))
    parts = [pooled_maps(frames[s:s + chunk].unsqueeze(0)) for s in range(0, len(frames), chunk)]
    a = torch.cat([p[0] for p in parts], dim=2)[0].permute(1, 0, 2)
    ph = torch.cat([p[1] for p in parts], dim=2)[0].permute(1, 0, 2)
    return a.contiguous().numpy(), ph.contiguous().numpy()


def build(split):
    rows = read_index(split)
    for k, row in enumerate(rows, 1):
        path = map_path(split, row["identifier"])
        if path.exists():
            continue
        csi, ati = load_raw(split, row["identifier"])
        a, ph = trace_maps(csi)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, a=a, ph=ph, ati=ati.astype(np.float32))
        print(f"[{k}/{len(rows)}] {row['identifier']}", flush=True)


class MapBank:
    """Every trace's maps of one split in memory, with window access at block granularity."""

    def __init__(self, split):
        self.rows = read_index(split)
        maps = [np.load(map_path(split, r["identifier"])) for r in self.rows]
        self.a = torch.from_numpy(np.stack([m["a"] for m in maps]))  # (N, 1200, 3, 128)
        self.ph = torch.from_numpy(np.stack([m["ph"] for m in maps]))  # (N, 1200, 4, 128)
        self.ati = torch.from_numpy(np.stack([m["ati"] for m in maps]))  # (N, 12000)
        self.max_block = self.a.shape[1] - WINDOW_BLOCKS

    def windows(self, traces, blocks, device="cpu"):
        """Trace indices + start blocks -> (a (B, 3, 60, 128), ph (B, 4, 60, 128), intervals (B, 600))."""
        offsets = torch.arange(WINDOW_BLOCKS)
        idx = torch.as_tensor(blocks)[:, None] + offsets[None, :]
        tr = torch.as_tensor(traces)[:, None]
        a = self.a[tr, idx].permute(0, 2, 1, 3)
        ph = self.ph[tr, idx].permute(0, 2, 1, 3)
        frames = torch.as_tensor(blocks)[:, None] * BLOCK + torch.arange(WINDOW)[None, :]
        ati = self.ati[tr, frames].clone()
        ati[:, -1] = 0
        return a.to(device), ph.to(device), ati.to(device)


def attenuation_shift(rng, batch, width=(12, 24), depth=(0.90, 0.95)):
    """Map-space version of P13's training augmentation -> (B, 1, 1, 128) shift of the log amplitude.

    P13 multiplies the raw CSI by a smooth dip g(f) <= 1 over 12-24 subcarriers.
    With |CSI| in the hundreds, log1p(|CSI| g) = log1p(|CSI|) + log g to within
    ~1e-3, so the dip becomes an additive shift of the pooled log amplitude.
    """
    profile = torch.ones(batch, 1024)
    for b in range(batch):
        w = int(rng.integers(width[0], width[1] + 1))
        start = int(rng.integers(6, 1019 - w + 1))
        d = float(rng.uniform(*depth))
        window = 0.5 - 0.5 * torch.cos(torch.linspace(0, 2 * math.pi, w))
        profile[b, start:start + w] = 1 - (1 - d) * window / window.max()
    log_g = torch.log(profile[:, 6:1019])
    log_g = torch.cat([log_g, log_g[:, -1:].expand(-1, 11)], dim=1)  # replicate pad to 1024
    return log_g.reshape(batch, 128, 8).mean(-1)[:, None, None, :]


@torch.no_grad()
def check(split="labeled", n=4):
    """Windows from the maps must equal the raw-input features."""
    bank = MapBank(split)
    worst = 0.0
    for k in range(n):
        trace, block = k * (len(bank.rows) // n), 37 * (k + 1)
        csi, ati = load_raw(split, bank.rows[trace]["identifier"])
        s = block * BLOCK
        frames = torch.from_numpy(raw_to_frames(csi[s:s + WINDOW])).unsqueeze(0)
        intervals = torch.from_numpy(ati[s:s + WINDOW].copy()).unsqueeze(0)
        intervals[:, -1] = 0
        f_raw, e_raw = input_features(frames, intervals)
        f_map, e_map = features_from_maps(*bank.windows([trace], [block]))
        worst = max(worst, float((f_raw - f_map).abs().max()), float((e_raw - e_map).abs().max()))
    print(f"maps vs raw features, {n} windows: max |diff| = {worst:.2e}")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="labeled", choices=["labeled", "unlabeled"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    build(args.split)
    if args.check:
        check(args.split)


if __name__ == "__main__":
    main()
