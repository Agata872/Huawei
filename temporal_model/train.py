"""Retrain P13 with other seeds: the same recipe on every labeled trace.

P13 (public 0.50) is a single training run, so part of its score may be luck
of the seed. Several runs of the same recipe, averaged with the original
(temporal_model.export --checkpoint ...), should be steadier on new rooms.

The recipe follows Reference p13_fast.py:
  - encoder from the P11B EMA weights, new temporal head
  - 1600 steps of 8 windows drawn class-uniform -> trace-uniform -> start-uniform
  - AdamW, encoder lr 2e-5, head lr 1e-4, weight decay 1e-3; encoder frozen for the first 200 steps
  - loss (CE clean + 0.5 CE attenuated) / 1.5, label smoothing 0.05, gradient clip 1.0
  - EMA 0.99 of all weights (the frozen encoder keeps its EMA equal to P11B during warm-up)
Windows come from the block maps (temporal_model.maps), so starts are multiples
of 10 frames rather than any frame.

    python -m temporal_model.train --p11b <P11B checkpoint.pt> --seeds 1,2,3,4,5
    python -m temporal_model.train ... --exclude-rooms a1 --name p13_noa1
    python -m temporal_model.train ... --exclude-rooms a1 --ehunam-share 0.5 --name p13_noa1_ehunam

--exclude-rooms leaves the traces of those rooms out of training.
--ehunam-share mixes in the EHUNAM captures (temporal_model.ehunam): after the
class is drawn, a window comes from them with this probability (always when
the competition rooms left have no trace of that class).

Meant for a GPU (colab/train_p13.ipynb, colab/train_ehunam.ipynb), about a
minute per seed. Writes runs/temporal_model/<name>_seed<k>.pt with the EMA
weights under "state_dict".
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common.data import NUM_CLASSES, RUNS_ROOT
from temporal_model.loro import forward
from temporal_model.maps import MapBank, attenuation_shift
from temporal_model.model import P11B_CHECKPOINT, TemporalNet, load_encoder

OUT = RUNS_ROOT / "temporal_model"


def sample_windows(banks, pools, external_share, rng, batch, device):
    """Class-uniform -> source (competition or external) -> trace-uniform -> start-uniform windows."""
    y = rng.integers(NUM_CLASSES, size=batch)
    source = np.zeros(batch, int)
    if external_share > 0:
        source = ((rng.random(batch) < external_share) | np.array([len(pools[0][c]) == 0 for c in y])).astype(int)
    traces = [int(rng.choice(pools[s][c])) for s, c in zip(source, y)]
    blocks = rng.integers(0, np.array([banks[s].max_block for s in source]) + 1)
    parts, order = [], []
    for s in np.unique(source):
        m = np.flatnonzero(source == s)
        parts.append(banks[s].windows([traces[i] for i in m], blocks[m], device))
        order.append(m)
    a, ph, ati = (torch.cat([p[k] for p in parts]) for k in range(3))
    y = torch.as_tensor(y[np.concatenate(order)], dtype=torch.long, device=device)
    return a, ph, ati, y


def train(banks, keep, encoder_path, seed, device, steps=1600, batch=8, warmup=200, external_share=0.0):
    """banks: the labeled bank, then optionally the EHUNAM bank; keep: per bank, a mask of usable traces."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = []
    for bank, k in zip(banks, keep):
        labels = np.array([r["label"] for r in bank.rows])
        pools.append({c: np.flatnonzero((labels == c) & k) for c in range(NUM_CLASSES)})
    model = load_encoder(TemporalNet(), encoder_path).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    model.encoder.requires_grad_(False)
    opt = torch.optim.AdamW([dict(params=model.encoder.parameters(), lr=2e-5),
                             dict(params=model.temporal.parameters(), lr=1e-4)], weight_decay=1e-3)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        if step == warmup + 1:
            model.encoder.requires_grad_(True)
        a, ph, ati, y = sample_windows(banks, pools, external_share, rng, batch, device)
        model.train()
        clean = forward(model, a, ph, ati)
        augmented = forward(model, a + attenuation_shift(rng, batch).to(device), ph, ati)
        loss = (F.cross_entropy(clean, y, label_smoothing=0.05)
                + 0.5 * F.cross_entropy(augmented, y, label_smoothing=0.05)) / 1.5
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), model.parameters()):
                e.mul_(0.99).add_(p, alpha=0.01)
        if step % 400 == 0:
            print(f"    step {step}: loss {loss.item():.3f} ({time.perf_counter() - started:.0f} s)", flush=True)
    return ema.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p11b", type=Path, default=P11B_CHECKPOINT, help="P11B checkpoint (its 'ema' encoder)")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exclude-rooms", default="", help="comma-separated labeled rooms left out, e.g. a1")
    parser.add_argument("--ehunam-share", type=float, default=0.0, help="share of windows from EHUNAM captures")
    parser.add_argument("--name", default="p13", help="checkpoint name prefix")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = [r for r in args.exclude_rooms.split(",") if r]
    banks = [MapBank("labeled")]
    keep = [np.array([r["room"] not in excluded for r in banks[0].rows])]
    if args.ehunam_share > 0:
        from temporal_model.ehunam import SPLIT, read_ehunam_index  # needs the download extras
        banks.append(MapBank(SPLIT, read_ehunam_index()))
        keep.append(np.ones(len(banks[1].rows), bool))
    for bank, k in zip(banks, keep):
        labels = np.array([r["label"] for r in bank.rows])[k]
        print(f"{bank.rows[0]['split']}: {k.sum()} traces, per class {np.bincount(labels, minlength=NUM_CLASSES).tolist()}")
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"seed {seed}", flush=True)
        ema = train(banks, keep, args.p11b, seed, args.device, args.steps, external_share=args.ehunam_share)
        path = OUT / f"{args.name}_seed{seed}.pt"
        config = dict(seed=seed, steps=args.steps, exclude_rooms=excluded, ehunam_share=args.ehunam_share)
        torch.save({"state_dict": ema.state_dict(), "config": config}, path)
        print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
