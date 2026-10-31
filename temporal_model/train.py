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

Meant for a GPU (colab/train_p13.ipynb), about a minute per seed. Writes
runs/temporal_model/p13_seed<k>.pt with the EMA weights under "state_dict".
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


def train(bank, encoder_path, seed, device, steps=1600, batch=8, warmup=200):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = np.array([r["label"] for r in bank.rows])
    by_class = {c: np.flatnonzero(labels == c) for c in range(NUM_CLASSES)}
    model = load_encoder(TemporalNet(), encoder_path).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    model.encoder.requires_grad_(False)
    opt = torch.optim.AdamW([dict(params=model.encoder.parameters(), lr=2e-5),
                             dict(params=model.temporal.parameters(), lr=1e-4)], weight_decay=1e-3)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        if step == warmup + 1:
            model.encoder.requires_grad_(True)
        y = rng.integers(NUM_CLASSES, size=batch)
        traces = [int(rng.choice(by_class[c])) for c in y]
        blocks = rng.integers(0, bank.max_block + 1, size=batch)
        a, ph, ati = bank.windows(traces, blocks, device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
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
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bank = MapBank("labeled")
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"seed {seed}", flush=True)
        ema = train(bank, args.p11b, seed, args.device, args.steps)
        path = OUT / f"p13_seed{seed}.pt"
        torch.save({"state_dict": ema.state_dict(), "config": dict(seed=seed, steps=args.steps)}, path)
        print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
