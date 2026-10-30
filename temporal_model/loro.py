"""Does the P13 network learn headcount structure that carries over to an unseen room?

P13 starts from an encoder that saw every labeled trace, so its training
accuracy says nothing about new rooms. This script trains the same
architecture from scratch on the precomputed maps (temporal_model.maps) and
scores windows it never saw:

  room:c1     every c1 trace held out: cross-room (c1 then has 1-4 people, a1 is
              the only training room with 2-4)
  traces:c1   one trace per headcount of c1 held out, the rest of c1 kept: same
              room, unseen recordings. The gap between the two is the room effect.

    python -m temporal_model.loro --holdout room:c1,room:a1,traces:c1,traces:a1 --seeds 0,1,2

Meant for a GPU (e.g. Colab, see colab/loro.ipynb). Each run is 3000 steps of
batch 16 and takes a few minutes. Per run it prints balanced accuracy over the
held-out classes, the confusion matrix and the Spearman correlation between
the expected headcount and the label (an ordering that per-room calibration
could still exploit when the argmax is off). Results with per-window logits go
to runs/temporal_model/loro_<holdout>_seed<k>.json.
"""

import argparse
import copy
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

from common.data import NUM_CLASSES, RUNS_ROOT
from temporal_model.maps import WINDOW_BLOCKS, MapBank, attenuation_shift
from temporal_model.model import TemporalNet, features_from_maps

OUT = RUNS_ROOT / "temporal_model"


def split_traces(rows, holdout, seed):
    """holdout 'room:<r>' or 'traces:<r>' -> (train indices, test indices)."""
    kind, room = holdout.split(":")
    in_room = [k for k, r in enumerate(rows) if r["room"] == room]
    if not in_room:
        raise ValueError(f"no labeled traces in room {room}")
    if kind == "room":
        test = in_room
    elif kind == "traces":
        rng = np.random.default_rng(seed)
        test = [int(rng.choice([k for k in in_room if rows[k]["label"] == c]))
                for c in sorted({rows[k]["label"] for k in in_room})]
    else:
        raise ValueError(holdout)
    train = [k for k in range(len(rows)) if k not in test]
    return train, test


def forward(model, a, ph, ati):
    features, extras = features_from_maps(a, ph, ati)
    return model.forward_features(features, extras)


def train(bank, train_idx, seed, device, steps, batch, lr):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = np.array([bank.rows[k]["label"] for k in train_idx])
    by_class = {c: [train_idx[i] for i in np.flatnonzero(labels == c)] for c in range(NUM_CLASSES)}
    classes = [c for c in by_class if by_class[c]]
    model = TemporalNet().to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        # class-uniform -> trace-uniform -> start uniform, as in P13
        y = rng.choice(classes, size=batch)
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
        sched.step()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), model.parameters()):
                e.mul_(0.99).add_(p, alpha=0.01)
        if step % 500 == 0:
            print(f"    step {step}: loss {loss.item():.3f} ({time.perf_counter() - started:.0f} s)", flush=True)
    return ema


@torch.no_grad()
def evaluate(model, bank, test_idx, device):
    records = []
    starts = list(range(0, bank.max_block + 1, WINDOW_BLOCKS))  # the 20 non-overlapping windows
    for k in test_idx:
        logits = forward(model, *bank.windows([k] * len(starts), starts, device)).cpu().numpy()
        records += [dict(ident=bank.rows[k]["identifier"], label=bank.rows[k]["label"], start=s * 10,
                         logits=z.tolist()) for s, z in zip(starts, logits)]
    return records


def summarize(records):
    y = np.array([r["label"] for r in records])
    z = np.array([r["logits"] for r in records])
    pred = z.argmax(1)
    p = np.exp(z - z.max(1, keepdims=True))
    ey = (p / p.sum(1, keepdims=True)) @ np.arange(NUM_CLASSES)
    present = np.unique(y)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    np.add.at(cm, (y, pred), 1)
    idents = list(dict.fromkeys(r["ident"] for r in records))
    trace_y = np.array([y[[r["ident"] == i for r in records]][0] for i in idents])
    trace_ey = np.array([ey[[r["ident"] == i for r in records]].mean() for i in idents])
    return dict(
        balanced=float(np.mean([np.mean(pred[y == c] == c) for c in present])),
        chance=1 / len(present),
        spearman_window=float(spearmanr(y, ey).statistic) if len(present) > 1 else float("nan"),
        spearman_trace=float(spearmanr(trace_y, trace_ey).statistic) if len(present) > 1 else float("nan"),
        confusion=cm.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", default="room:c1,room:a1,traces:c1,traces:a1")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bank = MapBank("labeled")
    table = []
    for holdout in args.holdout.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            train_idx, test_idx = split_traces(bank.rows, holdout, seed)
            print(f"{holdout} seed {seed}: train {len(train_idx)} traces, test "
                  f"{[bank.rows[k]['identifier'] for k in test_idx]}", flush=True)
            model = train(bank, train_idx, seed, args.device, args.steps, args.batch, args.lr)
            records = evaluate(model, bank, test_idx, args.device)
            summary = summarize(records)
            print(f"  balanced {summary['balanced']:.3f} (chance {summary['chance']:.2f}), Spearman "
                  f"window {summary['spearman_window']:+.2f} trace {summary['spearman_trace']:+.2f}", flush=True)
            for c, row in enumerate(summary["confusion"]):
                if sum(row):
                    print(f"    true {c}: {row}")
            path = OUT / f"loro_{holdout.replace(':', '_')}_seed{seed}.json"
            path.write_text(json.dumps(dict(config=vars(args) | dict(holdout=holdout, seed=seed),
                                            summary=summary, records=records)))
            table.append((holdout, seed, summary))
    print(f"\n{'holdout':<12}{'seeds':>6}{'balanced':>10}{'chance':>8}{'rho win':>9}{'rho trace':>10}")
    for holdout in dict.fromkeys(h for h, _, _ in table):
        s = [x for h, _, x in table if h == holdout]
        print(f"{holdout:<12}{len(s):>6}{np.mean([x['balanced'] for x in s]):>10.3f}{s[0]['chance']:>8.2f}"
              f"{np.mean([x['spearman_window'] for x in s]):>+9.2f}{np.mean([x['spearman_trace'] for x in s]):>+10.2f}")


if __name__ == "__main__":
    main()
