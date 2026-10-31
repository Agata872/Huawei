"""Do P13 runs with other seeds agree with P13 in the unlabeled rooms? (CPU only)

The unlabeled rooms are new to every run, so disagreement there shows how much
of a single run's predictions comes from its seed, and how far a mean-logit
ensemble moves away from P13.

    python -m temporal_model.maps --split unlabeled     # once
    python -m analysis.seed_agreement                    # P13 + runs/temporal_model/p13_seed*.pt

Writes runs/analysis/seed_agreement.txt.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from analysis.headcount_separability import OUT
from common.data import NUM_CLASSES, RUNS_ROOT
from temporal_model.loro import forward
from temporal_model.maps import WINDOW_BLOCKS, MapBank
from temporal_model.model import P13_CHECKPOINT, load_net


@torch.no_grad()
def window_logits(net, bank, chunk=40):
    """Logits of the 20 non-overlapping windows of every trace -> (traces * 20, 5)."""
    starts = list(range(0, bank.max_block + 1, WINDOW_BLOCKS))
    pairs = [(k, s) for k in range(len(bank.rows)) for s in starts]
    out = []
    for i in range(0, len(pairs), chunk):
        traces, blocks = zip(*pairs[i:i + chunk])
        out.append(forward(net, *bank.windows(list(traces), list(blocks))).numpy())
    return np.concatenate(out)


def shares(pred):
    return " ".join(f"{v:.2f}" for v in np.bincount(pred, minlength=NUM_CLASSES) / len(pred))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="*", type=Path,
                        default=sorted((RUNS_ROOT / "temporal_model").glob("p13_seed*.pt")))
    args = parser.parse_args()
    torch.set_num_threads(8)
    bank = MapBank("unlabeled")
    room = np.repeat([r["room"] for r in bank.rows], (bank.max_block + WINDOW_BLOCKS) // WINDOW_BLOCKS)
    names = ["P13"] + [p.stem for p in args.checkpoints]
    logits = [window_logits(load_net(p), bank) for p in [P13_CHECKPOINT] + list(args.checkpoints)]
    pred = [z.argmax(1) for z in logits]
    ensemble = np.mean(logits, axis=0).argmax(1)

    lines = [f"{len(names)} models on {len(room)} unlabeled windows: {', '.join(names)}", "",
             f"{'model':<14}{'agree with P13':>15}  predicted share 0..4"]
    for name, p in zip(names, pred):
        lines.append(f"{name:<14}{np.mean(p == pred[0]):>15.3f}  {shares(p)}")
    lines.append(f"{'ensemble':<14}{np.mean(ensemble == pred[0]):>15.3f}  {shares(ensemble)}")
    if len(pred) > 2:
        pairs = [np.mean(pred[i] == pred[j]) for i in range(1, len(pred)) for j in range(i + 1, len(pred))]
        lines.append(f"new seeds agree with each other: {np.mean(pairs):.3f} on average "
                     f"(min {np.min(pairs):.3f})")
    lines += ["", f"{'room':<10}{'windows':>8}  {'P13 share 0..4':<26}{'ensemble share 0..4':<26}{'ens = P13':>9}"]
    for r in sorted(set(room)):
        m = room == r
        lines.append(f"{r:<10}{m.sum():>8}  {shares(pred[0][m]):<26}{shares(ensemble[m]):<26}"
                     f"{np.mean(ensemble[m] == pred[0][m]):>9.3f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "seed_agreement.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
