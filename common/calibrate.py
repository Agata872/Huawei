"""Per-room prior calibration (label-shift correction) and its offline check on LORO logits.

The public test set is exactly class-balanced and its rooms may appear in the
unlabeled data. A model that ranks windows sensibly within a room but is
biased per room (e.g. the CNN calls most u1 windows 3-4 people and almost
never says 2) can be corrected with a per-room logit bias b chosen so that
the mean softmax over that room's windows matches the target prior
(Saerens et al. 2002; Lipton et al. 2018, BBSE).

    python -m common.calibrate runs/deep_model/<run>.json     # offline check on a LORO run
    python -m common.calibrate --export runs/deep_model/cnn_base.pt   # -> submissions/deep_cnn_base_cal.onnx

At inference the room is recognised by the static-channel fingerprint of the
window (normalised |mean CSI| over antennas x subcarrier groups), matched to a
bank of unlabeled windows; nearest-neighbour room ID is 96-100% correct for
rooms in the bank. Windows whose best similarity is below MIN_SIMILARITY
(rooms not in the bank) keep the uncalibrated logits.
"""

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

NUM_CLASSES = 5
MIN_SIMILARITY = 0.998  # seen rooms: >= 0.9992 for 99% of windows (v1: 95%); unseen rooms: <= 0.9975
MIN_TRACES = 10  # only rooms with enough unlabeled traces are assumed class-balanced
BANK_PER_TRACE = 10


def fingerprint(h):
    """(B, T, A, K, 2) pooled CSI -> (B, A*K) unit-norm static amplitude shape."""
    st = h.mean(dim=1)
    a = torch.sqrt((st * st).sum(dim=-1) + 1e-12)
    a = a / torch.sqrt((a * a).mean(dim=-1, keepdim=True))
    z = a.flatten(1)
    return z / torch.sqrt((z * z).sum(dim=1, keepdim=True))


class RoomCalibration(nn.Module):
    """core logits + the bias of the bank room nearest to the window's fingerprint."""

    def __init__(self, core, bank, bank_room, biases, min_similarity=MIN_SIMILARITY):
        super().__init__()
        self.core = core
        self.min_similarity = min_similarity
        self.register_buffer("bank", torch.as_tensor(bank, dtype=torch.float32))        # (N, D)
        self.register_buffer("bank_room", torch.as_tensor(bank_room, dtype=torch.long))  # (N,)
        self.register_buffer("biases", torch.as_tensor(biases, dtype=torch.float32))    # (R, 5)

    def forward(self, h, ati):
        logits = self.core(h, ati)
        sim = fingerprint(h) @ self.bank.T
        best, idx = sim.max(dim=1)
        bias = self.biases[self.bank_room[idx]]
        return logits + torch.where((best >= self.min_similarity)[:, None], bias, torch.zeros_like(bias))


def fit_bias(logits, target, iters=300):
    """Bias b (classes with target 0 -> -inf) so that mean softmax(logits + b) == target."""
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    live = target > 0
    b = np.where(live, 0.0, -np.inf)
    for _ in range(iters):
        z = logits + b
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        b[live] += np.log(target[live]) - np.log(p[:, live].mean(axis=0) + 1e-12)
        b[live] -= b[live].mean()
    return b


def fit_bias_hard(logits, target, iters=2000, step=0.05):
    """Bias so that the share of argmax predictions matches target (soft fit as the start)."""
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    live = target > 0
    b = fit_bias(logits, target)
    best, best_err = b.copy(), np.inf
    for _ in range(iters):
        share = np.bincount((logits + b).argmax(1), minlength=len(target)) / len(logits)
        err = np.abs(share - target).sum()
        if err < best_err:
            best, best_err = b.copy(), err
        b[live] += step * (np.log(target[live]) - np.log(share[live] + 0.5 / len(logits)))
    return best


def balanced_recall(labels, preds, classes):
    return float(np.mean([np.mean(preds[labels == c] == c) for c in classes]))


def evaluate(records, combos=30, seed=0):
    """Mimic a balanced test room: one trace per class; bias from the first half of
    each trace's windows, scored on the second half."""
    rng = np.random.default_rng(seed)
    by_fold = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_fold[r["fold"]][r["ident"]].append(r)
    rows = []
    for fold, traces in sorted(by_fold.items()):
        by_class = defaultdict(list)
        for ident, rs in traces.items():
            by_class[rs[0]["label"]].append(ident)
        classes = sorted(by_class)
        if len(classes) < 2:
            continue
        all_combos = list(itertools.product(*[by_class[c] for c in classes]))
        picks = [all_combos[i] for i in rng.permutation(len(all_combos))[:combos]]
        target = np.zeros(NUM_CLASSES)
        target[classes] = 1.0 / len(classes)
        base, cal, cal_trans, cal_hard = [], [], [], []
        for combo in picks:
            fit_l, ev_l, ev_y = [], [], []
            for ident in combo:
                rs = traces[ident]
                half = len(rs) // 2
                fit_l += [r["logits"] for r in rs[:half]]
                ev_l += [r["logits"] for r in rs[half:]]
                ev_y += [r["label"] for r in rs[half:]]
            ev_l, ev_y = np.array(ev_l), np.array(ev_y)
            base.append(balanced_recall(ev_y, ev_l.argmax(1), classes))
            b = fit_bias(fit_l, target)
            cal.append(balanced_recall(ev_y, (ev_l + b).argmax(1), classes))
            b_all = fit_bias(np.vstack([fit_l, ev_l]), target)
            cal_trans.append(balanced_recall(ev_y, (ev_l + b_all).argmax(1), classes))
            b_hard = fit_bias_hard(fit_l, target)
            cal_hard.append(balanced_recall(ev_y, (ev_l + b_hard).argmax(1), classes))
        rows.append((fold, classes, len(picks), np.mean(base), np.mean(cal), np.mean(cal_hard), np.mean(cal_trans)))
    print(f"{'fold':8} {'classes':16} {'combos':>6} | {'raw':>6} | {'soft cal':>8} | {'hard cal':>8} | {'transductive':>12}")
    for fold, classes, n, b, c, hc, t in rows:
        note = "" if len(classes) == NUM_CLASSES else "  (only these classes; optimistic)"
        print(f"{fold:8} {str(classes):16} {n:6d} | {b:6.3f} | {c:8.3f} | {hc:8.3f} | {t:12.3f}{note}")
    return rows


@torch.no_grad()
def build_calibrated(core, device="cpu", min_traces=MIN_TRACES, batch=100):
    """Fit per-room biases of `core` on the unlabeled rooms and wrap it in RoomCalibration."""
    from common.cv import eval_starts
    from common.data import load_traces

    core = core.to(device).eval()
    traces = load_traces("unlabeled", mmap=True)
    size = Counter(t.room for t in traces)
    rooms = sorted({t.room if t.room != "unknown" else t.ident for t in traces})
    logits, fps, room_of = defaultdict(list), [], []
    for t in traces:
        key = t.room if t.room != "unknown" else t.ident
        starts = eval_starts(t)
        for k in range(0, len(starts), batch):
            s = starts[k:k + batch]
            h = torch.from_numpy(np.stack([t.csi[i:i + 600] for i in s])).to(device)
            ati = torch.from_numpy(np.stack([t.ati[i:i + 600] for i in s])).to(device)
            ati[:, -1] = 0
            logits[key] += core(h, ati).float().cpu().tolist()
            z = fingerprint(h.float()).cpu().numpy()
            keep = np.linspace(0, len(z) - 1, min(BANK_PER_TRACE, len(z))).astype(int) if k == 0 else []
            fps += list(z[keep])
            room_of += [rooms.index(key)] * len(keep)
    biases = np.zeros((len(rooms), NUM_CLASSES))
    for r, key in enumerate(rooms):
        if key in size and size[key] >= min_traces:
            biases[r] = fit_bias(logits[key], np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
            before = np.bincount(np.argmax(logits[key], 1), minlength=5) / len(logits[key])
            after = np.bincount(np.argmax(np.array(logits[key]) + biases[r], 1), minlength=5) / len(logits[key])
            print(f"  {key}: bias {np.round(biases[r], 2)} | predicted share {np.round(before, 2)} -> {np.round(after, 2)}")
    print(f"bank: {len(fps)} windows from {len(rooms)} rooms; calibrated rooms: "
          f"{[k for k in rooms if size.get(k, 0) >= min_traces]}")
    return RoomCalibration(core.cpu(), np.array(fps), np.array(room_of), biases)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="LORO run JSON with per-window logits, or a checkpoint with --export")
    parser.add_argument("--export", action="store_true", help="Build and export the calibrated ONNX model")
    parser.add_argument("--gate", action="store_true", help="Also apply the quiet-window gate")
    parser.add_argument("--min-traces", type=int, default=MIN_TRACES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.export:
        records = json.load(open(args.path))["records"]
        if "logits" not in records[0]:
            raise SystemExit("run has no logits; rerun LORO with the current deep_model.train")
        evaluate(records)
        return
    from common.export import export_onnx
    from common.gate import QuietGate
    from deep_model.train import build_model

    ckpt = torch.load(args.path, map_location="cpu")
    core = build_model(ckpt["cfg"])
    core.load_state_dict(ckpt["state_dict"])
    model, suffix = build_calibrated(core, args.device, args.min_traces), "_cal"
    suffix += "_gate" if args.gate else ""
    if args.gate:
        model = QuietGate(model)
    output = args.output or Path("submissions") / f"deep_{Path(args.path).stem}{suffix}.onnx"
    export_onnx(model.eval(), output)


if __name__ == "__main__":
    main()
