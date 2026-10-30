"""How does P13 behave in the unlabeled rooms, and where do they sit on the CSI features?

The test rooms are among the unlabeled rooms (d1, h1, i1, j1, u1, v1, unknown)
and P13 never saw their labels, so its behaviour there is the closest
label-free view of the leaderboard. Per room:

  window share   share of windows P13 assigns to 0..4 people
  trace share    the same per trace (argmax of its windows' mean logits); a
                 class-balanced room would give about 0.2 each
  agreement      mean share of a trace's windows that match the trace's majority
                 class; headcount is constant within a trace
  confidence     mean max-softmax probability
  E[y] spread    std over traces of P13's expected headcount

and, for rooms with >= 10 traces, the Spearman correlation over traces between
P13's expected headcount and each key feature, next to the true-label
correlations in a1 and c1: it shows which room's rule P13 applies there.
Labeled rooms are in P13's training set, so their rows are in-sample.

    python -m analysis.unlabeled_rooms [--recompute]

CPU only, about 5 minutes the first time. Writes to runs/analysis/:
  unlabeled_rooms.txt   the tables above
  unlabeled_rooms.png   per-trace P13 expected headcount and key feature medians, every room
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from analysis.headcount_separability import (
    GRID, HEADCOUNT_RAMP, INK, INK_2, OUT, SURFACE, compute_windows, style,
)
from common.data import WINDOW
from temporal_model.model import P13_CHECKPOINT, load_net

KEY_FEATURES = ("sf_lag1", "coh_freq16", "pr_all3", "dynpow_all", "dop_0.3-1Hz", "level_logamp")
MIN_TRACES = 10
LABELED_ROOMS = ("a1", "c1", "b1", "e1", "g1", "unknown")


def load(split, recompute):
    path = OUT / f"windows_{split}_p13.npz"
    if path.exists() and not recompute:
        return {k: v for k, v in np.load(path).items()}
    print(f"computing {split} features and P13 logits (CPU)...")
    d = compute_windows(WINDOW, split, load_net(P13_CHECKPOINT))
    np.savez(path, **d)
    return d


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def per_trace(d):
    """One row per trace: feature medians, P13 prediction, agreement, confidence, expected headcount."""
    p = softmax(d["logits"])
    ey = p @ np.arange(5)
    out = {k: [] for k in ("ident", "room", "label", "X", "pred", "agreement", "confidence", "ey")}
    for ident in dict.fromkeys(d["ident"]):
        m = d["ident"] == ident
        first = np.flatnonzero(m)[0]
        window_pred = d["logits"][m].argmax(1)
        out["ident"].append(ident)
        out["room"].append(d["room"][first])
        out["label"].append(d["label"][first])
        out["X"].append(np.median(d["X"][m], axis=0))
        out["pred"].append(int(d["logits"][m].mean(0).argmax()))
        out["agreement"].append(np.bincount(window_pred, minlength=5).max() / m.sum())
        out["confidence"].append(p[m].max(1).mean())
        out["ey"].append(ey[m].mean())
    return {k: np.array(v) for k, v in out.items()}


def room_table(splits):
    lines = [f"{'room':<10}{'traces':>7}  {'window share 0..4':<32}{'trace share 0..4':<32}"
             f"{'agree':>6}{'conf':>6}{'E[y] sd':>8}"]
    for split, (d, t) in splits.items():
        rooms = LABELED_ROOMS if split == "labeled" else sorted(set(t["room"]), key=lambda r: (r == "unknown", r))
        for room in rooms:
            mw, mt = d["room"] == room, t["room"] == room
            ws = np.bincount(d["logits"][mw].argmax(1), minlength=5) / mw.sum()
            ts = np.bincount(t["pred"][mt], minlength=5) / mt.sum()
            tag = f"{room}{'*' if split == 'labeled' else ''}"
            fmt = lambda s: " ".join(f"{v:.2f}" for v in s)
            lines.append(f"{tag:<10}{mt.sum():>7}  {fmt(ws):<32}{fmt(ts):<32}"
                         f"{t['agreement'][mt].mean():>6.2f}{t['confidence'][mt].mean():>6.2f}{t['ey'][mt].std():>8.2f}")
    lines.append("* labeled rooms are in P13's training set (in-sample)")
    return lines


def rule_table(lab, unl, names):
    rooms = [r for r in sorted(set(unl["room"])) if r != "unknown" and (unl["room"] == r).sum() >= MIN_TRACES]
    lines = ["Spearman over traces. a1/c1: true headcount; other rooms: P13 expected headcount",
             f"{'feature':<16}{'a1 true':>9}{'c1 true':>9}" + "".join(f"{r + ' P13':>10}" for r in rooms)]
    for name in KEY_FEATURES:
        j = names.index(name)
        cells = []
        for r in ("a1", "c1"):
            m = lab["room"] == r
            cells.append(spearmanr(lab["label"][m], lab["X"][m, j]).statistic)
        for r in rooms:
            m = unl["room"] == r
            cells.append(spearmanr(unl["ey"][m], unl["X"][m, j]).statistic)
        lines.append(f"{name:<16}" + "".join(f"{c:>9.2f}" for c in cells[:2]) + "".join(f"{c:>10.2f}" for c in cells[2:]))
    return lines


def plot(lab, unl, names, path):
    order = [("labeled", r) for r in LABELED_ROOMS] + \
            [("unlabeled", r) for r in sorted(set(unl["room"]), key=lambda r: (r == "unknown", r))]
    panels = [("P13 expected headcount (labeled rooms in-sample)", None)] + [(n, names.index(n)) for n in KEY_FEATURES]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 2.6 * len(panels)), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for ax, (title, j) in zip(axes, panels):
        style(ax)
        for x, (split, room) in enumerate(order):
            t = lab if split == "labeled" else unl
            m = np.flatnonzero(t["room"] == room)
            ys = t["ey"][m] if j is None else t["X"][m, j]
            # labeled: filled, colour = true headcount; unlabeled: hollow, colour = P13's trace prediction
            classes = t["label"][m] if split == "labeled" else t["pred"][m]
            colors = [HEADCOUNT_RAMP[c] for c in classes]
            xs = x + rng.uniform(-0.28, 0.28, len(m))
            if split == "labeled":
                ax.scatter(xs, ys, s=30, c=colors, edgecolor=SURFACE, linewidth=0.8, zorder=3)
            else:
                ax.scatter(xs, ys, s=30, facecolor="none", edgecolor=colors, linewidth=1.4, zorder=3)
        ax.axvline(len(LABELED_ROOMS) - 0.5, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{r}\n({(lab if s == 'labeled' else unl)['room'].tolist().count(r)})" for s, r in order],
                           fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].set_ylim(-0.2, 4.2)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=f"{k}")
               for k, c in enumerate(HEADCOUNT_RAMP)]
    handles += [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=HEADCOUNT_RAMP[2], markeredgecolor=SURFACE,
                           markersize=7, label="filled: true label (labeled rooms)"),
                plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=HEADCOUNT_RAMP[2],
                           markeredgewidth=1.4, markersize=7, label="hollow: P13 prediction (unlabeled rooms)")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
               title="headcount (one dot per trace; left of the dashed line labeled, right unlabeled; (n) traces)",
               title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="ignore the cached window features and logits")
    args = parser.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    data = {split: load(split, args.recompute) for split in ("labeled", "unlabeled")}
    traces = {split: per_trace(d) for split, d in data.items()}
    names = list(data["labeled"]["names"])

    lines = room_table({s: (data[s], traces[s]) for s in data})
    lines += [""] + rule_table(traces["labeled"], traces["unlabeled"], names)
    text = "\n".join(lines)
    print(text)
    (OUT / "unlabeled_rooms.txt").write_text(text + "\n")
    plot(traces["labeled"], traces["unlabeled"], names, OUT / "unlabeled_rooms.png")
    print(f"\nwrote {OUT / 'unlabeled_rooms.txt'} and unlabeled_rooms.png")


if __name__ == "__main__":
    main()
