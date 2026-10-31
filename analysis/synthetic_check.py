"""Do synthetic multi-person windows look like real ones? (CPU only)

a1 and c1 have real 1-4 person recordings, so synthetic k-person windows built
there (temporal_model.synthesis) can be compared with the real thing:

  1p+n   a one-person base window plus the dynamics of n more one-person windows
  0p+n   (a1 only) the empty-room window plus n one-person dynamics
  raw / lp30   source dynamics added as recorded, or low-passed at 30 Hz

Three judges, each fitted on the real windows of the same room:
  P13         the Reference network (trained on these rooms); predicted share and mean
              expected headcount. It also knows each recording's static channel, which a
              synthetic window inherits from its base, so it may lean towards the base's count.
  features    logistic regression on the 48 dynamic features (absolute level left out)
  trends      for features whose trace medians follow headcount in the room (|rho| >= 0.5),
              whether they move the same way from the real one-person windows to 1p+1..3

    python -m analysis.synthetic_check [--n 40]

Writes runs/analysis/synthetic_check.txt and synthetic_check.png.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from analysis.headcount_separability import INK, INK_2, OUT, SURFACE, extra_features, style
from analysis.unlabeled_rooms import load, softmax
from common.data import WINDOW, load_raw, read_index
from common.preprocess import pool_subcarriers
from feature_model.features import FeatureExtractor
from temporal_model.model import P13_CHECKPOINT, load_net
from temporal_model.synthesis import synthesize

ROOMS = ("a1", "c1")
VARIANTS = {"raw": None, "lp30": 30.0}
PLOT_FEATURES = ("dynpow_all", "sf_lag1", "coh_freq16", "pr_all3")
SERIES_STYLE = {"real": ("#2a78d6", "o"), "raw": ("#eb6834", "D"), "lp30": ("#1baf7a", "D")}  # palette slots 1-3


def random_windows(rng, traces, n, exclude=()):
    """n (trace index, start) pairs whose windows do not overlap each other or `exclude`."""
    picks = list(exclude)
    while len(picks) < len(exclude) + n:
        t, s = int(rng.integers(len(traces))), int(rng.integers(0, 12000 - WINDOW + 1))
        if all(t != pt or abs(s - ps) >= WINDOW for pt, ps in picks):
            picks.append((t, s))
    return picks[len(exclude):]


def build_sets(room, n, rng):
    """Synthetic sets of one room, one at a time: (name, (complex windows (n, 600, 1, 3, 1024), intervals (n, 600), k))."""
    rows = [r for r in read_index("labeled") if r["room"] == room]
    load_trace = lambda r: load_raw("labeled", r["identifier"])
    one = [load_trace(r) for r in rows if r["label"] == 1]
    empty = [load_trace(r) for r in rows if r["label"] == 0]
    window = lambda tr, s: (tr[0][s:s + WINDOW], tr[1][s:s + WINDOW].copy())
    bases = [("1p", one, 1)] + ([("0p", empty, 0)] if empty else [])
    for base_name, base_traces, base_k in bases:
        for added in range(1, 5 - base_k):
            for variant, cutoff in VARIANTS.items():
                windows, intervals = [], []
                for _ in range(n):
                    (bt, bs), = random_windows(rng, base_traces, 1)
                    exclude = [(bt, bs)] if base_traces is one else []
                    sources = random_windows(rng, one, added, exclude)
                    base, ati = window(base_traces[bt], bs)
                    src = [window(one[t], s) for t, s in sources]
                    windows.append(synthesize(base, [w for w, _ in src], cutoff, [a for _, a in src]))
                    ati[-1] = 0
                    intervals.append(ati)
                yield f"{base_name}+{added} {variant}", (np.stack(windows), np.stack(intervals), base_k + added)


@torch.no_grad()
def judge_inputs(windows, intervals, net, extractor, chunk=8):
    """Complex windows -> (features (n, 49), P13 logits (n, 5))."""
    feats, logits = [], []
    for k in range(0, len(windows), chunk):
        frames = torch.from_numpy(np.ascontiguousarray(windows[k:k + chunk]).view(np.float32)
                                  .reshape(-1, WINDOW, 1, 3, 1024, 2))
        a = torch.from_numpy(intervals[k:k + chunk])
        pooled = pool_subcarriers(frames)
        _, extra, _ = extra_features(pooled, a)
        feats.append(torch.cat([extractor(pooled, a), extra], dim=1).numpy())
        logits.append(net(frames, a).numpy())
    return np.concatenate(feats), np.concatenate(logits)


def share(pred):
    return " ".join(f"{v:.2f}" for v in np.bincount(pred, minlength=5) / len(pred))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=40, help="synthetic windows per set")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    rng = np.random.default_rng(args.seed)
    real = load("labeled", False)  # cached real window features + P13 logits
    names = list(real["names"])
    dyn_cols = [j for j, n in enumerate(names) if n != "level_logamp"]
    net, extractor = load_net(P13_CHECKPOINT), FeatureExtractor().eval()
    lines, results = [], {}

    for room in ROOMS:
        m = real["room"] == room
        Xr, yr, zr, idr = real["X"][m], real["label"][m], real["logits"][m], real["ident"][m]
        mu, sd = Xr[:, dyn_cols].mean(0), Xr[:, dyn_cols].std(0) + 1e-12
        clf = LogisticRegression(C=0.1, max_iter=5000).fit((Xr[:, dyn_cols] - mu) / sd, yr)
        lines += [f"\n=== {room} ===",
                  f"{'set':<14}{'k':>3}  {'P13 share 0..4':<26}{'P13 E[y]':>9}  {'feature clf share 0..4':<26}{'clf mean':>9}"]
        ey_real = softmax(zr) @ np.arange(5)
        for k in sorted(set(yr)):
            mk = yr == k
            lines.append(f"{'real':<14}{k:>3}  {share(zr[mk].argmax(1)):<26}{ey_real[mk].mean():>9.2f}  "
                         f"{'(in-sample)':<26}{'':>9}")
        room_results = {"real": (Xr, yr, ey_real, idr)}
        for name, (windows, intervals, k) in build_sets(room, args.n, rng):
            print(f"  {room} {name}", flush=True)
            X, z = judge_inputs(windows, intervals, net, extractor)
            del windows
            pred_clf = clf.predict((X[:, dyn_cols] - mu) / sd)
            ey = softmax(z) @ np.arange(5)
            lines.append(f"{name:<14}{k:>3}  {share(z.argmax(1)):<26}{ey.mean():>9.2f}  "
                         f"{share(pred_clf):<26}{pred_clf.mean():>9.2f}")
            room_results[name] = (X, np.full(len(X), k), ey, None)

        # trend agreement: real trace medians vs synthetic 1p+n (with the real one-person windows as k = 1)
        traces = list(dict.fromkeys(idr))
        med = np.stack([np.median(Xr[idr == t], axis=0) for t in traces])
        lab = np.array([yr[idr == t][0] for t in traces])
        keep = lab >= 1
        for variant in VARIANTS:
            syn = [room_results[f"1p+{n} {variant}"] for n in (1, 2, 3)]
            Xs = np.concatenate([Xr[yr == 1]] + [s[0] for s in syn])
            ks = np.concatenate([np.ones((yr == 1).sum())] + [s[1] for s in syn])
            agree = total = 0
            for j in dyn_cols:
                rho_real = spearmanr(lab[keep], med[keep, j]).statistic
                if abs(rho_real) >= 0.5:
                    total += 1
                    agree += int(np.sign(spearmanr(ks, Xs[:, j]).statistic) == np.sign(rho_real))
            lines.append(f"trend agreement ({variant}): {agree} of {total} headcount-following features move the same way")
        results[room] = room_results

    text = "\n".join(lines)
    print(text)
    (OUT / "synthetic_check.txt").write_text(text + "\n")
    plot(results, names, OUT / "synthetic_check.png")
    print(f"\nwrote {OUT / 'synthetic_check.txt'} and synthetic_check.png")


def plot(results, names, path):
    panels = [("P13 expected headcount", None)] + [(n, names.index(n)) for n in PLOT_FEATURES]
    fig, axes = plt.subplots(len(ROOMS), len(panels), figsize=(4 * len(panels), 3.4 * len(ROOMS)), facecolor=SURFACE)
    for row, room in zip(axes, ROOMS):
        res = results[room]
        for ax, (title, j) in zip(row, panels):
            style(ax)
            X, y, ey, ident = res["real"]
            for t in dict.fromkeys(ident):  # real: one dot per recording
                m = ident == t
                v = ey[m].mean() if j is None else np.median(X[m, j])
                ax.scatter(y[m][0] - 0.12, v, s=34, color=SERIES_STYLE["real"][0], edgecolor=SURFACE, linewidth=0.8, zorder=3)
            for variant, dx in (("raw", 0.08), ("lp30", 0.22)):
                color, marker = SERIES_STYLE[variant]
                for name, (Xs, ks, eys, _) in res.items():
                    if not name.endswith(variant):
                        continue
                    v = eys if j is None else Xs[:, j]
                    lo, mid, hi = np.percentile(v, [25, 50, 75])
                    face = "none" if name.startswith("0p") else color  # hollow = empty-room base
                    ax.plot([ks[0] + dx] * 2, [lo, hi], color=color, linewidth=1.2, zorder=2)
                    ax.scatter(ks[0] + dx, mid, s=34, marker=marker, facecolor=face, edgecolor=color, linewidth=1.4, zorder=3)
            ax.set_xticks(range(5))
            ax.set_xlabel("headcount (real) / k (synthetic)", fontsize=8, color=INK_2)
            ax.set_title(f"{room}: {title}", fontsize=10, color=INK, loc="left")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=SERIES_STYLE["real"][0], markersize=7, label="real recording"),
               plt.Line2D([], [], marker="D", linestyle="", color=SERIES_STYLE["raw"][0], markersize=7, label="synthetic, raw dynamics"),
               plt.Line2D([], [], marker="D", linestyle="", color=SERIES_STYLE["lp30"][0], markersize=7, label="synthetic, low-passed 30 Hz"),
               plt.Line2D([], [], marker="D", linestyle="", markerfacecolor="none", markeredgecolor=INK_2, markersize=7,
                          label="hollow: empty-room base (a1)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9,
               title="synthetic: median and IQR of windows", title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
