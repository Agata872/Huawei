"""Does the CSI change with headcount in the same way in different rooms?

Compares the labeled windows of each headcount, room by room, on the
hand-crafted features of feature_model.features plus a few raw descriptors
(absolute level, Doppler spectrum of the antenna-ratio CSI). Only a1 (0-4
people) and c1 (1-4) contain 2-4 people, so the decisive question is whether a
feature orders headcounts the same way in both rooms and whether a rule fitted
in one of them still holds in the other.

    python -m analysis.headcount_separability [--stride 600] [--recompute]

Runs on CPU in about a minute. Writes to runs/analysis/:
  separability.csv           one row per feature, metrics below
  feature_distributions.png  per-trace medians by headcount and room
  doppler_spectra.png        mean Doppler spectrum by headcount, per room
  windows.npz                window features, reused unless --recompute

A trace is one 60 s acquisition; its windows are strongly correlated, so the
correlations below use one median per trace. Metrics:
  rho_a1, rho_c1   Spearman correlation of trace medians with headcount in a1 / c1
  eta_class        share of the trace-median variance (a1 + c1, 1-4 people) explained by headcount
  eta_room         the same share explained by room; transferable features have eta_room << eta_class
  auc01_all        0 vs 1 person AUC, windows of every room pooled (one threshold for all rooms)
  auc01_g1         the same inside g1 (7 empty and 10 one-person traces)
  xfer_raw         balanced accuracy on 1-4 people of a nearest-class-mean rule fitted in one of
                   a1 / c1 and scored in the other, averaged over both directions (chance 0.25)
  xfer_z           the same after standardising each room by its own windows, as the unlabeled
                   traces of a test room would allow at inference
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common.data import RUNS_ROOT, WINDOW, load_raw, raw_to_frames, read_index
from common.preprocess import EPS, abs2, cmul, conj, pool_subcarriers
from feature_model.features import FeatureExtractor

OUT = RUNS_ROOT / "analysis"
KEY_ROOMS = ("a1", "c1")  # the only rooms with 2-4 people
BAND_EDGES_HZ = (0.3, 1, 2, 4, 8, 16, 32, 100)
CONTROLS = ("level_logamp", "ati_mean_ms", "dynpow_all")

# Validated reference palette (dataviz skill, references/palette.md), light mode.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ROOM_STYLE = {  # categorical slots 1-3; the 0/1-only rooms share slot 3, told apart by marker
    "a1": ("#2a78d6", "o"), "c1": ("#eb6834", "o"),
    "b1": ("#1baf7a", "s"), "e1": ("#1baf7a", "^"), "g1": ("#1baf7a", "D"), "unknown": ("#1baf7a", "X"),
}
HEADCOUNT_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")  # blue 250..650, ordinal


def extra_features(h, ati):
    """Raw descriptors not in FeatureExtractor: (B, T, 3, K, 2) pooled CSI -> names, (B, F), spectra (B, T//2)."""
    level = torch.log10(torch.sqrt(abs2(h)).mean(dim=(1, 2, 3)) + EPS)

    # Rx1/Rx0 and Rx2/Rx0: the ratio cancels the per-frame phase offset and gain shared by the chains.
    ref = h[:, :, :1]
    q = cmul(h[:, :, 1:], conj(ref)) / (abs2(ref).unsqueeze(-1) + EPS)
    q = torch.complex(q[..., 0], q[..., 1]).flatten(2)  # (B, T, 2K)
    static = q.mean(dim=1, keepdim=True)
    dyn = q - static
    ratio_pow = (dyn.abs() ** 2).mean(dim=1) / (static.abs()[:, 0] ** 2 + EPS)
    ratio_dynpow = torch.log10(ratio_pow.median(dim=1).values + EPS)

    # Doppler spectrum: each channel normalised to unit power, then averaged; +f and -f folded.
    power = torch.fft.fft(dyn, dim=1).abs() ** 2  # (B, T, 2K)
    power = power / (power[:, 1:].sum(dim=1, keepdim=True) + EPS)
    t = power.shape[1]
    negative = power[:, t // 2 + 1:].flip(1)  # bins -1, -2, ... -(T//2 - 1)
    spectrum = power[:, 1:t // 2].mean(dim=2) + negative.mean(dim=2)  # (B, T//2 - 1)
    fs = 1e6 / ati[:, :-1].mean(dim=1)  # frames per second of each window
    freqs = torch.arange(1, t // 2)[None, :] * fs[:, None] / t
    bands = [((freqs >= lo) & (freqs < hi)).float() for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])]
    band_share = [(spectrum * m).sum(dim=1) / (spectrum.sum(dim=1) + EPS) for m in bands]
    centroid = (spectrum * freqs).sum(dim=1) / (spectrum.sum(dim=1) + EPS)

    names = (["level_logamp", "ratio_dynpow"]
             + [f"dop_{lo:g}-{hi:g}Hz" for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])]
             + ["dop_centroid_hz"])
    values = torch.stack([level, ratio_dynpow, *band_share, centroid], dim=1)
    return names, values, spectrum


@torch.no_grad()
def compute_windows(stride, split="labeled", net=None, chunk=5):
    """Features of every window of a split -> dict of arrays (also the spectrum per window).

    With `net` (a raw-input model such as temporal_model.TemporalNet), its logits
    for the same windows are added under "logits".
    """
    extractor = FeatureExtractor().eval()
    rows, feats, spectra, logits, names = [], [], [], [], None
    for row in read_index(split):
        csi, ati = load_raw(split, row["identifier"])
        frames = torch.from_numpy(raw_to_frames(csi))
        pooled = pool_subcarriers(frames)  # (12000, 3, 126, 2)
        starts = list(range(0, len(ati) - WINDOW + 1, stride))
        h = torch.stack([pooled[s:s + WINDOW] for s in starts])
        a = torch.from_numpy(np.stack([ati[s:s + WINDOW] for s in starts]))
        a[:, -1] = 0
        extra_names, extra, spectrum = extra_features(h, a)
        feats.append(torch.cat([extractor(h, a), extra], dim=1).numpy())
        spectra.append(spectrum.numpy())
        if net is not None:
            for k in range(0, len(starts), chunk):
                batch = torch.stack([frames[s:s + WINDOW] for s in starts[k:k + chunk]])
                logits.append(net(batch, a[k:k + chunk]).numpy())
        names = list(extractor.names) + extra_names
        rows += [(row["identifier"], row["room"], row["label"], s) for s in starts]
        print(f"  {row['identifier']}: {len(starts)} windows", flush=True)
    ident, room, label, start = zip(*rows)
    out = dict(X=np.concatenate(feats), spectra=np.concatenate(spectra), names=np.array(names),
               ident=np.array(ident), room=np.array(room), label=np.array(label), start=np.array(start))
    if net is not None:
        out["logits"] = np.concatenate(logits)
    return out


def trace_medians(d):
    idents = list(dict.fromkeys(d["ident"]))
    med = np.stack([np.median(d["X"][d["ident"] == i], axis=0) for i in idents])
    first = [np.flatnonzero(d["ident"] == i)[0] for i in idents]
    return dict(ident=np.array(idents), X=med, room=d["room"][first], label=d["label"][first])


def eta(x, groups):
    """Share of the variance of x explained by the group means."""
    total = ((x - x.mean()) ** 2).sum()
    within = sum(((x[groups == g] - x[groups == g].mean()) ** 2).sum() for g in np.unique(groups))
    return float(1 - within / total) if total > 0 else np.nan


def balanced_accuracy(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def nearest_mean_transfer(x, y, room, standardise):
    """Mean over a1->c1 and c1->a1 of the balanced accuracy of a 1-D nearest-class-mean rule."""
    x = x.astype(np.float64).copy()
    if standardise:
        for r in KEY_ROOMS:
            m = room == r
            x[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-12)
    scores = []
    for src, dst in (KEY_ROOMS, KEY_ROOMS[::-1]):
        classes = np.unique(y[room == src])
        means = np.array([x[(room == src) & (y == c)].mean() for c in classes])
        test = room == dst
        pred = classes[np.abs(x[test][:, None] - means[None, :]).argmin(axis=1)]
        scores.append(balanced_accuracy(y[test], pred))
    return float(np.mean(scores))


def multivariate_transfer(X, y, room, standardise):
    """Logistic regression on all features, a1 -> c1 and c1 -> a1, 1-4 people."""
    X = X.astype(np.float64).copy()
    if standardise:
        for r in KEY_ROOMS:
            m = room == r
            X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-12)
    scores = {}
    for src, dst in (KEY_ROOMS, KEY_ROOMS[::-1]):
        tr, te = room == src, room == dst
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        clf = LogisticRegression(C=0.1, max_iter=5000).fit((X[tr] - mu) / sd, y[tr])
        scores[f"{src}->{dst}"] = balanced_accuracy(y[te], clf.predict((X[te] - mu) / sd))
    return scores


def score_features(d, t):
    names = d["names"]
    key_w = np.isin(d["room"], KEY_ROOMS) & (d["label"] >= 1)
    key_t = np.isin(t["room"], KEY_ROOMS) & (t["label"] >= 1)
    w01 = d["label"] <= 1
    g1 = w01 & (d["room"] == "g1")
    rows = []
    for j, name in enumerate(names):
        x = d["X"][:, j]
        rho = {}
        for r in KEY_ROOMS:
            m = t["room"] == r
            rho[r] = spearmanr(t["label"][m], t["X"][m, j]).statistic
        rows.append(dict(
            feature=name, rho_a1=rho["a1"], rho_c1=rho["c1"],
            eta_class=eta(t["X"][key_t, j], t["label"][key_t]),
            eta_room=eta(t["X"][key_t, j], t["room"][key_t]),
            auc01_all=roc_auc_score(d["label"][w01], x[w01]),
            auc01_g1=roc_auc_score(d["label"][g1], x[g1]),
            xfer_raw=nearest_mean_transfer(x[key_w], d["label"][key_w], d["room"][key_w], False),
            xfer_z=nearest_mean_transfer(x[key_w], d["label"][key_w], d["room"][key_w], True),
        ))
    rows = [{k: (float(np.nan_to_num(v, nan=0.0)) if k != "feature" else v) for k, v in r.items()} for r in rows]
    return sorted(rows, key=lambda r: -r["xfer_z"])


def print_table(rows, top):
    cols = ["rho_a1", "rho_c1", "eta_class", "eta_room", "auc01_all", "auc01_g1", "xfer_raw", "xfer_z"]
    print(f"\n{'feature':<22}" + "".join(f"{c:>10}" for c in cols))
    for r in rows[:top]:
        print(f"{r['feature']:<22}" + "".join(f"{r[c]:>10.3f}" for c in cols))
    agree = [r for r in rows if r["rho_a1"] * r["rho_c1"] > 0 and min(abs(r["rho_a1"]), abs(r["rho_c1"])) >= 0.5]
    print(f"\nfeatures with |rho| >= 0.5 and the same sign in a1 and c1: {len(agree)} of {len(rows)}")
    for r in agree:
        print(f"  {r['feature']:<22} rho_a1 {r['rho_a1']:+.2f}  rho_c1 {r['rho_c1']:+.2f}  "
              f"eta class/room {r['eta_class']:.2f}/{r['eta_room']:.2f}")


def save_csv(rows, path):
    keys = list(rows[0])
    lines = [",".join(keys)] + [",".join(r["feature"] if k == "feature" else f"{r[k]:.4f}" for k in keys) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)


def plot_distributions(d, t, rows, path, top=9, controls=CONTROLS):
    chosen = [r["feature"] for r in rows[:top]]
    chosen += [c for c in controls if c not in chosen]
    names = list(d["names"])
    by_name = {r["feature"]: r for r in rows}
    ncol = 4
    nrow = -(-len(chosen) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), facecolor=SURFACE)
    offsets = {"a1": -0.18, "c1": 0.18}
    rng = np.random.default_rng(0)
    for ax, name in zip(axes.flat, chosen):
        j = names.index(name)
        style(ax)
        for k, ident in enumerate(t["ident"]):
            room, label = t["room"][k], t["label"][k]
            color, marker = ROOM_STYLE[room]
            xpos = label + offsets.get(room, rng.uniform(-0.08, 0.08))
            w = d["X"][d["ident"] == ident, j]
            lo, hi = np.percentile(w, [25, 75])
            ax.plot([xpos, xpos], [lo, hi], color=color, linewidth=1.0, alpha=0.6, zorder=2)
            ax.scatter(xpos, t["X"][k, j], s=36, color=color, marker=marker,
                       edgecolor=SURFACE, linewidth=1.0, zorder=3)
        r = by_name[name]
        ax.set_title(name, fontsize=10, color=INK, loc="left", pad=16)
        ax.text(0.0, 1.02, f"rho a1 {r['rho_a1']:+.2f}  c1 {r['rho_c1']:+.2f}   xfer z {r['xfer_z']:.2f}",
                transform=ax.transAxes, fontsize=7.5, color=INK_2, va="bottom", ha="left")
        ax.set_xticks(range(5))
        ax.set_xlabel("headcount", fontsize=8, color=INK_2)
    for ax in axes.flat[len(chosen):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="", markersize=7, label=room)
               for room, (c, m) in ROOM_STYLE.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
               title="room (dot = trace median, bar = window IQR)", title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_spectra(d, path):
    fs = 1e6 / 5000.0  # nominal frame rate for the axis; per-window rates only shift it slightly
    t = d["spectra"].shape[1]
    freqs = np.arange(1, t + 1) * fs / (2 * (t + 1))
    panels = [("a1", "a1"), ("c1", "c1"), ("g1", "g1"), ("others", "b1, e1, unknown")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), facecolor=SURFACE, sharey=True)
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        rooms = KEY_ROOMS + ("g1",)
        m_room = ~np.isin(d["room"], rooms) if key == "others" else d["room"] == key
        for c in range(5):
            m = m_room & (d["label"] == c)
            if not m.any():
                continue
            spec = d["spectra"][m].mean(axis=0)
            ax.plot(freqs, spec, color=HEADCOUNT_RAMP[c], linewidth=2, label=f"{c} people ({m.sum()} windows)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.3, 100)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Doppler frequency (Hz)", fontsize=8, color=INK_2)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("share of dynamic power per bin", fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=WINDOW, help="frames between window starts")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached window features")
    parser.add_argument("--top", type=int, default=20, help="rows of the printed table")
    args = parser.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"windows_stride{args.stride}.npz"
    if cache.exists() and not args.recompute:
        d = {k: v for k, v in np.load(cache).items()}
    else:
        print("computing window features (CPU)...")
        d = compute_windows(args.stride)
        np.savez(cache, **d)
    bad = ~np.isfinite(d["X"]).all(axis=1)
    if bad.any():
        print(f"dropping {bad.sum()} windows with non-finite features")
        d = {k: (v[~bad] if k != "names" else v) for k, v in d.items()}
    t = trace_medians(d)
    print(f"{len(d['X'])} windows, {len(t['ident'])} traces, {len(d['names'])} features")

    rows = score_features(d, t)
    print_table(rows, args.top)
    save_csv(rows, OUT / "separability.csv")

    key = np.isin(d["room"], KEY_ROOMS) & (d["label"] >= 1)
    print("\nall features, logistic regression, 1-4 people (chance 0.25):")
    for standardise in (False, True):
        scores = multivariate_transfer(d["X"][key], d["label"][key], d["room"][key], standardise)
        tag = "per-room standardised" if standardise else "raw"
        print(f"  {tag:<22}" + "  ".join(f"{k} {v:.3f}" for k, v in scores.items()))

    plot_distributions(d, t, rows, OUT / "feature_distributions.png")
    plot_spectra(d, OUT / "doppler_spectra.png")
    print(f"\nwrote {OUT / 'separability.csv'}, feature_distributions.png, doppler_spectra.png")


if __name__ == "__main__":
    main()
