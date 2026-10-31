"""Do SHARP-style Doppler features order headcounts the same way in a1 and c1? (CPU only)

SHARP (Meneghello et al., IEEE TMC 2023, github.com/francescamen/SHARP)
recognises activities in rooms it was not trained in from Doppler traces of
80 MHz CSI. Its ablation found the phase sanitization decisive: every Rx
antenna is cleaned on its own by referencing it to its strongest (assumed
static) path. Referencing one antenna to another, as the antenna-ratio Doppler
of analysis.headcount_separability does, dropped its accuracy in an unseen lab
from 96% to 60%.

Pipeline, per 600-frame window as at inference:
  1. every frame and antenna: divide by the mean amplitude, find the strongest path (peak of
     the 8x zero-padded impulse response, parabolic refinement; SHARP fits a Lasso) and
     multiply by the conjugate of that path's response, which removes the common phase
     (CFO, PPO) and the linear phase (SFO, PDD) offsets
  2. average groups of 8 subcarriers (126 groups), resample to a uniform grid at the
     median frame interval, subtract the window mean (the static part)
  3. Doppler vectors as in SHARP: 31-frame Hanning windows (hop 2) zero-padded to 100
     bins, power summed over subcarriers -> a (vectors x 100) Doppler trace per antenna
     covering +-98 Hz
The static part is removed before step 3 because people who sit or stand move
slowly, and the static leakage would otherwise cover the lowest ~12 Hz.

Window features (median over the 3 antennas) of the noise-corrected trace, with
the noise floor taken from the bins above 75 Hz:
  sh_dyn_db             dynamic power relative to the static power
  sh_snr_db             mean power below 40 Hz relative to the noise floor
  sh_rms_hz             RMS Doppler spread of the mean profile
  sh_entropy            normalised spectral entropy of the mean profile
  sh_gt{5,10,20,40}hz   share of the mean profile above that |frequency|
  sh_asym               |positive - negative| share of the mean profile
  sh_spread_med, _iqr   RMS spread of each Doppler vector: median and IQR over time
  sh_burst              log10 of the 90th / 10th percentile of the vector power
  sh_active             share of vectors whose motion power exceeds the noise power

They are scored like analysis.headcount_separability (same metrics, see
there), and a logistic regression on 1-4 people is fitted in a1 and scored in
c1 and vice versa, on these features, on the existing window features and on
both.

    python -m analysis.sharp_doppler [--recompute]

Runs in about 10 minutes. Writes runs/analysis/sharp_doppler.txt,
sharp_separability.csv, sharp_distributions.png and sharp_profiles.png; window
features are cached in sharp_windows.npz.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from numpy.lib.stride_tricks import sliding_window_view

from analysis.headcount_separability import (HEADCOUNT_RAMP, INK, INK_2, KEY_ROOMS, OUT, SURFACE,
                                             multivariate_transfer, plot_distributions, save_csv,
                                             score_features, style, trace_medians)
from common.data import WINDOW, load_raw, read_index
from common.preprocess import NUM_GROUPS, SC_GROUP, SC_START, SC_STOP

OVERSAMPLE = 8
N_WIN, HOP, N_DOPPLER = 31, 2, 100
NOISE_HZ, SIGNAL_HZ = 75, 40
SHARE_HZ = (5, 10, 20, 40)
NAMES = (["sh_dyn_db", "sh_snr_db", "sh_rms_hz", "sh_entropy"] + [f"sh_gt{f}hz" for f in SHARE_HZ]
         + ["sh_asym", "sh_spread_med", "sh_spread_iqr", "sh_burst", "sh_active"])
HANN = np.hanning(N_WIN).astype(np.float32)


def sanitize(csi, chunk=500):
    """Raw CSI (T, 1, 3, 1024) -> (T, 3, 126) complex64, each frame and antenna referenced to its strongest path."""
    out = np.empty((len(csi), 3, NUM_GROUPS), np.complex64)
    j = np.arange(SC_STOP - SC_START, dtype=np.float32)
    n = OVERSAMPLE * 1024
    for s in range(0, len(csi), chunk):
        h = csi[s:s + chunk, 0, :, SC_START:SC_STOP]  # (B, 3, 1008)
        h = h / np.abs(h).mean(axis=-1, keepdims=True)
        cir = np.abs(scipy.fft.ifft(h, n=n, axis=-1, workers=-1))
        m = cir.argmax(axis=-1)
        y0 = np.take_along_axis(cir, m[..., None], -1)[..., 0]
        ym = np.take_along_axis(cir, ((m - 1) % n)[..., None], -1)[..., 0]
        yp = np.take_along_axis(cir, ((m + 1) % n)[..., None], -1)[..., 0]
        delta = 0.5 * (ym - yp) / (ym - 2 * y0 + yp - 1e-12)
        theta = (2 * np.pi * (m + delta) / n).astype(np.float32)  # phase slope per subcarrier
        steer = np.exp(1j * theta[..., None] * j)  # conjugate of the path's linear phase
        r = (h * steer).mean(axis=-1, keepdims=True)  # complex gain of the strongest path
        clean = h * steer * (np.conj(r) / np.abs(r))
        out[s:s + chunk] = clean.reshape(*clean.shape[:-1], NUM_GROUPS, SC_GROUP).mean(axis=-1)
    return out


def uniform(x, ati):
    """Frames (N, ...) with intervals ati (us, last ignored) -> resampled at the median interval, frame rate."""
    t = np.concatenate([[0.0], np.cumsum(ati[:-1], dtype=np.float64)])
    dt = float(np.median(ati[:-1]))
    tu = np.arange(0.0, t[-1] + 1e-6, dt)
    i = np.clip(np.searchsorted(t, tu, side="right") - 1, 0, len(t) - 2)
    w = ((tu - t[i]) / (t[i + 1] - t[i])).astype(np.float32).reshape(-1, *[1] * (x.ndim - 1))
    return (1 - w) * x[i] + w * x[i + 1], 1e6 / dt


def doppler_trace(x):
    """Static-removed frames (N, 3, K) -> Doppler trace (vectors, 3, 100), bins -50..49."""
    seg = sliding_window_view(x, N_WIN, axis=0)[::HOP]  # (V, 3, K, 31)
    spec = scipy.fft.fft(seg * HANN, n=N_DOPPLER, axis=-1, workers=-1)
    return np.fft.fftshift((np.abs(spec) ** 2).sum(axis=2), axes=-1)


def window_features(x, ati):
    """Sanitized pooled frames of one window (600, 3, 126) -> features (len(NAMES),), mean profile (100,)."""
    x, fs = uniform(x, ati)
    static = x.mean(axis=0)
    x = x - static
    dyn_db = 10 * np.log10((np.abs(x) ** 2).mean(axis=(0, 2)) / (np.abs(static) ** 2).mean(axis=1))
    D = doppler_trace(x)  # (V, 3, 100)
    f = (np.arange(N_DOPPLER) - N_DOPPLER // 2) * fs / N_DOPPLER
    noise = D[..., np.abs(f) > NOISE_HZ].mean(axis=(0, 2))  # (3,) per bin
    Dc = np.clip(D - noise[None, :, None], 0, None)
    P = Dc.mean(axis=0)  # (3, 100)
    p = P / (P.sum(axis=1, keepdims=True) + 1e-20)
    e = Dc.sum(axis=2)  # (V, 3) motion power per vector
    spread = np.sqrt((Dc * f ** 2).sum(axis=2) / (e + 1e-20))
    raw = D.sum(axis=2)
    feats = np.stack([
        dyn_db,
        10 * np.log10(D[..., np.abs(f) < SIGNAL_HZ].mean(axis=(0, 2)) / noise),
        np.sqrt((p * f ** 2).sum(axis=1)),
        -(p * np.log(p + 1e-20)).sum(axis=1) / np.log(N_DOPPLER),
        *[p[:, np.abs(f) > hz].sum(axis=1) for hz in SHARE_HZ],
        np.abs(p[:, f > 0].sum(axis=1) - p[:, f < 0].sum(axis=1)),
        np.median(spread, axis=0),
        np.subtract(*np.percentile(spread, [75, 25], axis=0)),
        np.log10(np.percentile(raw, 90, axis=0) / np.percentile(raw, 10, axis=0)),
        (e > noise * N_DOPPLER).mean(axis=0),
    ])  # (features, 3)
    return np.median(feats, axis=1), p.mean(axis=0)


def compute_windows(split="labeled"):
    rows, feats, profiles = [], [], []
    for row in read_index(split):
        csi, ati = load_raw(split, row["identifier"])
        h = sanitize(csi)
        for s in range(0, len(ati) - WINDOW + 1, WINDOW):
            x, p = window_features(h[s:s + WINDOW], ati[s:s + WINDOW])
            feats.append(x)
            profiles.append(p)
            rows.append((row["identifier"], row["room"], row["label"], s))
        print(f"  {row['identifier']}", flush=True)
    ident, room, label, start = zip(*rows)
    return dict(X=np.array(feats), profiles=np.array(profiles), names=np.array(NAMES), ident=np.array(ident),
                room=np.array(room), label=np.array(label), start=np.array(start))


def transfer_lines(d, base):
    """a1 <-> c1 logistic-regression transfer on SHARP, existing and combined features."""
    keys = {(i, s): k for k, (i, s) in enumerate(zip(base["ident"], base["start"]))}
    order = np.array([keys[(i, s)] for i, s in zip(d["ident"], d["start"])])
    sets = {"SHARP": d["X"], "existing": base["X"][order], "both": np.hstack([d["X"], base["X"][order]])}
    ok = np.isfinite(sets["both"]).all(axis=1)
    key = ok & np.isin(d["room"], KEY_ROOMS) & (d["label"] >= 1)
    lines = ["Logistic regression on 1-4 people, fitted in one of a1 / c1 and scored in the other "
             "(balanced accuracy, chance 0.25)"]
    for name, X in sets.items():
        for standardise in (False, True):
            scores = multivariate_transfer(X[key], d["label"][key], d["room"][key], standardise)
            tag = "per-room standardised" if standardise else "raw"
            lines.append(f"  {name:<9}{X.shape[1]:>4} features  {tag:<22}"
                         + "  ".join(f"{k} {v:.3f}" for k, v in scores.items()))
    return lines


def plot_profiles(d, path):
    f = (np.arange(N_DOPPLER) - N_DOPPLER // 2) * (1e6 / 5087) / N_DOPPLER  # median frame rate for the axis
    panels = [("a1", "a1"), ("c1", "c1"), ("g1", "g1"), ("others", "b1, e1, unknown")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), facecolor=SURFACE, sharey=True)
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        rooms = KEY_ROOMS + ("g1",)
        m_room = ~np.isin(d["room"], rooms) if key == "others" else d["room"] == key
        for c in range(5):
            m = m_room & (d["label"] == c)
            if m.any():
                ax.plot(f, d["profiles"][m].mean(axis=0), color=HEADCOUNT_RAMP[c], linewidth=2,
                        label=f"{c} people ({m.sum()} windows)")
        ax.set_yscale("log")
        ax.set_xlim(-NOISE_HZ, NOISE_HZ)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Doppler frequency (Hz)", fontsize=8, color=INK_2)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("share of motion power per bin", fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="ignore the cached window features")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "sharp_windows.npz"
    if cache.exists() and not args.recompute:
        d = {k: v for k, v in np.load(cache).items()}
    else:
        print("computing SHARP Doppler features (CPU)...")
        d = compute_windows()
        np.savez(cache, **d)
    bad = ~np.isfinite(d["X"]).all(axis=1)
    if bad.any():
        print(f"dropping {bad.sum()} windows with non-finite features")
        d = {k: (v[~bad] if k != "names" else v) for k, v in d.items()}
    t = trace_medians(d)
    rows = score_features(d, t)

    cols = ["rho_a1", "rho_c1", "eta_class", "eta_room", "auc01_all", "auc01_g1", "xfer_raw", "xfer_z"]
    lines = [f"{len(d['X'])} windows, {len(t['ident'])} traces, {len(NAMES)} SHARP features "
             "(metrics as in analysis.headcount_separability; xfer chance 0.25)", "",
             f"{'feature':<16}" + "".join(f"{c:>10}" for c in cols)]
    lines += [f"{r['feature']:<16}" + "".join(f"{r[c]:>10.3f}" for c in cols) for r in rows]
    agree = [r for r in rows if r["rho_a1"] * r["rho_c1"] > 0 and min(abs(r["rho_a1"]), abs(r["rho_c1"])) >= 0.5]
    lines += ["", f"|rho| >= 0.5 with the same sign in a1 and c1: {len(agree)} of {len(rows)}"]
    lines += [f"  {r['feature']:<16} rho_a1 {r['rho_a1']:+.2f}  rho_c1 {r['rho_c1']:+.2f}  "
              f"eta class/room {r['eta_class']:.2f}/{r['eta_room']:.2f}" for r in agree]
    base_path = OUT / f"windows_stride{WINDOW}.npz"
    if base_path.exists():
        lines += [""] + transfer_lines(d, dict(np.load(base_path)))
    else:
        lines += ["", f"{base_path} missing: run python -m analysis.headcount_separability for the comparison"]
    text = "\n".join(lines)
    print(text)
    (OUT / "sharp_doppler.txt").write_text(text + "\n")
    save_csv(rows, OUT / "sharp_separability.csv")
    plot_distributions(d, t, rows, OUT / "sharp_distributions.png", top=len(rows), controls=())
    plot_profiles(d, OUT / "sharp_profiles.png")
    print(f"\nwrote {OUT / 'sharp_doppler.txt'}, sharp_separability.csv, sharp_distributions.png, sharp_profiles.png")


if __name__ == "__main__":
    main()
