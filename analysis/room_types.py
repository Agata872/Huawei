"""Do the rooms fall into a few types, and do the unlabeled rooms resemble a labeled one? (CPU only)

Headcount models do not transfer between a1 and c1, but a test room might share
its type with a labeled room (the task says rooms of one layout have identical
router deployments). This compares rooms through descriptors of the static
environment, computed per recording from every 10th frame:

  level_db        mean received amplitude (dB)
  rx1_db, rx2_db  Rx 1 and Rx 2 level relative to Rx 0
  select_db       spread of the static amplitude across subcarriers
  coh_bw_mhz      frequency lag at which the static amplitude decorrelates to 0.5
  delay_rms_ns    RMS delay spread of the power delay profile (peak-aligned)
  k_db            power of the strongest delay tap relative to the rest
  ati_ms, ati_slow  median frame interval, share of intervals > 20 ms (channel load); listed
                  but not used for distances: g1's frame clock runs 0.3% slow, which alone
                  would put it ~10 within-room spreads from every other room

Distances between rooms are in units of the typical within-room spread
(the pooled spread of one room's recordings, which differ in headcount), so a
distance near 1 means two rooms are as alike as two recordings of one room.
The static fingerprint of common.calibrate gives a second, finer comparison.

    python -m analysis.room_types

Writes runs/analysis/room_types.txt and room_types.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.headcount_separability import GRID, INK, INK_2, OUT, ROOM_STYLE, SURFACE, style
from common.calibrate import fingerprint
from common.data import load_raw, read_index
from common.preprocess import pool_subcarriers

LABELED_ROOMS = ("a1", "c1", "b1", "e1", "g1")
UNLABELED_ROOMS = ("d1", "h1", "i1", "j1", "u1", "v1")
NAMES = ("level_db", "rx1_db", "rx2_db", "select_db", "coh_bw_mhz", "delay_rms_ns", "k_db", "ati_ms", "ati_slow")
PROPAGATION = 7  # the first 7 descriptors are used for distances
SUBCARRIER_MHZ = 80 / 1024
TAP_NS = 1e3 / 80  # delay resolution of a 1024-point IFFT over 80 MHz
UNLABELED_GRAY = "#8a8985"


def descriptors(csi, ati, step=10):
    """One recording -> (descriptor vector, static fingerprint)."""
    h = csi[::step, 0, :, 6:1019].astype(np.complex64)  # (T, 3, 1013)
    static = np.abs(h.mean(axis=0))  # (3, 1013)
    level = 20 * np.log10(static.mean(axis=1))
    select = (20 * np.log10(static + 1e-9)).std(axis=1).mean()
    # frequency correlation of the static amplitude
    s = static - static.mean(axis=1, keepdims=True)
    corr = np.array([np.mean([np.corrcoef(s[r, :-k], s[r, k:])[0, 1] for r in range(3)]) for k in range(1, 200)])
    coh_bw = (np.argmax(corr < 0.5) + 1 if (corr < 0.5).any() else 200) * SUBCARRIER_MHZ
    # power delay profile, averaged over frames and antennas, aligned on its peak
    taps = np.fft.ifft(h * np.hanning(h.shape[-1]), n=1024, axis=-1)
    pdp = np.roll((np.abs(taps) ** 2).mean(axis=(0, 1)), -int(np.argmax((np.abs(taps) ** 2).mean(axis=(0, 1)))))
    noise = np.median(pdp[300:700])
    p = np.clip(pdp[np.r_[-4:41]] - noise, 0, None)
    tau = np.r_[-4:41] * TAP_NS
    mean_tau = (p * tau).sum() / p.sum()
    delay_rms = np.sqrt((p * (tau - mean_tau) ** 2).sum() / p.sum())
    k = 10 * np.log10(p[4] / (p.sum() - p[4]))
    dt = ati[:-1] / 1e3
    vec = [level.mean(), level[1] - level[0], level[2] - level[0], select, coh_bw, delay_rms, k,
           np.median(dt), np.mean(dt > 20)]
    frames = np.ascontiguousarray(csi[::step]).view(np.float32).reshape(-1, 1, 3, 1024, 2)
    fp = fingerprint(pool_subcarriers(torch.from_numpy(frames)).unsqueeze(0))[0].numpy()
    return np.array(vec), fp


def collect():
    rows, vecs, fps = [], [], []
    for split in ("labeled", "unlabeled"):
        for r in read_index(split):
            vec, fp = descriptors(*load_raw(split, r["identifier"]))
            rows.append(r)
            vecs.append(vec)
            fps.append(fp)
            print(f"  {r['identifier']}", flush=True)
    return rows, np.array(vecs), np.array(fps)


def room_distances(X, room, rooms):
    """Median descriptors per room, and distances in units of the pooled within-room spread."""
    within = [X[room == r] - np.median(X[room == r], axis=0) for r in rooms if (room == r).sum() >= 3]
    spread = np.sqrt(np.mean(np.concatenate(within) ** 2, axis=0))
    centre = np.array([np.median(X[room == r], axis=0) for r in rooms])
    z = centre / spread
    dist = np.sqrt(((z[:, None] - z[None]) ** 2).mean(-1))
    # a recording vs its own room's median, as the same-room reference
    own = np.median([np.sqrt((((x - centre[rooms.index(r)]) / spread) ** 2).mean())
                     for x, r in zip(X, room) if r in rooms and (room == r).sum() >= 3])
    return centre, spread, dist, own


def main():
    rows, X, F = collect()
    room = np.array([r["room"] for r in rows])
    split = np.array([r["split"] for r in rows])
    rooms = list(LABELED_ROOMS + UNLABELED_ROOMS)
    centre = np.array([np.median(X[room == r], axis=0) for r in rooms])
    _, spread, dist, own = room_distances(X[:, :PROPAGATION], room, rooms)

    # fingerprint dissimilarity between rooms: 1e3 * (1 - median cosine over recording pairs)
    S = F @ F.T
    fpd = np.array([[1e3 * (1 - np.median(S[np.ix_(room == a, room == b)][~np.eye((room == a).sum(), dtype=bool)]
                                          if a == b else S[np.ix_(room == a, room == b)]))
                     if (a != b or (room == a).sum() > 1) else np.nan for b in rooms] for a in rooms])

    lines = ["Room medians (recordings: labeled + unlabeled)",
             f"{'room':<7}{'n':>4}" + "".join(f"{n:>14}" for n in NAMES)]
    for r, c in zip(rooms, centre):
        lines.append(f"{r:<7}{(room == r).sum():>4}" + "".join(f"{v:>14.2f}" for v in c))
    lines.append(f"{'within-room spread':<11}" + "".join(f"{v:>14.2f}" for v in spread))  # propagation only
    lines += ["", f"Descriptor distance between rooms, in within-room spreads (one recording vs its own "
                  f"room: {own:.2f} median)", f"{'':<6}" + "".join(f"{r:>7}" for r in rooms)]
    lines += [f"{a:<6}" + "".join(f"{dist[i, j]:>7.1f}" for j in range(len(rooms))) for i, a in enumerate(rooms)]
    lines += ["", "Fingerprint dissimilarity, 1e3 x (1 - median cosine); diagonal = recordings of the same room",
              f"{'':<6}" + "".join(f"{r:>7}" for r in rooms)]
    lines += [f"{a:<6}" + "".join(f"{fpd[i, j]:>7.2f}" for j in range(len(rooms))) for i, a in enumerate(rooms)]
    lines += ["", "Nearest labeled room of each unlabeled room"]
    for i, r in enumerate(UNLABELED_ROOMS, len(LABELED_ROOMS)):
        j = int(np.argmin(dist[i, :len(LABELED_ROOMS)]))
        jf = int(np.nanargmin(fpd[i, :len(LABELED_ROOMS)]))
        lines.append(f"  {r}: descriptors -> {LABELED_ROOMS[j]} ({dist[i, j]:.1f} spreads), "
                     f"fingerprint -> {LABELED_ROOMS[jf]} ({fpd[i, jf]:.2f} vs own {fpd[i, i]:.2f})")
    lines.append(f"  for reference a1 <-> c1: {dist[0, 1]:.1f} spreads, fingerprint {fpd[0, 1]:.2f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "room_types.txt").write_text(text + "\n")
    plot(X[:, :PROPAGATION], room, split, spread, rooms, dist, OUT / "room_types.png")
    print(f"\nwrote {OUT / 'room_types.txt'} and room_types.png")


def plot(X, room, split, spread, rooms, dist, path):
    Z = (X - X.mean(0)) / spread
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    P = (Z - Z.mean(0)) @ vt[:2].T
    fig, (ax, hx) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=(1.15, 1)))
    style(ax)
    unl = split == "unlabeled"
    ax.scatter(P[unl, 0], P[unl, 1], s=16, color=UNLABELED_GRAY, alpha=0.55, linewidth=0, zorder=2)
    for r in LABELED_ROOMS + ("unknown",):
        m = (room == r) & ~unl
        color, marker = ROOM_STYLE[r]
        ax.scatter(P[m, 0], P[m, 1], s=40, color=color, marker=marker, edgecolor=SURFACE, linewidth=0.8, zorder=3,
                   label=f"{r} (labeled)")
    for r in rooms:
        c = np.median(P[room == r], axis=0)
        ax.annotate(r, c, xytext=(6, 6), textcoords="offset points", fontsize=9, color=INK, zorder=4)
    ax.scatter([], [], s=16, color=UNLABELED_GRAY, label="unlabeled recording")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_xlabel("principal component 1", fontsize=9, color=INK_2)
    ax.set_ylabel("principal component 2", fontsize=9, color=INK_2)
    ax.set_title("One dot per recording, static-environment descriptors", fontsize=10, color=INK, loc="left")

    hx.set_facecolor(SURFACE)
    im = hx.imshow(dist, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f2f6fc", "#86b6ef", "#2a78d6", "#104281"]), vmin=0)
    hx.set_xticks(range(len(rooms)), rooms, fontsize=9, color=INK_2)
    hx.set_yticks(range(len(rooms)), rooms, fontsize=9, color=INK_2)
    for i in range(len(rooms)):
        for j in range(len(rooms)):
            v = dist[i, j]
            hx.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if v > 0.6 * dist.max() else INK)
    for s in hx.spines.values():
        s.set_visible(False)
    hx.axhline(len(LABELED_ROOMS) - 0.5, color=GRID, linewidth=2)
    hx.axvline(len(LABELED_ROOMS) - 0.5, color=GRID, linewidth=2)
    hx.set_title("Room-to-room distance, in within-room spreads", fontsize=10, color=INK, loc="left")
    fig.colorbar(im, ax=hx, fraction=0.046, pad=0.03).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
