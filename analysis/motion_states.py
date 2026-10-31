"""Do motion states found without labels (K-means, then an HMM) count people? (CPU only)

Steps 1 and 2 of a K-means -> HMM -> LSTM pipeline. Both steps use every room,
labeled and unlabeled, so the states are not shaped by a1 and c1 alone; only
the check against headcount uses labels.

Each 600-frame window is cut into 60 blocks of 10 frames, as in the P13 block
maps. Block descriptors come from the static-removed CSI of a 30-frame segment
centred on the block (median over the 3 antennas where per antenna):
  snr_db     power below 40 Hz relative to the window's noise floor (bins above 75 Hz)
  rel_db     block power relative to the window's median block, in dB
  spread_hz  RMS Doppler spread
  hi_share   share of the power above 20 Hz
  asym       |positive - negative| share of the power
The Doppler shape (spread_hz, hi_share, asym) is taken from the block spectrum
with its noise: a block without motion has a flat spectrum rather than an
undefined shape, which keeps K-means and the Gaussian HMM away from point
masses; snr_db tells the two apart.
  coherence  mean |correlation| of the dynamic CSI between antenna pairs
  rank       participation ratio of the segment's time x subcarrier matrix
             (effective number of independent components)

Step 1: K-means (K=6) on the standardised blocks. A state used by one room
  only describes that room rather than a kind of motion, so the script prints
  each room's state usage and how much of the state sequence room and
  headcount explain (normalised mutual information).
Step 2: a diagonal Gaussian HMM whose K states start from the K-means centres,
  fitted on all 60-block windows. Per window: posterior state shares, switches
  per second and mean dwell time of the Viterbi path.

The window features of both steps are scored as in
analysis.headcount_separability (Spearman in a1 / c1, transfer between them),
plus 0 vs 1 person with a1, e1 and g1 each held out in turn, against the
existing window features.

    python -m analysis.motion_states [--recompute]

About 15 minutes on CPU the first time (block descriptors of every trace are
cached in runs/analysis/motion_blocks.npz). Writes runs/analysis/motion_states.txt.
"""

import argparse

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score

from analysis.headcount_separability import (KEY_ROOMS, OUT, balanced_accuracy, multivariate_transfer,
                                             score_features, trace_medians)
from common.data import WINDOW, load_raw, read_index
from common.preprocess import NUM_GROUPS, SC_GROUP, SC_START, SC_STOP

BLOCK, SEG, N_FFT = 10, 30, 64
NOISE_HZ, SIGNAL_HZ, HIGH_HZ = 75, 40, 20
BLOCK_NAMES = ("snr_db", "rel_db", "spread_hz", "hi_share", "asym", "coherence", "rank")
K = 6
HANN = np.hanning(SEG).astype(np.float32)
BLOCK_S = BLOCK * 5087e-6  # median frame interval
HELD_OUT_01 = ("a1", "e1", "g1")  # the rooms with both 0 and 1 person


def block_descriptors(x, ati):
    """Pooled CSI of one window (600, 3, 126) -> (60, len(BLOCK_NAMES))."""
    fs = 1e6 / np.median(ati[:-1])
    x = x - x.mean(axis=0)
    nb = len(x) // BLOCK
    starts = np.clip(np.arange(nb) * BLOCK + BLOCK // 2 - SEG // 2, 0, len(x) - SEG)
    seg = x[starts[:, None] + np.arange(SEG)]  # (60, 30, 3, 126)
    spec = np.abs(np.fft.fft(seg * HANN[:, None, None], n=N_FFT, axis=1)) ** 2
    spec = np.fft.fftshift(spec.sum(axis=-1), axes=1)  # (60, 64, 3)
    f = (np.arange(N_FFT) - N_FFT // 2) * fs / N_FFT
    noise = spec[:, np.abs(f) > NOISE_HZ].mean(axis=(0, 1))  # (3,) per bin
    e = spec.sum(axis=1)  # (60, 3)
    snr = 10 * np.log10(spec[:, np.abs(f) < SIGNAL_HZ].mean(axis=1) / noise)
    rel = 10 * np.log10(e) - np.median(10 * np.log10(e), axis=0)
    spread = np.sqrt((spec * f[None, :, None] ** 2).sum(axis=1) / e)
    hi = spec[:, np.abs(f) > HIGH_HZ].sum(axis=1) / e
    asym = np.abs(spec[:, f > 0].sum(axis=1) - spec[:, f < 0].sum(axis=1)) / e
    v = seg.transpose(0, 2, 1, 3).reshape(nb, 3, -1)  # (60, 3, 30 * 126)
    gram = np.einsum("bik,bjk->bij", v, v.conj())
    norm = np.sqrt(np.abs(np.einsum("bii->bi", gram)))
    cc = np.abs(gram) / (norm[:, :, None] * norm[:, None, :] + 1e-20)
    coherence = (cc[:, 0, 1] + cc[:, 0, 2] + cc[:, 1, 2]) / 3
    sv = np.linalg.svd(seg.transpose(0, 2, 1, 3), compute_uv=False) ** 2  # (60, 3, 30)
    rank = sv.sum(axis=-1) ** 2 / (sv ** 2).sum(axis=-1)
    per_antenna = np.median(np.stack([snr, rel, spread, hi, asym, rank], axis=-1), axis=1)  # (60, 6)
    return np.column_stack([per_antenna[:, :5], coherence, per_antenna[:, 5]]).astype(np.float32)


def compute_blocks():
    """Block descriptors of every non-overlapping window of both splits."""
    blocks, meta = [], []
    for split in ("labeled", "unlabeled"):
        for row in read_index(split):
            csi, ati = load_raw(split, row["identifier"])
            h = csi[:, 0, :, SC_START:SC_STOP].reshape(len(csi), 3, NUM_GROUPS, SC_GROUP).mean(axis=-1)
            for s in range(0, len(ati) - WINDOW + 1, WINDOW):
                blocks.append(block_descriptors(h[s:s + WINDOW], ati[s:s + WINDOW]))
                meta.append((row["identifier"], split, row["room"], row["label"], s))
            print(f"  {row['identifier']}", flush=True)
    ident, split, room, label, start = zip(*meta)
    return dict(blocks=np.array(blocks), ident=np.array(ident), split=np.array(split), room=np.array(room),
                label=np.array(label), start=np.array(start))


def path_features(path):
    """Hard state paths (windows, 60) -> switches per second, mean dwell time (s)."""
    switches = (np.diff(path, axis=1) != 0).sum(axis=1)
    duration = path.shape[1] * BLOCK_S
    return switches / duration, duration / (switches + 1)


def window_sets(b, km, hmm, Z):
    """Window features of the K-means and HMM steps -> {set name: (names, X)}."""
    n, t, _ = b["blocks"].shape
    hard = km.predict(Z.reshape(n * t, -1)).reshape(n, t)
    km_share = np.stack([(hard == k).mean(axis=1) for k in range(K)], axis=1)
    km_rate, km_dwell = path_features(hard)
    lengths = [t] * n
    post = hmm.predict_proba(Z.reshape(n * t, -1), lengths).reshape(n, t, K)
    viterbi = hmm.predict(Z.reshape(n * t, -1), lengths).reshape(n, t)
    share = post.mean(axis=1)
    rate, dwell = path_features(viterbi)
    entropy = -(share * np.log(share + 1e-12)).sum(axis=1) / np.log(K)
    return {
        "kmeans": ([f"km_share{k}" for k in range(K)] + ["km_switch_per_s", "km_dwell_s"],
                   np.column_stack([km_share, km_rate, km_dwell])),
        "hmm": ([f"hmm_share{k}" for k in range(K)] + ["hmm_switch_per_s", "hmm_dwell_s", "hmm_entropy"],
                np.column_stack([share, rate, dwell, entropy])),
    }


def held_out_01(X, label, room):
    """0 vs 1 person: logistic regression trained without one room, scored in it (balanced accuracy)."""
    w = (label <= 1) & np.isfinite(X).all(axis=1)
    out = {}
    for r in HELD_OUT_01:
        tr, te = w & (room != r), w & (room == r)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        clf = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit((X[tr] - mu) / sd, label[tr])
        out[r] = balanced_accuracy(label[te], clf.predict((X[te] - mu) / sd))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="ignore the cached block descriptors")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "motion_blocks.npz"
    if cache.exists() and not args.recompute:
        b = dict(np.load(cache))
    else:
        print("computing block descriptors (CPU)...")
        b = compute_blocks()
        np.savez(cache, **b)
    n, t, _ = b["blocks"].shape
    flat = b["blocks"].reshape(n * t, -1).astype(np.float64)
    mu, sd = flat.mean(0), flat.std(0)
    Z = ((flat - mu) / sd).reshape(n, t, -1)
    room_key = np.where(b["split"] == "labeled", b["room"], np.char.add(b["room"], "*"))  # * = unlabeled

    # Step 1: K-means on all blocks
    km = KMeans(K, n_init=4, random_state=0).fit(Z.reshape(n * t, -1))
    order = np.argsort(km.cluster_centers_[:, 0])  # states by snr_db, quietest first
    km.cluster_centers_ = km.cluster_centers_[order]
    hard = km.predict(Z.reshape(n * t, -1))
    centres = km.cluster_centers_ * sd + mu
    lines = [f"{n} windows x {t} blocks from {len(set(b['ident']))} recordings; K = {K}", "",
             "Step 1, K-means state centres (original units)",
             f"  {'state':<7}" + "".join(f"{c:>11}" for c in BLOCK_NAMES)]
    lines += [f"  {k:<7}" + "".join(f"{v:>11.2f}" for v in c) for k, c in enumerate(centres)]
    rooms = list(dict.fromkeys(room_key))
    block_room = np.repeat(room_key, t)
    lines += ["", "State usage per room (% of the room's blocks; * = unlabeled)",
              f"  {'room':<10}" + "".join(f"{k:>7}" for k in range(K))]
    for r in rooms:
        m = block_room == r
        lines.append(f"  {r:<10}" + "".join(f"{100 * np.mean(hard[m] == k):>7.1f}" for k in range(K)))
    block_label = np.repeat(b["label"], t)
    lab = block_label >= 0
    lines += ["", "Normalised mutual information with the block states:",
              f"  room (all rooms)            {normalized_mutual_info_score(block_room, hard):.3f}",
              f"  headcount (labeled rooms)   {normalized_mutual_info_score(block_label[lab], hard[lab]):.3f}"]
    for r in KEY_ROOMS:
        m = block_room == r
        lines.append(f"  headcount inside {r}        {normalized_mutual_info_score(block_label[m], hard[m]):.3f}")

    # Step 2: HMM initialised from the K-means states
    hmm = GaussianHMM(K, covariance_type="diag", n_iter=50, tol=1e-3, random_state=0, init_params="", params="stmc")
    hmm.startprob_ = np.full(K, 1 / K)
    hmm.transmat_ = np.full((K, K), 0.1 / (K - 1)) + np.eye(K) * (0.9 - 0.1 / (K - 1))
    hmm.means_ = km.cluster_centers_
    hmm.covars_ = np.stack([Z.reshape(n * t, -1)[hard == k].var(0) + 1e-3 for k in range(K)])
    hmm.fit(Z.reshape(n * t, -1), [t] * n)
    stay = np.diag(hmm.transmat_)
    lines += ["", f"Step 2, HMM ({hmm.monitor_.iter} EM iterations, converged {hmm.monitor_.converged})",
              f"  {'state':<7}" + "".join(f"{c:>11}" for c in BLOCK_NAMES) + f"{'stay prob':>11}{'dwell s':>9}"]
    lines += [f"  {k:<7}" + "".join(f"{v:>11.2f}" for v in hmm.means_[k] * sd + mu)
              + f"{stay[k]:>11.3f}{BLOCK_S / (1 - stay[k]):>9.2f}" for k in range(K)]

    # window features vs headcount, labeled windows only
    sets = window_sets(b, km, hmm, Z)
    labeled = b["split"] == "labeled"
    base_path = OUT / f"windows_stride{WINDOW}.npz"
    base = dict(np.load(base_path))
    keys = {(i, s): k for k, (i, s) in enumerate(zip(base["ident"], base["start"]))}
    order = np.array([keys[(i, s)] for i, s in zip(b["ident"][labeled], b["start"][labeled])])
    names = sets["kmeans"][0] + sets["hmm"][0]
    d = dict(X=np.hstack([sets["kmeans"][1], sets["hmm"][1]])[labeled], names=np.array(names),
             ident=b["ident"][labeled], room=b["room"][labeled], label=b["label"][labeled],
             start=b["start"][labeled])
    rows = score_features(d, trace_medians(d))
    cols = ["rho_a1", "rho_c1", "eta_class", "eta_room", "auc01_all", "auc01_g1", "xfer_raw", "xfer_z"]
    lines += ["", "Window features vs headcount (labeled windows; metrics as in analysis.headcount_separability)",
              f"  {'feature':<17}" + "".join(f"{c:>10}" for c in cols)]
    lines += [f"  {r['feature']:<17}" + "".join(f"{r[c]:>10.3f}" for c in cols) for r in rows]
    agree = [r["feature"] for r in rows
             if r["rho_a1"] * r["rho_c1"] > 0 and min(abs(r["rho_a1"]), abs(r["rho_c1"])) >= 0.5]
    lines.append(f"  |rho| >= 0.5 with the same sign in a1 and c1: {len(agree)} of {len(rows)} {agree or ''}")

    existing = base["X"][order]
    candidates = {"K-means": sets["kmeans"][1][labeled], "HMM": sets["hmm"][1][labeled], "existing": existing,
                  "existing + HMM": np.hstack([existing, sets["hmm"][1][labeled]])}
    y, room = d["label"], d["room"]
    key = np.isin(room, KEY_ROOMS) & (y >= 1)
    lines += ["", "Logistic regression, 1-4 people, fitted in one of a1 / c1 and scored in the other (chance 0.25)"]
    for name, X in candidates.items():
        ok = key & np.isfinite(X).all(axis=1)
        s = multivariate_transfer(X[ok], y[ok], room[ok], standardise=False)
        lines.append(f"  {name:<16}" + "  ".join(f"{k} {v:.3f}" for k, v in s.items()))
    lines += ["", "0 vs 1 person, logistic regression with one room held out (balanced accuracy, chance 0.50)"]
    for name, X in candidates.items():
        s = held_out_01(X, y, room)
        lines.append(f"  {name:<16}" + "  ".join(f"{k} {v:.3f}" for k, v in s.items())
                     + f"   mean {np.mean(list(s.values())):.3f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "motion_states.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
