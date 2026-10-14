"""Hand-crafted window features, written in PyTorch so they export to ONNX.

Every feature is computed from one 600-frame window and is designed to be
insensitive to the absolute signal level, because absolute amplitude differs
between sessions of the same room. Feature names are prefixed by group so
groups can be ablated with `train.py --exclude`:

  dynpow_  power of the time-varying part relative to the static channel
  ampcv_   per-channel temporal coefficient of variation of |CSI|
  sf_      complex structure function at several lags (how fast the channel decorrelates)
  sfamp_   the same on relative amplitude only
  pr_      participation ratios of the dynamic covariance (effective number of
           independent motion components)
  coh_     coherence of the dynamic part across antennas / subcarrier offsets
  nonstat  how much dynamic power changes across the window
  ati_     acquisition-interval statistics
"""

import torch
from torch import nn

from common.preprocess import (
    EPS, abs2, cmul, conj, interval_ratio, normalise_frames, pool_groups,
    relative_amplitude, split_static_dynamic,
)

LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FREQ_OFFSETS = (1, 4, 16)
ANTENNA_PAIRS = ((0, 1), (0, 2), (1, 2))
NUM_CHUNKS = 4
GAP_US = 7000.0
SHORT_US = 4500.0


def _log(x):
    return torch.log10(x + EPS)


def _std(x, dim):
    mean = x.mean(dim=dim, keepdim=True)
    return torch.sqrt(((x - mean) ** 2).mean(dim=dim) + EPS)


def _covariance(x):
    """x (B, T, C, 2) -> Hermitian covariance (B, C, C, 2), C_ij = mean_t conj(x_i) x_j."""
    xr, xi = x[..., 0], x[..., 1]
    t = x.shape[1]
    cr = (xr.transpose(1, 2) @ xr + xi.transpose(1, 2) @ xi) / t
    ci = (xr.transpose(1, 2) @ xi - xi.transpose(1, 2) @ xr) / t
    return torch.stack((cr, ci), dim=-1)


def _participation(cov):
    """Effective rank estimates of a Hermitian (B, C, C, 2) matrix from its traces.

    pr2 = (sum l)^2 / sum l^2 and pr3 = sqrt((sum l)^3 / sum l^3), where l are
    eigenvalues; both range from 1 (one dominant component) to C (white).
    The matrix is first scaled to unit trace: dynamic covariances are ~1e-8,
    so working on raw values would let EPS and float32 rounding dominate.
    """
    tr1 = torch.diagonal(cov[..., 0], dim1=-2, dim2=-1).sum(-1)
    cov = cov / tr1.clamp(min=1e-30)[:, None, None, None]
    cr, ci = cov[..., 0], cov[..., 1]
    tr2 = (cr ** 2 + ci ** 2).sum(dim=(-1, -2))
    c2r = cr @ cr - ci @ ci
    c2i = cr @ ci + ci @ cr
    tr3 = (c2r * cr + c2i * ci).sum(dim=(-1, -2))  # Re tr(C^2 C^H) = tr(C^3) for Hermitian C
    pr2 = 1.0 / tr2.clamp(min=1e-6)
    pr3 = torch.sqrt(1.0 / tr3.clamp(min=1e-6))
    return pr2, pr3


def _coherence(a, b):
    """Mean over the last channel axis of |sum_t a conj(b)| / sqrt(sum|a|^2 sum|b|^2).

    a, b: (B, T, K, 2) -> (B,)
    """
    cross = cmul(a, conj(b)).sum(dim=1)  # (B, K, 2)
    num = torch.sqrt(abs2(cross) + EPS)
    den = torch.sqrt(abs2(a).sum(dim=1) * abs2(b).sum(dim=1) + EPS)
    return (num / den).mean(dim=-1)


class FeatureExtractor(nn.Module):
    """(B, T, 3, 126, 2) pooled CSI + (B, T) intervals -> (B, F) features."""

    def __init__(self, cov_pool=9):
        super().__init__()
        self.cov_pool = cov_pool
        self.names = (
            ["dynpow_all"] + [f"dynpow_rx{a}" for a in range(3)]
            + ["ampcv_mean", "ampcv_std", "ampcv_max"]
            + [f"sf_lag{l}" for l in LAGS]
            + [f"sfamp_lag{l}" for l in LAGS]
            + ["pr_all2", "pr_all3", "pr_freq", "pr_ant"]
            + ["coh_ant"] + [f"coh_freq{d}" for d in FREQ_OFFSETS]
            + ["nonstat"]
            + ["ati_mean_ms", "ati_cv", "ati_gap_frac", "ati_short_frac", "ati_logmax"]
        )

    @property
    def num_features(self):
        return len(self.names)

    def forward(self, h, ati):
        h = normalise_frames(h)
        _, dyn, _ = split_static_dynamic(h)  # (B, T, A, K, 2)
        ramp = relative_amplitude(h)  # (B, T, A, K)
        feats = []

        # dynamic power relative to the static channel
        p_ant = abs2(dyn).mean(dim=(1, 3))  # (B, A)
        p_all = p_ant.mean(dim=1)
        feats += [_log(p_all)] + [_log(p_ant[:, a]) for a in range(3)]

        # per-channel temporal coefficient of variation of |CSI|
        log_cv = _log(torch.sqrt((ramp ** 2).mean(dim=1) + EPS)).flatten(1)  # (B, A*K)
        feats += [log_cv.mean(dim=1), _std(log_cv, dim=1), log_cv.amax(dim=1)]

        # structure functions: 0 = identical after `lag` frames, ~1 = decorrelated
        p_amp = (ramp ** 2).mean(dim=(1, 2, 3))
        for lag in LAGS:
            d = abs2(dyn[:, lag:] - dyn[:, :-lag]).mean(dim=(1, 2, 3))
            feats.append(d / (2 * p_all + EPS))
        for lag in LAGS:
            d = ((ramp[:, lag:] - ramp[:, :-lag]) ** 2).mean(dim=(1, 2, 3))
            feats.append(d / (2 * p_amp + EPS))

        # effective number of independent components of the dynamic part
        coarse = pool_groups(dyn, self.cov_pool)  # (B, T, A, K', 2)
        pr2, pr3 = _participation(_covariance(coarse.flatten(2, 3)))
        pr_freq = torch.stack(
            [_participation(_covariance(coarse[:, :, a]))[0] for a in range(3)], dim=1
        ).mean(dim=1)
        pr_ant, _ = _participation(_covariance(coarse.mean(dim=3)))
        feats += [_log(pr2), _log(pr3), _log(pr_freq), _log(pr_ant)]

        # coherence of the dynamic part across antennas and subcarrier offsets
        feats.append(torch.stack(
            [_coherence(dyn[:, :, a], dyn[:, :, b]) for a, b in ANTENNA_PAIRS], dim=1
        ).mean(dim=1))
        for off in FREQ_OFFSETS:
            a = dyn[:, :, :, :-off].flatten(2, 3)
            b = dyn[:, :, :, off:].flatten(2, 3)
            feats.append(_coherence(a, b))

        # non-stationarity of dynamic power within the window
        t = dyn.shape[1] // NUM_CHUNKS * NUM_CHUNKS
        chunks = abs2(dyn[:, :t]).reshape(dyn.shape[0], NUM_CHUNKS, -1).mean(dim=2)
        feats.append(_std(_log(chunks), dim=1))

        # acquisition intervals (last value is padding)
        raw = ati[:, :-1]
        mean = raw.mean(dim=1)
        feats += [
            mean / 1000.0,
            _std(raw, dim=1) / (mean + EPS),
            (raw > GAP_US).float().mean(dim=1),
            (raw < SHORT_US).float().mean(dim=1),
            interval_ratio(ati).amax(dim=1),
        ]
        return torch.stack(feats, dim=1)
