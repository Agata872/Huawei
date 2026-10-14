"""ONNX-exportable CSI preprocessing shared by training and the submitted models.

Complex numbers are kept as a trailing dimension of size 2 (real, imag),
because ONNX has no complex dtype. Every function works on any number of
leading batch dimensions, so the same code runs on GPU batches during
training and inside the exported single-sample ONNX graph.
"""

import torch

# Subcarriers 0-5 and 1019-1023 are always zero. 6..1013 gives 1008 = 126 x 8.
SC_START, SC_STOP, SC_GROUP = 6, 1014, 8
NUM_GROUPS = (SC_STOP - SC_START) // SC_GROUP  # 126
NOMINAL_INTERVAL_US = 5000.0
EPS = 1e-12


def pool_subcarriers(frames):
    """Raw frames (..., T, 1, 3, 1024, 2) -> (..., T, 3, 126, 2).

    Drops the null edge subcarriers and averages complex CSI over groups of 8
    adjacent subcarriers. The phase step between neighbours is <= 0.03 rad, so
    averaging keeps >99.9% of the magnitude.
    """
    x = frames[..., 0, :, SC_START:SC_STOP, :]
    x = x.reshape(*x.shape[:-2], NUM_GROUPS, SC_GROUP, 2)
    return x.mean(dim=-2)


def pool_groups(h, factor):
    """Average adjacent subcarrier groups: (..., K, 2) -> (..., K // factor, 2)."""
    k = h.shape[-2] // factor * factor
    h = h[..., :k, :]
    return h.reshape(*h.shape[:-2], k // factor, factor, 2).mean(dim=-2)


def cmul(a, b):
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack((ar * br - ai * bi, ar * bi + ai * br), dim=-1)


def conj(a):
    return torch.stack((a[..., 0], -a[..., 1]), dim=-1)


def abs2(a):
    return a[..., 0] ** 2 + a[..., 1] ** 2


def cabs(a):
    return torch.sqrt(abs2(a) + EPS)


def normalise_frames(h, gain=True, phase=True):
    """Remove per-frame gain and common phase jitter from (B, T, A, K, 2) CSI.

    gain:  divide each frame by its RMS over antennas and subcarriers (AGC).
    phase: rotate each frame so that the subcarrier-sum of antenna 0 has zero
           phase. The receive chains share one oscillator, so this cancels the
           common phase offset without touching inter-antenna phase.
    """
    if gain:
        rms = torch.sqrt(abs2(h).mean(dim=(-1, -2), keepdim=True) + EPS)  # (B, T, 1, 1)
        h = h / rms.unsqueeze(-1)
    if phase:
        ref = h[..., 0, :, :].sum(dim=-2)  # (B, T, 2)
        unit = ref / cabs(ref).unsqueeze(-1)
        h = cmul(h, conj(unit)[..., None, None, :])
    return h


def split_static_dynamic(h):
    """Split (B, T, A, K, 2) CSI into its time-average and the deviation from it.

    Returns (static, dynamic_rel, static_power):
      static       (B, 1, A, K, 2)  mean over time, the room "fingerprint"
      dynamic_rel  (B, T, A, K, 2)  (h - static) / RMS(static), scale-free
      static_power (B,)             mean |static|^2
    """
    static = h.mean(dim=1, keepdim=True)
    static_power = abs2(static).mean(dim=(1, 2, 3))
    dynamic = (h - static) / torch.sqrt(static_power + EPS)[:, None, None, None, None]
    return static, dynamic, static_power


def relative_amplitude(h):
    """|h| / mean_t |h| - 1 for (B, T, A, K, 2) -> (B, T, A, K)."""
    amp = cabs(h)
    return amp / (amp.mean(dim=1, keepdim=True) + EPS) - 1.0


def interval_ratio(ati):
    """(B, T) intervals in us -> (B, T - 1) log ratio to the nominal 5 ms, padding dropped."""
    x = ati[:, :-1].clamp(min=1.0)
    return torch.log(x / NOMINAL_INTERVAL_US)
