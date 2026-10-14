"""CNN over the dynamic (time-varying) part of the CSI.

Input to the network is built inside the model so the exported ONNX graph
contains the full preprocessing:
  1. per-frame gain and common-phase normalisation
  2. remove the static channel (time mean) -> relative dynamic CSI
  3. stack real, imag of the dynamic part and relative amplitude per antenna
     -> 9 channels over (time, subcarrier group)
  4. optionally rescale by the window's own RMS, so the network sees the
     pattern of the variation rather than its room-dependent strength
"""

import torch
from torch import nn

from common.preprocess import (
    EPS, interval_ratio, normalise_frames, pool_groups, relative_amplitude, split_static_dynamic,
)

GAP_US = 7000.0
SHORT_US = 4500.0


def _std(x, dim):
    mean = x.mean(dim=dim, keepdim=True)
    return torch.sqrt(((x - mean) ** 2).mean(dim=dim) + EPS)


class InputFeatures(nn.Module):
    """(B, T, 3, 126, 2) pooled CSI -> (B, 9, T, K') map and (B, 1) log RMS."""

    def __init__(self, freq_pool=2, rms_norm=True):
        super().__init__()
        self.freq_pool = freq_pool
        self.rms_norm = rms_norm

    def forward(self, h):
        # Pooling is linear, so doing it first barely changes the result and
        # makes the per-frame normalisation `freq_pool` times cheaper.
        h = normalise_frames(pool_groups(h, self.freq_pool))  # (B, T, A, K', 2)
        _, dyn, _ = split_static_dynamic(h)
        ramp = relative_amplitude(h)
        x = torch.cat((dyn[..., 0], dyn[..., 1], ramp), dim=2)  # (B, T, 3A, K')
        rms = torch.sqrt((x ** 2).mean(dim=(1, 2, 3)) + EPS)  # (B,)
        if self.rms_norm:
            x = x / rms[:, None, None, None]
        return x.permute(0, 2, 1, 3), torch.log10(rms)[:, None]


def conv_block(cin, cout, kernel, stride):
    pad = (kernel[0] // 2, kernel[1] // 2)
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, pad, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class CSINet(nn.Module):
    """(B, T, 3, 126, 2) pooled CSI + (B, T) intervals -> (B, 5) logits."""

    def __init__(self, width=32, dropout=0.3, freq_pool=2, rms_norm=True,
                 use_ati=True, use_scale=True):
        super().__init__()
        self.inputs = InputFeatures(freq_pool, rms_norm)
        self.use_ati = use_ati
        self.use_scale = use_scale
        w = width
        # (T, K') = (600, 63) -> (150, 32) -> (75, 16) -> (38, 16) -> (19, 8) -> (10, 4)
        self.body = nn.Sequential(
            conv_block(9, w, (7, 3), (4, 2)),
            conv_block(w, 2 * w, (5, 3), (2, 2)),
            conv_block(2 * w, 2 * w, (5, 3), (2, 1)),
            conv_block(2 * w, 4 * w, (5, 3), (2, 2)),
            conv_block(4 * w, 4 * w, (3, 3), (2, 2)),
        )
        self.embed_dim = 8 * w + 4 * use_ati + 1 * use_scale
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5),
        )

    def forward(self, h, ati):
        return self.head(self.embed(h, ati))

    def embed(self, h, ati):
        """(B, embed_dim) window embedding; the domain-adversarial branch attaches here."""
        # Under autocast only conv/linear run in float16; the preprocessing is
        # elementwise maths and reductions, which stay float32 (dynamic part ~1e-3).
        x, log_rms = self.inputs(h)
        z = self.body(x).flatten(2).float()  # (B, C, T' * K')
        parts = [z.mean(dim=-1), _std(z, dim=-1)]
        if self.use_ati:
            r = interval_ratio(ati)
            raw = ati[:, :-1]
            parts.append(torch.stack((
                r.mean(dim=1), _std(r, dim=1),
                (raw > GAP_US).float().mean(dim=1), (raw < SHORT_US).float().mean(dim=1),
            ), dim=1))
        if self.use_scale:
            parts.append(log_rms)
        return torch.cat(parts, dim=1)
