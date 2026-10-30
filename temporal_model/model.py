"""The Reference P13 "temporal evidence" network (public 0.50).

Raw frames (B, 600, 1, 3, 1024, 2) + intervals (B, 600)
  -> 18 maps of 60 time x 128 frequency bins + 7 window statistics
  -> 3-block CNN encoder, (B, 64, 8, 16)
  -> 8 time tokens -> one 4-head self-attention block
  -> mean / std / learned pooling over tokens -> logits (B, 5).

The arithmetic follows Reference/roomsense_p13_fast/baseline/p13_model.py and
p2_fast.py, and the parameter names match, so the Reference checkpoints load
unchanged.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
REFERENCE_RUNS = REPO / "Reference" / "roomsense_p13_fast" / "experiments"
P11B_CHECKPOINT = REFERENCE_RUNS / "runs_p11_fast" / "P11B" / "checkpoint.pt"
P13_CHECKPOINT = REFERENCE_RUNS / "runs_p13_fast" / "checkpoint.pt"


def _pool(v):
    """(B, T, C, K) -> (B, C, T / 10, 128): means over 10 frames x 8 subcarriers."""
    return F.avg_pool2d(F.pad(v.permute(0, 2, 1, 3), (0, 11, 0, 0), mode="replicate"), (10, 8))


def input_features(frames, intervals):
    """Raw window batch -> (features (B, 18, 60, 128), extras (B, 7))."""
    return features_from_maps(*pooled_maps(frames), intervals)


def pooled_maps(frames):
    """Raw frames (B, T, 1, 3, 1024, 2) -> log amplitude (B, 3, T/10, 128), phase increments (B, 4, T/10, 128).

    Every value is a mean over one 10-frame x 8-subcarrier block, so the maps of
    a whole trace can be computed once and cut into windows at multiples of 10 frames.
    """
    z = frames[:, :, 0, :, 6:1019, :]
    r, i = z[..., 0], z[..., 1]
    mag = torch.sqrt(torch.clamp(r * r + i * i, min=1e-8))
    a = _pool(torch.log1p(mag))
    # Phase of Rx j relative to Rx 0, differenced across 8 subcarriers, cancels per-antenna offsets.
    phase = []
    for j in (1, 2):
        pr = r[:, :, j] * r[:, :, 0] + i[:, :, j] * i[:, :, 0]
        pi = i[:, :, j] * r[:, :, 0] - r[:, :, j] * i[:, :, 0]
        den = torch.sqrt(torch.clamp(pr * pr + pi * pi, min=1e-8))
        valid = (mag[:, :, j] * mag[:, :, 0] > 1.0).float()
        c, s = pr / den * valid, pi / den * valid
        cr = c[:, :, 8:] * c[:, :, :-8] + s[:, :, 8:] * s[:, :, :-8]
        sr = s[:, :, 8:] * c[:, :, :-8] - c[:, :, 8:] * s[:, :, :-8]
        phase += [F.pad(cr, (0, 8), mode="replicate"), F.pad(sr, (0, 8), mode="replicate")]
    return a, _pool(torch.stack(phase, 2))


def features_from_maps(a, ph, intervals):
    """Pooled maps of one 600-frame window + its intervals -> (features (B, 18, 60, 128), extras (B, 7))."""
    local = a - F.avg_pool2d(F.pad(a, (4, 4, 0, 0), mode="replicate"), (1, 9), stride=1)
    rxrel = a - a.mean(1, keepdim=True)
    residual = a - a.mean(2, keepdim=True)
    scale = torch.sqrt(residual.square().mean((2, 3), keepdim=True) + 0.000025)
    residual = torch.clamp(residual / scale, -6, 6)
    variability = torch.sqrt((a - a.mean(2, keepdim=True)).square().mean(2, keepdim=True) + 1e-8).expand_as(a)
    # intervals[k] is frame k -> k + 1 in microseconds; the padded last value is dropped.
    raw = intervals[:, :599] * 1e-6
    dt = torch.clamp(raw, min=0.001, max=0.020)
    logdt = torch.log(dt / 0.0051)
    tt = torch.cat((torch.zeros_like(dt[:, :1]), torch.cumsum(dt, 1)), 1)
    tt = tt / torch.clamp(tt[:, -1:], min=0.001)
    lt = torch.cat((logdt, torch.zeros_like(logdt[:, :1])), 1)
    times = torch.stack((lt, tt), 1).reshape(-1, 2, 60, 10).mean(3).unsqueeze(-1).expand(-1, -1, -1, 128)
    features = torch.cat((local, rxrel, residual, torch.log1p(variability * 100), ph, times), 1)
    absolute = (a.mean((2, 3)) - 6.0) * 0.1
    atistat = torch.stack((logdt.mean(1), logdt.std(1, unbiased=False), (raw > 0.020).float().mean(1),
                           torch.log1p(torch.clamp(raw, max=5).sum(1))), 1) * 0.1
    return features, torch.cat((absolute, atistat), 1)


def build_encoder():
    layers, ci = [], 18
    for co in (32, 48, 64):
        layers += [nn.Conv2d(ci, co, 3, stride=2, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
                   nn.Conv2d(co, co, 3, padding=1, groups=co), nn.Conv2d(co, co, 1), nn.GroupNorm(8, co), nn.SiLU()]
        ci = co
    return nn.Sequential(*layers)


class TemporalHead(nn.Module):
    """(B, 64, 8, 16) encoder maps + (B, 7) extras -> (B, 5) logits."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(1024, 128), nn.LayerNorm(128), nn.SiLU())
        position = torch.zeros(8, 128)
        index = torch.arange(8, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, 128, 2, dtype=torch.float32) * (-math.log(10000.0) / 128))
        position[:, 0::2] = torch.sin(index * div)
        position[:, 1::2] = torch.cos(index * div)
        self.register_buffer("position", 0.02 * position.unsqueeze(0))
        self.qkv = nn.Linear(128, 384)
        self.attention_output = nn.Linear(128, 128)
        self.attention_norm = nn.LayerNorm(128)
        self.ffn_norm = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128))
        self.dropout = nn.Dropout(0.1)
        self.pool_score = nn.Linear(128, 1)
        self.classifier = nn.Sequential(nn.Linear(391, 128), nn.SiLU(), nn.Dropout(0.25), nn.Linear(128, 5))

    def forward(self, h, extras):
        # Time is the sequence axis; each token flattens channels, then frequency.
        tokens = self.projection(h.permute(0, 2, 1, 3).reshape(-1, 8, 1024)) + self.position
        q, k, v = self.qkv(tokens).reshape(-1, 8, 3, 4, 32).permute(2, 0, 3, 1, 4)
        attention = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * (32 ** -0.5), dim=-1)
        attended = torch.matmul(attention, v).permute(0, 2, 1, 3).reshape(-1, 8, 128)
        tokens = self.attention_norm(tokens + self.dropout(self.attention_output(attended)))
        tokens = self.ffn_norm(tokens + self.dropout(self.ffn(tokens)))
        mean = tokens.mean(1)
        std = (tokens - mean.unsqueeze(1)).square().mean(1).add(1e-6).sqrt()
        weighted = (tokens * torch.softmax(self.pool_score(tokens), dim=1)).sum(1)
        return self.classifier(torch.cat((mean, std, weighted, extras), 1))


class TemporalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.temporal = TemporalHead()

    def forward_features(self, features, extras):
        return self.temporal(self.encoder(features), extras)

    def forward(self, frames, intervals):
        return self.forward_features(*input_features(frames, intervals))


class Ensemble(nn.Module):
    """Mean logits of several TemporalNets; the parameter-free features are computed once."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, frames, intervals):
        features, extras = input_features(frames, intervals)
        return torch.stack([m.forward_features(features, extras) for m in self.members]).mean(0)


def load_net(path, key="ema"):
    """A TemporalNet from a Reference checkpoint (key 'ema') or one of ours (key 'state_dict')."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = TemporalNet()
    net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt[key])
    return net.eval()


def load_encoder(net, path=P11B_CHECKPOINT):
    """Copy the P11B EMA encoder into `net` (P13's initialisation)."""
    ema = torch.load(path, map_location="cpu", weights_only=False)["ema"]
    net.encoder.load_state_dict({k[len("encoder."):]: v for k, v in ema.items() if k.startswith("encoder.")})
    return net
