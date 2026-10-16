"""Quiet-window gate: predict 0 people when a window's dynamics are pure noise.

The statistic is the lag-1 autocorrelation of the detrended complex CSI,
median over antennas x subcarrier groups. It is about signal / (signal +
noise), so it does not depend on the room's gain or noise level (WiDetect,
Zhang et al. 2019). Empty labelled rooms with clean channels (b1, g1) stay
below 0.2; windows below the threshold are labelled 0 and every other
window keeps the wrapped model's prediction.

    python -m common.gate runs/deep_model/cnn_base.pt     # -> submissions/deep_cnn_base_gate.onnx
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from common.data import NUM_CLASSES
from common.export import export_onnx
from common.preprocess import EPS

QUIET_THRESHOLD = 0.2


def quiet_statistic(h):
    """(B, T, A, K, 2) pooled CSI -> (B,) median lag-1 autocorrelation of the detrended dynamics."""
    t = torch.arange(h.shape[1], dtype=h.dtype, device=h.device)
    t = (t - t.mean())[None, :, None, None, None]
    d = h - h.mean(dim=1, keepdim=True)
    d = d - t * (t * d).sum(dim=1, keepdim=True) / (t * t).sum()
    num = (d[:, 1:] * d[:, :-1]).sum(dim=(1, 4))  # Re(sum d_t conj(d_{t-1})), (B, A, K)
    den = (d * d).sum(dim=(1, 4)) + EPS
    r = (num / den).flatten(1)
    return torch.sort(r, dim=1).values[:, (r.shape[1] - 1) // 2]


class QuietGate(nn.Module):
    """Wrap a core model: quiet windows -> class 0, others -> the core's logits."""

    def __init__(self, core, threshold=QUIET_THRESHOLD):
        super().__init__()
        self.core = core
        self.threshold = threshold
        zero = torch.full((NUM_CLASSES,), -10.0)
        zero[0] = 10.0
        self.register_buffer("zero_logits", zero)

    def forward(self, h, ati):
        logits = self.core(h, ati)
        quiet = (quiet_statistic(h) < self.threshold)[:, None]
        return torch.where(quiet, self.zero_logits.expand_as(logits), logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="deep_model checkpoint (.pt)")
    parser.add_argument("--threshold", type=float, default=QUIET_THRESHOLD)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    from deep_model.train import build_model

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    core = build_model(ckpt["cfg"])
    core.load_state_dict(ckpt["state_dict"])
    output = args.output or Path("submissions") / f"deep_{args.checkpoint.stem}_gate.onnx"
    export_onnx(QuietGate(core.eval(), args.threshold).eval(), output)


if __name__ == "__main__":
    main()
