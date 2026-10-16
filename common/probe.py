"""Constant-prediction models for measuring the public test class distribution.

A model that always predicts k people scores exactly the share of class k
in the public test set, so submitting k = 0..3 gives the whole distribution
(class 4 is the remainder). The task states that the public and private
sets follow the same distribution.

    python -m common.probe            # -> submissions/probe_const{0,1,2,3}.onnx
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from common.data import NUM_CLASSES
from common.export import export_onnx


class ConstantCore(nn.Module):
    """Pooled batch -> the same logits for every sample, favouring class `k`."""

    def __init__(self, k):
        super().__init__()
        logits = torch.zeros(NUM_CLASSES)
        logits[k] = 1.0
        self.register_buffer("logits", logits)

    def forward(self, h, ati):
        # Tie the output shape to the batch so both inputs stay in the graph.
        return torch.zeros_like(ati[:, :NUM_CLASSES]) + self.logits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", default="0,1,2,3")
    parser.add_argument("--out-dir", type=Path, default=Path("submissions"))
    args = parser.parse_args()
    for k in map(int, args.classes.split(",")):
        export_onnx(ConstantCore(k).eval(), args.out_dir / f"probe_const{k}.onnx", n_check=8)


if __name__ == "__main__":
    main()
