"""Average several CNN checkpoints (softmax mean) into one ONNX model, optionally with the quiet gate.

    python -m common.ensemble runs/deep_model/final_*_s?.pt --gate --output submissions/ens6_gate.onnx

Five-class and coarse (0 / 1 / 2+) checkpoints can be mixed: coarse models put
the 2+ probability on class 2. On LORO the five-class + coarse ensemble with
the gate scored 0.372 mean per-class recall (single CNN + gate: 0.345).
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from common.export import export_onnx
from common.gate import QuietGate


class SoftmaxEnsemble(nn.Module):
    def __init__(self, cores):
        super().__init__()
        self.cores = nn.ModuleList(cores)

    def forward(self, h, ati):
        probs = torch.stack([torch.softmax(c(h, ati), dim=1) for c in self.cores]).mean(dim=0)
        return torch.log(probs + 1e-8)


def load_core(path):
    from deep_model.train import build_model

    ckpt = torch.load(path, map_location="cpu")
    core = build_model(ckpt["cfg"])
    core.load_state_dict(ckpt["state_dict"])
    return core.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--gate", action="store_true", help="Wrap with the quiet-window gate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = SoftmaxEnsemble([load_core(p) for p in args.checkpoints])
    if args.gate:
        model = QuietGate(model)
    print(f"ensemble of {len(args.checkpoints)}: {[p.name for p in args.checkpoints]}")
    export_onnx(model.eval(), args.output)


if __name__ == "__main__":
    main()
