"""Export a trained CNN checkpoint to the submission ONNX format.

    python -m deep_model.train --final --name base
    python -m deep_model.export runs/deep_model/base.pt --output submissions/deep_base.onnx
"""

import argparse
from pathlib import Path

import torch

from common.export import export_onnx
from deep_model.train import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path("submissions") / f"deep_{args.checkpoint.stem}.onnx"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["cfg"])
    model.load_state_dict(ckpt["state_dict"])
    export_onnx(model.eval(), output)


if __name__ == "__main__":
    main()
