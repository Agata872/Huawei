"""Load a compressed RoomSense sample and export an untrained PyTorch model.

Run from this folder with Python 3.14:
    python -m pip install -r requirements.txt
    python example.py /path/to/data/labeled --output model.onnx

Replace ExampleModel with your own trained model when preparing a submission.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import blosc2
import numpy as np
import onnxruntime as ort
import torch
from torch import nn


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 5)

    def forward(self, frames, intervals):
        components = frames.abs().mean(dim=(1, 2, 3)).mean(dim=0)
        interval = intervals.mean().reshape(1)
        return self.classifier(torch.cat((components, interval)))


def load_sample(folder, identifier):
    csi = blosc2.load_array(str(folder / f"CSI_{identifier}.b2"))
    ati = blosc2.load_array(str(folder / f"ATI_{identifier}.b2"))
    return csi, ati


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir", type=Path, help="Extracted labeled or unlabeled folder"
    )
    parser.add_argument("--output", type=Path, default=Path("model.onnx"))
    args = parser.parse_args()
    with (args.data_dir / "samples.csv").open(newline="") as stream:
        identifier = next(csv.DictReader(stream))["identifier"]
    csi, ati = load_sample(args.data_dir, identifier)

    # Each complex64 value becomes two float32 values: real, then imaginary.
    frames = csi[:600].view(np.float32).reshape(600, 1, 3, 1024, 2)
    intervals = ati[:600].copy()
    intervals[-1] = 0
    example = (torch.from_numpy(frames), torch.from_numpy(intervals))
    model = ExampleModel().eval()
    torch.onnx.export(
        model,
        example,
        str(args.output),
        input_names=["frames", "intervals"],
        output_names=["logits"],
        opset_version=18,
        external_data=False,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = {"frames": frames, "intervals": intervals}
    session.run(["logits"], inputs)
    start = perf_counter()
    logits = session.run(["logits"], inputs)[0]
    elapsed = perf_counter() - start
    if logits.shape != (5,) or logits.dtype != np.float32:
        raise ValueError("Model output must be float32 with shape (5,)")
    size = args.output.stat().st_size
    if size >= 100_000_000:
        raise ValueError("The model must be smaller than 100 MB")
    print(f"Saved {args.output} ({size:,} bytes)")
    print(f"Predicted headcount: {logits.argmax()} (untrained example model)")
    print(f"CPU prediction time: {elapsed:.3f} s/sample (after warm-up)")
    if elapsed >= 0.7:
        print(
            "Warning: prediction exceeds the 0.7 s/sample CPU budget. "
            "Estimate excludes loading overhead; server CPU speed may differ."
        )


if __name__ == "__main__":
    main()
