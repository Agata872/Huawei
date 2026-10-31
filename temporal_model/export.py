"""Export the P13 network to the submission ONNX format, optionally with 2-4 people merged.

    python -m temporal_model.export                     # P13 as is -> submissions/p13.onnx
    python -m temporal_model.export --merge-to 3        # -> submissions/p13_merge3.onnx
    python -m temporal_model.export --checkpoint a.pt b.pt c.pt   # mean logits -> submissions/p13_ens3.onnx

--merge-to c sends every window the network calls 2, 3 or 4 people to class
c (the other two crowd classes are never predicted); 0 and 1 are unchanged.
It only relabels the argmax. On the class-balanced leaderboard, comparing
c = 2, 3, 4 with the unmerged score shows whether the 2-4 split carries
information and which crowd class is detected best.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common.export import check_onnx, raw_samples
from temporal_model.model import P13_CHECKPOINT, Ensemble, load_net

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"


class MergeCrowd(nn.Module):
    """Logits (B, 5) -> the same, with the largest of classes 2-4 moved to class `target`."""

    def __init__(self, core, target):
        super().__init__()
        self.core = core
        self.target = target

    def forward(self, frames, intervals):
        z = self.core(frames, intervals)
        crowd = z[:, 2:].max(dim=1, keepdim=True).values
        never = torch.full_like(crowd, -1e4)
        return torch.cat([z[:, :2]] + [crowd if c == self.target else never for c in (2, 3, 4)], dim=1)


class Submission(nn.Module):
    """Official single-sample interface: frames (600, 1, 3, 1024, 2), intervals (600,) -> logits (5,)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames, intervals):
        return self.model(frames.unsqueeze(0), intervals.unsqueeze(0))[0]


def export(model, output, samples):
    """Batched model -> single-sample ONNX file at `output`, checked on `samples`; returns the wrapped model."""
    wrapped = Submission(model).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    example = (torch.from_numpy(samples[0][0]), torch.from_numpy(samples[0][1]))
    torch.onnx.export(wrapped, example, str(output), input_names=["frames", "intervals"],
                      output_names=["logits"], opset_version=17)
    check_onnx(output, wrapped, samples)
    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=[P13_CHECKPOINT],
                        help="one or more checkpoints; several are averaged (mean logits)")
    parser.add_argument("--merge-to", type=int, choices=(2, 3, 4), default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    nets = [load_net(p) for p in args.checkpoint]
    core = nets[0] if len(nets) == 1 else Ensemble(nets).eval()
    model = MergeCrowd(core, args.merge_to) if args.merge_to else core
    name = "p13" if len(nets) == 1 else f"p13_ens{len(nets)}"
    output = args.output or SUBMISSIONS / (f"{name}_merge{args.merge_to}.onnx" if args.merge_to else f"{name}.onnx")
    samples = raw_samples(24)
    wrapped = export(model, output, samples)

    # The merged model must agree with the plain network everywhere except the 2-4 relabelling.
    with torch.no_grad():
        for frames, intervals, _ in samples:
            f, a = torch.from_numpy(frames), torch.from_numpy(intervals)
            plain = int(Submission(core)(f, a).argmax())
            expected = plain if plain < 2 or not args.merge_to else args.merge_to
            assert int(wrapped(f, a).argmax()) == expected, (plain, expected)
    print(f"  relabelling checked on {len(samples)} windows")


if __name__ == "__main__":
    main()
