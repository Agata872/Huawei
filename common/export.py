"""Export a model to the submission ONNX format and check it like the platform does.

Both routes build a `core` module that maps a pooled batch
(B, T, 3, 126, 2) + intervals (B, T) to logits (B, 5). `RawInputWrapper`
adds the subcarrier pooling so the exported graph takes the raw
`frames` (600, 1, 3, 1024, 2) and `intervals` (600,) inputs.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from common.data import WINDOW, load_raw, raw_to_frames, read_index
from common.preprocess import pool_subcarriers

MAX_BYTES = 100_000_000
TIME_BUDGET_S = 0.7


class RawInputWrapper(nn.Module):
    def __init__(self, core, squeeze=True):
        super().__init__()
        self.core = core
        self.squeeze = squeeze

    def forward(self, frames, intervals):
        pooled = pool_subcarriers(frames).unsqueeze(0)
        out = self.core(pooled, intervals.unsqueeze(0))
        return out[0] if self.squeeze else out


def raw_samples(n, split="labeled", seed=0):
    """n raw (frames, intervals, label) windows from distinct traces, as the platform feeds them."""
    rows = read_index(split)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    out = []
    for i in picks:
        row = rows[i]
        csi, ati = load_raw(split, row["identifier"])
        start = int(rng.integers(0, (len(ati) - WINDOW) // WINDOW + 1)) * WINDOW
        frames = raw_to_frames(csi[start:start + WINDOW]).copy()
        intervals = ati[start:start + WINDOW].copy()
        intervals[-1] = 0
        out.append((frames, intervals, row["label"]))
    return out


def export_onnx(core, path, n_check=24):
    """Export `core` wrapped for raw inputs, then validate the file.

    Checks the interface, the 100 MB limit, agreement with PyTorch on real
    samples and the average CPU time per sample (4 intra-op threads).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = RawInputWrapper(core).cpu().eval()
    samples = raw_samples(n_check)
    example = (torch.from_numpy(samples[0][0]), torch.from_numpy(samples[0][1]))
    torch.onnx.export(
        model, example, str(path),
        input_names=["frames", "intervals"], output_names=["logits"],
        opset_version=18, external_data=False,
    )
    return check_onnx(path, model, samples)


def check_onnx(path, torch_model=None, samples=None):
    path = Path(path)
    onnx.checker.check_model(str(path))
    size = path.stat().st_size
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()], [o.name for o in session.get_outputs()]
    assert names == (["frames", "intervals"], ["logits"]), names

    samples = samples or raw_samples(24)
    feed = lambda s: {"frames": s[0], "intervals": s[1]}
    session.run(["logits"], feed(samples[0]))  # warm-up
    times, max_diff, agree, correct = [], 0.0, 0, 0
    for s in samples:
        start = perf_counter()
        logits = session.run(["logits"], feed(s))[0]
        times.append(perf_counter() - start)
        assert logits.shape == (5,) and logits.dtype == np.float32, (logits.shape, logits.dtype)
        assert np.isfinite(logits).all(), "non-finite logits"
        if torch_model is not None:
            with torch.no_grad():
                ref = torch_model(torch.from_numpy(s[0]), torch.from_numpy(s[1])).numpy()
            max_diff = max(max_diff, float(np.abs(ref - logits).max()))
            agree += int(ref.argmax() == logits.argmax())
        correct += int(logits.argmax() == s[2])
    mean_t = float(np.mean(times))
    print(f"ONNX {path}: {size / 1e6:.2f} MB, {mean_t * 1000:.1f} ms/sample (CPU, 4 threads)")
    if torch_model is not None:
        print(f"  torch vs onnx: max |diff| = {max_diff:.2e}, argmax agree {agree}/{len(samples)}")
    print(f"  accuracy on {len(samples)} labeled training windows (seen data, sanity only): "
          f"{correct}/{len(samples)}")
    if size >= MAX_BYTES:
        raise ValueError(f"model is {size:,} bytes, limit is {MAX_BYTES:,}")
    if mean_t >= TIME_BUDGET_S:
        print(f"  WARNING: {mean_t:.3f} s/sample exceeds the {TIME_BUDGET_S} s budget")
    return {"bytes": size, "seconds_per_sample": mean_t, "max_abs_diff": max_diff}
