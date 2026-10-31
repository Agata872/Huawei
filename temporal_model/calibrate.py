"""Per-room calibration of the P13 network on the unlabeled rooms.

Some public test windows come from rooms in the unlabeled data (common.calibrate).
For every unlabeled room with at least MIN_TRACES traces, a logit bias is fitted
so that P13's mean softmax over the room's windows matches a balanced test room;
at inference the room is recognised by its static-channel fingerprint.

  full     one bias per class, target 0.2 each
  coarse   one bias for 0, one for 1 and one shared by 2-4, target 0.2 / 0.2 / 0.6;
           keeps P13's own split of the crowd, which the merged probes
           (temporal_model.export --merge-to) showed to be informative on the public set

    python -m temporal_model.calibrate        # -> submissions/p13_cal_full.onnx, p13_cal_coarse.onnx

Leaderboard probe: with --probe k, every window matched to a bank room (or to
--probe-rooms) is sent to class k and all others keep P13's output. An
unchanged public score means no public window comes from those rooms.

    python -m temporal_model.calibrate --probe 0                    # -> submissions/p13_probe0_bank.onnx
    python -m temporal_model.calibrate --probe 0 --probe-rooms v1   # -> submissions/p13_probe0_v1.onnx
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common.calibrate import (BANK_PER_TRACE, MIN_TRACES, NUM_CLASSES, RoomCalibration, fingerprint,
                              fit_group_bias)
from common.data import WINDOW, load_raw, raw_to_frames, read_index
from common.export import raw_samples
from common.preprocess import pool_subcarriers
from temporal_model.export import SUBMISSIONS, export
from temporal_model.model import P13_CHECKPOINT, load_net

TARGETS = {
    "full": ([[c] for c in range(NUM_CLASSES)], [0.2] * NUM_CLASSES),
    "coarse": ([[0], [1], [2, 3, 4]], [0.2, 0.2, 0.6]),
}


class RawRoomCalibration(RoomCalibration):
    """RoomCalibration around a core that takes raw frames; the room is matched on the pooled CSI."""

    def forward(self, frames, intervals):
        return self.core(frames, intervals) + self.room_bias(pool_subcarriers(frames))


def raw_window(csi, ati, start):
    frames = raw_to_frames(csi[start:start + WINDOW]).copy()
    intervals = ati[start:start + WINDOW].copy()
    intervals[-1] = 0
    return frames, intervals


@torch.no_grad()
def unlabeled_windows(net, chunk=5):
    """P13 logits of every non-overlapping unlabeled window, and the fingerprint bank.

    Each 'unknown' trace counts as a room of its own.
    """
    rows = read_index("unlabeled")
    keys = [r["room"] if r["room"] != "unknown" else r["identifier"] for r in rows]
    rooms = sorted(set(keys))
    window_room, logits, bank, bank_room = [], [], [], []
    for row, key in zip(rows, keys):
        csi, ati = load_raw("unlabeled", row["identifier"])
        z, fp = [], []
        starts = list(range(0, len(ati) - WINDOW + 1, WINDOW))
        for k in range(0, len(starts), chunk):
            frames, intervals = map(np.stack, zip(*[raw_window(csi, ati, s) for s in starts[k:k + chunk]]))
            frames, intervals = torch.from_numpy(frames), torch.from_numpy(intervals)
            z.append(net(frames, intervals).numpy())
            fp.append(fingerprint(pool_subcarriers(frames)).numpy())
        fp = np.concatenate(fp)
        keep = np.linspace(0, len(fp) - 1, BANK_PER_TRACE).astype(int)
        logits.append(np.concatenate(z))
        bank.append(fp[keep])
        window_room += [rooms.index(key)] * len(starts)
        bank_room += [rooms.index(key)] * len(keep)
    return {"rooms": rooms, "traces": Counter(keys), "probes": [(r["identifier"], rooms.index(k)) for r, k in zip(rows, keys)],
            "window_room": np.array(window_room), "logits": np.concatenate(logits),
            "bank": np.concatenate(bank), "bank_room": np.array(bank_room)}


def fit_biases(data, target, min_traces=MIN_TRACES):
    groups, shares = TARGETS[target]
    biases = np.zeros((len(data["rooms"]), NUM_CLASSES))
    print(f"{target}: {'room':8}{'traces':>7}  {'bias':<34}{'predicted share 0..4 before -> after'}")
    for r, key in enumerate(data["rooms"]):
        if data["traces"][key] < min_traces:
            continue
        z = data["logits"][data["window_room"] == r]
        biases[r] = fit_group_bias(z, groups, shares)
        share = lambda logits: " ".join(f"{v:.2f}" for v in np.bincount(logits.argmax(1), minlength=NUM_CLASSES) / len(z))
        print(f"  {key:12}{data['traces'][key]:>5}  {np.array2string(biases[r], precision=2):<34}"
              f"{share(z)} -> {share(z + biases[r])}")
    return biases


def probe_biases(data, cls, rooms=None):
    """Biases that send every window of `rooms` (default: all bank rooms) to class `cls`."""
    biases = np.zeros((len(data["rooms"]), NUM_CLASSES))
    for key in rooms or data["rooms"]:
        biases[data["rooms"].index(key)] = -1e4
        biases[data["rooms"].index(key), cls] = 0
    return biases


def check_rooms(path, net, data, biases, per_room=3):
    """ONNX on windows halfway between bank windows: is each matched to its own room's bias?"""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    seen, hits, total = Counter(), 0, 0
    for ident, r in data["probes"]:
        if seen[r] == per_room:
            continue
        seen[r] += 1
        csi, ati = load_raw("unlabeled", ident)
        frames, intervals = raw_window(csi, ati, 900)
        out = session.run(["logits"], {"frames": frames, "intervals": intervals})[0]
        with torch.no_grad():
            plain = net(torch.from_numpy(frames)[None], torch.from_numpy(intervals)[None])[0].numpy()
        hits += int(np.abs(out - plain - biases[r]).max() < 5e-3)
        total += 1
    print(f"  unlabeled windows given their own room's bias: {hits}/{total}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=P13_CHECKPOINT)
    parser.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    parser.add_argument("--min-traces", type=int, default=MIN_TRACES)
    parser.add_argument("--probe", type=int, choices=range(NUM_CLASSES), default=None,
                        help="export a leaderboard probe instead: matched windows go to this class")
    parser.add_argument("--probe-rooms", nargs="+", default=None, help="rooms the probe applies to (default: all)")
    args = parser.parse_args()
    torch.set_num_threads(8)
    net = load_net(args.checkpoint)
    data = unlabeled_windows(net)
    print(f"bank: {len(data['bank'])} windows from {len(data['rooms'])} rooms")
    if args.probe is None:
        models = [(f"p13_cal_{t}", fit_biases(data, t, args.min_traces)) for t in args.targets]
    else:
        name = "_".join(args.probe_rooms or ["bank"])
        models = [(f"p13_probe{args.probe}_{name}", probe_biases(data, args.probe, args.probe_rooms))]
    samples = raw_samples(24)
    for name, biases in models:
        model = RawRoomCalibration(net, data["bank"], data["bank_room"], biases)
        output = SUBMISSIONS / f"{name}.onnx"
        export(model, output, samples)
        check_rooms(output, net, data, biases)


if __name__ == "__main__":
    main()
