"""EHUNAM people-counting captures as extra training data for the P13 network.

EHUNAM (de Armas et al., Scientific Data 12, 2025; CC BY 4.0; figshare
10.6084/m9.figshare.28541225) records 802.11ac CSI at 5 GHz over 80 MHz with
Raspberry Pi 4 monitors (bcm43455c0, Nexmon) in six rooms, with 0-4 people
(and more), at ~1000 packets/s for 60 s per capture and receiver. Every capture
without machines that 3 or more receivers recorded at >= 400 packets/s
becomes one 3-antenna trace in the competition format (receivers 1-3):

  1. per receiver: repair the timestamps (a few jumps), take the packet nearest
     to each frame time of a competition ATI sequence (a random stretch of a
     labeled trace), so frame timing and per-frame noise look like ours
  2. flip the sign of k >= -64 (a hardware artefact in every frame) and reference
     every frame to its strongest path as in analysis.sharp_doppler: Nexmon phase
     is random from frame to frame, the competition CSI phase is not
  3. interpolate the DC gap (k = -1..2) and stretch k = -122..122 over
     subcarriers 6..1018 of the 1024 grid
  4. scale the capture to a mean |CSI| level drawn from the labeled rooms
     (51-57 dB), each receiver within +-1.5 dB of it
  5. shrink the time-varying part around the static channel by a factor drawn
     log-uniformly from 0.04-0.10 per capture: block-to-block changes of the log
     amplitude are 15-25x larger than in the competition data (0.007 vs
     0.0005 in empty rooms, 0.04-0.08 vs 0.002-0.003 with people), while their
     ratio between empty and occupied rooms is about the same
  6. block maps as in temporal_model.maps, 1100 blocks (11000 frames, 56 s) per trace

    python -m temporal_model.ehunam [--workers 4]

Streams the selected files (about 25 GB) from the figshare archive one by one
with HTTP range requests and keeps only the maps: data/cache/maps/ehunam/
<capture>.npz plus index.csv (identifier, layout, room, headcount). Needs the
labeled maps (python -m temporal_model.maps) for the ATI sequences.
"""

import argparse
import csv
import io
import struct
import threading
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import inflate64
import numpy as np
import pandas as pd
import scipy.fft
import scipy.io
from remotezip import RemoteZip

from common.data import read_index
from temporal_model.maps import MAP_ROOT, BLOCK, map_path, trace_maps

ARCHIVE = "https://ndownloader.figshare.com/files/52814369"  # EHUNAM.zip, 78 GB
SUMMARY = "https://ndownloader.figshare.com/files/52834283"  # Summary.xlsx
SPLIT = "ehunam"
FRAMES = 11000
START_S = 0.5  # skip the first half second of every capture
VALID = slice(6, 251)  # FFT-shifted bins 6..250 = k -122..122
FLIP = 58  # positions >= 58 (k >= -64) carry a sign flip
GAP = np.arange(121, 125)  # k = -1..2: DC nulls and a distorted neighbour
LEVEL_DB = (51.0, 57.0)
RECEIVER_DB = 1.5
DYNAMIC_SCALE = (0.04, 0.10)
OVERSAMPLE = 8
_local = threading.local()


def select(summary):
    """Summary table -> one row per usable capture with its three receiver files."""
    s = summary[(summary.BW == 80) & (summary.N_Machine == 0) & summary.Application.isin(["PC", "E"])
                & (summary.N_People <= 4) & (summary.N_CSI / summary.T_Meas >= 400) & summary.Rx.isin([1, 2, 3])]
    s = s.assign(capture=s.File.str.replace(r"^(MC\d_\d+[AB]?)_\d+_", r"\1_", regex=True).str.replace(".mat", ""))
    rows = []
    for capture, g in s.groupby("capture"):
        if sorted(g.Rx) == [1, 2, 3]:
            g = g.sort_values("Rx")
            rows.append(dict(identifier=capture.replace("#", "x"), room=g.Enviroment.iloc[0].lower().replace(" ", "_"),
                             label=int(g.N_People.iloc[0]), files=list(g.File)))
    return rows


def fetch(name):
    """One archive member -> bytes, by range requests (the archive uses Deflate64)."""
    if not hasattr(_local, "zip"):
        _local.zip = RemoteZip(ARCHIVE)
    z = _local.zip
    info = z.getinfo(name)
    z.fp.seek(info.header_offset)
    n, e = struct.unpack("<HH", z.fp.read(30)[26:30])
    z.fp.seek(info.header_offset + 30 + n + e)
    data = inflate64.Inflater().inflate(z.fp.read(info.compress_size))
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise IOError(f"{name}: corrupt download")
    return data


def strongest_path_reference(h):
    """(T, K) complex CSI -> the same with each frame's strongest path at zero delay and phase."""
    k = np.arange(h.shape[1], dtype=np.float32)
    n = OVERSAMPLE * 256
    cir = np.abs(scipy.fft.ifft(h, n=n, axis=-1, workers=-1))
    m = cir.argmax(axis=-1)
    rows = np.arange(len(m))
    y0, ym, yp = cir[rows, m], cir[rows, (m - 1) % n], cir[rows, (m + 1) % n]
    theta = (2 * np.pi * (m + 0.5 * (ym - yp) / (ym - 2 * y0 + yp - 1e-12)) / n).astype(np.float32)
    steer = np.exp(1j * theta[:, None] * k)
    r = (h * steer).mean(axis=-1, keepdims=True)
    return h * steer * (np.conj(r) / np.abs(r))


def receiver(mat, times):
    """One receiver file + frame times (s) -> (FRAMES, 1024) complex64 on the competition subcarrier grid."""
    ts = mat["TimeStamp"][:, 0].astype(np.float64)
    dt = np.diff(ts)
    bad = (dt <= 0) | (dt > 1.0)
    dt[bad] = np.median(dt[~bad])
    t = np.concatenate([[0.0], np.cumsum(dt)])
    i = np.clip(np.searchsorted(t, times), 1, len(t) - 1)
    i -= (times - t[i - 1]) < (t[i] - times)  # nearest packet
    h = mat["CSI"][i, VALID].astype(np.complex64)
    h[:, FLIP:] *= -1
    h[:, GAP] = 0
    h = strongest_path_reference(h)
    left, right = GAP[0] - 1, GAP[-1] + 1
    w = ((GAP - left) / (right - left)).astype(np.float32)
    h[:, GAP] = h[:, [left]] * (1 - w) + h[:, [right]] * w
    p = (np.arange(6, 1019) - 6) * (h.shape[1] - 1) / 1012  # stretch k = -122..122 over bins 6..1018
    p0 = np.minimum(p.astype(int), h.shape[1] - 2)
    wp = (p - p0).astype(np.float32)
    out = np.zeros((len(h), 1024), np.complex64)
    out[:, 6:1019] = h[:, p0] * (1 - wp) + h[:, p0 + 1] * wp
    return out


def convert(row, ati_bank, rng):
    """Capture row -> (a, ph, ati) block maps of a 3-antenna trace."""
    src = ati_bank[rng.integers(len(ati_bank))]
    s = rng.integers(len(src) - FRAMES)
    ati = src[s:s + FRAMES].copy()
    ati[-1] = 0
    times = START_S + np.concatenate([[0.0], np.cumsum(ati[:-1], dtype=np.float64)]) * 1e-6
    level = rng.uniform(*LEVEL_DB)
    shrink = np.float32(np.exp(rng.uniform(*np.log(DYNAMIC_SCALE))))
    antennas = []
    for name in row["files"]:
        mat = scipy.io.loadmat(io.BytesIO(fetch(name)), variable_names=["CSI", "TimeStamp"])
        if mat["TimeStamp"].shape[0] < 2 or times[-1] > np.sum(np.clip(np.diff(mat["TimeStamp"][:, 0]), 0, 1.0)):
            raise ValueError(f"{name}: shorter than {times[-1]:.1f} s")
        h = receiver(mat, times)
        gain = 10 ** ((level + rng.uniform(-RECEIVER_DB, RECEIVER_DB)) / 20) / np.abs(h[:, 6:1019]).mean()
        static = h.mean(axis=0)
        antennas.append((static + shrink * (h - static)) * np.float32(gain))
    csi = np.stack(antennas, axis=1)[:, None]  # (T, 1, 3, 1024)
    a, ph = trace_maps(csi)
    return a, ph, ati.astype(np.float32)


def read_ehunam_index():
    """The converted captures, as rows like common.data.read_index."""
    with (MAP_ROOT / SPLIT / "index.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["split"] = SPLIT
        row["label"] = int(row["headcount"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4, help="captures downloaded in parallel")
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N captures (testing)")
    args = parser.parse_args()
    with urllib.request.urlopen(SUMMARY) as response:
        summary = pd.read_excel(io.BytesIO(response.read()))
    rows = select(summary)[:args.limit]
    labeled = read_index("labeled")
    ati_bank = [np.load(map_path("labeled", r["identifier"]))["ati"] for r in labeled]
    out = MAP_ROOT / SPLIT
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} captures: " + ", ".join(f"{c} people {sum(r['label'] == c for r in rows)}" for c in range(5)),
          flush=True)

    def work(k):
        row = rows[k]
        path = map_path(SPLIT, row["identifier"])
        if not path.exists():
            a, ph, ati = convert(row, ati_bank, np.random.default_rng(zlib.crc32(row["identifier"].encode())))
            np.savez(path, a=a, ph=ph, ati=ati)
        print(f"  [{k + 1}/{len(rows)}] {row['identifier']} ({row['room']}, {row['label']})", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, range(len(rows))))
    with (out / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "layout", "room", "acquisition", "headcount"])
        w.writerows([r["identifier"], SPLIT, r["room"], r["identifier"], r["label"]] for r in rows)
    print(f"wrote {len(rows)} traces to {out}")


if __name__ == "__main__":
    main()
