"""Build the pooled-CSI cache used by both routes.

    python -m common.build_cache [--workers 16] [--force]

Writes data/cache/<split>/<identifier>_csi.npy, (T, 3, 126, 2) float32, and
<identifier>_ati.npy, (T,) float32. About 36 MB per trace and 7 GB in total.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from common.data import SPLITS, cache_paths, load_raw, raw_to_frames, read_index
from common.preprocess import pool_subcarriers


def build_one(job):
    split, ident, force = job
    csi_path, ati_path = cache_paths(split, ident)
    if csi_path.exists() and ati_path.exists() and not force:
        return ident, "cached"
    torch.set_num_threads(2)
    csi, ati = load_raw(split, ident)
    frames = torch.from_numpy(raw_to_frames(csi))
    pooled = pool_subcarriers(frames).numpy().astype(np.float32)
    csi_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(csi_path, pooled)
    np.save(ati_path, ati.astype(np.float32))
    return ident, f"{pooled.shape}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="Rebuild existing files")
    args = parser.parse_args()
    jobs = [(split, row["identifier"], args.force) for split in SPLITS for row in read_index(split)]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (ident, status) in enumerate(pool.map(build_one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {ident}: {status}", flush=True)


if __name__ == "__main__":
    main()
