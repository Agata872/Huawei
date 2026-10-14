"""Train the CNN with leave-one-room-out CV, or on all rooms for export.

    python -m deep_model.train                      # LORO over all labeled rooms
    python -m deep_model.train --folds a1,c1        # selected folds only
    python -m deep_model.train --final --name base  # all rooms -> runs/deep_model/base.pt

Labeled traces are kept on the GPU; each step samples random 600-frame
windows class-balanced over traces. Reported accuracy is from the last step
(no early stopping on the held-out room).
"""

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch import nn

from common.cv import eval_starts, save_run, summarize
from common.data import RUNS_ROOT, WINDOW, load_traces
from deep_model.model import CSINet

MODEL_KEYS = ("width", "dropout", "freq_pool", "rms_norm", "use_ati", "use_scale")


def build_model(cfg):
    return CSINet(**{k: cfg[k] for k in MODEL_KEYS})


class GpuTraces:
    """All traces of one split stacked on the device for fast window sampling."""

    def __init__(self, traces, device):
        self.traces = traces
        self.csi = torch.from_numpy(np.stack([t.csi for t in traces])).to(device)
        self.ati = torch.from_numpy(np.stack([t.ati for t in traces])).to(device)
        self.labels = torch.tensor([t.label for t in traces], device=device)
        self.offsets = torch.arange(WINDOW, device=device)

    def gather(self, trace_idx, starts):
        idx = starts[:, None] + self.offsets
        h = self.csi[trace_idx[:, None], idx]
        ati = self.ati[trace_idx[:, None], idx].clone()
        ati[:, -1] = 0
        return h, ati


def augment(h, ati, cfg):
    b = h.shape[0]
    dev = h.device
    if cfg["aug_antenna"] > 0:
        perm = torch.argsort(torch.rand(b, 3, device=dev), dim=1)
        keep = torch.rand(b, device=dev) >= cfg["aug_antenna"]
        perm[keep] = torch.arange(3, device=dev)
        h = torch.gather(h, 2, perm[:, None, :, None, None].expand_as(h))
    if cfg["aug_freq_flip"] > 0:
        flip = torch.rand(b, device=dev) < cfg["aug_freq_flip"]
        h = torch.where(flip[:, None, None, None, None], h.flip(3), h)
    if cfg["aug_time_flip"] > 0:
        flip = torch.rand(b, device=dev) < cfg["aug_time_flip"]
        h = torch.where(flip[:, None, None, None, None], h.flip(1), h)
        rev = torch.cat((ati[:, :-1].flip(1), ati[:, -1:]), dim=1)
        ati = torch.where(flip[:, None], rev, ati)
    return h, ati


def train_model(data, train_idx, cfg, device, monitor=None):
    """Train on the traces data.traces[train_idx]; `monitor(model, step)` is called periodically."""
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"],
                                                pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    use_amp = cfg["amp"] and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    train_idx = torch.as_tensor(train_idx, device=device)
    labels = data.labels[train_idx]
    counts = torch.bincount(labels, minlength=5).float()
    weights = 1.0 / counts[labels]  # every class equally likely per draw
    max_start = data.csi.shape[1] - WINDOW + 1

    for step in range(1, cfg["steps"] + 1):
        model.train()
        pick = train_idx[torch.multinomial(weights, cfg["batch"], replacement=True)]
        starts = torch.randint(0, max_start, (cfg["batch"],), device=device)
        h, ati = augment(*data.gather(pick, starts), cfg)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(h, ati), data.labels[pick])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if monitor and (step % cfg["eval_every"] == 0 or step == cfg["steps"]):
            monitor(model, step, loss.item())
    return model


@torch.no_grad()
def predict_windows(model, data, trace_ids, batch=128):
    """Predictions for the non-overlapping windows of the given traces."""
    model.eval()
    jobs = [(i, s) for i in trace_ids for s in eval_starts(data.traces[i])]
    preds = []
    for k in range(0, len(jobs), batch):
        ids, starts = zip(*jobs[k:k + batch])
        dev = data.csi.device
        h, ati = data.gather(torch.tensor(ids, device=dev), torch.tensor(starts, device=dev))
        preds += model(h, ati).argmax(dim=1).tolist()
    return [(data.traces[i], p) for (i, _), p in zip(jobs, preds)]


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def build_parser(description=__doc__):
    """Model, optimisation and augmentation options shared with deep_model.adapt."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", default="all", help="'all' or comma-separated rooms")
    p.add_argument("--final", action="store_true", help="Train on all labeled rooms and save")
    p.add_argument("--name", default="cnn")
    p.add_argument("--devices", default=None,
                   help="Comma-separated devices; LORO folds are spread over them "
                        "(default: all GPUs, else cpu)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable float16 autocast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--freq-pool", type=int, default=2)
    p.add_argument("--no-rms-norm", dest="rms_norm", action="store_false")
    p.add_argument("--no-ati", dest="use_ati", action="store_false")
    p.add_argument("--no-scale", dest="use_scale", action="store_false")
    p.add_argument("--aug-antenna", type=float, default=0.5)
    p.add_argument("--aug-freq-flip", type=float, default=0.5)
    p.add_argument("--aug-time-flip", type=float, default=0.5)
    return p


def run_folds(rooms, cfg, device):
    """Train and evaluate the given LORO folds on one device; returns window records."""
    data = GpuTraces(load_traces("labeled"), device)
    records = []
    for room in rooms:
        held = [i for i, t in enumerate(data.traces) if t.room == room]
        train = [i for i in range(len(data.traces)) if i not in held]
        start = time.time()

        def monitor(model, step, loss):
            out = predict_windows(model, data, held)
            acc = np.mean([p == t.label for t, p in out])
            print(f"  [{room}] step {step:5d} loss {loss:.3f} held-out acc {acc:.3f}", flush=True)

        print(f"fold {room} on {device}: train {len(train)} traces, held-out {len(held)}", flush=True)
        model = train_model(data, train, cfg, device, monitor)
        records += [dict(fold=room, ident=t.ident, label=t.label, pred=int(p))
                    for t, p in predict_windows(model, data, held)]
        print(f"  [{room}] fold time {time.time() - start:.0f}s", flush=True)
    return records


def default_devices():
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] or ["cpu"]


def main():
    args = parse_args()
    cfg = vars(args)
    devices = args.devices.split(",") if args.devices else default_devices()

    if args.final:
        data = GpuTraces(load_traces("labeled"), devices[0])
        model = train_model(data, list(range(len(data.traces))), cfg, devices[0],
                            monitor=lambda m, s, l: print(f"step {s} loss {l:.3f}", flush=True))
        path = RUNS_ROOT / "deep_model" / f"{args.name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": cfg, "state_dict": model.cpu().state_dict()}, path)
        print(f"saved {path}")
        return

    rooms = sorted({t.room for t in load_traces("labeled", mmap=True)})
    if args.folds != "all":
        rooms = args.folds.split(",")
    # Spread folds round-robin over devices, one worker process per device.
    shards = [(rooms[i::len(devices)], dev) for i, dev in enumerate(devices) if rooms[i::len(devices)]]
    if len(shards) == 1:
        records = run_folds(shards[0][0], cfg, shards[0][1])
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(len(shards), mp_context=ctx) as pool:
            futures = [pool.submit(run_folds, r, cfg, d) for r, d in shards]
            records = [rec for f in futures for rec in f.result()]
    summary = summarize(records, f"deep_model LORO ({args.name})")
    save_run("deep_model", args.name, cfg, summary, records)


if __name__ == "__main__":
    main()
