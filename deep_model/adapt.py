"""Route B with unlabeled data: domain-adversarial training + trace-level pseudo-labels.

    python -m deep_model.adapt --folds a1,c1,g1,e1 --steps 2000 --name dann_pl
    python -m deep_model.adapt --adv none --rounds 2 --name pl_only     # pseudo-labels only
    python -m deep_model.adapt --final --name dann_pl                   # all rooms -> runs/deep_model/dann_pl.pt
    python -m deep_model.export runs/deep_model/dann_pl.pt

Each round trains a fresh CNN on the labeled traces plus the pool traces that
were pseudo-labeled after the previous round (none in round 0). Every step
also draws a domain-balanced batch of pool windows. With --adv, a domain
classifier behind a gradient-reversal layer tries to tell rooms apart from
the window embedding, which pushes the CNN towards room-invariant features
(dann), optionally conditioned on the class prediction (cdan). LORO is
transductive: see common/pseudo.py.
"""

import math
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch
from torch import nn

from common import pseudo
from common.cv import eval_starts, save_run, summarize
from common.data import RUNS_ROOT, WINDOW, load_traces
from deep_model.train import (
    GpuTraces, augment, build_model, build_parser, default_devices, predict_windows,
)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coeff * grad, None


class DomainHead(nn.Module):
    """Predicts the room of a window embedding through a gradient-reversal layer."""

    def __init__(self, embed_dim, num_domains, conditional):
        super().__init__()
        self.conditional = conditional
        in_dim = embed_dim * 5 if conditional else embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_domains),
        )

    def forward(self, emb, logits, coeff):
        x = GradReverse.apply(emb, coeff)
        if self.conditional:  # CDAN: embedding (x) class probabilities
            p = torch.softmax(logits.detach().float(), dim=1)
            x = (x[:, :, None] * p[:, None, :]).flatten(1)
        return self.net(x)


class CpuTraces:
    """Pool traces kept in host memory; they do not fit on the shared GPUs."""

    def __init__(self, traces):
        self.traces = traces
        self.csi = torch.from_numpy(np.stack([t.csi for t in traces]))
        self.ati = torch.from_numpy(np.stack([t.ati for t in traces]))
        self.offsets = torch.arange(WINDOW)
        self.index = {t.ident: i for i, t in enumerate(traces)}

    def gather(self, trace_idx, starts, device):
        idx = starts[:, None] + self.offsets
        h = self.csi[trace_idx[:, None], idx].pin_memory().to(device, non_blocking=True)
        ati = self.ati[trace_idx[:, None], idx].clone()
        ati[:, -1] = 0
        return h, ati.to(device)


@torch.no_grad()
def pool_probs(model, cpu, pool_idx, device, batch=128):
    """{ident: mean softmax over the trace's non-overlapping windows}."""
    model.eval()
    jobs = [(i, s) for i in pool_idx for s in eval_starts(cpu.traces[i])]
    sums, counts = defaultdict(lambda: np.zeros(5)), Counter()
    for k in range(0, len(jobs), batch):
        ids, starts = zip(*jobs[k:k + batch])
        h, ati = cpu.gather(torch.tensor(ids), torch.tensor(starts), device)
        probs = torch.softmax(model(h, ati).float(), dim=1).cpu().numpy()
        for i, p in zip(ids, probs):
            sums[i] += p
            counts[i] += 1
    return {cpu.traces[i].ident: sums[i] / counts[i] for i in sums}


def train_round(gpu, train_idx, cpu, pool_idx, pseudo_labels, domains, cfg, device, seed):
    """One fresh model trained on labeled + pseudo-labeled traces (+ domain adversary)."""
    gpu_dom, cpu_dom, num_domains = domains
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    params = list(model.parameters())
    dom_head = None
    if cfg["adv"] != "none":
        dom_head = DomainHead(model.embed_dim, num_domains, cfg["adv"] == "cdan").to(device)
        params += list(dom_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"],
                                                pct_start=0.1)
    use_amp = cfg["amp"] and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    max_start = gpu.csi.shape[1] - WINDOW + 1

    train_t = torch.as_tensor(train_idx, device=device)
    labels = gpu.labels[train_t]
    class_w = (1.0 / torch.bincount(labels, minlength=5).float())[labels]

    pool_t = torch.as_tensor(pool_idx)
    pool_dom = cpu_dom[pool_t]
    dom_count = Counter(pool_dom.tolist())
    pool_w = torch.tensor([1.0 / dom_count[d] for d in pool_dom.tolist()])
    pool_y = torch.tensor([pseudo_labels.get(i, -1) for i in pool_idx])
    use_pool = dom_head is not None or bool(pseudo_labels)
    bu = cfg["pool_batch"]

    def sample_pool():
        pick = torch.multinomial(pool_w, bu, replacement=True)
        starts = torch.randint(0, max_start, (bu,))
        return pick, cpu.gather(pool_t[pick], starts, device)

    prefetch = ThreadPoolExecutor(1)
    future = prefetch.submit(sample_pool) if use_pool else None
    for step in range(1, cfg["steps"] + 1):
        model.train()
        pick = train_t[torch.multinomial(class_w, cfg["batch"], replacement=True)]
        starts = torch.randint(0, max_start, (cfg["batch"],), device=device)
        h, ati = augment(*gpu.gather(pick, starts), cfg)
        y = gpu.labels[pick]
        if use_pool:
            upick, (hu, atiu) = future.result()
            future = prefetch.submit(sample_pool)
            hu, atiu = augment(hu, atiu, cfg)
            yu = pool_y[upick].to(device)
            h, ati = torch.cat((h, hu)), torch.cat((ati, atiu))
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            emb = model.embed(h, ati)
            logits = model.head(emb)
            n = len(y)
            loss = ce(logits[:n], y)
            if use_pool and (yu >= 0).any():
                mask = yu >= 0
                loss = loss + cfg["pseudo_weight"] * ce(logits[n:][mask], yu[mask])
            if dom_head is not None:
                progress = step / cfg["steps"]
                coeff = cfg["adv_weight"] * (2.0 / (1.0 + math.exp(-10 * progress)) - 1.0)
                d = torch.cat((gpu_dom[pick], pool_dom[upick].to(device)))
                loss = loss + nn.functional.cross_entropy(dom_head(emb, logits, coeff).float(), d)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
    prefetch.shutdown()
    return model


def run_fold(room, gpu, cpu, cfg, device):
    """Rounds of training + pseudo-labeling for one held-out room (None = final model)."""
    tag = room or "final"
    labeled = [t for t in cpu.traces if t.label >= 0]
    unlabeled = [t for t in cpu.traces if t.label < 0]
    train_traces, pool_traces = pseudo.fold_pool(labeled, unlabeled, room, cfg["transductive"])
    train_ids = {t.ident for t in train_traces}
    train_idx = [i for i, t in enumerate(gpu.traces) if t.ident in train_ids]
    held_idx = [i for i, t in enumerate(gpu.traces) if room is not None and t.room == room]
    pool_idx = [cpu.index[t.ident] for t in pool_traces]

    keys = sorted({pseudo.domain_key(t) for t in train_traces + pool_traces})
    key_id = {k: i for i, k in enumerate(keys)}
    gpu_dom = torch.tensor([key_id.get(pseudo.domain_key(t), 0) for t in gpu.traces], device=device)
    cpu_dom = torch.tensor([key_id.get(pseudo.domain_key(t), 0) for t in cpu.traces])
    domains = (gpu_dom, cpu_dom, len(keys))
    print(f"[{tag}] {device}: {len(train_idx)} labeled, pool {len(pool_idx)} traces, "
          f"{len(keys)} domains", flush=True)

    pseudo_labels, rounds, chosen, model = {}, [], {}, None
    for r in range(cfg["rounds"]):
        start = time.time()
        model = train_round(gpu, train_idx, cpu, pool_idx, pseudo_labels, domains, cfg, device,
                            seed=cfg["seed"] + r)
        records = [dict(fold=room, ident=t.ident, label=t.label, pred=int(p))
                   for t, p in predict_windows(model, gpu, held_idx)]
        acc = np.mean([x["pred"] == x["label"] for x in records]) if records else float("nan")
        probs = pool_probs(model, cpu, pool_idx, device)
        chosen = pseudo.select(probs, cfg["pseudo_threshold"], cfg["pseudo_max_per_class"])
        print(f"[{tag}] round {r}: held-out window acc {acc:.3f}, {time.time() - start:.0f}s",
              flush=True)
        pseudo.report(chosen, probs, pool_traces, prefix=f"[{tag}] round {r}: ")
        pseudo_labels = {cpu.index[i]: c for i, c in chosen.items()}
        rounds.append(records)
    return rounds, model, chosen


def worker(rooms, cfg, device):
    gpu = GpuTraces(load_traces("labeled"), device)
    cpu = CpuTraces(load_traces("labeled") + load_traces("unlabeled"))
    return [run_fold(room, gpu, cpu, cfg, device)[0] for room in rooms]


def parse_args():
    p = build_parser(__doc__)
    p.add_argument("--adv", default="dann", choices=["none", "dann", "cdan"])
    p.add_argument("--adv-weight", type=float, default=0.3, help="Max gradient-reversal coefficient")
    p.add_argument("--rounds", type=int, default=2, help="1 = no pseudo-labels")
    p.add_argument("--pool-batch", type=int, default=64)
    p.add_argument("--pseudo-weight", type=float, default=0.5)
    p.add_argument("--pseudo-threshold", type=float, default=0.5)
    p.add_argument("--pseudo-max-per-class", type=int, default=10)
    p.add_argument("--no-transductive", dest="transductive", action="store_false",
                   help="Keep the held-out room out of the unlabeled pool")
    p.set_defaults(name="adapt")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = vars(args)
    devices = args.devices.split(",") if args.devices else default_devices()

    if args.final:
        gpu = GpuTraces(load_traces("labeled"), devices[0])
        cpu = CpuTraces(load_traces("labeled") + load_traces("unlabeled"))
        _, model, chosen = run_fold(None, gpu, cpu, cfg, devices[0])
        path = RUNS_ROOT / "deep_model" / f"{args.name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": cfg, "state_dict": model.cpu().state_dict(), "pseudo_labels": chosen}, path)
        print(f"saved {path}")
        return

    rooms = sorted({t.room for t in load_traces("labeled", mmap=True)})
    if args.folds != "all":
        rooms = args.folds.split(",")
    shards = [(rooms[i::len(devices)], d) for i, d in enumerate(devices) if rooms[i::len(devices)]]
    with ProcessPoolExecutor(len(shards), mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(worker, r, cfg, d) for r, d in shards]
        per_fold = [rounds for f in futures for rounds in f.result()]
    summaries = []
    for r in range(args.rounds):
        records = [rec for rounds in per_fold for rec in rounds[r]]
        summaries.append(summarize(records, f"deep_model adapt ({args.name}) round {r}"))
    final_records = [rec for rounds in per_fold for rec in rounds[-1]]
    save_run("deep_model", args.name, cfg, {"rounds": summaries}, final_records)


if __name__ == "__main__":
    main()
