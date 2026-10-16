"""Pretrain the CNN on WiMANS headcounts (external data), then check transfer.

    python -m common.wimans                                   # build the WiMANS cache once
    python -m deep_model.pretrain --name wimans --holdout meeting_room   # cross-room check inside WiMANS
    python -m deep_model.pretrain --name wimans                          # all three rooms -> checkpoint
    python -m deep_model.train --init runs/deep_model/wimans.pt ...      # fine-tune on RoomSense

Every evaluation also runs the model zero-shot on all RoomSense labeled
windows: if a WiMANS-only model beats chance there, headcount information
transfers between the datasets. Scores are mean per-class recall (the public
test set is class-balanced). ATI and signal-scale inputs default to off,
because their distributions differ between the two hardware setups.
"""

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from common.cv import eval_starts
from common.data import RUNS_ROOT, WINDOW, load_traces
from common.preprocess import NUM_GROUPS
from common.wimans import interp_matrix, load_cache, to_groups
from deep_model.train import augment, build_model, build_parser


def balanced(labels, preds, classes=range(5)):
    labels, preds = np.asarray(labels), np.asarray(preds)
    rec = [np.mean(preds[labels == c] == c) for c in classes if (labels == c).any()]
    return float(np.mean(rec)), [round(float(np.mean(preds[labels == c] == c)), 2) if (labels == c).any() else None
                                 for c in classes]


class WimansData:
    """WiMANS windows on the GPU (float16, 30 subcarriers), interpolated to 126 groups per batch."""

    def __init__(self, device, bands=("2.4", "5")):
        csi, ati, meta = load_cache()
        keep = meta["ok"].to_numpy() & meta["wifi_band"].astype(float).isin([float(b) for b in bands]).to_numpy()
        self.meta = meta[keep].reset_index(drop=True)
        self.csi = torch.from_numpy(np.ascontiguousarray(csi[np.flatnonzero(keep)])).to(device)
        self.ati = torch.from_numpy(ati[keep]).to(device)
        self.users = torch.tensor(self.meta["users"].to_numpy(), device=device)
        self.matrix = interp_matrix(NUM_GROUPS).to(device)
        self.device = device

    def gather(self, idx, tx):
        h = to_groups(self.csi[idx, tx].float(), self.matrix)  # (B, 600, 3, 126, 2)
        return h, self.ati[idx]


def sample_weights(meta, idx):
    """Balanced over headcount within each room."""
    sub = meta.iloc[idx]
    counts = sub.groupby(["environment", "users"]).size()
    return torch.tensor([1.0 / counts[(e, u)] for e, u in zip(sub["environment"], sub["users"])])


@torch.no_grad()
def eval_wimans(model, data, idx, batch=256):
    model.eval()
    preds = []
    for k in range(0, len(idx), batch):
        b = torch.as_tensor(idx[k:k + batch], device=data.device)
        h, ati = data.gather(b, torch.zeros_like(b))
        preds += model(h, ati).argmax(1).tolist()
    return balanced(data.meta["users"].to_numpy()[idx], preds)


@torch.no_grad()
def eval_roomsense(model, device, batch=128):
    """Zero-shot on all RoomSense labeled windows."""
    model.eval()
    labels, preds = [], []
    for t in load_traces("labeled", mmap=True):
        starts = eval_starts(t)
        h = torch.from_numpy(np.stack([t.csi[s:s + WINDOW] for s in starts])).to(device)
        ati = torch.from_numpy(np.stack([t.ati[s:s + WINDOW] for s in starts])).to(device)
        ati[:, -1] = 0
        preds += model(h, ati).argmax(1).tolist()
        labels += [t.label] * len(starts)
    score, rec = balanced(labels, preds)
    coarse = [np.mean(np.array(preds)[np.array(labels) == c] == c) for c in (0, 1)]
    ge2 = np.array(labels) >= 2
    coarse.append(np.mean(np.array(preds)[ge2] >= 2))
    return score, rec, [round(float(c), 2) for c in coarse]


def main():
    p = build_parser(__doc__)
    p.add_argument("--holdout", default=None, help="WiMANS room kept out of training (cross-room check)")
    p.add_argument("--bands", default="2.4,5")
    p.set_defaults(name="wimans", steps=6000, batch=128, use_ati=False, use_scale=False)
    cfg = vars(p.parse_args())
    device = (cfg["devices"] or "cuda:0").split(",")[0]
    torch.manual_seed(cfg["seed"])

    data = WimansData(device, tuple(cfg["bands"].split(",")))
    meta = data.meta
    train_idx = np.flatnonzero(meta["environment"] != cfg["holdout"]) if cfg["holdout"] else np.arange(len(meta))
    held_idx = np.flatnonzero(meta["environment"] == cfg["holdout"]) if cfg["holdout"] else np.array([], int)
    print(f"WiMANS: {len(meta)} samples ({cfg['bands']} GHz), train {len(train_idx)}, held-out "
          f"{cfg['holdout']} {len(held_idx)}; users per class {meta['users'].value_counts().sort_index().tolist()}",
          flush=True)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    use_amp = cfg["amp"] and device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    weights = sample_weights(meta, train_idx)
    train_t = torch.as_tensor(train_idx, device=device)
    start = time.time()
    for step in range(1, cfg["steps"] + 1):
        model.train()
        pick = train_t[torch.multinomial(weights, cfg["batch"], replacement=True).to(device)]
        tx = torch.randint(0, 3, (len(pick),), device=device)
        h, ati = augment(*data.gather(pick, tx), cfg)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(h, ati), data.users[pick])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            msg = f"step {step} loss {loss.item():.3f} ({time.time() - start:.0f}s)"
            if len(held_idx):
                s, rec = eval_wimans(model, data, held_idx)
                msg += f" | WiMANS {cfg['holdout']} balanced {s:.3f} recall {rec}"
            s, rec, coarse = eval_roomsense(model, device)
            msg += f" | RoomSense zero-shot balanced {s:.3f} recall {rec} 0/1/2+ {coarse}"
            print(msg, flush=True)
    path = RUNS_ROOT / "deep_model" / f"{cfg['name']}.pt"
    torch.save({"cfg": cfg, "state_dict": model.cpu().state_dict()}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
