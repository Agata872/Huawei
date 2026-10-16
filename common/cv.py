"""Leave-one-room-out (LORO) cross-validation shared by both routes.

Each fold holds out every labeled trace of one room. The held-out traces are
cut into non-overlapping 600-frame windows (20 per 60 s trace), which mimic
the test samples. Windows from one trace never appear in both training and
evaluation.
"""

import json
import time
from collections import Counter, defaultdict

import numpy as np

from common.data import NUM_CLASSES, RUNS_ROOT, WINDOW


def eval_starts(trace, length=WINDOW):
    return list(range(0, trace.num_frames - length + 1, length))


def summarize(records, title=""):
    """Print and return accuracy tables.

    records: iterable of dicts with keys fold, ident, label, pred (one per window).
    """
    records = list(records)
    by_fold = defaultdict(list)
    for r in records:
        by_fold[r["fold"]].append(r)

    print(f"\n=== {title} ===" if title else "")
    print(f"{'fold':<10}{'windows':>8}{'acc':>8}{'trace-vote':>12}   labels")
    fold_acc = {}
    for fold in sorted(by_fold):
        rs = by_fold[fold]
        acc = np.mean([r["pred"] == r["label"] for r in rs])
        vote = _trace_vote_acc(rs)
        fold_acc[fold] = float(acc)
        labels = dict(sorted(Counter(r["label"] for r in rs).items()))
        print(f"{fold:<10}{len(rs):>8}{acc:>8.3f}{vote:>12.3f}   {labels}")
    pooled = float(np.mean([r["pred"] == r["label"] for r in records]))
    macro = float(np.mean(list(fold_acc.values())))
    print(f"{'pooled':<10}{len(records):>8}{pooled:>8.3f}{_trace_vote_acc(records):>12.3f}")
    print(f"{'fold-mean':<10}{'':>8}{macro:>8.3f}")
    always_one = np.mean([r["label"] == 1 for r in records])
    print(f"reference: always predicting 1 person scores {always_one:.3f} pooled")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for r in records:
        cm[r["label"], r["pred"]] += 1
    print("confusion (rows = true 0..4, cols = predicted 0..4):")
    for i, row in enumerate(cm):
        print(f"  {i}: " + " ".join(f"{v:5d}" for v in row))
    # The public test set is exactly class-balanced (constant probes all score
    # 0.20), so its accuracy equals the mean per-class recall.
    recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    balanced = float(recall[cm.sum(axis=1) > 0].mean())
    print(f"per-class recall {np.round(recall, 3).tolist()} -> balanced accuracy {balanced:.3f} "
          f"(estimate of the leaderboard score)")
    return {"pooled": pooled, "fold_mean": macro, "per_fold": fold_acc, "confusion": cm.tolist(),
            "recall": recall.tolist(), "balanced": balanced}


def _trace_vote_acc(records):
    votes = defaultdict(list)
    labels = {}
    for r in records:
        votes[r["ident"]].append(r["pred"])
        labels[r["ident"]] = r["label"]
    hits = [Counter(v).most_common(1)[0][0] == labels[i] for i, v in votes.items()]
    return float(np.mean(hits))


def save_run(route, name, config, summary, records):
    folder = RUNS_ROOT / route
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}.json"
    with path.open("w") as f:
        json.dump({"config": config, "summary": summary, "records": records}, f, indent=1)
    print(f"saved {path}")
    return path
