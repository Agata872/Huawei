"""Route A with unlabeled data: trace-level self-training.

    python -m feature_model.adapt --model logreg --rounds 3
    python -m feature_model.adapt --model lgbm --rounds 3 --final   # -> submissions/feature_lgbm_selftrain.onnx

Round 0 fits on the labeled windows of the training rooms. After each round,
the pool traces are labeled by averaging window probabilities per trace, and
confident traces join the next round's training set with weight
--pseudo-weight. The pool is the unlabeled data plus, in LORO, the
held-out room with its labels hidden (see common/pseudo.py).
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import pseudo
from common.cv import save_run, summarize
from common.data import WINDOW
from feature_model.export import export_classifier
from feature_model.extract import load_features
from feature_model.train import make_classifier, select_columns


@dataclass
class TraceInfo:
    ident: str
    split: str
    room: str
    label: int


def fit(kind, X, y, w, seed=0):
    clf = make_classifier(kind, seed)
    if kind == "logreg":
        return clf.fit(X, y, logisticregression__sample_weight=w)
    return clf.fit(X, y, sample_weight=w)


def trace_probs(clf, X, ident, is_eval):
    """{ident: mean predict_proba over its non-overlapping windows}."""
    probs = clf.predict_proba(X[is_eval])
    full = np.zeros((len(probs), 5))
    full[:, clf.classes_] = probs  # a class absent from training still gets a column
    sums, counts = defaultdict(lambda: np.zeros(5)), defaultdict(int)
    for i, p in zip(ident[is_eval], full):
        sums[i] += p
        counts[i] += 1
    return {i: sums[i] / counts[i] for i in sums}


def run(room, lab, unl, cols, args):
    """Self-training for one held-out room (None = all labeled rooms train)."""
    tag = room or "final"
    traces = {}
    for d, split in ((lab, "labeled"), (unl, "unlabeled")):
        for i, r, l in zip(d["ident"], d["room"], d["label"]):
            traces[i] = TraceInfo(str(i), split, str(r), int(l))
    labeled = [t for t in traces.values() if t.split == "labeled"]
    unlabeled = [t for t in traces.values() if t.split == "unlabeled"]
    train_traces, pool_traces = pseudo.fold_pool(labeled, unlabeled, room, args.transductive)
    train_ids = {t.ident for t in train_traces}
    pool_ids = {t.ident for t in pool_traces}

    X = np.concatenate((lab["X"], unl["X"]))[:, cols]
    ident = np.concatenate((lab["ident"], unl["ident"]))
    y_true = np.concatenate((lab["label"], unl["label"]))
    is_eval = np.concatenate((lab["start"], unl["start"])) % WINDOW == 0
    in_train = np.isin(ident, list(train_ids))
    in_pool = np.isin(ident, list(pool_ids))
    held = np.array([room is not None and traces[i].room == room and traces[i].split == "labeled"
                     for i in ident])

    chosen, rounds, clf = {}, [], None
    for r in range(args.rounds):
        pl = np.array([chosen.get(i, -1) for i in ident])
        use = in_train | (in_pool & (pl >= 0))
        y = np.where(in_train, y_true, pl)
        w = np.where(in_train, 1.0, args.pseudo_weight)
        clf = fit(args.model, X[use], y[use], w[use])
        test = held & is_eval
        records = [dict(fold=room, ident=str(i), label=int(l), pred=int(p))
                   for i, l, p in zip(ident[test], y_true[test], clf.predict(X[test]))]
        probs = trace_probs(clf, X[in_pool], ident[in_pool], is_eval[in_pool])
        chosen = pseudo.select(probs, args.pseudo_threshold, args.pseudo_max_per_class)
        if records:
            acc = np.mean([x["pred"] == x["label"] for x in records])
            print(f"[{tag}] round {r}: held-out window acc {acc:.3f}")
        pseudo.report(chosen, probs, pool_traces, prefix=f"[{tag}] round {r}: ")
        rounds.append(records)
    return rounds, clf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="logreg", choices=["lgbm", "logreg"])
    p.add_argument("--exclude", default="")
    p.add_argument("--rounds", type=int, default=3, help="1 = no pseudo-labels")
    p.add_argument("--pseudo-weight", type=float, default=0.5)
    p.add_argument("--pseudo-threshold", type=float, default=0.5)
    p.add_argument("--pseudo-max-per-class", type=int, default=10)
    p.add_argument("--no-transductive", dest="transductive", action="store_false")
    p.add_argument("--final", action="store_true", help="Train on all rooms and export ONNX")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--name", default="")
    args = p.parse_args()

    lab, unl = load_features("labeled"), load_features("unlabeled")
    cols = select_columns(lab["names"], args.exclude)
    if args.final:
        _, clf = run(None, lab, unl, cols, args)
        output = args.output or Path("submissions") / f"feature_{args.model}_selftrain.onnx"
        output.parent.mkdir(parents=True, exist_ok=True)
        export_classifier(args.model, clf, cols, output)
        return

    per_fold = [run(room, lab, unl, cols, args)[0] for room in sorted(set(lab["room"]))]
    summaries = []
    for r in range(args.rounds):
        records = [rec for rounds in per_fold for rec in rounds[r]]
        summaries.append(summarize(records, f"feature_model self-training ({args.model}) round {r}"))
    final_records = [rec for rounds in per_fold for rec in rounds[-1]]
    save_run("feature_model", args.name or f"{args.model}_selftrain", vars(args) | {"output": None},
             {"rounds": summaries}, final_records)


if __name__ == "__main__":
    main()
