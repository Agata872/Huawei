"""Leave-one-room-out evaluation of the feature route.

    python -m feature_model.train [--model lgbm|logreg] [--exclude ati_,sfamp_]

Trains on all (overlapping) windows of the other rooms and scores the
non-overlapping 600-frame windows of the held-out room.
"""

import argparse
from collections import defaultdict

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.cv import save_run, summarize
from common.data import WINDOW
from feature_model.extract import load_features


def make_classifier(kind, seed=0):
    if kind == "lgbm":
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            class_weight="balanced", random_state=seed, verbose=-1,
        )
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000),
        )
    raise ValueError(kind)


def select_columns(names, exclude):
    prefixes = [p for p in exclude.split(",") if p]
    keep = [i for i, n in enumerate(names) if not any(n.startswith(p) for p in prefixes)]
    return np.array(keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "logreg"])
    parser.add_argument("--exclude", default="", help="Comma-separated feature-name prefixes to drop")
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    data = load_features("labeled")
    cols = select_columns(data["names"], args.exclude)
    X, y, room = data["X"][:, cols], data["label"], data["room"]
    is_eval = data["start"] % WINDOW == 0
    print(f"model={args.model}, {len(cols)} features, excluded: {args.exclude or 'none'}")

    records, importance = [], defaultdict(list)
    for fold in sorted(set(room)):
        train = room != fold
        test = (room == fold) & is_eval
        clf = make_classifier(args.model).fit(X[train], y[train])
        pred = clf.predict(X[test])
        records += [dict(fold=str(fold), ident=str(i), label=int(l), pred=int(p))
                    for i, l, p in zip(data["ident"][test], y[test], pred)]
        if args.model == "lgbm":
            gain = clf.booster_.feature_importance("gain")
            for n, g in zip(data["names"][cols], gain / gain.sum()):
                importance[n].append(g)

    summary = summarize(records, f"feature_model LORO ({args.model})")
    if importance:
        top = sorted(importance.items(), key=lambda kv: -np.mean(kv[1]))[:12]
        print("top features by mean gain share:", ", ".join(f"{n} {np.mean(g):.2f}" for n, g in top))
    config = vars(args) | {"features": data["names"][cols].tolist()}
    save_run("feature_model", args.name or args.model, config, summary, records)


if __name__ == "__main__":
    main()
