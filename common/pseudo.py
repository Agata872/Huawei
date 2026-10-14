"""Trace-level pseudo-labels and domain ids for the unlabeled data (both routes).

Every trace has a constant headcount, so window probabilities are averaged
per trace before a label is assigned. Selection is capped per class so the
model's favourite class cannot flood the next round.

Transductive LORO: in each fold the held-out room's traces join the
unlabeled pool with their labels hidden. This mimics a test room that also
appears in the unlabeled set (allowed by the rules), and the hidden labels
measure how accurate the pseudo-labels are.
"""

from collections import Counter, defaultdict

import numpy as np


def domain_key(trace):
    """Room-level domain id. 'unknown' traces are grouped per split."""
    return f"{trace.split}:{trace.room}"


def fold_pool(labeled, unlabeled, heldout_room, transductive=True):
    """(train_labeled, pool) traces for one fold. heldout_room=None means all labeled rooms train."""
    train = [t for t in labeled if t.room != heldout_room]
    pool = list(unlabeled)
    if transductive and heldout_room is not None:
        pool += [t for t in labeled if t.room == heldout_room]
    return train, pool


def select(trace_probs, threshold, max_per_class):
    """{ident: mean prob (5,)} -> {ident: label}.

    Keeps traces whose top mean probability is >= threshold, at most
    `max_per_class` per predicted class, most confident first.
    """
    by_class = defaultdict(list)
    for ident, p in trace_probs.items():
        c = int(np.argmax(p))
        by_class[c].append((float(p[c]), ident))
    chosen = {}
    for c, items in by_class.items():
        for conf, ident in sorted(items, reverse=True)[:max_per_class]:
            if conf >= threshold:
                chosen[ident] = c
    return chosen


def report(chosen, trace_probs, pool, prefix=""):
    """Print predicted-class counts, selection counts, and accuracy where labels are known."""
    by_ident = {t.ident: t for t in pool}
    predicted = Counter(int(np.argmax(p)) for p in trace_probs.values())
    picked = Counter(chosen.values())
    known = [i for i in trace_probs if by_ident[i].label >= 0]
    msg = (f"{prefix}pool argmax per class {dict(sorted(predicted.items()))}, "
           f"selected {len(chosen)}: {dict(sorted(picked.items()))}")
    if known:
        acc_all = np.mean([int(np.argmax(trace_probs[i])) == by_ident[i].label for i in known])
        sel = [i for i in known if i in chosen]
        acc_sel = np.mean([chosen[i] == by_ident[i].label for i in sel]) if sel else float("nan")
        msg += (f" | held-out traces: argmax acc {acc_all:.3f} ({len(known)}), "
                f"selected {len(sel)} acc {acc_sel:.3f}")
    print(msg, flush=True)
    return {"selected": len(chosen), "per_class": dict(picked)}
