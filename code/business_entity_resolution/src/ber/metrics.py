"""Macro-averaged F_beta (beta = 0.5) exactly as defined in the challenge."""
from __future__ import annotations

from typing import Iterable, Mapping


def f_beta(pred: set[str], truth: set[str], beta: float = 0.5) -> float:
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred: Mapping[str, Iterable[str]], truth: Mapping[str, set[str]]) -> dict:
    """Score every S1 id in `truth`; a missing prediction counts as an empty list."""
    total = 0.0
    n = 0
    single_ok = single_n = 0
    tp = fp = fn = 0
    for s1, t in truth.items():
        p = set(pred.get(s1, ()))
        total += f_beta(p, t)
        n += 1
        if not t:
            single_n += 1
            single_ok += int(not p)
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "macro_f05": total / max(1, n),
        "n": n,
        "pair_precision": prec,
        "pair_recall": rec,
        "singleton_acc": single_ok / max(1, single_n),
    }
