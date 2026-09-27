"""Learn token equivalences from the labelled training pairs.

For every true (S1, S2/S3) pair we look at address tokens (and name tokens)
that occur on only one side.  Tokens that systematically replace each other
(e.g. `mh` <-> `maharashtra`, `dili` <-> `delhi` (romanised Devanagari),
`praivet` <-> `private` (romanised Hindi), `il` <-> `illinois`) are collected
and mapped to a single canonical spelling (the most frequent Source-1 form).

Everything is learned from the provided training data only; no external
resources are involved.  The resulting map is applied as a final
normalisation step to both queries and pool records, in every country
(the map simply has no effect on tokens it does not contain).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl


def _mine(
    pairs_a: list[str], pairs_b: list[str], min_count: int, min_cond: float, max_diff: int,
    min_rev: float = 0.2, min_len: int = 2,
) -> dict[str, str]:
    """pairs_a[i], pairs_b[i]: normalised token strings of the two sides of a true pair.

    A pair (x on side A, y on side B) becomes an equivalence when y is replaced by x
    in at least `min_cond` of the cases where y is unmatched, and x by y in at least
    `min_rev` of the cases where x is unmatched (near-symmetric, avoids mapping to
    generic tokens or halves of multi-token expansions).
    """
    co: Counter = Counter()
    co2: Counter = Counter()          # (y, "x1 x2"): single side-B token replaced by two side-A tokens
    only_a: Counter = Counter()
    only_b: Counter = Counter()
    freq_a: Counter = Counter()
    for a, b in zip(pairs_a, pairs_b):
        la = a.split()
        ta, tb = set(la), set(b.split())
        freq_a.update(ta)
        da, db = ta - tb, tb - ta
        if not da or not db or len(da) > max_diff or len(db) > max_diff:
            continue
        for x in da:
            only_a[x] += 1
        for y in db:
            only_b[y] += 1
        if len(da) == 1 and len(db) == 1:      # strict 1:1 replacement only
            co[(next(iter(da)), next(iter(db)))] += 1
        if len(db) == 1 and len(da) == 2:
            y = next(iter(db))
            co2[(y, " ".join(t for t in la if t in da))] += 1
    mapping: dict[str, str] = {}
    # 1) two-token expansions (ny -> new york, wv -> west virginia, ap -> andhra pradesh)
    for (y, xs), c in co2.items():
        if c < min_count or y.isdigit() or len(y) < min_len:
            continue
        if c / only_b[y] >= min_cond:
            if y not in mapping or c > co2[(y, mapping[y])]:
                mapping[y] = xs
    # 2) single-token replacements: for each side-B token, its dominant side-A replacement.
    best_for_b: dict[str, tuple[str, int]] = {}
    for (x, y), c in co.items():
        if y in mapping or c < min_count or x.isdigit() or y.isdigit() or len(x) < min_len or len(y) < min_len:
            continue
        if c / only_b[y] < min_cond:      # y is usually replaced by x
            continue
        if c / only_a[x] < min_rev:       # and x is usually replaced by y
            continue
        if y not in best_for_b or c > best_for_b[y][1]:
            best_for_b[y] = (x, c)
    for y, (x, _) in best_for_b.items():
        if x == y or x in mapping:
            continue
        # canonical form = the one Source-1 uses more often
        if freq_a[y] > freq_a[x]:
            mapping[x] = y
        else:
            mapping[y] = x
    # resolve chains a->b->c (single-token values only)
    for k in list(mapping):
        v = mapping[k]
        seen = {k}
        while " " not in v and v in mapping and v not in seen:
            seen.add(v)
            v = mapping[v]
        mapping[k] = v
    return {k: v for k, v in mapping.items() if k != v}


def _mine_aligned(pairs_a: list[str], pairs_b: list[str], min_count: int = 5, min_cond: float = 0.5, min_len: int = 3) -> dict[str, str]:
    """Position-aligned token mining for transliterated names.

    Romanised Indic names keep the word order of the English name, so when both sides
    have the same number of tokens we pair token i with token i.  Returns {romanised: english}.
    """
    co: Counter = Counter()
    cnt_b: Counter = Counter()
    for a, b in zip(pairs_a, pairs_b):
        la, lb = a.split(), b.split()
        if len(la) != len(lb) or not la:
            continue
        for x, y in zip(la, lb):
            if x == y:
                continue
            cnt_b[y] += 1
            co[(x, y)] += 1
    best: dict[str, tuple[str, int]] = {}
    for (x, y), c in co.items():
        if c < min_count or len(y) < min_len or y.isdigit() or x.isdigit():
            continue
        if c / cnt_b[y] < min_cond:
            continue
        if y not in best or c > best[y][1]:
            best[y] = (x, c)
    return {y: x for y, (x, _) in best.items() if x != y}


def learn(s1: pl.DataFrame, pool: pl.DataFrame, truth: dict[str, set[str]]) -> dict[str, dict[str, str]]:
    tp = pl.DataFrame(
        {"s1_id": [k for k, v in truth.items() for _ in v], "cand_id": [x for v in truth.values() for x in v]}
    )
    a = s1.select(["entity_id", "addr_norm", "name_norm"]).rename({"entity_id": "s1_id", "addr_norm": "a1", "name_norm": "n1"})
    b = pool.select(["entity_id", "addr_norm", "name_norm", "name_nonlatin"]).rename(
        {"entity_id": "cand_id", "addr_norm": "a2", "name_norm": "n2"}
    )
    j = tp.join(a, on="s1_id").join(b, on="cand_id")
    addr_map = _mine(j["a1"].to_list(), j["a2"].to_list(), min_count=30, min_cond=0.5, max_diff=2, min_rev=0.2)
    jn = j.filter(pl.col("name_nonlatin") == 1)
    # romanised Indic tokens -> English spelling; applied to non-Latin-script names only
    name_map = _mine_aligned(jn["n1"].to_list(), jn["n2"].to_list())
    return {"addr": addr_map, "name": name_map}


def save(maps: dict, path: Path) -> None:
    path.write_text(json.dumps(maps, ensure_ascii=False, indent=0), encoding="utf-8")


def load(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {"addr": {}, "name": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_map(df: pl.DataFrame, maps: dict[str, dict[str, str]]) -> pl.DataFrame:
    """Rewrite tokens in the normalised columns using the learned maps."""
    def rewrite(col: str, m: dict[str, str]) -> pl.Expr:
        if not m:
            return pl.col(col)
        return (
            pl.col(col).str.split(" ")
            .list.eval(pl.element().replace(m))
            .list.join(" ")
        )
    am, nm = maps.get("addr", {}), maps.get("name", {})
    nonlatin = pl.col("name_nonlatin") == 1
    out = df.with_columns(
        rewrite("addr_norm", am).alias("addr_norm"),
        rewrite("addr_words", am).alias("addr_words"),
        pl.when(nonlatin).then(rewrite("name_norm", nm)).otherwise(pl.col("name_norm")).alias("name_norm"),
        pl.when(nonlatin).then(rewrite("name_core", nm)).otherwise(pl.col("name_core")).alias("name_core"),
    )
    return out.with_columns(pl.col("name_core").str.replace_all(" ", "").alias("name_nospace"))
