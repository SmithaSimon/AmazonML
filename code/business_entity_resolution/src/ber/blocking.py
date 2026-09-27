"""Candidate generation.

Per country, three sparse TF-IDF cosine top-k channels between Source-1 queries
and the Source-2 ∪ Source-3 pool:

  * all  : word unigrams over  name_core + address words + address numbers
  * name : char 3-grams (within word boundaries) over name_core   (typos, domains, transliterations)
  * addr : word unigrams over house number + address words + numbers (renamed businesses)

The union of the channels is the candidate set.  Matches never cross countries
in the training data, so country is a hard block; the code never enumerates
which countries exist.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from . import config


@dataclass
class Channel:
    name: str
    text_expr: pl.Expr
    analyzer: str
    ngram: tuple[int, int]
    topk: int
    max_df: float
    min_df: int = 2
    pool_filter: pl.Expr | None = None   # restrict the pool side to a subset (e.g. domain / non-Latin names)


CHANNELS = [
    Channel(
        "all",
        pl.concat_str([pl.col("name_core"), pl.col("addr_words"), pl.col("addr_nums")], separator=" "),
        "word", (1, 1), config.TOPK_ALL, 0.02,
    ),
    Channel("name", pl.col("name_core"), "word", (1, 1), config.TOPK_NAME, 0.02),
    Channel(
        "addr",
        pl.concat_str([pl.col("house_num"), pl.col("addr_words"), pl.col("addr_nums")], separator=" "),
        "word", (1, 1), config.TOPK_ADDR, 0.02,
    ),
    # character 3-grams on the space-less core name, against only the pool records whose
    # name is a domain (`sjdcement.com`) or in a non-Latin script (romanised): these are the
    # records word-level channels miss, and the subset is small enough for char n-grams.
    Channel(
        "namechar", pl.col("name_nospace"), "char", (3, 3), config.TOPK_NAMECHAR, 0.05,
        pool_filter=(pl.col("is_domain") == 1) | (pl.col("name_nonlatin") == 1),
    ),
]


def _vectorizer(ch: Channel) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=ch.analyzer,
        ngram_range=ch.ngram,
        token_pattern=r"\S+" if ch.analyzer == "word" else None,
        lowercase=False,
        min_df=ch.min_df,
        max_df=ch.max_df,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2",
    )


def topk_channel(
    q_text: list[str], p_text: list[str], ch: Channel, n_threads: int | None = None, log: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (query_idx, pool_idx, score) for the top-k pool rows per query."""
    t0 = time.time()
    vec = _vectorizer(ch)
    P = vec.fit_transform(p_text).tocsr()
    PT = P.T.tocsr()
    del P
    if log:
        print(f"    [{ch.name}] pool matrix {PT.shape[1]} x {PT.shape[0]} nnz={PT.nnz:,} fit {time.time()-t0:.0f}s", flush=True)
    qi_all, pi_all, sc_all = [], [], []
    starts = list(range(0, len(q_text), config.QUERY_CHUNK))
    next_report = 0.05
    for k, start in enumerate(starts, 1):
        Q = vec.transform(q_text[start : start + config.QUERY_CHUNK]).tocsr()
        C = sp_matmul_topn(Q, PT, top_n=ch.topk, threshold=0.05, sort=True, n_threads=n_threads or config.N_JOBS)
        C = C.tocoo()
        qi_all.append(C.row.astype(np.int64) + start)
        pi_all.append(C.col.astype(np.int64))
        sc_all.append(C.data.astype(np.float32))
        if log and k / len(starts) >= next_report:
            print(f"    [{ch.name}] {k/len(starts):4.0%} of queries ({time.time()-t0:.0f}s)", flush=True)
            next_report += 0.05
    if log:
        print(f"    [{ch.name}] done {len(q_text)} queries in {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(qi_all), np.concatenate(pi_all), np.concatenate(sc_all)


def block_country(q: pl.DataFrame, p: pl.DataFrame, channels: list[Channel] | None = None) -> pl.DataFrame:
    """Candidates for one country.  Returns columns s1_id, cand_id, cos_all, cos_name, cos_addr."""
    channels = channels or CHANNELS
    frames = []
    for ch in channels:
        qt = q.select(ch.text_expr.alias("t"))["t"].to_list()
        if ch.pool_filter is not None:
            sub_idx = p.with_row_index("_i").filter(ch.pool_filter)["_i"].to_numpy()
            if len(sub_idx) == 0:
                frames.append(pl.DataFrame({"qi": [], "pi": [], f"cos_{ch.name}": []}, schema={"qi": pl.Int64, "pi": pl.Int64, f"cos_{ch.name}": pl.Float32}))
                continue
            pt = p[sub_idx.tolist()].select(ch.text_expr.alias("t"))["t"].to_list()
            qi, pi, sc = topk_channel(qt, pt, ch)
            pi = sub_idx[pi]
        else:
            pt = p.select(ch.text_expr.alias("t"))["t"].to_list()
            qi, pi, sc = topk_channel(qt, pt, ch)
        frames.append(pl.DataFrame({"qi": qi, "pi": pi, f"cos_{ch.name}": sc}))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on=["qi", "pi"], how="full", coalesce=True)
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in out.columns if c.startswith("cos_")])
    out = out.with_columns(
        pl.Series("s1_id", q["entity_id"].to_numpy()[out["qi"].to_numpy()]),
        pl.Series("cand_id", p["entity_id"].to_numpy()[out["pi"].to_numpy()]),
    ).drop(["qi", "pi"])
    return out


def block_all(q: pl.DataFrame, p: pl.DataFrame) -> pl.DataFrame:
    """Run blocking country by country (open set of country labels)."""
    outs = []
    for country in q["country"].unique().sort().to_list():
        qc = q.filter(pl.col("country") == country)
        pc = p.filter(pl.col("country") == country)
        print(f"  country={country}: {qc.height:,} queries vs {pc.height:,} pool", flush=True)
        if pc.height == 0:
            continue
        outs.append(block_country(qc, pc))
    return pl.concat(outs) if outs else pl.DataFrame()


def candidate_recall(cands: pl.DataFrame, truth: dict[str, set[str]], s1_ids: list[str]) -> dict:
    """Recall of true pairs among candidates + candidate count stats, restricted to s1_ids."""
    have = cands.group_by("s1_id").agg(pl.col("cand_id"))
    have_d = dict(zip(have["s1_id"], have["cand_id"]))
    tp = tot = 0
    n_cand = 0
    per_channel = {c: 0 for c in cands.columns if c.startswith("cos_")}
    cols = list(per_channel)
    ch_sets: dict[str, dict[str, set[str]]] = {}
    for c in cols:
        sub = cands.filter(pl.col(c) > 0).group_by("s1_id").agg(pl.col("cand_id"))
        ch_sets[c] = dict(zip(sub["s1_id"], (set(x) for x in sub["cand_id"])))
    for s1 in s1_ids:
        t = truth.get(s1, set())
        h = set(have_d.get(s1, []))
        n_cand += len(h)
        tot += len(t)
        tp += len(t & h)
        for c in cols:
            per_channel[c] += len(t & ch_sets[c].get(s1, set()))
    res = {"pair_recall": tp / max(1, tot), "cands_per_s1": n_cand / max(1, len(s1_ids))}
    for c in cols:
        res[f"recall_{c}"] = per_channel[c] / max(1, tot)
    return res
