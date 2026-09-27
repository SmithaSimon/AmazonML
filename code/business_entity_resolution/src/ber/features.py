"""Pairwise features for (Source-1 entity, candidate) pairs.

All features are similarity-type quantities (no country identity), so a model
trained on the training countries transfers to unseen ones.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import math

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from . import config

# per-worker IDF tables (token -> idf), loaded once by the pool initializer
_IDF_NAME: dict[str, float] = {}
_IDF_ADDR: dict[str, float] = {}


def _init_idf(path: str) -> None:
    global _IDF_NAME, _IDF_ADDR
    t = pl.read_parquet(path)
    _IDF_NAME = dict(zip(t.filter(pl.col("kind") == "name")["token"], t.filter(pl.col("kind") == "name")["idf"]))
    _IDF_ADDR = dict(zip(t.filter(pl.col("kind") == "addr")["token"], t.filter(pl.col("kind") == "addr")["idf"]))


def _widf(t1: set, t2: set, idf: dict[str, float], default: float) -> tuple[float, float, float]:
    """IDF-weighted Jaccard, max idf of a shared token, max idf of an unshared token."""
    if not t1 and not t2:
        return 0.0, 0.0, 0.0
    shared = t1 & t2
    union = t1 | t2
    ws = sum(idf.get(t, default) for t in shared)
    wu = sum(idf.get(t, default) for t in union)
    mx_s = max((idf.get(t, default) for t in shared), default=0.0)
    mx_u = max((idf.get(t, default) for t in union - shared), default=0.0)
    return (ws / wu if wu else 0.0), mx_s, mx_u

REC_COLS = [
    "entity_id", "name_norm", "name_core", "name_nospace", "is_domain", "name_nonlatin",
    "addr_norm", "addr_words", "addr_nums", "house_num", "addr_empty",
]


def _jacc(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _pair_features(rows: list[tuple]) -> dict[str, list]:
    """rows: (n1_norm, n1_core, n1_ns, d1, nl1, a1_norm, a1_words, a1_nums, h1, e1,
              n2_norm, n2_core, n2_ns, d2, nl2, a2_norm, a2_words, a2_nums, h2, e2)"""
    out: dict[str, list] = {k: [] for k in FEATURE_NAMES_PAIR}
    ap = out.__getitem__
    for (n1n, n1c, n1s, d1, nl1, a1n, a1w, a1u, h1, e1,
         n2n, n2c, n2s, d2, nl2, a2n, a2w, a2u, h2, e2) in rows:
        t1, t2 = set(n1c.split()), set(n2c.split())
        inter = len(t1 & t2)
        ap("name_jacc").append(_jacc(t1, t2))
        ap("name_inter").append(inter)
        ap("name_cov1").append(inter / max(1, len(t1)))
        ap("name_cov2").append(inter / max(1, len(t2)))
        ap("name_len1").append(len(t1))
        ap("name_len2").append(len(t2))
        ap("name_exact").append(int(n1c == n2c))
        ap("name_ratio").append(fuzz.ratio(n1c, n2c))
        ap("name_tsr").append(fuzz.token_set_ratio(n1c, n2c))
        ap("name_tsort").append(fuzz.token_sort_ratio(n1c, n2c))
        ap("name_partial").append(fuzz.partial_ratio(n1c, n2c) if (n1c and n2c) else 0)
        ap("name_ns_ratio").append(fuzz.ratio(n1s, n2s))
        ap("name_ns_partial").append(fuzz.partial_ratio(n1s, n2s) if (n1s and n2s) else 0)
        ap("name_jw").append(JaroWinkler.similarity(n1c, n2c))
        f1 = n1c.split(" ", 1)[0] if n1c else ""
        f2 = n2c.split(" ", 1)[0] if n2c else ""
        ap("name_first_eq").append(int(bool(f1) and f1 == f2))
        ap("name_first_in").append(int(bool(f2) and f2 in t1))
        suf1 = set(n1n.split()) - t1
        suf2 = set(n2n.split()) - t2
        ap("suffix_jacc").append(_jacc(suf1, suf2) if (suf1 or suf2) else -1.0)
        ap("cand_domain").append(d2)
        ap("cand_nonlatin").append(nl2)
        ap("s1_domain").append(d1)
        # address
        w1, w2 = set(a1w.split()), set(a2w.split())
        u1, u2 = set(a1u.split()), set(a2u.split())
        ap("addr_empty").append(e2)
        ap("addr_words_jacc").append(_jacc(w1, w2))
        ap("addr_words_inter").append(len(w1 & w2))
        ap("addr_words_cov1").append(len(w1 & w2) / max(1, len(w1)))
        ap("addr_words_cov2").append(len(w1 & w2) / max(1, len(w2)))
        ap("addr_nums_jacc").append(_jacc(u1, u2))
        ap("addr_nums_inter").append(len(u1 & u2))
        ap("addr_nums_n1").append(len(u1))
        ap("addr_nums_n2").append(len(u2))
        ap("house_eq").append(int(bool(h1) and h1 == h2))
        ap("house_both").append(int(bool(h1) and bool(h2)))
        ap("house_in_nums").append(int(bool(h1) and h1 in u2))
        ap("addr_tsr").append(fuzz.token_set_ratio(a1n, a2n) if (a1n and a2n) else 0)
        ap("addr_ratio").append(fuzz.ratio(a1n, a2n) if (a1n and a2n) else 0)
        ap("addr_partial").append(fuzz.partial_ratio(a1n, a2n) if (a1n and a2n) else 0)
        ap("addr_len2").append(len(a2n.split()))
        # --- finer house-number / number features (hard negatives differ by a digit) ---
        both = bool(h1) and bool(h2)
        lev = Levenshtein.distance(h1, h2) if both else -1
        ap("house_lev").append(lev)
        ap("house_one_edit").append(int(both and lev == 1))
        ap("house_prefix").append(int(both and h1 != h2 and (h1.endswith(h2) or h2.endswith(h1) or h1.startswith(h2) or h2.startswith(h1))))
        ap("house_neq").append(int(both and h1 != h2))
        s1n, s2n = u1 - {h1}, u2 - {h2}
        ap("sec_nums_inter").append(len(s1n & s2n))
        ap("sec_nums_jacc").append(_jacc(s1n, s2n))
        near = 0
        for x in u1 - u2:
            if len(x) >= 2:
                for y in u2 - u1:
                    if len(y) >= 2 and Levenshtein.distance(x, y) <= 1:
                        near += 1
                        break
        ap("nums_near").append(near)
        ap("nums_ratio").append(fuzz.ratio(a1u, a2u) if (a1u and a2u) else 0)
        # --- IDF-weighted overlaps: a shared rare token is strong evidence, an unshared rare token counts against ---
        wj, mxs, mxu = _widf(t1, t2, _IDF_NAME, 12.0)
        ap("name_widf_jacc").append(wj)
        ap("name_shared_maxidf").append(mxs)
        ap("name_unshared_maxidf").append(mxu)
        ap("name2_idf_sum").append(sum(_IDF_NAME.get(t, 12.0) for t in t2))
        wj, mxs, mxu = _widf(w1, w2, _IDF_ADDR, 12.0)
        ap("addr_widf_jacc").append(wj)
        ap("addr_shared_maxidf").append(mxs)
        ap("addr_unshared_maxidf").append(mxu)
    return out


FEATURE_NAMES_PAIR = [
    "name_jacc", "name_inter", "name_cov1", "name_cov2", "name_len1", "name_len2", "name_exact",
    "name_ratio", "name_tsr", "name_tsort", "name_partial", "name_ns_ratio", "name_ns_partial", "name_jw",
    "name_first_eq", "name_first_in", "suffix_jacc", "cand_domain", "cand_nonlatin", "s1_domain",
    "addr_empty", "addr_words_jacc", "addr_words_inter", "addr_words_cov1", "addr_words_cov2",
    "addr_nums_jacc", "addr_nums_inter", "addr_nums_n1", "addr_nums_n2", "house_eq", "house_both",
    "house_in_nums", "addr_tsr", "addr_ratio", "addr_partial", "addr_len2",
    "house_lev", "house_one_edit", "house_prefix", "house_neq", "sec_nums_inter", "sec_nums_jacc",
    "nums_near", "nums_ratio",
    "name_widf_jacc", "name_shared_maxidf", "name_unshared_maxidf", "name2_idf_sum",
    "addr_widf_jacc", "addr_shared_maxidf", "addr_unshared_maxidf",
]
FEATURE_NAMES_BLOCK = ["cos_all", "cos_name", "cos_addr", "cos_namechar", "cos_emb"]
FEATURE_NAMES_CTX = [
    "n_cands", "score0", "rank0", "margin0", "best0_gap",
    "s1_addr_dup", "s1_name_dup", "cand_name_n_s1", "s1_name_addr_key_dup", "src_is_s3",
    # candidate-competition context over the FULL candidate table (all S1 entities are queries
    # in both train and test, so these are distributed identically):
    "cand_n_s1", "cand_best_cos", "cand_gap_cos", "cand_rank_cos",
    # fuzzy name ambiguity: how many S1 entities retrieved this candidate with a high name cosine,
    # and how many pool records carry (nearly) the S1 entity's name
    "cand_n_s1_name_hi", "s1_n_name_hi",
]


def add_candidate_context(cands: pl.DataFrame) -> pl.DataFrame:
    """How many S1 entities compete for each candidate, and how this pair ranks among them."""
    return cands.with_columns(
        pl.len().over("cand_id").alias("cand_n_s1"),
        pl.col("cos_all").max().over("cand_id").alias("cand_best_cos"),
        pl.col("cos_all").rank("ordinal", descending=True).over("cand_id").alias("cand_rank_cos"),
        (pl.col("cos_name") >= 0.7).sum().over("cand_id").alias("cand_n_s1_name_hi"),
        (pl.col("cos_name") >= 0.9).sum().over("s1_id").alias("s1_n_name_hi"),
    ).with_columns((pl.col("cand_best_cos") - pl.col("cos_all")).alias("cand_gap_cos"))


FEATURE_NAMES = FEATURE_NAMES_BLOCK + FEATURE_NAMES_PAIR + FEATURE_NAMES_CTX


def idf_table(s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Document-frequency based IDF of name-core and address-word tokens over all records of a split."""
    out = []
    for kind, col in (("name", "name_core"), ("addr", "addr_words")):
        docs = pl.concat([s1.select(pl.col(col).alias("t")), pool.select(pl.col(col).alias("t"))])
        n = docs.height
        df = (
            docs.with_columns(pl.col("t").str.split(" ").list.unique())
            .explode("t").filter(pl.col("t") != "")
            .group_by("t").len()
        )
        out.append(df.select(
            pl.lit(kind).alias("kind"), pl.col("t").alias("token"),
            (math.log(n) - (pl.col("len") + 1).log()).cast(pl.Float32).alias("idf"),
        ))
    return pl.concat(out)


def _chunk_rows(pairs: pl.DataFrame) -> list[tuple]:
    cols = [
        "n1_norm", "n1_core", "n1_ns", "d1", "nl1", "a1_norm", "a1_words", "a1_nums", "h1", "e1",
        "n2_norm", "n2_core", "n2_ns", "d2", "nl2", "a2_norm", "a2_words", "a2_nums", "h2", "e2",
    ]
    return list(pairs.select(cols).iter_rows())


def _worker(args: tuple) -> str:
    in_path, out_path = args
    rows = _chunk_rows(pl.read_parquet(in_path))
    feats = _pair_features(rows)
    pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32, strict=False) for k, v in feats.items()}).write_parquet(out_path)
    Path(in_path).unlink()
    return out_path


def build_features(cands: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Join record fields onto candidate pairs and compute all features.

    Returns cands columns + FEATURE_NAMES.
    """
    ren1 = {
        "name_norm": "n1_norm", "name_core": "n1_core", "name_nospace": "n1_ns", "is_domain": "d1",
        "name_nonlatin": "nl1", "addr_norm": "a1_norm", "addr_words": "a1_words", "addr_nums": "a1_nums",
        "house_num": "h1", "addr_empty": "e1",
    }
    ren2 = {k: v.replace("1", "2") for k, v in ren1.items()}
    s1r = s1.select(REC_COLS).rename(ren1).rename({"entity_id": "s1_id"})
    pr = pool.select(REC_COLS).rename(ren2).rename({"entity_id": "cand_id"})
    for c in FEATURE_NAMES_BLOCK:            # optional channels (e.g. cos_emb) default to 0
        if c not in cands.columns:
            cands = cands.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(c))
    df = cands.join(s1r, on="s1_id", how="left").join(pr, on="cand_id", how="left")

    # Ambiguity context computed from the FULL Source-1 table (identical in train and test):
    #   s1_addr_dup          how many S1 entities (same country) share this S1's normalised address
    #   s1_name_dup          how many S1 entities (same country) share this S1's core name
    #   s1_name_addr_key_dup how many share core name + house number
    #   cand_name_n_s1       how many S1 entities (same country) have exactly the candidate's core name
    s1k = s1.select(["entity_id", "country", "name_core", "addr_norm", "house_num"])
    dup_addr = s1k.group_by(["country", "addr_norm"]).len().rename({"len": "s1_addr_dup"})
    dup_name = s1k.group_by(["country", "name_core"]).len().rename({"len": "s1_name_dup"})
    dup_key = s1k.group_by(["country", "name_core", "house_num"]).len().rename({"len": "s1_name_addr_key_dup"})
    ctx = (
        s1k.join(dup_addr, on=["country", "addr_norm"], how="left")
        .join(dup_name, on=["country", "name_core"], how="left")
        .join(dup_key, on=["country", "name_core", "house_num"], how="left")
        .select(["entity_id", "s1_addr_dup", "s1_name_dup", "s1_name_addr_key_dup"])
        .rename({"entity_id": "s1_id"})
    )
    df = df.join(ctx, on="s1_id", how="left")
    cand_names = pool.select(["entity_id", "country", "name_core"]).rename({"entity_id": "cand_id"})
    cand_names = cand_names.join(dup_name.rename({"s1_name_dup": "cand_name_n_s1"}), on=["country", "name_core"], how="left")
    df = df.join(cand_names.select(["cand_id", "cand_name_n_s1"]), on="cand_id", how="left")
    df = df.with_columns(
        pl.col("cand_name_n_s1").fill_null(0),
        pl.col("cand_id").str.starts_with("S3-").cast(pl.Int8).alias("src_is_s3"),
    )

    # pairwise string features (multiprocess)
    n = df.height
    chunk = 100_000
    tmp = config.WORK_DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    in_cols = [
        "n1_norm", "n1_core", "n1_ns", "d1", "nl1", "a1_norm", "a1_words", "a1_nums", "h1", "e1",
        "n2_norm", "n2_core", "n2_ns", "d2", "nl2", "a2_norm", "a2_words", "a2_nums", "h2", "e2",
    ]
    jobs = []
    for i in range(0, n, chunk):
        in_path = str(tmp / f"pairs_{i//chunk:06d}.parquet")
        df.slice(i, chunk).select(in_cols).write_parquet(in_path)
        jobs.append((in_path, str(tmp / f"feat_{i//chunk:06d}.parquet")))
    idf_path = str(tmp / "idf.parquet")
    if not Path(idf_path).exists():
        idf_table(s1, pool).write_parquet(idf_path)
    with ProcessPoolExecutor(max_workers=config.N_JOBS, initializer=_init_idf, initargs=(idf_path,)) as ex:
        paths = list(ex.map(_worker, jobs))
    feats = pl.concat([pl.read_parquet(p) for p in paths]) if paths else pl.DataFrame({k: [] for k in FEATURE_NAMES_PAIR})
    for p in paths:
        Path(p).unlink()
    df = pl.concat([df, feats], how="horizontal")

    # context features from a cheap combined score
    df = df.with_columns(
        (
            0.4 * pl.col("name_tsr") / 100 + 0.3 * pl.col("addr_tsr") / 100
            + 0.3 * pl.col("cos_all") + 0.1 * pl.col("house_eq")
        ).alias("score0")
    )
    df = df.with_columns(
        pl.len().over("s1_id").alias("n_cands"),
        pl.col("score0").rank("ordinal", descending=True).over("s1_id").alias("rank0"),
        (pl.col("score0").max().over("s1_id") - pl.col("score0")).alias("margin0"),
        # gap between best and second best for this s1
        (pl.col("score0").max().over("s1_id") - pl.col("score0").sort(descending=True).shift(-1).first().over("s1_id"))
        .fill_null(1.0).alias("best0_gap"),
    )
    drop = [c for c in df.columns if c[:2] in ("n1", "n2", "a1", "a2", "d1", "d2", "h1", "h2", "e1", "e2") or c in ("nl1", "nl2")]
    return df.drop(drop)
