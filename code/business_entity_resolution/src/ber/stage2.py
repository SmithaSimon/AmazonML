"""Second pass: cluster-consistency re-scoring.

The first-pass model scores each (S1, candidate) pair in isolation.  But the
confident matches of an S1 entity describe it better than the S1 record alone:
a renamed / domain-name record gains support if its address agrees with the
entity's other confident matches, an empty-address record with a misspelled
name gains support if its name tokens agree with theirs.

For every pair we build a leave-one-out "profile" of the entity from its other
confident candidates (first-pass prob >= CONF) and measure how much of the
candidate's name / address / number tokens the profile supports, plus
probability context (rank, mass of the other candidates, best competing S1 for
the same record).  A second LightGBM is trained on these features + the
first-pass probability, using out-of-fold first-pass probabilities on the
training entities so the second model never sees in-sample scores.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import config

CONF = 0.9
STAGE2_FEATURES = [
    "p", "p_logit", "p_rank", "p_sum_others", "p_max_other", "n_conf_others",
    "cand_p_max_other_s1", "cand_p_gap",
    "prof_name_cov", "prof_addr_cov", "prof_nums_cov", "prof_house_support", "conf_house_agree",
    "prof_name_n", "prof_addr_n",
]


def _explode_tokens(df: pl.DataFrame, col: str, name: str) -> pl.DataFrame:
    return (
        df.select(["s1_id", "cand_id", pl.col(col).str.split(" ").list.unique().alias("tok")])
        .explode("tok").filter(pl.col("tok") != "").rename({"tok": name})
    )


def cand_top2(prob_paths: list) -> pl.DataFrame:
    """Top-2 first-pass probabilities per candidate record over ALL pairs (streamed from parquet parts)."""
    lf = pl.concat([pl.scan_parquet(str(p)).select(["cand_id", "p"]) for p in prob_paths])
    return (
        lf.group_by("cand_id").agg(pl.col("p").top_k(2).alias("t"))
        .select("cand_id", pl.col("t").list.get(0).alias("max1"), pl.col("t").list.get(1, null_on_oob=True).fill_null(0.0).alias("max2"))
        .collect()
    )


def cluster_features(pairs: pl.DataFrame, pool: pl.DataFrame, top2: pl.DataFrame | None = None) -> pl.DataFrame:
    """pairs: s1_id, cand_id, p (first-pass probability) for ALL pairs of the entities involved
    (whole S1 groups).  top2: cand_id, max1, max2 over the full pair table (None -> computed from `pairs`).

    Returns s1_id, cand_id + STAGE2_FEATURES.
    """
    p = pairs.select(["s1_id", "cand_id", "p"])
    if top2 is None:
        top2 = p.group_by("cand_id").agg(pl.col("p").top_k(2).alias("t")).select(
            "cand_id", pl.col("t").list.get(0).alias("max1"), pl.col("t").list.get(1, null_on_oob=True).fill_null(0.0).alias("max2"))
    # probability context
    p = p.with_columns(
        (pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log().alias("p_logit"),
        pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("p_rank"),
        (pl.col("p").sum().over("s1_id") - pl.col("p")).alias("p_sum_others"),
        (pl.col("p") >= CONF).cast(pl.Int32).alias("is_conf"),
    )
    p = p.with_columns(
        ((pl.col("is_conf").sum().over("s1_id")) - pl.col("is_conf")).alias("n_conf_others"),
        pl.col("p").sort(descending=True).shift(-1).first().over("s1_id").fill_null(0.0).alias("_second"),
    )
    p = p.join(top2, on="cand_id", how="left").with_columns(
        pl.when(pl.col("p") == pl.col("p").max().over("s1_id")).then(pl.col("_second")).otherwise(pl.col("p").max().over("s1_id")).alias("p_max_other"),
        # best probability of the same record under a DIFFERENT S1 (one-to-one competition)
        pl.when(pl.col("p") >= pl.col("max1").fill_null(0.0)).then(pl.col("max2").fill_null(0.0)).otherwise(pl.col("max1").fill_null(0.0)).alias("cand_p_max_other_s1"),
    ).with_columns((pl.col("p") - pl.col("cand_p_max_other_s1")).alias("cand_p_gap"))

    # token profiles from confident members, leave-one-out
    rec = pool.select(["entity_id", "name_core", "addr_words", "addr_nums", "house_num"]).rename({"entity_id": "cand_id"})
    pr = p.select(["s1_id", "cand_id", "is_conf"]).join(rec, on="cand_id", how="left")
    feats = p.select(["s1_id", "cand_id"])
    for col, name in (("name_core", "name"), ("addr_words", "addr"), ("addr_nums", "nums")):
        tok = _explode_tokens(pr, col, "tok").join(pr.select(["s1_id", "cand_id", "is_conf"]), on=["s1_id", "cand_id"])
        # support count of each token among confident members of the entity
        sup = tok.filter(pl.col("is_conf") == 1).group_by(["s1_id", "tok"]).len().rename({"len": "c"})
        tok = tok.join(sup, on=["s1_id", "tok"], how="left").with_columns(
            (pl.col("c").fill_null(0) - pl.col("is_conf")).alias("support")   # leave-one-out
        )
        agg = tok.group_by(["s1_id", "cand_id"]).agg(
            (pl.col("support") >= 1).mean().alias(f"prof_{name}_cov"),
            pl.len().alias(f"prof_{name}_n"),
        )
        feats = feats.join(agg, on=["s1_id", "cand_id"], how="left")
    feats = feats.with_columns([pl.col(c).fill_null(0.0) for c in feats.columns if c.startswith("prof_")])
    feats = feats.drop("prof_nums_n")

    # house-number agreement with the other confident members
    h = pr.select(["s1_id", "cand_id", "is_conf", "house_num"])
    hs = h.filter((pl.col("is_conf") == 1) & (pl.col("house_num") != "")).group_by(["s1_id", "house_num"]).len().rename({"len": "hc"})
    h = h.join(hs, on=["s1_id", "house_num"], how="left").with_columns(
        pl.when(pl.col("house_num") == "").then(0).otherwise(pl.col("hc").fill_null(0) - pl.col("is_conf")).alias("prof_house_support")
    )
    n_conf_h = h.filter((pl.col("is_conf") == 1) & (pl.col("house_num") != "")).group_by("s1_id").len().rename({"len": "nch"})
    h = h.join(n_conf_h, on="s1_id", how="left").with_columns(
        (pl.col("prof_house_support") / (pl.col("nch").fill_null(0) - pl.col("is_conf") * (pl.col("house_num") != "").cast(pl.Int32)).clip(1, None)).alias("conf_house_agree")
    )
    feats = feats.join(h.select(["s1_id", "cand_id", "prof_house_support", "conf_house_agree"]), on=["s1_id", "cand_id"], how="left")
    out = p.select(["s1_id", "cand_id", "p", "p_logit", "p_rank", "p_sum_others", "p_max_other", "n_conf_others",
                    "cand_p_max_other_s1", "cand_p_gap"]).join(feats, on=["s1_id", "cand_id"], how="left")
    return out.select(["s1_id", "cand_id"] + STAGE2_FEATURES).fill_null(0.0)
