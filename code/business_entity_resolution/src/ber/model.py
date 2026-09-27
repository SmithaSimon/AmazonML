"""LightGBM pairwise matcher + decision rule tuned for macro F_0.5."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .features import FEATURE_NAMES
from .metrics import macro_f05

PARAMS = dict(
    objective="binary",
    learning_rate=0.1,
    num_leaves=255,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=config.N_JOBS,
    seed=config.SEED,
)


def train(feats: pl.DataFrame, valid: pl.DataFrame | None = None, rounds: int = 2000,
          feature_names: list[str] | None = None) -> lgb.Booster:
    names = feature_names or FEATURE_NAMES
    X = feats.select(names).to_numpy().astype(np.float32)
    y = feats["label"].to_numpy()
    dtrain = lgb.Dataset(X, y, feature_name=names)
    valid_sets = [dtrain]
    if valid is not None:
        Xv = valid.select(names).to_numpy().astype(np.float32)
        dvalid = lgb.Dataset(Xv, valid["label"].to_numpy(), reference=dtrain)
        valid_sets = [dvalid]
    booster = lgb.train(
        PARAMS, dtrain, num_boost_round=rounds, valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )
    return booster


def predict_proba(booster: lgb.Booster, feats: pl.DataFrame, feature_names: list[str] | None = None) -> np.ndarray:
    names = feature_names or FEATURE_NAMES
    X = feats.select(names).to_numpy().astype(np.float32)
    return booster.predict(X, num_iteration=booster.best_iteration or None)


def _one_to_one(pairs: pl.DataFrame) -> pl.DataFrame:
    """Each S2/S3 record belongs to at most one S1 entity: keep it only for its best-scoring S1."""
    df = pairs.filter(pl.col("prob") == pl.col("prob").max().over("cand_id"))
    return df.unique(subset=["cand_id"], keep="first")


def decide(pairs: pl.DataFrame, threshold: float, one_to_one: bool = True, top_threshold: float | None = None) -> dict[str, list[str]]:
    """Global-threshold rule.  pairs: s1_id, cand_id, prob.  Returns {s1_id: [cand ids]}.

    top_threshold (optional, lower than threshold): if no candidate of an entity reaches
    `threshold`, its single best candidate is still accepted when it reaches `top_threshold`
    (an empty prediction scores 0 on a non-singleton, so a moderately confident best guess
    has positive expected value there).
    """
    df = _one_to_one(pairs) if one_to_one else pairs
    keep = pl.col("prob") >= threshold
    if top_threshold is not None:
        best = pl.col("prob") == pl.col("prob").max().over("s1_id")
        keep = keep | (best & (pl.col("prob") >= top_threshold))
    df = df.filter(keep)
    g = df.sort("prob", descending=True).group_by("s1_id").agg(pl.col("cand_id"))
    return dict(zip(g["s1_id"], g["cand_id"].to_list()))


def decide_expected_f(pairs: pl.DataFrame, scale: float = 1.0, min_prob: float = 0.05) -> dict[str, list[str]]:
    """Per-entity expected-F_0.5 maximisation using the (calibrated) match probabilities.

    For an entity with candidates sorted by probability p_1 >= p_2 >= ..., predicting the
    top-k has plug-in expected F_0.5 = 1.25 * sum_{i<=k} p_i / (0.25 * k + sum_all p_i);
    predicting nothing scores prod(1 - p_i) (the probability the entity is a singleton).
    `scale` optionally tempers the probabilities (p^scale) to correct calibration.
    """
    df = _one_to_one(pairs).filter(pl.col("prob") >= min_prob)
    if scale != 1.0:
        df = df.with_columns(pl.col("prob") ** scale)
    df = df.sort(["s1_id", "prob"], descending=[False, True])
    df = df.with_columns(
        pl.col("prob").cum_sum().over("s1_id").alias("cum"),
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
        pl.col("prob").sum().over("s1_id").alias("tot"),
        (1 - pl.col("prob")).log().sum().over("s1_id").exp().alias("p_empty"),
    )
    df = df.with_columns((1.25 * pl.col("cum") / (0.25 * pl.col("k") + pl.col("tot"))).alias("ef"))
    best = df.group_by("s1_id").agg(pl.col("ef").max().alias("ef_max"), pl.col("p_empty").first())
    df = df.join(best, on="s1_id", how="left", suffix="_r")
    # choose k* = argmax ef, and only if it beats predicting empty
    chosen = df.filter((pl.col("ef") == pl.col("ef_max")) & (pl.col("ef_max") > pl.col("p_empty")))
    kstar = chosen.group_by("s1_id").agg(pl.col("k").min().alias("kstar"))
    out = df.join(kstar, on="s1_id", how="inner").filter(pl.col("k") <= pl.col("kstar"))
    g = out.group_by("s1_id").agg(pl.col("cand_id"))
    return dict(zip(g["s1_id"], g["cand_id"].to_list()))


def tune_threshold(pairs: pl.DataFrame, truth: dict[str, set[str]], s1_ids: list[str]) -> tuple[dict, dict]:
    """Try the decision rules on validation entities; return (rule, metrics) of the best."""
    sub_truth = {s: truth[s] for s in s1_ids}
    best_rule, best_m = {"rule": "threshold", "threshold": 0.5}, {"macro_f05": -1}
    for t in np.arange(0.30, 0.96, 0.02):
        m = macro_f05(decide(pairs, float(t)), sub_truth)
        if m["macro_f05"] > best_m["macro_f05"]:
            best_rule, best_m = {"rule": "threshold", "threshold": float(t)}, m
    t_hi = best_rule["threshold"]
    for t_lo in np.arange(0.10, t_hi, 0.04):
        m = macro_f05(decide(pairs, t_hi, top_threshold=float(t_lo)), sub_truth)
        if m["macro_f05"] > best_m["macro_f05"]:
            best_rule, best_m = {"rule": "threshold", "threshold": t_hi, "top_threshold": float(t_lo)}, m
    for scale in (0.8, 1.0, 1.25, 1.5, 2.0):
        m = macro_f05(decide_expected_f(pairs, scale=scale), sub_truth)
        print(f"  expected-F rule scale={scale}: {m['macro_f05']:.5f}")
        if m["macro_f05"] > best_m["macro_f05"]:
            best_rule, best_m = {"rule": "expected_f", "scale": scale}, m
    return best_rule, best_m


def apply_rule(pairs: pl.DataFrame, rule: dict) -> dict[str, list[str]]:
    if rule["rule"] == "expected_f":
        return decide_expected_f(pairs, scale=rule.get("scale", 1.0))
    return decide(pairs, rule["threshold"], top_threshold=rule.get("top_threshold"))


def save(booster: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
