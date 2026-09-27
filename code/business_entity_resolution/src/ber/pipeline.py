"""Command-line entry point.  Run from code/business_entity_resolution/src.

    python -m ber.pipeline prepare   --split train
    python -m ber.pipeline prepare   --split test
    python -m ber.pipeline block-eval --n 50000            # blocking recall on a train sample (dev)
    python -m ber.pipeline block     --split train --n 400000
    python -m ber.pipeline block     --split test
    python -m ber.pipeline featurize --split train
    python -m ber.pipeline featurize --split test
    python -m ber.pipeline train
    python -m ber.pipeline predict
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from . import config
from .io import read_ground_truth, read_split, write_id_list_file
from .metrics import macro_f05
from .normalize import normalise_frame


def _norm_chunk(args: tuple) -> str:
    """Worker: normalise one chunk and write it to disk (returning big frames over pipes is fragile on Windows)."""
    df, out_path = args
    normalise_frame(df).write_parquet(out_path)
    return out_path


def prepare(split: str) -> None:
    t0 = time.time()
    s1, pool = read_split(config.DATA_DIR, split)
    print(f"read {split}: s1={s1.height:,} pool={pool.height:,} ({time.time()-t0:.0f}s)", flush=True)
    tmp = config.WORK_DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, df in (("s1", s1), ("pool", pool)):
        n = df.height
        chunk = 100_000
        jobs = [(df.slice(i, chunk), str(tmp / f"{split}_{name}_{i//chunk:05d}.parquet")) for i in range(0, n, chunk)]
        with ProcessPoolExecutor(max_workers=config.N_JOBS) as ex:
            paths = list(ex.map(_norm_chunk, jobs))
        out = pl.concat([pl.read_parquet(p) for p in paths])
        path = config.WORK_DIR / f"{split}_{name}.parquet"
        out.write_parquet(path)
        for p in paths:
            Path(p).unlink()
        print(f"wrote {path} rows={out.height:,} ({time.time()-t0:.0f}s)", flush=True)


def load_prepared(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    s1 = pl.read_parquet(config.WORK_DIR / f"{split}_s1.parquet")
    pool = pl.read_parquet(config.WORK_DIR / f"{split}_pool.parquet")
    return s1, pool


def _truth() -> dict[str, set[str]]:
    return read_ground_truth(config.DATA_DIR / "train" / "train_ground_truth.tsv")


def _sample_s1(s1: pl.DataFrame, n: int | None) -> pl.DataFrame:
    if n is None or n >= s1.height:
        return s1
    rng = np.random.default_rng(config.SEED)
    idx = np.sort(rng.choice(s1.height, size=n, replace=False))
    return s1[idx.tolist()]


def equiv_step() -> None:
    """Learn token equivalences from the training pairs and apply them to every prepared parquet."""
    from . import equiv

    s1, pool = load_prepared("train")
    maps = equiv.learn(s1, pool, _truth())
    equiv.save(maps, config.WORK_DIR / "equiv.json")
    print(f"learned {len(maps['addr'])} address and {len(maps['name'])} name token equivalences", flush=True)
    del s1, pool
    for split in ("train", "test"):
        for name in ("s1", "pool"):
            path = config.WORK_DIR / f"{split}_{name}.parquet"
            if path.exists():
                equiv.apply_map(pl.read_parquet(path), maps).write_parquet(path)
                print(f"applied to {path}", flush=True)


def block_eval(n: int, country: str | None) -> None:
    from .blocking import block_all, candidate_recall

    s1, pool = load_prepared("train")
    truth = _truth()
    if country:
        s1 = s1.filter(pl.col("country") == country)
    q = _sample_s1(s1, n)
    t0 = time.time()
    cands = block_all(q, pool)
    print(f"blocking took {time.time()-t0:.0f}s, {cands.height:,} candidate pairs", flush=True)
    print(candidate_recall(cands, truth, q["entity_id"].to_list()))
    cands.write_parquet(config.WORK_DIR / "dev_cands.parquet")


def block(split: str, n: int | None) -> None:
    from .blocking import block_all, candidate_recall

    s1, pool = load_prepared(split)
    q = _sample_s1(s1, n)
    t0 = time.time()
    cands = block_all(q, pool)
    print(f"blocking took {time.time()-t0:.0f}s, {cands.height:,} candidate pairs for {q.height:,} S1", flush=True)
    cands.write_parquet(config.WORK_DIR / f"{split}_cands.parquet")
    q.select("entity_id").write_parquet(config.WORK_DIR / f"{split}_queries.parquet")
    if split == "train":
        print(candidate_recall(cands, _truth(), q["entity_id"].to_list()))


PAIRS_PER_PART = 6_000_000


def _feat_parts(split: str) -> list[Path]:
    parts = sorted(config.WORK_DIR.glob(f"{split}_feats_*.parquet"))
    single = config.WORK_DIR / f"{split}_feats.parquet"
    return parts or ([single] if single.exists() else [])


def featurize(split: str) -> None:
    """Compute features in parts split at S1-entity boundaries (memory-bounded).

    Writes work/<split>_feats_NN.parquet.  Context features only depend on the
    full S1 table and on the candidates of the same S1 entity, so splitting by
    S1 entity is exact.
    """
    from .features import add_candidate_context, build_features

    s1, pool = load_prepared(split)
    cands = add_candidate_context(pl.read_parquet(config.WORK_DIR / f"{split}_cands.parquet")).sort("s1_id")
    lab = None
    if split == "train":
        truth = _truth()
        lab = pl.DataFrame(
            {"s1_id": [k for k, v in truth.items() for _ in v], "cand_id": [x for v in truth.values() for x in v], "label": 1}
        ).with_columns(pl.col("label").cast(pl.Int8))
    for old in _feat_parts(split):
        old.unlink()
    idf_cache = config.WORK_DIR / "tmp" / "idf.parquet"   # IDF is computed per split (test includes France)
    if idf_cache.exists():
        idf_cache.unlink()
    # part boundaries: cumulative pair count per S1 entity
    sizes = cands.group_by("s1_id", maintain_order=True).len()
    cum = sizes["len"].cum_sum()
    part_of_s1 = (cum // PAIRS_PER_PART).cast(pl.Int32)
    cands = cands.join(pl.DataFrame({"s1_id": sizes["s1_id"], "part": part_of_s1}), on="s1_id", how="left")
    t0 = time.time()
    n_pos = n_all = 0
    for k, part in enumerate(sorted(cands["part"].unique().to_list())):
        c = cands.filter(pl.col("part") == part).drop("part")
        feats = build_features(c, s1, pool)
        if lab is not None:
            feats = feats.join(lab, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
            n_pos += int(feats["label"].sum())
        n_all += feats.height
        feats.write_parquet(config.WORK_DIR / f"{split}_feats_{k:02d}.parquet")
        print(f"  part {k}: {feats.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
    if lab is not None:
        print("positives:", n_pos, "of", n_all)
    print(f"features for {n_all:,} pairs in {time.time()-t0:.0f}s", flush=True)


def train(n_train: int = config.TRAIN_S1, n_val: int = config.VAL_S1) -> None:
    """Fit on the pairs of `n_train` random S1 entities, tune the decision rule on `n_val` others.

    Feature parts are scanned lazily and filtered, so memory stays bounded even when the
    candidate table covers every training S1 entity.
    """
    from . import model as M

    queries = pl.read_parquet(config.WORK_DIR / "train_queries.parquet")["entity_id"].to_list()
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(len(queries))
    n_val = min(n_val, len(queries) // 4)
    val_ids = [queries[i] for i in perm[:n_val]]
    tr_ids = [queries[i] for i in perm[n_val : n_val + n_train]]
    val_s = pl.Series("s1_id", val_ids)
    tr_s = pl.Series("s1_id", tr_ids)
    ftr = pl.concat([pl.scan_parquet(p).filter(pl.col("s1_id").is_in(tr_s)).collect() for p in _feat_parts("train")])
    fva = pl.concat([pl.scan_parquet(p).filter(pl.col("s1_id").is_in(val_s)).collect() for p in _feat_parts("train")])
    print(f"train pairs {ftr.height:,} (pos {ftr['label'].sum():,}) from {len(tr_ids):,} S1;  val pairs {fva.height:,} from {len(val_ids):,} S1", flush=True)
    t0 = time.time()
    booster = M.train(ftr, fva)
    print(f"trained {booster.best_iteration} rounds in {time.time()-t0:.0f}s", flush=True)
    imp = sorted(zip(booster.feature_name(), booster.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(k, int(v)) for k, v in imp[:15]])

    truth = _truth()
    fva = fva.with_columns(pl.Series("prob", M.predict_proba(booster, fva)))
    pairs = fva.select(["s1_id", "cand_id", "prob"])
    val_list = sorted(val_ids)
    rule, met = M.tune_threshold(pairs, truth, val_list)
    print(f"best rule {rule} -> {met}")
    # recall ceiling of the candidate set on the validation entities
    pos = fva.filter(pl.col("label") == 1).group_by("s1_id").agg(pl.col("cand_id"))
    ceil = macro_f05(dict(zip(pos["s1_id"], pos["cand_id"].to_list())), {s: truth[s] for s in val_list})
    print(f"oracle (perfect classifier on candidates): {ceil}")
    M.save(booster, config.WORK_DIR / "model.txt")
    (config.WORK_DIR / "threshold.json").write_text(json.dumps({"rule": rule, "val": met}))
    pairs.write_parquet(config.WORK_DIR / "val_pairs.parquet")


def _split_ids() -> tuple[list[str], list[str]]:
    """Deterministic train / validation S1 ids (same as train())."""
    queries = pl.read_parquet(config.WORK_DIR / "train_queries.parquet")["entity_id"].to_list()
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(len(queries))
    n_val = min(config.VAL_S1, len(queries) // 4)
    return [queries[i] for i in perm[n_val : n_val + config.TRAIN_S1]], [queries[i] for i in perm[:n_val]]


def train2(n_parts: int | None = None) -> None:
    """Second pass (cluster consistency).  Needs train feature parts and model.txt from train().

    Stage-2 is trained on the fold-A entities (out-of-fold probabilities from the model fitted
    on fold B); fold-B entities get their probabilities from the fold-A model, validation and
    remaining entities from the full first-pass model, so every probability the second model
    ever sees is out-of-sample.  `n_parts` limits the feature parts used (smoke tests).
    """
    from . import model as M
    from .features import FEATURE_NAMES
    from .stage2 import STAGE2_FEATURES, cand_top2, cluster_features, prune

    tr_ids, val_ids = _split_ids()
    tr_s, val_s = pl.Series("s1_id", tr_ids), pl.Series("s1_id", val_ids)
    parts = _feat_parts("train")[:n_parts] if n_parts else _feat_parts("train")
    ftr = pl.concat([pl.scan_parquet(p).filter(pl.col("s1_id").is_in(tr_s)).collect() for p in parts])
    # 1. two cross-fitted first-pass models -> out-of-fold probabilities on the training entities
    rng = np.random.default_rng(config.SEED + 1)
    fold_a = set(np.array(tr_ids)[rng.random(len(tr_ids)) < 0.5].tolist())
    in_a = ftr["s1_id"].is_in(pl.Series(list(fold_a)))
    fa, fb = ftr.filter(in_a), ftr.filter(~in_a)
    t0 = time.time()
    model_a = M.train(fa, fb, rounds=1500)          # trained on A, early-stopped on B
    model_b = M.train(fb, fa, rounds=1500)
    print(f"cross-fit models: {model_a.best_iteration} / {model_b.best_iteration} rounds in {time.time()-t0:.0f}s", flush=True)
    full = M.load(config.WORK_DIR / "model.txt")
    del ftr, fa, fb
    # 2. first-pass probabilities for EVERY training pair: OOF inside the train subset, full model elsewhere
    prob_paths = []
    for k, p in enumerate(parts):
        f = pl.read_parquet(p)
        is_tr_a = f["s1_id"].is_in(pl.Series(list(fold_a))).to_numpy()
        is_tr = f["s1_id"].is_in(tr_s).to_numpy()
        prob = M.predict_proba(full, f)
        pa = M.predict_proba(model_b, f)   # fold A entities are out-of-sample for model_b
        pb = M.predict_proba(model_a, f)
        prob = np.where(is_tr & is_tr_a, pa, np.where(is_tr, pb, prob))
        out = f.select(["s1_id", "cand_id", "label"]).with_columns(pl.Series("p", prob.astype(np.float32)))
        path = config.WORK_DIR / f"train_prob_{k:02d}.parquet"
        out.write_parquet(path)
        prob_paths.append(path)
        print(f"  probs part {k}", flush=True)
    top2 = cand_top2(prob_paths)
    # 3. stage-2 features per part, keep train-subset + validation pairs
    s1, pool = load_prepared("train")
    fold_a_s = pl.Series("s1_id", list(fold_a))
    rows_tr, rows_va = [], []
    n_before = n_after = 0
    for k, (p, pp) in enumerate(zip(parts, prob_paths)):
        f = pl.read_parquet(p)
        pr = pl.read_parquet(pp)
        n_before += pr.height
        pr = prune(pr)                      # last filtering stage: same rule at train and test time
        n_after += pr.height
        cf = cluster_features(pr, pool, top2)
        f = f.join(cf, on=["s1_id", "cand_id"], how="inner")
        rows_tr.append(f.filter(pl.col("s1_id").is_in(fold_a_s)))     # OOF probabilities (model_b)
        rows_va.append(f.filter(pl.col("s1_id").is_in(val_s)))
        print(f"  stage-2 features part {k}", flush=True)
    n_s1 = pl.read_parquet(config.WORK_DIR / "train_queries.parquet").height
    print(f"pruning kept {n_after:,} of {n_before:,} pairs ({n_after/n_s1:.2f} per S1 entity)", flush=True)
    ftr2, fva2 = pl.concat(rows_tr), pl.concat(rows_va)
    feats2 = FEATURE_NAMES + STAGE2_FEATURES
    print(f"stage-2 train pairs {ftr2.height:,}  val pairs {fva2.height:,}", flush=True)
    t0 = time.time()
    booster2 = M.train(ftr2, fva2, rounds=2000, feature_names=feats2)
    print(f"stage-2 trained {booster2.best_iteration} rounds in {time.time()-t0:.0f}s", flush=True)
    imp = sorted(zip(booster2.feature_name(), booster2.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(k, int(v)) for k, v in imp[:15]])
    truth = _truth()
    pairs1 = fva2.select(["s1_id", "cand_id", pl.col("p").alias("prob")])
    r1, m1 = M.tune_threshold(pairs1, truth, sorted(val_ids))
    print(f"stage-1 on val: {r1} -> {m1}")
    pairs2 = fva2.select(["s1_id", "cand_id"]).with_columns(pl.Series("prob", M.predict_proba(booster2, fva2, feats2)))
    r2, m2 = M.tune_threshold(pairs2, truth, sorted(val_ids))
    print(f"stage-2 on val: {r2} -> {m2}")
    M.save(booster2, config.WORK_DIR / "model2.txt")
    (config.WORK_DIR / "threshold2.json").write_text(json.dumps({"rule": r2, "val": m2, "stage1_val": m1}))
    pairs2.write_parquet(config.WORK_DIR / "val_pairs2.parquet")


def predict2() -> None:
    """Apply the second pass on test: needs test feature parts, model.txt, model2.txt."""
    from . import model as M
    from .features import FEATURE_NAMES
    from .stage2 import STAGE2_FEATURES, cand_top2, cluster_features, prune

    full = M.load(config.WORK_DIR / "model.txt")
    booster2 = M.load(config.WORK_DIR / "model2.txt")
    rule = json.loads((config.WORK_DIR / "threshold2.json").read_text())["rule"]
    feats2 = FEATURE_NAMES + STAGE2_FEATURES
    parts = _feat_parts("test")
    prob_paths = []
    for k, p in enumerate(parts):
        f = pl.read_parquet(p)
        path = config.WORK_DIR / f"test_prob_{k:02d}.parquet"
        f.select(["s1_id", "cand_id"]).with_columns(pl.Series("p", M.predict_proba(full, f).astype(np.float32))).write_parquet(path)
        prob_paths.append(path)
    top2 = cand_top2(prob_paths)
    _, pool = load_prepared("test")
    outs = []
    n_before = 0
    for k, (p, pp) in enumerate(zip(parts, prob_paths)):
        pr = pl.read_parquet(pp)
        n_before += pr.height
        pr = prune(pr)                      # last filtering stage -> this is the candidate set
        f = pl.read_parquet(p).join(cluster_features(pr, pool, top2), on=["s1_id", "cand_id"], how="inner")
        outs.append(f.select(["s1_id", "cand_id"]).with_columns(pl.Series("prob", M.predict_proba(booster2, f, feats2))))
        print(f"  stage-2 scored part {k}", flush=True)
    pairs = pl.concat(outs)
    n_s1 = pl.read_parquet(config.WORK_DIR / "test_s1.parquet").height
    print(f"pruning kept {pairs.height:,} of {n_before:,} pairs ({pairs.height/n_s1:.2f} per S1 entity)", flush=True)
    pairs.write_parquet(config.WORK_DIR / "test_pairs2.parquet")
    _write_outputs(pairs, rule, candidates=pairs)


def _write_outputs(pairs: pl.DataFrame, rule: dict, candidates: pl.DataFrame | None = None) -> None:
    """candidates: the exact pair set the final model scored (defaults to the blocking output)."""
    from . import model as M

    matches = M.apply_rule(pairs, rule)
    if candidates is None:
        candidates = pl.read_parquet(config.WORK_DIR / "test_cands.parquet")
    cands = candidates.group_by("s1_id").agg(pl.col("cand_id"))
    cand_lists = dict(zip(cands["s1_id"], cands["cand_id"].to_list()))
    s1_ids = pl.read_parquet(config.WORK_DIR / "test_s1.parquet")["entity_id"].to_list()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_id_list_file(config.OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    write_id_list_file(config.OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, cand_lists, "candidate_entity_ids")
    n_match = sum(len(v) for v in matches.values())
    print(f"wrote outputs: {len(s1_ids):,} S1 rows, {n_match:,} matched ids, rule {rule}")
    validator = config.DATA_DIR.parent / "utils" / "validate_submission.py"
    if validator.exists():
        subprocess.run([sys.executable, str(validator), "--matching", str(config.OUTPUT_DIR / "matching_results.tsv"),
                        "--candidate", str(config.OUTPUT_DIR / "candidate_pairs.tsv"), "--test-dir", str(config.DATA_DIR / "test")])


def predict() -> None:
    from . import model as M

    booster = M.load(config.WORK_DIR / "model.txt")
    rule = json.loads((config.WORK_DIR / "threshold.json").read_text())["rule"]
    parts = []
    for p in _feat_parts("test"):
        feats = pl.read_parquet(p)
        parts.append(feats.select(["s1_id", "cand_id"]).with_columns(pl.Series("prob", M.predict_proba(booster, feats))))
        print(f"  scored {p.name}", flush=True)
    pairs = pl.concat(parts)
    pairs.write_parquet(config.WORK_DIR / "test_pairs.parquet")
    matches = M.apply_rule(pairs, rule)
    cands = pl.read_parquet(config.WORK_DIR / "test_cands.parquet").group_by("s1_id").agg(pl.col("cand_id"))
    cand_lists = dict(zip(cands["s1_id"], cands["cand_id"].to_list()))
    s1_ids = pl.read_parquet(config.WORK_DIR / "test_s1.parquet")["entity_id"].to_list()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_id_list_file(config.OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    write_id_list_file(config.OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, cand_lists, "candidate_entity_ids")
    n_match = sum(len(v) for v in matches.values())
    print(f"wrote outputs: {len(s1_ids):,} S1 rows, {n_match:,} matched ids, rule {rule}")
    validator = config.DATA_DIR.parent / "utils" / "validate_submission.py"
    if validator.exists():
        subprocess.run([sys.executable, str(validator), "--matching", str(config.OUTPUT_DIR / "matching_results.tsv"),
                        "--candidate", str(config.OUTPUT_DIR / "candidate_pairs.tsv"), "--test-dir", str(config.DATA_DIR / "test")])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--split", required=True)
    sub.add_parser("equiv")
    me = sub.add_parser("merge-emb"); me.add_argument("--split", required=True)
    b = sub.add_parser("block-eval"); b.add_argument("--n", type=int, default=50000); b.add_argument("--country", default=None)
    bl = sub.add_parser("block"); bl.add_argument("--split", required=True); bl.add_argument("--n", type=int, default=None)
    f = sub.add_parser("featurize"); f.add_argument("--split", required=True)
    t = sub.add_parser("train"); t.add_argument("--n-train", type=int, default=config.TRAIN_S1); t.add_argument("--n-val", type=int, default=config.VAL_S1)
    sub.add_parser("predict")
    t2 = sub.add_parser("train2"); t2.add_argument("--parts", type=int, default=None)
    sub.add_parser("predict2")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.split)
    elif a.cmd == "equiv":
        equiv_step()
    elif a.cmd == "merge-emb":
        from .embed import merge
        merge(a.split)
        if a.split == "train":
            from .blocking import candidate_recall
            cands = pl.read_parquet(config.WORK_DIR / "train_cands.parquet")
            print(candidate_recall(cands, _truth(), pl.read_parquet(config.WORK_DIR / "train_queries.parquet")["entity_id"].to_list()))
    elif a.cmd == "block-eval":
        block_eval(a.n, a.country)
    elif a.cmd == "block":
        block(a.split, a.n)
    elif a.cmd == "featurize":
        featurize(a.split)
    elif a.cmd == "train":
        train(a.n_train, a.n_val)
    elif a.cmd == "predict":
        predict()
    elif a.cmd == "train2":
        train2(a.parts)
    elif a.cmd == "predict2":
        predict2()


if __name__ == "__main__":
    main()
