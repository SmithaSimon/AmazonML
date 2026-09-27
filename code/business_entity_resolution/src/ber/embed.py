"""Experiment 2: dense multilingual name embeddings as an extra blocking channel.

Targets the record types the sparse channels miss: Indic-script names
(transliterations), domain names and empty-address records whose name is
misspelled.  Reduced scope so it also runs on a CPU: every Source-1 name is
encoded, but on the pool side only the records of those three kinds (~14% of
the pool).

Model: a small multilingual sentence encoder with a permissive licence and far
under 8B parameters (default `intfloat/multilingual-e5-small`, MIT, 118M params).
Nothing external is looked up: the model only maps strings to vectors.

Usage:
    python -m ber.embed encode --split train      # resumable, 100k-name chunks under work/emb/
    python -m ber.embed encode --split test
    python -m ber.embed block  --split train      # writes work/<split>_cands_emb.parquet (s1_id, cand_id, cos_emb)
    python -m ber.embed block  --split test
    python -m ber.pipeline merge-emb --split train / test   (adds column cos_emb, unions the pairs)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import config

MODEL = "intfloat/multilingual-e5-small"
BATCH = 256
CHUNK = 100_000
TOPK_ALWAYS = 3       # always keep the 3 nearest pool records
TOPK_MAX = 6          # keep up to 6 when the cosine is high
HIGH_COS = 0.93       # e5 cosines are compressed (unrelated names often score ~0.85-0.9)
POOL_FILTER = (pl.col("is_domain") == 1) | (pl.col("name_nonlatin") == 1) | (pl.col("addr_empty") == 1)


def _texts(df: pl.DataFrame) -> list[str]:
    # raw (un-normalised) name so the multilingual model sees the original script;
    # e5 expects a "query: " prefix for symmetric similarity tasks.
    return ("query: " + df["business_name"].str.strip_chars()).to_list()


def _load_model(model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(config.N_JOBS)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 32          # business names are short; this doubles throughput
    print(f"model {model_name} on {device}", flush=True)
    return model


def encode(split: str, model_name: str = MODEL, out_dir: Path | None = None, limit: int | None = None) -> None:
    out_dir = out_dir or (config.WORK_DIR / "emb")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_name)
    for name in ("s1", "pool"):
        df = pl.read_parquet(config.WORK_DIR / f"{split}_{name}.parquet",
                             columns=["entity_id", "business_name", "is_domain", "name_nonlatin", "addr_empty"])
        if name == "pool":
            df = df.with_row_index("_i").filter(POOL_FILTER)
            np.save(out_dir / f"{split}_pool_idx.npy", df["_i"].to_numpy().astype(np.int64))
        if limit:
            df = df.head(limit)
        texts = _texts(df)
        n = len(texts)
        t0 = time.time()
        for k, start in enumerate(range(0, n, CHUNK)):
            part = out_dir / f"{split}_{name}_{k:04d}.npy"
            if part.exists():
                continue
            emb = model.encode(texts[start : start + CHUNK], batch_size=BATCH, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
            np.save(part, emb)
            done = min(start + CHUNK, n)
            rate = done / max(1, time.time() - t0)
            print(f"  {split}_{name}: {done:,}/{n:,} ({done/n:4.0%}) {rate:.0f} names/s, eta {(n-done)/max(rate,1)/60:.0f} min", flush=True)
        (out_dir / f"{split}_{name}_DONE").write_text(str(n))
        print(f"  {split}_{name}: {n:,} names encoded ({time.time()-t0:.0f}s)", flush=True)


def _load_emb(split: str, name: str, out_dir: Path) -> np.ndarray:
    parts = sorted(out_dir.glob(f"{split}_{name}_[0-9]*.npy"))
    return np.concatenate([np.load(p) for p in parts])


IVF_MIN_POOL = 50_000   # below this, exact search is cheap enough
IVF_NLIST = 2048
IVF_NPROBE = 64         # ~94% agreement with exact top-1 on this data, ~6x faster than exact on CPU


def _build_index(P: np.ndarray):
    import faiss

    faiss.omp_set_num_threads(config.N_JOBS)
    d = P.shape[1]
    if len(P) < IVF_MIN_POOL:
        index = faiss.IndexFlatIP(d)
    else:
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, min(IVF_NLIST, len(P) // 40), faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(config.SEED)
        index.train(P[rng.choice(len(P), min(100_000, len(P)), replace=False)])
        index.nprobe = IVF_NPROBE
    index.add(P)
    return index


def block(split: str, out_dir: Path | None = None) -> None:
    """Per-country cosine top-k (inner product on unit vectors) with a FAISS IVF index
    (exact search for small pools)."""
    out_dir = out_dir or (config.WORK_DIR / "emb")
    s1 = pl.read_parquet(config.WORK_DIR / f"{split}_s1.parquet", columns=["entity_id", "country"])
    pool = pl.read_parquet(config.WORK_DIR / f"{split}_pool.parquet", columns=["entity_id", "country"])
    e1 = _load_emb(split, "s1", out_dir)
    ep = _load_emb(split, "pool", out_dir)
    pidx = np.load(out_dir / f"{split}_pool_idx.npy")          # rows of `pool` that were encoded
    assert len(e1) == s1.height and len(ep) == len(pidx), "embedding parts incomplete"
    pool_sub = pool[pidx.tolist()]
    outs = []
    for country in s1["country"].unique().sort().to_list():
        qi = np.flatnonzero((s1["country"] == country).to_numpy())
        pj = np.flatnonzero((pool_sub["country"] == country).to_numpy())
        if len(pj) == 0:
            continue
        t0 = time.time()
        index = _build_index(np.ascontiguousarray(ep[pj].astype(np.float32)))
        D, I = [], []
        for start in range(0, len(qi), 50_000):
            d, i = index.search(np.ascontiguousarray(e1[qi[start : start + 50_000]].astype(np.float32)), TOPK_MAX)
            D.append(d); I.append(i)
            if (start // 50_000) % 4 == 3:
                print(f"    {country}: {min(start+50_000, len(qi)):,}/{len(qi):,} queries ({time.time()-t0:.0f}s)", flush=True)
        D, I = np.concatenate(D), np.concatenate(I)
        rank = np.tile(np.arange(TOPK_MAX), (len(qi), 1))
        keep = (rank < TOPK_ALWAYS) | (D >= HIGH_COS)
        keep &= I >= 0
        rows = np.repeat(qi, TOPK_MAX).reshape(len(qi), TOPK_MAX)[keep]
        cols = pidx[pj[I[keep]]]
        outs.append(pl.DataFrame({
            "s1_id": s1["entity_id"].to_numpy()[rows],
            "cand_id": pool["entity_id"].to_numpy()[cols],
            "cos_emb": D[keep].astype(np.float32),
        }))
        print(f"  {country}: {len(qi):,} queries vs {len(pj):,} pool, {int(keep.sum()):,} pairs in {time.time()-t0:.0f}s", flush=True)
    out = pl.concat(outs)
    out.write_parquet(config.WORK_DIR / f"{split}_cands_emb.parquet")
    print(f"wrote {out.height:,} embedding candidate pairs ({out.height / s1.height:.1f} per S1)", flush=True)


def merge(split: str) -> None:
    """Union the embedding pairs into <split>_cands.parquet and add the cos_emb column."""
    cands = pl.read_parquet(config.WORK_DIR / f"{split}_cands.parquet")
    emb = pl.read_parquet(config.WORK_DIR / f"{split}_cands_emb.parquet")
    if "cos_emb" in cands.columns:
        cands = cands.drop("cos_emb")
    merged = cands.join(emb, on=["s1_id", "cand_id"], how="full", coalesce=True)
    merged = merged.with_columns([pl.col(c).fill_null(0.0) for c in merged.columns if c.startswith("cos_")])
    merged.write_parquet(config.WORK_DIR / f"{split}_cands.parquet")
    print(f"{split}: {cands.height:,} sparse + {emb.height:,} embedding pairs -> {merged.height:,} union", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("--split", required=True); e.add_argument("--model", default=MODEL)
    e.add_argument("--limit", type=int, default=None); e.add_argument("--out-dir", default=None)
    b = sub.add_parser("block"); b.add_argument("--split", required=True); b.add_argument("--out-dir", default=None)
    m = sub.add_parser("merge"); m.add_argument("--split", required=True)
    a = ap.parse_args()
    if a.cmd == "encode":
        encode(a.split, a.model, Path(a.out_dir) if a.out_dir else None, a.limit)
    elif a.cmd == "block":
        block(a.split, Path(a.out_dir) if a.out_dir else None)
    elif a.cmd == "merge":
        merge(a.split)


if __name__ == "__main__":
    main()
