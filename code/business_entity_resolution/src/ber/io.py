"""Reading the challenge TSVs and writing submission files."""
from __future__ import annotations

from pathlib import Path

import polars as pl

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: str | Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        schema_overrides={c: pl.Utf8 for c in COLS},
        null_values=None,
        encoding="utf8",
    )
    return df.select(COLS).fill_null("")


def read_split(data_dir: str | Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (source1, pool) where pool = source2 ∪ source3 for the given split."""
    d = Path(data_dir) / split
    s1 = read_source(d / f"{split}_source1.tsv")
    s2 = read_source(d / f"{split}_source2.tsv")
    s3 = read_source(d / f"{split}_source3.tsv")
    return s1, pl.concat([s2, s3])


def read_ground_truth(path: str | Path) -> dict[str, set[str]]:
    df = pl.read_csv(
        path, separator="\t", quote_char=None,
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    ).fill_null("")
    return {
        s1: (set(m.split(",")) if m else set())
        for s1, m in zip(df["source1_entity_id"], df["matched_entity_ids"])
    }


def write_id_list_file(path: str | Path, s1_ids: list[str], lists: dict[str, list[str]], col: str) -> None:
    """Write a two-column TSV (source1_entity_id, <col>) with one row per S1 id, in the given order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in s1_ids:
            ids = lists.get(s1, [])
            # dedupe while preserving order, drop anything that is not S2/S3
            seen: set[str] = set()
            clean = []
            for x in ids:
                if x.startswith(("S2-", "S3-")) and x not in seen:
                    seen.add(x)
                    clean.append(x)
            f.write(f"{s1}\t{','.join(clean)}\n")
