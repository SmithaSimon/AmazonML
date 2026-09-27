"""Paths and tunables. Override with environment variables where useful."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[4]))
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", ROOT / "output"))

N_JOBS = int(os.environ.get("BER_N_JOBS", max(1, os.cpu_count() or 4)))
SEED = 42

# Blocking
TOPK_ALL = 15      # combined name+address channel
TOPK_NAME = 6      # name-only channel
TOPK_ADDR = 8      # address-only channel
TOPK_NAMECHAR = 8  # char n-gram name channel vs domain / non-Latin pool subset
TRAIN_S1 = 700_000   # S1 entities whose candidate pairs are used to fit the model
VAL_S1 = 100_000     # held-out S1 entities for threshold tuning / reporting
QUERY_CHUNK = 20000
