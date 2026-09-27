#!/usr/bin/env bash
# Full pipeline, resumable: each stage is skipped when its output already exists.
# Run from src/ with nohup so it survives closing the browser tab:
#   nohup bash ../aws/run_all.sh > $BER_ROOT/run.log 2>&1 &
set -euo pipefail
W="${BER_WORK_DIR:?set BER_WORK_DIR}"
O="${BER_OUTPUT_DIR:?set BER_OUTPUT_DIR}"
stage() { echo "=== $(date '+%F %T') $1"; }

[ -f "$W/train_pool.parquet" ] || { stage "prepare train"; python -m ber.pipeline prepare --split train; }
[ -f "$W/test_pool.parquet" ]  || { stage "prepare test";  python -m ber.pipeline prepare --split test; }
[ -f "$W/equiv.json" ]         || { stage "equiv";         python -m ber.pipeline equiv; }
[ -f "$W/train_cands.parquet" ]|| { stage "block train";   python -m ber.pipeline block --split train; }
ls "$W"/train_feats_*.parquet >/dev/null 2>&1 || { stage "featurize train"; python -m ber.pipeline featurize --split train; }
[ -f "$W/model.txt" ]          || { stage "train";         python -m ber.pipeline train; }
[ -f "$W/test_cands.parquet" ] || { stage "block test";    python -m ber.pipeline block --split test; }
ls "$W"/test_feats_*.parquet >/dev/null 2>&1  || { stage "featurize test";  python -m ber.pipeline featurize --split test; }
[ -f "$O/matching_results.tsv" ] || { stage "predict (first pass)"; python -m ber.pipeline predict; }
[ -f "$W/model2.txt" ]         || { stage "train2 (cluster-consistency second pass)"; python -m ber.pipeline train2; }
stage "predict2"; python -m ber.pipeline predict2
stage "done"; ls -la "$O"
# keep a copy of the outputs + model in S3 (set BER_S3_BUCKET to enable)
if [ -n "${BER_S3_BUCKET:-}" ]; then
  aws s3 cp "$O/matching_results.tsv" "s3://$BER_S3_BUCKET/runs/$(date +%Y%m%d_%H%M)/matching_results.tsv"
  aws s3 cp "$W/model.txt"            "s3://$BER_S3_BUCKET/runs/$(date +%Y%m%d_%H%M)/model.txt"
  aws s3 cp "$W/threshold.json"       "s3://$BER_S3_BUCKET/runs/$(date +%Y%m%d_%H%M)/threshold.json"
fi
