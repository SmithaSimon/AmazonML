#!/usr/bin/env bash
# Experiment 2 on a GPU instance: dense name embeddings -> FAISS top-k -> upload the pairs to S3.
# Needs work/{train,test}_{s1,pool}.parquet (pulled from s3://<bucket>/work by bootstrap.sh).
# Run from src/:   nohup bash ../aws/run_embed.sh > $BER_ROOT/embed.log 2>&1 &
set -euo pipefail
W="${BER_WORK_DIR:?}"; B="${BER_S3_BUCKET:?}"
stage() { echo "=== $(date '+%F %T') $1"; }
nvidia-smi || echo "WARNING: no GPU visible; this will be slow"
for split in train test; do
  [ -f "$W/emb/${split}_pool_DONE" ]     || { stage "encode $split"; python -m ber.embed encode --split $split; }
  [ -f "$W/${split}_cands_emb.parquet" ] || { stage "block $split";  python -m ber.embed block  --split $split; }
  aws s3 cp "$W/${split}_cands_emb.parquet" "s3://$B/work/${split}_cands_emb.parquet"
done
stage "done — on the laptop run:  aws s3 sync s3://$B/work work --exclude '*' --include '*_cands_emb.parquet'"
