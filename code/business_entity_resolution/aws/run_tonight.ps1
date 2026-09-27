# Overnight CPU run: embedding blocking channel (reduced scope) + full retrain chain (v6).
# Run from code/business_entity_resolution/src:
#   powershell -ExecutionPolicy Bypass -File ..\aws\run_tonight.ps1 > ..\..\..\work\tonight.log 2>&1
$ErrorActionPreference = "Continue"
function Stage($msg) { Write-Output ("=== {0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg) }
$W = "..\..\..\work"

Stage "encode train";   python -m ber.embed encode --split train;   if (-not $?) { exit 1 }
Stage "encode test";    python -m ber.embed encode --split test;    if (-not $?) { exit 1 }
Stage "block train";    python -m ber.embed block --split train;    if (-not $?) { exit 1 }
Stage "block test";     python -m ber.embed block --split test;     if (-not $?) { exit 1 }
Copy-Item "$W\train_cands.parquet" "$W\train_cands_v5.parquet" -Force
Copy-Item "$W\test_cands.parquet"  "$W\test_cands_v5.parquet"  -Force
Stage "merge train";    python -m ber.pipeline merge-emb --split train; if (-not $?) { exit 1 }
Stage "merge test";     python -m ber.pipeline merge-emb --split test;  if (-not $?) { exit 1 }
Stage "featurize train"; python -m ber.pipeline featurize --split train; if (-not $?) { exit 1 }
Stage "train";          python -m ber.pipeline train;                 if (-not $?) { exit 1 }
Stage "train2";         python -m ber.pipeline train2;                if (-not $?) { exit 1 }
Stage "featurize test"; python -m ber.pipeline featurize --split test; if (-not $?) { exit 1 }
Stage "predict2";       python -m ber.pipeline predict2;              if (-not $?) { exit 1 }
Copy-Item -Recurse -Force "..\..\..\output" "..\..\..\output_v6"
Stage "done"
