# One-shot environment setup for this project on a fresh Windows laptop.
# Run from the project root in PowerShell (Claude Code can run it):
#     powershell -ExecutionPolicy Bypass -File .\SETUP.ps1
# Idempotent: safe to re-run. Installs Python 3.11 + Git via winget if missing, the pinned
# Python packages, then verifies data/intermediates and prints what to do next.
$ErrorActionPreference = "Continue"
function Say($m) { Write-Host ("`n== {0}" -f $m) -ForegroundColor Cyan }
function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Say "1/5 Python 3.11"
$py = $null
foreach ($c in @("py -3.11", "python3.11", "python")) {
    try { $v = & cmd /c "$c --version 2>&1"; if ($v -match "3\.11") { $py = $c; break } } catch {}
}
if (-not $py) {
    winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements --silent
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $py = "py -3.11"
}
Write-Host "using: $py -> $(& cmd /c "$py --version 2>&1")"

Say "2/5 Git"
if (-not (Have git)) {
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements --silent
    $env:PATH = "C:\Program Files\Git\cmd;" + $env:PATH
}
git --version
git config user.name  "Smitha S"
git config user.email "smithasimon19@gmail.com"
git config core.autocrlf false
if (-not (git remote 2>$null | Select-String origin)) { git remote add origin https://github.com/SmithaSimon/AmazonML.git }

Say "3/5 Python packages (pinned)"
& cmd /c "$py -m pip install -q --upgrade pip"
& cmd /c "$py -m pip install -q -r code\business_entity_resolution\requirements.txt"
& cmd /c "$py -m pip install -q sentence-transformers faiss-cpu"
& cmd /c "$py -c ""import polars, lightgbm, rapidfuzz, sklearn, sparse_dot_topn, unidecode, sentence_transformers, faiss; print('packages OK')"""

Say "4/5 Data and intermediates"
$checks = @(
    @{p="student_resource\dataset\train\train_source1.tsv"; why="dataset"},
    @{p="student_resource\dataset\test\test_source1.tsv";   why="dataset"},
    @{p="work\train_s1.parquet";     why="normalised records (else run: prepare + equiv)"},
    @{p="work\test_pool.parquet";    why="normalised records (else run: prepare + equiv)"},
    @{p="work\equiv.json";           why="learned token map"},
    @{p="work\train_cands.parquet";  why="MERGED v6 candidates (else re-block: ~12 h)"},
    @{p="work\test_cands.parquet";   why="MERGED v6 candidates (else re-block: ~12 h)"},
    @{p="work\train_queries.parquet";why="training query ids"},
    @{p="output_v5\matching_results.tsv"; why="best submission so far (v5)"}
)
$missing = 0
foreach ($c in $checks) {
    if (Test-Path $c.p) { Write-Host ("  OK       {0}" -f $c.p) } else { Write-Host ("  MISSING  {0}   <- {1}" -f $c.p, $c.why) -ForegroundColor Yellow; $missing++ }
}
if (Test-Path work\test_cands.parquet) {
    Push-Location code\business_entity_resolution\src
    & cmd /c "$py -c ""import polars as pl; from ber import config; c=pl.read_parquet(config.WORK_DIR/'test_cands.parquet').columns; print('test_cands columns:', c); print('cos_emb present:', 'cos_emb' in c)"""
    Pop-Location
}

Say "5/5 Machine"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$ram = [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize/1MB, 1)
Write-Host ("  {0} | {1} cores / {2} threads | {3} GB RAM" -f $cpu.Name, $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors, $ram)
$gpu = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name
Write-Host ("  GPU: {0}" -f ($gpu -join ", "))
if ($gpu -match "NVIDIA") { Write-Host "  NVIDIA GPU found: CPU-only torch is installed; for GPU experiments install a CUDA build of torch (see HANDOFF.md)." }

Say "Done"
if ($missing -eq 0) {
    Write-Host "Everything is in place. Next (about 3 h): finish v6 - see HANDOFF.md, section 'To finish v6'."
} else {
    Write-Host "$missing item(s) missing - see HANDOFF.md 'Files to carry over'." -ForegroundColor Yellow
}
