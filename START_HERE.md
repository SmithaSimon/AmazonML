# START HERE (new laptop)

This folder is the complete, self-contained project: code + git history, dataset, the
intermediates that took hours to compute, the best submission, and the notes.

## For the human (2 manual steps)

1. Copy this whole folder onto the new laptop, e.g. to `C:\Projects\AmazonML`.
2. Install **Visual Studio Code** and the **Claude Code** extension, open this folder in VS Code,
   start Claude Code and paste:

   > Read START_HERE.md, then run SETUP.ps1 and fix anything it reports. Then read HANDOFF.md,
   > README.md and RESULTS.md and continue with finishing v6.

That's it. Python, Git and all packages are installed by the script.

## For Claude (what to do in the new session)

1. Run `powershell -ExecutionPolicy Bypass -File .\SETUP.ps1` from this folder. It installs
   Python 3.11 and Git via winget if they are missing, installs the pinned packages, checks that
   the dataset / intermediates / v5 outputs are present, and prints the machine's CPU/RAM/GPU.
   If winget prompts for elevation the user has to approve it once. Open a new terminal
   afterwards if `python` or `git` is still not found (PATH refresh).
2. Confirm `test_cands columns` printed by the script includes `cos_emb` (the merged v6
   candidates). If it does not, see HANDOFF.md "Files to carry over".
3. Read `HANDOFF.md` (state and next commands), `README.md` (pipeline), `RESULTS.md`
   (experiment log, v1–v5, what each change bought and why the next steps are what they are).
4. Continue: finish v6 (featurize train → train → train2 → featurize test → predict2), record
   the validation score in RESULTS.md, commit + tag `v6`, push (the user signs in to GitHub in
   the browser the first time; run the push in their terminal if the credential prompt cannot
   be shown), and package the release assets as described in HANDOFF.md.
5. Machine-specific: `ber.config.N_JOBS` defaults to all CPU cores; the pipeline is chunked to
   fit 16 GB RAM. If the new laptop has an NVIDIA GPU, the embedding step and the future
   cross-encoder experiment can run locally instead of on AWS (install a CUDA torch build).

## Folder contents

```
SETUP.ps1, START_HERE.md, HANDOFF.md, README.md, RESULTS.md, Documentation_template.md
code/business_entity_resolution/     pipeline (src/), requirements.txt, aws/ scripts
student_resource/                    organiser README + validator + dataset/ (3 GB)
work/                                normalised records, learned token map, MERGED v6 candidate
                                     tables, models/thresholds of v4/v5, release zips
output_v5/                           best submission so far (validator PASS, val F0.5 0.9746)
.git/                                full history; remote = https://github.com/SmithaSimon/AmazonML
```
