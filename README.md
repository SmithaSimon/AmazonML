# Amazon ML Challenge 2026 — Business Entity Resolution

Match noisy business records from two sources (S2, S3) to a deduplicated reference source (S1)
across US, India and an unseen country (France), scored by macro F_0.5 per S1 entity.

Full pipeline, runnable on a 16 GB laptop (no GPU needed):
normalisation → learned token equivalences → per-country sparse TF-IDF blocking (+ optional
multilingual embedding channel) → pairwise LightGBM → cluster-consistency second pass →
one-to-one assignment and F_0.5-tuned threshold.

| version | held-out macro F_0.5 | what changed |
|---|---|---|
| v1 | 0.9546 | baseline features + Source-1 ambiguity counts |
| v2 | 0.9625 | learned token equivalences, char channel for domains / Indic scripts |
| v3 | 0.9652 | blocking over all S1 entities, candidate-competition features, aligned transliteration mining |
| v4 | 0.9713 | house-number edit distance, IDF-weighted overlaps, fuzzy name-ambiguity counts |
| v5 | 0.9746 | cluster-consistency second pass on out-of-fold probabilities |

## Layout

```
code/business_entity_resolution/   the submission package (src/, README.md, requirements.txt, aws/)
Documentation_template.md          methodology write-up (challenge template, filled in)
student_resource/                  organiser's README, validator and (not committed) dataset
```

## Quick start

```
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
python -m ber.pipeline prepare --split train && python -m ber.pipeline prepare --split test
python -m ber.pipeline equiv
python -m ber.pipeline block --split train && python -m ber.pipeline featurize --split train
python -m ber.pipeline train
python -m ber.pipeline block --split test  && python -m ber.pipeline featurize --split test
python -m ber.pipeline predict            # first-pass submission
python -m ber.pipeline train2 && python -m ber.pipeline predict2   # final submission in output/
```

See `code/business_entity_resolution/README.md` for details and `code/business_entity_resolution/aws/`
for running on AWS (SageMaker / EC2) and the GPU embedding experiment.

The dataset is not in this repository: place the organiser's `dataset/` folder under
`student_resource/` (or point `BER_DATA_DIR` at it).
