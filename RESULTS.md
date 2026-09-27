# Experiment log

Validation = macro F_0.5 on 100k held-out training Source-1 entities (never used for fitting or
threshold tuning). Oracle = score of a perfect classifier on the candidate set (blocking ceiling).
Leaderboard column is filled in as submissions are uploaded.

| version | date | validation F_0.5 | precision / recall / singleton acc | blocking recall (oracle) | leaderboard (public) | tag |
|---|---|---|---|---|---|---|
| v1 | 2026-09-25 | 0.9546 | 0.990 / 0.900 / 0.961 | 94.9% (0.982) | | – |
| v2 | 2026-09-26 | 0.9625 | 0.991 / 0.917 / 0.964 | 96.1% (0.987) | | – |
| v3 | 2026-09-26 | 0.9652 | 0.991 / 0.923 / 0.967 | 96.3% (0.987) | | – |
| v4 | 2026-09-26 | 0.9713 | 0.9935 / 0.933 / 0.975 | 96.3% (0.987) | | – |
| v5 | 2026-09-26 | **0.9746** | 0.9946 / 0.942 / 0.977 | 96.3% (0.987) | | `v5` |
| v6 | 2026-09-27 | (running) | | | | |

## What changed in each version

**v1 — baseline.** Normalisation (romanisation, abbreviations, legal-suffix stripping, domain unpacking,
house-number parsing); three per-country sparse TF-IDF blocking channels (name+address words,
name words, address words) on a 400k-entity training sample; 36 pairwise string features
(RapidFuzz ratios, token overlaps, number overlaps) + per-entity rank/margin context + Source-1
ambiguity counts (same address / same name / same name+house number); LightGBM; one-to-one
assignment; global threshold tuned for macro F_0.5. 74-minute training on the laptop.

**v2 — learned normalisation.** Token equivalences mined from the training pairs (state codes,
Devanagari state names, romanised Hindi/Kannada/Telugu name tokens, typos); character-3-gram
channel restricted to domain-name and non-Latin pool records; leading-zero stripping.
India went from 0.927 to 0.946, US from 0.968 to 0.973.

**v3 — scale.** Blocking over all 2.2M training entities (66M pairs) so candidate-competition
features (how many S1 entities retrieved a record, this pair's rank/gap among them) are
distributed identically in train and test; 2.7x more training pairs (700k entities);
position-aligned transliteration mining (588 name tokens); stricter one-to-one/one-to-two
address-token mining. Decision-rule experiments (best-guess fallback, expected-F_0.5
maximisation) showed the global threshold is already optimal.

**v4 — hard-negative features.** House-number Levenshtein distance, one-edit and prefix flags,
secondary-number overlap, near-miss number count, ratio over sorted number lists; IDF-weighted
name/address Jaccard, rarest shared and rarest unshared token, candidate-name specificity;
fuzzy name-ambiguity counts (S1 entities retrieving the record with name cosine >= 0.7).
Largest single step: precision 0.991 -> 0.9935, singleton accuracy 0.967 -> 0.975.

**v5 — cluster-consistency second pass.** Two cross-fitted first-pass models give out-of-fold
probabilities on the training entities; a second LightGBM re-scores every pair with
leave-one-out profile features (coverage of the candidate's name / address / number tokens by
the entity's other confident matches, house-number agreement), probability context (rank,
mass of the other candidates, best competing S1 for the same record) and the original features.
Threshold 0.66 with the single best candidate accepted from 0.58.

**v6 — embedding blocking channel (in progress).** multilingual-e5-small name embeddings for all
S1 names and the domain / non-Latin / empty-address pool subset; per-country FAISS IVF top-3
(up to 6 above cosine 0.93) unioned into the candidate set; `cos_emb` as a feature; full
retrain of both stages.

## Loss analysis (v5 validation)

77% of the lost score is missed matches on entities with no false merge, 23% false merges.
Of missed pairs, 46% were never candidates (blocking recall by record type: plain 97.9%,
domain 96.5%, Indic-script 88%, empty-address 76%); of the rest, a third are empty-address
records with ambiguous names.

## Runtime (8-core / 16 GB laptop, CPU only)

prepare 2+3 min · equiv 3 min · block train (all S1) 143 min · featurize train 21 min ·
train 14 min · train2 25 min · block test 97 min · featurize test 17 min · predict2 ~70 min.
Embedding channel (v6): encoding 7.9 h on CPU (≈ 20 min on a g4dn.xlarge), IVF search ~1 h.

## Reproducing a tagged version

```
git checkout v5
cd code/business_entity_resolution/src
bash ../aws/run_all.sh        # or the command list in code/business_entity_resolution/README.md
```

Submission files and models for each tagged version are attached to the matching GitHub
Release (they are too large for the repository itself).
