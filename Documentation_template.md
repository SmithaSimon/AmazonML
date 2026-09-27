# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

We resolve Source-2/Source-3 records to Source-1 entities with a classic two-stage pipeline built for a 16 GB CPU-only laptop: country-blocked sparse TF-IDF candidate generation (three complementary channels, ~22 candidates per Source-1 entity, 95% pair recall) followed by a LightGBM pairwise matcher over ~45 string-similarity and ambiguity features, a one-to-one assignment step and a probability threshold tuned directly for macro F_0.5. Two data-driven components handle the noise without any external resources: token equivalences (state codes, Devanagari state names, romanised Hindi/Kannada/Telugu name tokens) mined from the training pairs, and ambiguity counts computed from the full Source-1 table that let the model abstain on generic names and address-less records.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the training split (2.2M Source-1 entities, 10.3M Source-2/3 records):

- Matches never cross countries; each Source-2/3 record matches at most one Source-1 entity (7.64M matched ids, all unique). 5.6% of Source-1 entities are singletons; the rest have 1–11 matches (median 3).
- **Address is the anchor.** Among true pairs, name and address both have zero token overlap only 0.0075% of the time. Names are frequently replaced wholesale (renames such as "Rizanex f/k/a Clear Consumer Global", domain names such as `anchorsmartcommunities.com`, Indic-script transliterations), while the address usually survives with abbreviations, reordering, dropped components or a digit typo.
- 4.4% of matched records have an **empty address**, and 98% of empty-address records are true matches, so they must be resolved on name alone. Because 30% of Source-1 names are duplicated within the country, an empty-address record is only safely matchable when the name is unique.
- ~5% of Source-2 and ~3% of Source-3 names are in Indic scripts (Devanagari, Kannada, Telugu, Tamil, Malayalam, Gujarati, Bengali), as are some address tokens (state names).
- Unmatched Source-2/3 records (27%) are mostly unrelated businesses, but the data also contains deliberate hard negatives: same name, house number off by a digit or two.
- France appears only in the test set (15% of test Source-1). Nothing in the pipeline enumerates countries; all dictionaries are generic, all features are similarities.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage pairwise classifier (LightGBM) + global one-to-one assignment  
**Core Innovation:** (a) token equivalences learned from the training pairs (no hand-written state/transliteration tables), (b) Source-1 ambiguity features (name / address duplication counts, fuzzy name-ambiguity counts) that let a precision-tuned model abstain, (c) candidate-context features (rank, margin among the entity's candidates, competition for the record among Source-1 entities) and a one-to-one assignment exploiting the "each S2/S3 record matches at most one S1" structure, (d) a **cluster-consistency second pass**: every pair is re-scored against a leave-one-out profile of the entity built from its other confident matches, using out-of-fold first-pass probabilities so the second model never sees in-sample scores.

---

## 3. Candidate Generation (Blocking)

Normalisation first: accent folding and script romanisation (`unidecode`, with a "squash" step that makes romanised Indic text resemble English spelling), lowercasing, `&`→`and`, punctuation removal, leading-zero stripping, a generic abbreviation table (Rd/Road, St/Street, R./Rue, Bd/Boulevard, Pvt/Private, Corp/Corporation …), legal-suffix separation, domain-name unpacking (`sjdcement.com` → `sjdcement`), house-number extraction from the first numbered comma-component, and finally the learned token-equivalence map.

Blocking runs per country (open set of labels) with `sparse_dot_topn` cosine top-k over TF-IDF vectors fitted on the Source-2∪3 pool of that country:

- **Blocking keys used:**
  1. `all` — word unigrams over core name + address words + address numbers, top-15
  2. `name` — word unigrams over core name, top-6 (empty-address records)
  3. `addr` — word unigrams over house number + address words + numbers, top-8 (renamed businesses)
  4. `namechar` — character 3-grams over the space-less core name, against only the pool records whose name is a domain or in a non-Latin script (~5% of the pool), top-8
  Tokens with document frequency > 2% are pruned (5% for the char channel); min cosine 0.05.
- **Candidate pairs generated:** 51.2M for the 1.73M test Source-1 entities (29.5 per entity); 65.9M for all 2.2M training Source-1 entities.
- **How you ensured true matches were not lost:** measured on the full training set: pair recall 96.3% (union), 94.7% from `all` alone, 74.6% `addr`, 45.0% `name`, 4.9% `namechar`; oracle macro F_0.5 with a perfect classifier on these candidates = 0.987. Blocking recall by record type: plain 97.9%, domain names 96.5%, Indic-script names 88%, empty-address records 76% (the last two are the remaining miss categories).

---

## 4. Matching Model

**Features used:**
- Name features: token Jaccard / overlap / coverage, RapidFuzz ratio, token-set, token-sort and partial ratios, Jaro-Winkler, no-space ratio (for domain names), first-token match, legal-suffix agreement, domain and non-Latin-script flags
- Address features: empty flag, word Jaccard / overlap / coverage, number-set Jaccard and overlap, house-number equality, house number contained in the other side's numbers, RapidFuzz token-set / ratio / partial on the full normalised address; **house-number edit distance, one-edit and prefix flags** (the data contains same-name hard negatives whose house number differs by one digit, while true matches also carry digit typos), secondary-number (unit/floor) overlap, count of near-miss numbers, ratio over the sorted number lists
- IDF-weighted overlaps (IDF from the split's own records): weighted Jaccard, the rarest shared token and the rarest unshared token for names and for addresses, and the total specificity of the candidate name (decisive for empty-address records)
- Blocking scores: cosine from each of the three channels
- Context: number of candidates, cheap combined score, its rank and margin to the best candidate, gap between best and second-best; Source-1 ambiguity counts from the full Source-1 table (entities sharing the same address, the same core name, the same name + house number; number of Source-1 entities carrying exactly the candidate's name); candidate-competition context over the full candidate table (how many Source-1 entities retrieved this record, the best cosine among them, this pair's gap and rank) — every Source-1 entity is a query in both train and test, so these are identically distributed; source flag

**Model type:** two LightGBM binary classifiers (255 leaves, lr 0.1, early stopping).
- *First pass* (878 rounds): 74 features, trained on 20.9M pairs from 700k Source-1 entities, validated on 3.0M pairs from 100k disjoint entities.
- *Second pass* (179 rounds): the same features plus 15 cluster-consistency features and the first-pass probability. Training data: the 350k "fold A" entities scored by a first-pass model fitted on the other 350k (out-of-fold); fold-B entities are scored by the fold-A model and everything else by the full model, so every probability the second model sees is out-of-sample. Cluster features: leave-one-out coverage of the candidate's name / address / number tokens by the entity's other confident (p ≥ 0.9) matches, house-number agreement with them, count and probability mass of the other candidates, rank, and the best competing Source-1 probability for the same record.

**Threshold selection method:** one-to-one assignment (each S2/S3 record kept only for its highest-probability S1), then the threshold that maximises macro F_0.5 on the validation entities (first pass 0.72; second pass 0.66 with the single best candidate accepted from 0.58). Per-entity expected-F_0.5 maximisation from the calibrated probabilities was evaluated and rejected (−0.005).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9746** on 100k held-out training Source-1 entities (pair precision 0.9946, pair recall 0.942, singleton accuracy 0.977). Progression: 0.951 (baseline features) → 0.955 (+ Source-1 ambiguity features) → 0.9625 (+ learned token equivalences, char channel, leading-zero normalisation; India went from 0.927 to 0.946, US from 0.968 to 0.973) → 0.9652 (+ blocking over all Source-1 entities with candidate-competition features, 2.7× more training pairs, token-aligned transliteration mining) → 0.9713 (+ house-number edit-distance, IDF-weighted overlaps, fuzzy name-ambiguity counts) → 0.9746 (+ cluster-consistency second pass). Oracle on the candidate set: 0.987.
- **Loss decomposition (validation, 100k entities):** 77% of the lost score comes from missed matches on entities with no false merge, 23% from false merges. Of the missed pairs, 46% were never generated as candidates; of those that were, 31% are empty-address records.
- **Common false positives (wrong merges):** empty-address records whose name is shared by several Source-1 entities; same-name records with a slightly different house number (hard negatives in the data).
- **Common false negatives (missed matches):** 48% of missed pairs were never generated as candidates (Indic-script names with truncated addresses, empty-address generic names); the rest scored below the threshold — mainly address-less records with typos and pairs where both the name is truncated and the address altered.

---

## 6. Conclusion

A carefully normalised sparse-blocking + two-stage gradient-boosting pipeline reaches macro F_0.5 ≈ 0.975 on held-out data while running end-to-end on a laptop in about eight hours. The decisive elements were address-driven blocking, learned token equivalences for abbreviations and transliterations, and ambiguity features that let the precision-weighted decision rule abstain.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/ber/`: `normalize.py` (text normalisation), `equiv.py` (token-equivalence mining), `blocking.py` (TF-IDF top-k channels), `features.py` (pairwise + context features), `stage2.py` (cluster-consistency features), `model.py` (LightGBM, one-to-one assignment, threshold tuning), `metrics.py` (macro F_0.5), `io.py`, `pipeline.py` (CLI). Reproduction: see `README.md` — `prepare → equiv → block → featurize → train → block/featurize (test) → predict → train2 → predict2`; `predict2` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the official validator.

### B. Additional Results

Learned token equivalences: 118 address tokens (strict one-to-one or one-to-two replacements only: `mh`→`maharashtra`, `up`→`uttar pradesh`, `andhrprdesh`→`andhra pradesh` from Devanagari, `tmillnatu`→`tamil nadu` from Tamil script, US state names→codes, US city→county pairs the data uses interchangeably) and 588 romanised Indic name tokens mined by position alignment (`praibhet`→`private` from Bengali, `knstrkshn`→`construction`, `intarnyashnal`→`international`, `solyaushns`→`solutions`). Runtime on an 8-core, 16 GB laptop: normalisation ~2 min per split, equivalence mining ~3 min, blocking all training entities 143 min, training featurisation 15 min, training 16 min, test blocking 97 min, test featurisation 11 min, scoring + writing ~60 min.
