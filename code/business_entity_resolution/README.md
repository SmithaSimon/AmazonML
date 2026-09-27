# Business Entity Resolution — ML Challenge 2026

End-to-end pipeline: TSV data → normalisation → blocking (candidate generation) →
pairwise LightGBM matcher → decision rule tuned for macro F_0.5 → output TSVs.

No external data, APIs or lookups are used anywhere. Every dictionary in the code
(abbreviations, legal suffixes) is a short generic list; nothing is fetched or looked up.

## Environment

```
python 3.11
pip install -r requirements.txt
```

Tested on a 16 GB RAM, 8-core laptop (CPU only). No GPU is needed.

## Layout

```
src/ber/normalize.py   text normalisation (romanisation, abbreviations, suffix stripping, address parsing)
src/ber/blocking.py    TF-IDF top-k candidate generation per country (3 channels)
src/ber/features.py    pairwise + context features
src/ber/model.py       LightGBM matcher, one-to-one assignment, threshold tuning
src/ber/stage2.py      second pass: entity profiles from confident matches, competition context
src/ber/embed.py       optional GPU experiment: multilingual name embeddings as a blocking channel
src/ber/metrics.py     macro F_0.5 scorer (challenge definition)
src/ber/io.py          TSV reading / submission writing
src/ber/pipeline.py    CLI orchestrating the steps
```

## Reproduce end-to-end

All commands are run from `src/`. Paths default to `../../../student_resource/dataset`
(data), `../../../work` (intermediates) and `../../../output` (submission files);
override with the environment variables `BER_DATA_DIR`, `BER_WORK_DIR`, `BER_OUTPUT_DIR`.

```
cd src
python -m ber.pipeline prepare   --split train
python -m ber.pipeline prepare   --split test
python -m ber.pipeline equiv                                # learn token equivalences from train pairs, apply to both splits
python -m ber.pipeline block     --split train              # candidates for every training S1 entity (~2.4 h)
python -m ber.pipeline featurize --split train
python -m ber.pipeline train                                # fits on 700k S1 entities, tunes the decision rule on 100k held-out ones
python -m ber.pipeline block     --split test               # ~1.6 h
python -m ber.pipeline featurize --split test
python -m ber.pipeline predict                              # first-pass submission (checkpoint), runs validator
python -m ber.pipeline train2                               # second pass: cross-fitted OOF probabilities + cluster-consistency model
python -m ber.pipeline predict2                             # FINAL: writes output/matching_results.tsv + candidate_pairs.tsv, runs validator
```

`aws/run_all.sh` runs the same sequence, resumably.

Total wall-clock on a 16 GB, 8-core laptop: about 6 hours. Peak memory is under 12 GB;
every heavy step is chunked (blocking per country in 20k-query chunks, features in
6M-pair parts split at S1 boundaries, training on a lazily filtered subset).

`candidate_pairs.tsv` is exactly the set of pairs the LightGBM model scores
(the output of `block --split test`); `matching_results.tsv` is a subset of it.
