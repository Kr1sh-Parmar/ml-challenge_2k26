# Business Entity Resolution — V4

Design, evidence and results: [`version_4.md`](version_4.md).

## Layout
```
version 4/
├── src/er4/          # the pipeline (config, normalize, mining, blocking, features, models, decide, pipeline, util)
├── anchor_v2/        # V2's pair model (scored on the mirror hold-out as the leaderboard anchor) + V2's mined tables
├── v4_run.ipynb      # driver notebook: one cell per stage
├── cache/            # checkpoints (created on first run, ~12-15 GB)
└── output/           # matching_results.tsv, candidate_pairs.tsv, probe_france_empty/
```

## Requirements
- Python 3.12, `pip install -r requirements.txt`
- 24 GB RAM, ~30 GB free disk. An NVIDIA GPU is used for XGBoost if present (CPU otherwise).
- Data at `<repo>/student_resource/dataset/{train,test}/` (found automatically by walking up from this folder).

## Run
End to end (≈3.2 h the first time; every stage is checkpointed, so a re-run resumes where it stopped):
```
cd "version 4/src"
python -m er4.pipeline --stage all
```
One stage: `--stage roles|mine|parse|blkfit|block|trainfeat|l1|score|l2|decide|export`.
To recompute a stage, delete its files in `cache/`.

Long runs: start them in their own terminal window (Claude Code stops background jobs when memory runs low).

Smoke test (≈2.5% of the data, separate `cache_smoke/` and `output_smoke/`, a few minutes):
```
set ER4_SMOKE=1          (PowerShell: $env:ER4_SMOKE="1")
python -m er4.pipeline --stage all
```

## Stages
| Stage | What | Output |
|---|---|---|
| roles | test mirror: 21% S1 dropout; MINE / TRAIN / HOLDOUT / context roles | `roles.parquet`, `truth_mirror.parquet` |
| mine | variant tables (train MINE + France self-mining), legal forms | `tables.pkl` |
| parse | parse + enrich per split and country | `parsed_{split}_{cty}_{s1,s23}.parquet` |
| blkfit | blocking scorer + n_cand on MINE at full scale | `blk_scorer.pkl` |
| block | all S1s vs full pools, top-N, competition features | `cand_{split}_{cty}.parquet` |
| trainfeat | pair features for TRAIN edges (+ V2 anchor score) | `train_X_{cty}.parquet` |
| l1 | XGBoost L1, 5-fold OOF + final, LOCO report | `l1.json`, `l1_oof.parquet`, `l1_report.json` |
| score | mirror competition closure + all test edges | `p_{split}_{cty}.parquet` |
| l2 | context re-ranker, 5-fold OOF + final | `l2.json`, `l2_oof.parquet` |
| decide | decision layer on TRAIN; HOLDOUT report (V4 vs V2 anchor) | `decision.pkl` |
| export | test predictions, both TSVs, France-empty probe, official validator `--check-ids` | `output/` |
