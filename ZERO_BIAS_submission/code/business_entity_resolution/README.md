# Business Entity Resolution — Team ZERO BIAS

End-to-end pipeline that produces `output/matching_results.tsv` (the leaderboard file, public LB 0.986 for V5 alone) and
`output/candidate_pairs.tsv` from the provided train/test TSVs. Methodology: `Documentation_template.md` (zip root).

**What the submitted file is**
| Country | Predictions |
|---|---|
| US, India | **V5** (V4 test-mirror pipeline + dense retrieval + cross-encoder + evidence pooling + Fellegi–Sunter EM + GBDT stacker) with its **learned cost-sensitive decision policy** |
| France (unseen) | pairs predicted by **≥ 2 of the 4 models V4.1 / V4.2 / V4.3 / V5** (majority vote, training-free) |

## Layout
```
src/
├── erpaths.py                 shared paths: ER_DATA (dataset), ER_WORK (caches / models / outputs)
├── v2/prepare.py              V2 base layers (the V2 notebook's code cells, verbatim): samples, mined variant tables,
│                              parsed samples, V2 blocking scorer, V2 features + anchor LightGBM  -> ER_WORK/v2/cache
├── v3/                        V3 package: dense bi-encoder (multilingual-e5-small) fine-tuning + GPU kNN, mDeBERTa-v3
│                              cross-encoder, Fellegi–Sunter EM, decision policy / expected-F, set transformer, v2_base
├── er4/                       the V4 -> V5 pipeline (test-mirror validation, full-scale per-country processing)
│   ├── config.py util.py      knobs, checkpoints, metric (self-checked against the README example)
│   ├── normalize.py mining.py parsing, name keys, variant-table mining (train + unsupervised France self-mining)
│   ├── blocking.py            5 inverted-index passes + reverse pass + duplicate-group mates, blocking scorer, dense union
│   ├── features.py models.py  109 pair features, GPU XGBoost L1 / L2 context re-ranker / singleton model
│   ├── decide.py              calibration, exclusivity, expected-F0.5 list selection
│   ├── pipeline.py            V4 / V4.1 stages (roles ... export)
│   ├── dense.py               V4.1 dense retrieval pass
│   ├── stack3.py              V4.2: evidence pooling (mates), cross-encoder, L3 stacker
│   ├── final.py               V4.3: band pair features, per-country EM, CE retrained on mirror TRAIN, 2 stacker rounds, policy
│   └── v5.py                  V5: +398k labelled context S1s, CE retrained on 1.09M pairs, 2-seed stacker, LOCO
├── er4_v5_policy_export.py    re-export V5 with the learned policy switched on
├── make_final.py              assembles the submitted matching_results.tsv + candidate_pairs.tsv, runs the validator
├── rebuild_candidates.py      CPU-only reconstruction of candidate_pairs.tsv (how the submitted candidate file was made)
└── anchor_v2/                 V2's mined tables (legal forms, train + France substitutions) and V2's LightGBM (anchor)
```

## Environment
- Python 3.12, `pip install -r requirements.txt` (install the CUDA torch wheel first, see the comment in the file).
- GPU stages (v3 training, dense pass, cross-encoders, EM, XGBoost on CUDA) ran on an **NVIDIA RTX A4000 16 GB, 64 GB RAM**.
  CPU stages (parse, mining, blocking, features) need **≥ 24 GB RAM, 16 threads**, ~40 GB free disk for `ER_WORK`.
- No internet lookups at any point; the only downloads are the two MIT-licensed pretrained checkpoints from Hugging Face
  (`intfloat/multilingual-e5-small`, `microsoft/mdeberta-v3-base`).

## Data and folders
- `ER_DATA` = folder with `train/` and `test/` (default: auto-found `student_resource/dataset` above this folder). The official
  validator is looked up at `ER_DATA/../utils/validate_submission.py` (the `student_resource` layout).
- `ER_WORK` = where every cache, model and stage output goes (default `./work`, tens of GB). All stages checkpoint there;
  re-running a command resumes; delete a file to recompute that step.

Environment variables below are bash syntax; in PowerShell use `$env:ER4_DENSE="1"; python ...` (and `Remove-Item Env:ER4_DENSE`).

## Reproduce end to end (run every command from `src/`)
| # | Command | Produces (in `ER_WORK`) | Device | Rough time* |
|---|---|---|---|---|
| 1 | `python v2/prepare.py` | `v2/cache/`: samples S/M, mined tables, parsed samples, V2 blocking scorer + features, anchor model | CPU | 1.5–2 h |
| 2 | `V3_TIER=M python -m v3.train` | `v3/models/dense_ft` (bi-encoder fine-tuned on sample M + self-supervised test noise variants), `v3/models/ce_fold0-1` (cross-encoder), V3 OOF ledger | GPU | 6–8 h |
| 3 | `ER4_DENSE=1 python -m er4.pipeline --stage all` | **V4.1**: test mirror, full-scale parse + blocking + dense union, L1/L2 XGBoost, decision layer, `output_dense/` | CPU+GPU | 4–5 h |
| 4 | `ER4_DENSE=1 python -m er4.stack3 --stage all` | **V4.2**: mates, cross-encoder band scores, L3 stacker, `output_v42/` | GPU | 2–3 h |
| 5 | `ER4_DENSE=1 python -m er4.final --stage all` | **V4.3**: band pair features, EM, CE retrained (`models/ce43_fold*`), stackers, policy, `output_final/` | GPU | 4–5 h |
| 6 | `ER4_V5=1 python -m er4.v5 --stage all` | **V5**: context labels, CE-v5 (`models_v5/ce5_half*`), stacker, LOCO report, `output_v5/` (+ `candidate_pairs.tsv`) | GPU | 5–6 h |
| 7 | `ER4_V5=1 python er4_v5_policy_export.py` | `output_v5_policy/` (V5 + learned policy) | CPU | 10 min |
| 8 | `python make_final.py --out ../../../output` | the two submission files + official validator (`--check-ids`) | CPU | 5 min |

\* GPU-stage times are estimates for the A4000 box above. Measured on a 16-thread / 24 GB laptop (CPU only): parsing all mirror + test records 12 min, blocking-scorer fit on MINE at full scale 9 min, blocking every test S1 11.5 min (US 4.6, France 0.8, India 6.1), writing `candidate_pairs.tsv` 1 min — i.e. `rebuild_candidates.py` ≈ 35 min end to end. Run long stages in their own terminal: on 24 GB RAM a second concurrent job pushes Windows into pagefile growth that can fill the disk.

Step 8 writes `matching_results.tsv` (US/India from step 7, France = ≥ 2-of-4 vote over steps 3–6) and copies V5's
`candidate_pairs.tsv`: the exact edge set the V4.1–V5 models scored (they share `cand_test_*`), so every submitted pair
is a candidate. Each stage also logs its test-mirror HOLDOUT report (`decide` stages), the number the team trusted.

### How the submitted `candidate_pairs.tsv` was produced (CPU-only reconstruction)
The GPU run's `output_v5/candidate_pairs.tsv` was not copied off the training machine. The submitted file was rebuilt
on a CPU laptop with
```
python rebuild_candidates.py --matching ../../../output/matching_results.tsv --out ../../../output
```
It re-runs the lexical half of the candidate generator with the same code and inputs (roles → tables → parse →
blocking scorer refit on MINE at full scale → blocking of every test S1 against its country's full pool, top-`n_cand`)
and adds the submitted matches that lie outside it (matched dense-retrieval edges). The file is a superset of the matches
but not bit-identical to the scored set: see Determinism below and the documentation, §3.

## Faster checks
- **Smoke run** (~2.5 % of S1, separate `cache_smoke*/`, minutes, CPU is enough for the lexical path):
  `ER4_SMOKE=1 python -m er4.pipeline --stage all`
- **Strict train-only rerun** (test records used only for inference and corpus statistics — no France self-mined tables,
  no test-text encoder adaptation): `ER4_STRICT=1 python -m er4.pipeline --stage all` (own `cache_strict/`).
- `python -m er4.util` imports the metric self-checks (README example 0.714, singleton rules).

## Determinism
Seeds are fixed (`seed=0`, S1-hash folds). Parse, mining and features are deterministic. **Blocking is not**: each pass keeps
its top-K by a sum of IDF weights, ties are common, and polars' multi-threaded `rank("ordinal")` breaks them in thread order, so
two runs on identical inputs keep different tied candidates (measured: ≈ 14 % of edges differ, totals within 0.1 %). Sorting by
(score, record index) before ranking in `er4/blocking.py:index_join` would make it deterministic. GPU training (XGBoost `hist` on CUDA, bf16 cross-encoders, InfoNCE fine-tuning) is not bit-exact, so re-runs land within a
few 1e-4 of the reported HOLDOUT scores rather than on identical files.
