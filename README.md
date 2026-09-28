# ML Challenge 2026 — Business Entity Resolution

Match every Source 1 business record to its Source 2 / Source 3 records (macro F0.5 per S1 entity).
Challenge brief and rules: [`student_resource/README.md`](student_resource/README.md).

## Versions

| Folder | What it is | Validation | Leaderboard |
|---|---|---|---|
| `version 1/` | First end-to-end design + notebooks (`version_1.md`, `er.py`) | — | ~0.95 |
| `version 2/` | Complete V1 design in one notebook (`v2_complete_model.ipynb`) | 10 % sample OOF 0.983 | — |
| `version 3/` | GPU pipeline package `v3/` (dense retrieval, cross-encoder, set transformer, learned policy); design `version_3.md` | 10 % sample OOF 0.994 (optimistic: sample, not test conditions) | — |
| `version 4/` | Test-mirror pipeline `src/er4/` (full-scale, test density); design `version_4.md` | mirror HOLDOUT | |
| &nbsp;&nbsp;V4.1 | V4 + dense-retrieval blocking (`ER4_DENSE=1`) | 0.9836 | |
| &nbsp;&nbsp;V4.2 | + evidence pooling, cross-encoder, L3 stacker (`er4/stack3.py`) | 0.9873 | |
| &nbsp;&nbsp;V4.3 | + pair features, per-country EM, retrained cross-encoder, policy (`er4/final.py`) | 0.98975 | 0.984 |
| &nbsp;&nbsp;V5 | + 398k extra labelled S1s, cross-encoder retrained on 1.09M pairs (`er4/v5.py`, `ER4_V5=1`) | 0.99016 | 0.986 |
| `final_submission/` | Best per country: V5 + learned policy (US, India) + France majority vote of V4.1–V5 | US 0.98969, India 0.99116 | |

Submission files: `version 4/submissions/v4.1 … v5/matching_results.tsv`, `final_submission/final_matching_results.tsv`.

## Environment
- Python 3.11, CUDA GPU (developed on an RTX A4000 16 GB, 64 GB RAM).
- `pip install -r requirements.txt` plus, for V3+: `torch` (CUDA build), `transformers`, `xgboost`, `sentencepiece`
  (see `version 4/requirements.txt`).
- Data goes in `student_resource/dataset/{train,test}/` (not committed).

## Reproduce
Each version's README has its commands. V4.x / V5 (run from `version 4/src`):
```
python -m er4.pipeline --stage all                       # V4
ER4_DENSE=1 python -m er4.dense && ER4_DENSE=1 python -m er4.pipeline --stage all   # V4.1
ER4_DENSE=1 python -m er4.stack3 --stage all             # V4.2
ER4_DENSE=1 python -m er4.final --stage all              # V4.3
ER4_V5=1 python -m er4.v5 --stage all                    # V5
python er4_final_combos.py && python er4_final_combo2.py # final training-free combinations
```
`ER4_SMOKE=1` runs any of them on ~2.5 % of the data in minutes. Caches, models and outputs are regenerated
(multi-GB, git-ignored).

## Rules followed
MIT / Apache-2.0 models only (multilingual-e5-small, mdeberta-v3-base, XGBoost/LightGBM), all ≤ 8B parameters;
no external data, APIs or lookups. Unlabeled test records are used only unsupervised (IDF, self-mined French
tables, encoder adaptation, per-country EM); `ER4_STRICT=1` gives a strictly train-only rerun.
