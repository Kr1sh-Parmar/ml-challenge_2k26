# Business Entity Resolution — Version 3 (V3.0.0)

Implementation of `version_3.md` as a Python package (`v3/`), GPU-first (CUDA required).
It reuses the V2 normalization, parsing, blocking and pair features verbatim (`v3/v2_base.py` is lifted from
`version 2/v2_complete_model.ipynb`), so the V2 caches in `version 2/cache/` are the G0 baseline.

## Layout

| Module | Layer(s) | What it does |
|---|---|---|
| `common.py` | L0, L3, L16 | config + compute-tier switch (`V3_TIER=S/M/L`), stage cache, metric, frozen + repeated folds, bootstrap |
| `v2_base.py` | L4, L7 (V2) | V2 library: normalization, parsing, mined substitutions, blocking passes, pair features |
| `data.py` | L1 | train samples S / M (V2), test countries (V2 parse + blocking; India computed here) |
| `dense.py`, `stage_retrieval.py`, `stage_test_dense.py` | L6, L7b | multilingual-e5 bi-encoder fine-tuned on sample M (+ self-supervised test noise variants), exact GPU kNN |
| `pool.py` | L7b, L8 | pool = V2 top-50 ∪ dense top-20 (+ reverse), pool features, GPU-XGBoost student, adaptive M |
| `clusters.py` | L7a, L9 | companion duplicate model, high-precision record clusters with size cap, F-CLUS |
| `fs_em.py` | L5, L9, L10 | Fellegi–Sunter comparison vectors, supervised m/u, per-country EM on GPU, F-EM + standalone score |
| `ce.py` | L10 (tier M) | cross-encoder `microsoft/mdeberta-v3-base` on the uncertainty band, OOF by S1 group, bf16 |
| `stack.py` | L9–L11 | shared feature assembly: level-1 matrix, F-UNC, token matrix |
| `settx.py` | L11 J2 | set transformer v2 with NULL slot + soft-duplicate attention bias, BCE + NULL + soft-F0.5 loss |
| `decision.py` | L12–L14 | monotone GBDT calibrator, expected-F DP, component decomposition + exact solve, learned policy |
| `train.py` | all | OOF training on sample S, adoption ledger (`ledger_train.tsv`), refits saved in `models/` |
| `infer.py` | L15 | per-country test inference → `output/matching_results.tsv`, `output/candidate_pairs.tsv`, validator |

## Run

```bash
# from "version 3/", with the repo venv (Python 3.11, torch CUDA)
python -m v3.data india             # V2 base layers for test countries missing from the V2 cache (CPU)
python -m v3.stage_retrieval        # fine-tune dense encoder on sample M, dense pass on S (GPU)
python -m v3.stage_test_dense       # test embeddings + dense pass (GPU)
V3_TIER=M python -m v3.train        # all OOF layers + refits (tier S skips the cross-encoder)
V3_TIER=M python -m v3.infer        # test inference, writes output/ and runs the official validator
```
Every stage checkpoints to `cache/`; delete a file (or `V3_REFRESH=name`) to recompute it.

## Compute tiers
- **S** — dense retrieval, student pruner, GBDTs, Fellegi–Sunter, clusters, set transformer, L13/L14.
- **M** — S + mDeBERTa-v3 cross-encoder on the student's uncertainty band (default).
- **L** — design lists Qwen3-Reranker / listwise Qwen3-8B; not run here: on one 16 GB A4000 an 8B model over
  ~19M test pairs is out of budget. The `ce_model` switch accepts a larger encoder (e.g. `FacebookAI/xlm-roberta-large`).

## Compliance
Models: multilingual-e5-small (MIT), mdeberta-v3-base (MIT); libraries XGBoost / LightGBM / PyTorch (Apache/MIT/BSD).
Only provided train/test records are used (test records only unsupervised: noise-variant positives, EM, IDF).
