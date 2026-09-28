# V4.2 submission (2026-09-27)
V4.1 (test-mirror pipeline + dense retrieval) + evidence pooling (record-vs-strongest-mate features) + V3's mDeBERTa
cross-encoder on uncertain edges + L3 GPU-XGBoost stacker; V4 decision layer refit on mirror TRAIN.
- `matching_results.tsv` — upload this (1,732,544 rows; official validator PASS with --check-ids)
Mirror HOLDOUT macro F0.5: 0.9873 (US 0.9866, India 0.9884); V4.1 0.9836.
