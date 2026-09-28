# V4.3 final submission (2026-09-27)
V4.2 + full pair features for uncertain edges + per-country Fellegi-Sunter EM (France adaptation) + mDeBERTa cross-encoder
retrained on mirror TRAIN + two evidence-pooling rounds + GPU-XGBoost stacker + learned cost-sensitive decision policy.
- `matching_results.tsv` — 1,732,544 rows; official validator PASS with --check-ids
Mirror HOLDOUT macro F0.5: 0.98975 (US 0.9892, India 0.9905); V4.2 0.9873; V4.1 0.9836.
