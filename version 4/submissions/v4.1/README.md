# V4.1 submission (2026-09-27)
V4 test-mirror pipeline + dense retrieval blocking (V3's fine-tuned multilingual-e5 bi-encoder), `ER4_DENSE=1`.
- `matching_results.tsv` — upload this (1,732,544 rows; official validator PASS with --check-ids)
- `candidate_pairs.tsv` — candidate set for the final zip
- `probe_france_empty/matching_results.tsv` — France predicted empty (leaderboard probe)

Mirror HOLDOUT macro F0.5: 0.9836 (US 0.9838, India 0.9832); V4 baseline 0.9653; V2 anchor 0.957.
