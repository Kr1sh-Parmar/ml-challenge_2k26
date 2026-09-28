# Business Entity Resolution — Model Version 4

**Goal:** close the gap between local validation and the leaderboard. V2 scored **0.9835** macro F0.5 in cross-validation
and **0.85** on the final evaluation. V4 is designed so that every model, threshold and reported number is produced
under the **same conditions as the test set**, and the local score is shown to predict the leaderboard before we trust it.

---

## 1. What went wrong in V2 (evidence)

The model was not the weak point: on its own candidates a perfect classifier would reach 0.9907, and V2 reached 0.9835.
The loss came from **differences between the data we validated on and the data we were scored on**.

| # | Problem | Evidence | Why it hurts on test |
|---|---|---|---|
| 1 | **Scale shift** | V2 trained and validated on sample S = 10% of train S1 *and* 10% of the S2/S3 pool (220k S1, 1.03M S2/S3). Test ran per country at full scale (US: 663k S1, 3.8M S2/S3). Feature gain: `x_prod` 42%, **`blk_rev_rank` 29%**, `blk` 11%. `blk_rev_rank` ranks an S1 among all S1s competing for the same S2/S3 record, and was computed over only 10% of the S1s. Chain / frequency counts (`f_core_*`, `f_addr_*`) were raw counts of a 10% corpus. Per-pass `top_k` was tuned at 10% scale. | Look-alike businesses (chains, same street, same mall) are ~6–10× denser at full scale. A feature value that meant "no competitor" in training is common at test, so the model gives confident scores to wrong candidates. Blocking lists fill up with look-alikes, pushing true matches out. |
| 2 | **Density shift (S1 dropout)** | Train: 2.20M S1 over 10.3M S2/S3 (4.7 per S1). Test: 1.73M S1 over 10.0M S2/S3 (5.8 per S1): the same pool size with **21% fewer S1**. V2's sample kept the *train* density. | Records of businesses whose S1 is absent have no rightful owner, so exclusivity and reverse rank stop protecting neighbours. V2 predicted 3.78 matches per US test S1 vs 3.35 in OOF: the extra ones are mostly false merges, which F0.5 punishes 4:1. |
| 3 | **Decision layer fitted on the wrong distribution** | Isotonic calibration, exclusivity margin, global threshold, expected-F guardrails and `m_miss` were all fitted on sample S. | Calibrated probabilities are over-confident at test, so expected-F0.5 picks lists that are too long. |
| 4 | **Unseen country (France, 15% of test)** | Leave-one-country-out: US→India 0.941 (−0.038 vs in-country), India→US 0.979 (−0.007). France self-mining found typo pairs only. | A second-order loss (≈ −0.005 to −0.01 overall), but real. |
| 5 | **No local check predicted the leaderboard** | Every reported number came from sample S. | The 0.13 gap was invisible until submission. |
| 6 | **Engineering** | OOM from Python row lists (fixed late in V2), 5 GB pickles filled the disk, the OpenCL LightGBM GPU build crashed on 2 of 5 folds (`best_split_info.left_count > 0`). | Lost runs, restarts, no GPU speed-up. |

---

## 2. Principles

1. **Test-mirror everything.** Candidates, features, models and thresholds are produced on full-scale, per-country pools
   with the test's S1 density. Nothing is fitted on a scaled-down sample.
2. **Prove the mirror first.** V2's model is re-scored on the mirror hold-out; if that lands near the leaderboard's 0.85,
   the mirror is trustworthy and V4's hold-out score is a real estimate of the leaderboard.
3. **Same code path for mirror and test.** The mirror is "the test pipeline run on labelled data".
4. **Scale-free features.** No feature whose meaning changes with corpus size.
5. **Keep what worked.** V2's normalization, mining, blocking passes and pair features are ported, not rewritten.
6. **Stay inside the machine.** 24 GB RAM, ~30 GB free disk, 6 GB GPU: stream per country and per chunk, parquet (zstd)
   checkpoints, no pickled frames.

---

## 3. Validation protocol: the test mirror

Built per country (US, India) from the **full** training data. Every train S1 gets a uniform number `u` (seed 0, the same
draw V2 used, so V2's sample S and mining sample M map onto known ranges):

| `u` range | Role | Share | Used for |
|---|---|---|---|
| `[0.79, 1.00)` | **Dropped** | 21% | Removed from the S1 table; their S2/S3 records stay in the pool as unowned businesses → test density |
| `[0.10, 0.20)` | **MINE** | 10% | Variant mining (same slice as V2's M) and fitting the blocking scorer |
| `[0.20, 0.32)` | **TRAIN** | 12% | Labels for the L1/L2 models (5-fold, grouped by S1) and the decision layer |
| `[0.32, 0.42)` | **HOLDOUT** | 10% | Final scores only. Never used to fit or tune anything |
| rest | context | 47% | Present in the S1 table as competitors, exactly like the unlabelled test S1s |

Resulting mirror: ≈1.74M S1 over the full 10.3M S2/S3 pool (≈5.9 per S1), per country, at full scale.

**Scores reported (all macro F0.5, per country and overall):**
- **Anchor:** V2's L1 model with V2's rule (threshold 0.7 + exclusivity) on HOLDOUT. Expected ≈0.85 if the mirror is right.
- **V4 OOF** on TRAIN (tuning score) and **V4 HOLDOUT** (the estimate we trust).
- **LOCO** on TRAIN: US-only model → India, India-only model → US. The France proxy.
- Blocking pair completeness and oracle F0.5 at full scale.

To keep the mirror affordable, hold-out and competitor scoring use the **competition closure**: every HOLDOUT edge,
plus, for each S2/S3 record in a TRAIN or HOLDOUT list, the edges of its top-3 competing S1s by blocking score.
Those are the only edges that can change an exclusivity decision.

---

## 4. Layer-by-layer design

### L1 · Normalization and parsing (ported from V2 + French gaps)
- V2 parser kept: NFKC + ligatures + Unidecode transliteration, elision split, numeric-token protection, seed
  contractions, mined substitution tables, DBA/URL/legal/landmark/postal/unit/house/street/segment parsing,
  name keys (sorted, joined, skeleton, transliteration, metaphone, acronym).
- V2 already handled rue/chemin/impasse/route, French legal forms and elisions. **Added for France:** French
  ordinals (`1er`, `2e`, `3eme`), `bis/ter/quater` after a house number, `cedex` removed, `arrondissement`→`arr`,
  `allees`→`all`, `boulevard`/`bd` both → `blvd` (already), `centre commercial`→`cc`.
- Parsing is batch-wise straight into Arrow; parsed frames are stored as zstd parquet per split and country.
- The legal-form set and the substitution tables are passed to worker processes explicitly (not module globals).

### L2 · Candidate generation (blocking)
- V2's passes kept: P1 name characters, P2 rare name tokens, P3 address anchor, P5 address tokens, P7 name+address,
  reverse pass (S2/S3 → S1), P6 duplicate-group mates; per-country IDF; posting caps as a fraction of the pool.
- **Every S1 of a country is blocked against the country's full pool**, for the mirror and for the test alike.
  The reverse pass always indexes *all* S1s, so reverse ranks mean the same thing everywhere.
- Blocking scorer (standardized logistic regression) refit on MINE edges at full scale.
- `n_cand` chosen from the full-scale completeness curve on MINE (smallest N within 0.002 of the best of 50/60/80).
- **Competition features** computed over all edges of the country: `blk_rev_rank` (rank of this S1 for the record),
  `rev_n` (number of S1s competing for the record), `blk_rev_gap` (best competing blocking score minus this one).

### L3 · Pair features
- V2's 101 features kept (name similarity, name keys, address components, address similarity, name×address cross,
  quality, provenance).
- **Scale-free frequencies:** chain/mall counts are converted to `log1p(count per 100k records of that corpus side)`.
  The raw counts are still computed (the V2 anchor model needs them), but V4 models never see them.
- New: `rev_n`, `blk_rev_gap`.

### L4 · Matching models (GPU)
- **L1: XGBoost on CUDA** (`tree_method=hist`, `device=cuda`, Apache-2.0), binary logistic, monotone increasing
  constraints on the core similarities (`n_tset`, `n_jw_core`, `n_c3`, `n_idf_ov`, `a_tset`, `a_c3`, `blk`) for
  robustness across countries. 5-fold OOF grouped by S1, early stopping, final model on all TRAIN with
  1.1 × mean best iteration. Chunked `inplace_predict`. Falls back to CPU automatically if CUDA is unavailable.
  Chosen over LightGBM because LightGBM's GPU build crashes here; parity was checked on V2's cached features.
- **L2: XGBoost context re-ranker (stacking).** Inputs are L1's score and its context, available for every edge:
  rank, gap to the best and count above 0.5 / 0.9 within the S1's list; the best competing S1's L1 score for the same
  record, the gap to it, the rank of this S1 for the record and the number of S1s above 0.5 for it; blocking score,
  rank, reverse rank and `rev_n`. This lets the model learn crowding and exclusivity from data. Trained on TRAIN
  (OOF, same folds). Kept only if it beats L1 on TRAIN OOF after the decision layer.
- **Singleton model:** entity-level XGBoost, P(S1 has at least one match), isotonic-calibrated.
- **Dropped from V2:** the companion duplicate-group model and propagation (+0.0002 in V2, extra cost and fragility).
- **Smoke mode:** `ER4_SMOKE=1` runs every stage on ~2.5% of S1 in minutes (separate `cache_smoke/`) to catch bugs
  before a multi-hour run.
- **Not chosen:** transformer cross-encoders / embedding blocking. The gap is distribution shift, not capacity, and the
  installed torch is CPU-only. Revisit only if HOLDOUT errors look capacity-bound.

### L5 · Decision layer (fitted on mirror TRAIN, evaluated on HOLDOUT)
1. Isotonic calibration of the chosen model's OOF scores.
2. Exclusivity with ambiguity margin (grid over ε, λ) using the scores of **all** competing S1s.
3. Singleton probability.
4. Expected-F0.5 list selection (Poisson-binomial DP over k = 0…max_k) with guardrails (score floor, max list size) and
   `m_miss` = blocking misses per S1 measured at full scale; compared with the plain global-threshold rule.
5. The rule with the best TRAIN OOF score is kept and applied unchanged to HOLDOUT and to test.

### L6 · Inference and resources
- Per country, per chunk; lean dtypes; zstd parquet checkpoints; intermediate edges deleted after use.
- Start-up guard: warn below 25 GB free disk, stop below 8 GB.
- LightGBM is only used to score the V2 anchor.

---

## 5. Pipeline stages (`python -m er4.pipeline --stage <name>` or `v4_run.ipynb`)

| Stage | Output (in `version 4/cache/`) | Notes |
|---|---|---|
| `roles` | `roles.parquet` | mirror roles per train S1 |
| `mine` | `tables.pkl` | variant tables (train MINE + France self-mining); the first full run was seeded with V2's tables (same MINE slice and RNG, so identical definition) |
| `parse` | `parsed_{split}_{cty}.parquet` | split ∈ {mirror, test} |
| `blkfit` | `blk_scorer.pkl`, completeness curve | fitted on MINE at full scale |
| `block` | `cand_{split}_{cty}.parquet` | all S1s, top-N, competition features |
| `trainfeat` | `train_X.parquet` | TRAIN edges |
| `l1` | `l1.json`, `l1_oof.parquet`, LOCO report | XGBoost CUDA |
| `score` | `p_{split}_{cty}.parquet` | mirror closure + all test edges; V2 anchor score on the mirror |
| `l2` | `l2.json`, `l2_oof.parquet` | context re-ranker |
| `decide` | `decision.pkl`, HOLDOUT report | tuned on TRAIN, scored on HOLDOUT |
| `export` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | + official validator `--check-ids` |

---

## 6. Resource budget (measured in V2, scaled)

| Step | Estimate |
|---|---|
| Parse + enrich (≈24M records) | ~20 min |
| Blocking (mirror + test, 4 country passes) | ~30 min |
| TRAIN features (~13M pairs) | ~15 min |
| L1 + L2 on GPU | ~15 min |
| Mirror closure scoring (~25M pairs) | ~25 min |
| Test scoring (~85M pairs) | ~85 min |
| Total, first run | ~3.2 h |

Disk: ≈12–15 GB of V4 checkpoints (V2's cache is deleted first).

---

## 7. Results log

| Milestone | Result |
|---|---|
| M0 GPU parity (V2's 10.1M × 101 features, same S1-grouped fold) | LightGBM CPU: logloss 0.003204, AUC 0.999974, 834 iters, 212 s. **XGBoost CUDA: logloss 0.003194, AUC 0.999975, 780 iters, 192 s** (no crashes) → adopted |
| M0 smoke run (`ER4_SMOKE=1`, ~2.5% of S1) | all 11 stages pass in 4.8 min. At this tiny scale the V2 anchor also scores ~0.98 on HOLDOUT — expected: crowding only exists at full scale, which is the point of the mirror. France S2/S3 parse coverage: house 0.93, street 0.93 (US 0.92 / 0.89). |
| M1 Anchor (V2 on mirror HOLDOUT) | *pending* |
| M2 Full-scale blocking | *pending* |
| M3 L1 OOF + LOCO | *pending* |
| M4 L2 + decision, HOLDOUT | *pending* |
| M5 France parse coverage | *pending* |
| M6 Test export + validator | *pending* |
