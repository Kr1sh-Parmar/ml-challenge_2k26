# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** ZERO BIAS  
**Team Members:** Krish Parmar, Varshil Rathod, Krrish Devani, Krissa Jiyani  
**Submission Date:** 28 September 2026

---

## 1. Executive Summary
A precision-first cascade: country-agnostic parsing and data-mined variant tables → 5 lexical inverted-index blocking passes plus a
reverse pass and a fine-tuned multilingual dense-retrieval pass → GPU XGBoost pair models and context re-rankers → an mDeBERTa-v3
cross-encoder, evidence pooling from each record's strongest duplicate and per-country unsupervised Fellegi–Sunter EM on the uncertain
band → a GBDT stacker → a decision layer that optimises expected per-entity F0.5 (calibration, many-to-one exclusivity, singleton model,
learned cost-sensitive list-length policy). The key innovation is the **test-mirror validation protocol**: after a 10 %-sample model
scored 0.9835 in CV but far lower on the leaderboard, every model, feature and threshold was rebuilt and tuned at full scale with the
test's S1 density, which made the local HOLDOUT score track the leaderboard (V4.3: HOLDOUT 0.98975 → LB 0.984; V5: 0.99016 → LB 0.986).
The unseen country (France) is predicted by a majority vote of four independently trained versions.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from the EDA (full training data unless stated):

| Finding | Value | Design consequence |
|---|---|---|
| Records | train 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3; test 1,732,544 / 4,887,273 / 5,082,316 | everything streamed per country and per chunk (polars, parquet checkpoints) |
| Matches per S1 | mean 3.46; 5.58 % singletons (US 5.58 %, India 5.59 %); max 11 | recall over the whole list matters; "empty" is a full-point decision → singleton model |
| Many-to-one | 0 of 7,638,365 true pairs put an S2/S3 record under two S1s | hard exclusivity: each S2/S3 record goes to at most one S1 |
| Distractors | 26.6 % of S2 and 25.4 % of S3 records match no S1 | rejection matters as much as retrieval |
| Duplicates inside a source | 59 % (S2) / 63 % (S3) of S1s have ≥ 2 records from the same source | evidence pooling between an S1's candidates |
| Country agreement | matched pairs share the country label 100 % | block within the country label (labels treated as an open set) |
| **Density shift** | train 4.7 S2/S3 per S1, test 5.5–5.8: test has **21 % fewer S1 over a same-size pool** | the decisive finding (see 2.2): validate with 21 % of train S1 removed |
| Unseen country | France = 259,452 test S1 (15 %); legal forms found by frequency, not hard-coded: sarl, sas, eurl, sa, sasu, sci, ei | country is never a feature; per-country corpus statistics; unsupervised adaptation |
| Postal codes | present in < 1.5 % of addresses (0.5 % of US S2/S3 after parsing) | address blocking anchors on house number + street instead |
| Missing / junk addresses | 2.4–3.7 % of S2/S3 (`None`, `<NULL>`, `null`) | missing is always "unknown", never a mismatch |

**Noise catalogue** (from matched pairs): native-script transliteration of names and states (Devanagari, Gujarati, Telugu, Kannada:
`Digital इंफ्रा Pvt Ltd`, `ಕರ್ನಾಟಕ`); names replaced by a URL / hashtag / unrelated word (`hbworld.com`, `#integratedtelecom`,
`New Life Catholic Church → Arctavo`); DBA forms (`Wexevo One dba Global Bright LLC`); injected accents (`Frontier Móbile`); legal-suffix
drift and relocation (`LLC Moncada Léarning Center`); `##`, leading zeros and ordinal typos in numbers (`##30920`, `04218`, `1ND`);
street-type and state abbreviation (`Rd/Road`, `NC/North Carolina`, `MH/Maharashtra`); component reordering and dropping; landmarks
(`Nr. Suketu Complex`, `Opp. Ranu Primary School`); unit / door / plot markers; random numeric IDs appended to names.

### 2.2 Solution Strategy
**Approach Type:** Blocking + cascaded classifiers (GBDT + multilingual transformer cross-encoder + probabilistic record linkage),
stacked, with a metric-aware global decision layer; hybrid lexical + dense retrieval.  
**Core Innovation:** the **test-mirror** — validate and fit everything under test conditions — plus a decision layer that optimises
the per-entity F0.5 directly (expected-F0.5 dynamic programme + a learned cost-sensitive policy under many-to-one exclusivity).

**How the design evolved (and why):**

| Version | What changed | Local score | Leaderboard |
|---|---|---|---|
| V1 | 1 blocking pass, 20 rapidfuzz features, LightGBM, global threshold | OOF 0.9596 (10 % sample) | ~0.95 |
| V2 | full parser, mined variant tables, 5 blocking passes + reverse + transitive, 101 features, monotone LightGBM, calibration, duplicate propagation, exclusivity margin, singleton model, expected-F0.5 | OOF 0.9835 (10 % sample) | 0.85 |
| V3 | GPU cascade on the 10 % sample: fine-tuned dense retrieval, distilled pruner, clusters, Fellegi–Sunter EM, mDeBERTa cross-encoder, set transformer, component solver, learned policy | OOF 0.9944 (sample) | — |
| **V4** | **test mirror**: full-scale per-country pools, 21 % S1 dropout, TRAIN / HOLDOUT roles; scale-free features; GPU XGBoost L1 + L2 context re-ranker | HOLDOUT 0.9653 | — |
| V4.1 | + V3's dense retrieval as a blocking pass | 0.9836 | — |
| V4.2 | + evidence pooling (mates), cross-encoder on the uncertain band, L3 stacker | 0.9873 | — |
| V4.3 | + band pair features, per-country EM, CE retrained on mirror TRAIN, 2 stacker rounds, learned policy | 0.98975 | 0.984 |
| V5 | + 398k extra labelled context S1s, CE retrained on 1.09 M band pairs, 2-seed stacker | 0.99016 | 0.986 |
| **Submitted** | US/India = V5 + learned policy; France = ≥ 2-of-4 vote (V4.1–V5) | US 0.98969, India 0.99116 | this file |

**Why V2 failed on the leaderboard (the V4 diagnosis).** V2 was trained and validated on a 10 % sample of S1 *and* of the pool. Its
top features were `x_prod` (42 % of gain) and `blk_rev_rank` (29 %), the rank of an S1 among the S1s competing for the same record —
computed over only 10 % of the S1s — and raw chain/mall counts of a 10 % corpus. At full scale look-alike businesses are 6–10× denser,
so "no competitor" became common and scores became over-confident; the test's missing S1s (21 %) leave records with no rightful owner,
so exclusivity no longer protects neighbours (V2 predicted 3.78 matches per US test S1 against 3.35 in OOF). V4 fixes this by
construction:

**The test mirror** (built per country from the full training data; every train S1 gets `u ~ U[0,1)`, seed 0):

| `u` range | Role | Use |
|---|---|---|
| [0.79, 1.00) | dropped (21 %) | removed from S1; their records stay in the pool as unowned businesses → test density (≈ 5.9 S2/S3 per S1) |
| [0.10, 0.20) | MINE | variant mining, blocking-scorer fit; also the dense encoder's training pairs |
| [0.20, 0.32) | TRAIN | labels for every model (5-fold, grouped by S1) and for the decision layer |
| [0.32, 0.42) | HOLDOUT | final scores only; never used to fit a model |
| [0.42, 0.60) | V5 context labels | extra labelled rows for V5's band models (never seen by earlier layers) |
| rest | context | present as competitors, like unlabelled test S1s |

The mirror is "the test pipeline run on labelled data": the same code path, the same full pools, the same competitor structure.
Scores are computed with the *competition closure* (every HOLDOUT edge plus, for each record in a TRAIN/HOLDOUT list, the edges of its
top-3 competing S1s), which are exactly the edges that can change an exclusivity decision.

**Pipeline (final system):**
```
raw TSVs ─► S0 safe ingestion (strings only, no NA/quote handling, integrity asserts)
        ─► S1 normalisation + parsing  ─► S2 variant tables (train-mined + France self-mined)
        ─► S3 blocking: 5 lexical passes + reverse + dup-group mates ─► blocking scorer ─► top-N  ∪  dense top-20 (+reverse)
        ─► S4 L1 XGBoost (107 pair features)  ─► L2 context re-ranker (list + competition context)
        ─► S5 uncertain band 0.003 ≤ p ≤ 0.997: pair features, mates, cross-encoders, Fellegi–Sunter EM ─► stacker (2 seeds)
        ─► S6 isotonic calibration ─► exclusivity ─► singleton model ─► expected-F0.5 lists ─► learned policy (K ≤ 8)
        ─► S7 France: ≥ 2-of-4 vote of V4.1 / V4.2 / V4.3 / V5 ─► export + official validator (--check-ids)
```

---

## 3. Candidate Generation (Blocking)
Every S1 of a country is blocked against the country's **full** S2 ∪ S3 pool (country labels are an open set; France is blocked
against the French pool). Each pass is an inverted index on hashed keys; keys with more than `cap = max(20, cap_frac · pool)` postings
are skipped as stop-like (so no stop-word list is needed); a candidate's pass score is the sum of IDF of the shared keys
(IDF computed per country on the corpus being blocked); top-K per pass.

- **Blocking keys used:**

| Pass | Keys | top-K | Catches |
|---|---|---|---|
| P1 name characters | 4-grams of the joined core name and of URL names; joined / consonant-skeleton / transliteration-folded keys | 30 | typos, joined words, URLs, transliterations (`saaphttveer` ≡ `software` → `sftvr`) |
| P2 rare name tokens | core + trade-name tokens, sorted-token key, Metaphone, acronym ↔ short name | 20 | word-order swaps, DBA / trade names, acronyms |
| P3 address anchor | `house|street`, `house|first name token`, postal, `postal|first name token`, `house|unit` | 20 | same premises, replaced names |
| P5 address tokens | address tokens + (number, next word) bigrams | 15 | corrupted or replaced names |
| P7 name + address | P2 ∪ P5 keys in one index | 20 | partial evidence on both fields |
| Reverse | P2 + name keys, S2/S3 → S1, indexing **all** S1s of the country | 5 per record | S1s with many matches that per-S1 truncation cuts |
| P6 duplicate mates | S2/S3 groups sharing the sorted core name (size 2–6) | ≤ 5 per S1 | weak duplicates of a strong candidate |
| Dense (V4.1+) | multilingual-e5-small fine-tuned with InfoNCE on MINE true pairs (batches grouped by country/region → hard in-batch negatives) + self-supervised noise-variant positives; exact GPU kNN per country | 20, + reverse top-3 | heavy name replacement, transliteration, component reordering |

  The union of the lexical passes is ranked by a **blocking scorer** (standardised logistic regression on 20 features: pass scores / log-ranks /
  found-flags / reverse rank / P6 flag / number of passes), fitted on MINE edges at full scale; the list size `n_cand` is the smallest N
  in {50, 60, 80} within 0.002 of the best completeness on MINE. Dense-only edges receive the scorer's value for "no lexical pass
  found it". Competition features are then computed over all edges of the country (`blk_rev_rank`, `rev_n`, `blk_rev_gap`).

- **Candidate pairs generated:** **124,112,134** pairs in `output/candidate_pairs.tsv` (71.6 per S1; 1 of 1,732,544 S1s has an empty
  list): the re-run lexical top-N set (123,769,532 pairs, 71.4 per S1, `n_cand` = 80) plus the 342,602 submitted matches (5.90 % of
  5,809,459) that lie outside it (see the note below). **Reduction ratio 0.99998** against all 6.72 × 10¹² within-country pairs.

  | Country | Test S1 | Pool (S2 ∪ S3) | Lexical / S1 | Candidates / S1 | Matches outside lexical | Reduction ratio |
  |---|---|---|---|---|---|---|
  | US | 663,106 | 3,817,031 | 70.0 | 70.1 | 63,659 (2.85 %) | 0.999982 |
  | India | 809,986 | 4,717,565 | 75.0 | 75.2 | 178,286 (6.52 %) | 0.999984 |
  | France | 259,452 | 1,434,993 | 64.1 | 64.5 | 100,657 (11.99 %) | 0.999955 |

  France relies most on the non-lexical edges — consistent with dense retrieval being most valuable where the lexical tables
  (mined on US/India) transfer least.
- **How we ensured true matches were not lost:**
  - multiple complementary passes (name characters, rare tokens, address anchors, address tokens, joint name+address), both
    directions (the reverse pass rescues S1s with many matches), duplicate-group expansion, and a dense pass for heavily corrupted names;
  - measured recall at every change. On the 10 % sample (V2 lexical top-50): pair completeness **0.9734**, entity-level ceiling
    0.9189, oracle macro F0.5 0.9907. Adding the dense pass (V3, same sample): pair completeness **0.99918** at 64.8 pairs per S1,
    oracle macro F0.5 **0.99976**. At full scale on MINE (lexical only, test density, measured in this rebuild) the task is much
    harder — the same passes give a union recall of 0.957 (US, 84.5 edges per S1) / 0.930 (India, 93.0) and, after the blocking
    scorer, completeness at N = 50 / 60 / 80 of 0.952 / 0.954 / 0.956 (US) and 0.916 / 0.921 / 0.927 (India), hence `n_cand` = 80.
    Full-scale crowding is exactly why the dense pass was added in V4.1 (+1.8 points of HOLDOUT F0.5 over V4);
  - pass contributions (V2 sample, recall alone / found only by this pass): P1 0.730 / 0.0070, P2 0.709 / 0.0002, P3 0.668 / 0.0072,
    P5 0.791 / 0.0037, P7 0.933 / 0.0014, reverse 0.730 / 0.0013, P6 0.0024 / 0.0024 — every pass contributes unique true pairs;
  - completeness vs list size (V2 sample): N = 10 / 20 / 30 / 50 / 80 → 0.942 / 0.959 / 0.967 / 0.973 / 0.976 (knee near 50).

**Note on `candidate_pairs.tsv` in this package.** The file that the V4.1–V5 models scored is lexical top-N ∪ dense top-20 (+ reverse
top-3). Our GPU training machine's copy of it was not preserved, and the dense half cannot be recomputed without the fine-tuned
encoder weights. The submitted file was therefore rebuilt on a CPU laptop with `src/rebuild_candidates.py`: the lexical half is
re-run with the same code and inputs (same tables, parser, passes, blocking scorer refit on MINE at full scale, same `n_cand`), and the
dense half is represented by its matched edges (every submitted pair was a candidate of the scoring run — each export asserts
`matches ⊆ candidates`), so the file is a superset of the matches.

It is *not* bit-identical to the scored set, for a reason we measured while rebuilding: each blocking pass keeps its top-K by a sum of
IDF weights, many candidates tie on that sum, and polars' multi-threaded `rank("ordinal")` breaks ties in a thread-order-dependent way,
so two consecutive runs on identical inputs keep different tied candidates (≈ 14 % of edges differ between runs, mostly low-ranked
look-alikes; totals agree within 0.1 %). Consequently the "matches outside the lexical set" count reported below mixes true
dense-retrieval edges with tie-break differences and is an upper bound on the dense contribution. A deterministic tie-breaker
(sort by score, then record index) would remove this; we did not change the code the submission was produced with. The full pipeline's
own file is `ER_WORK/output_v5/candidate_pairs.tsv` (copied by `make_final.py`).

---

## 4. Matching Model

**Features used** (107 per pair at L1; every feature is country-agnostic, missing information is encoded as −1 "unknown" and never as
a mismatch; no country one-hot, no IDs, no row order):
- **Name features (29):** fuzzy ratio, token-set / token-sort / partial ratios, Jaro-Winkler (full and core name), normalised
  Levenshtein, LCS, character-3-gram cosine, token Jaccard, soft Jaccard (JW ≥ 0.9), Monge-Elkan, IDF-weighted overlap and max shared
  IDF, first / last token equality, best DBA-variant similarity, length ratio, token-count difference; key equalities (sorted, joined,
  consonant skeleton, transliteration-folded, Metaphone), acronym match, URL-name match / JW, legal form same / conflict / unknown,
  numbers in the name.
- **Address features (25):** postal / postal prefix / house / house digits / unit / region / locality (each 3-valued), numeric-set
  Jaccard, street Jaccard, landmark flags and similarity, candidate name inside the other's landmark, missing-component counts;
  char-3-gram cosine, token-set / ratio / partial, IDF overlap, token Jaccard, length ratio, empty flags.
- **Other:** name × address product and minimum, name-inside-address both ways, field-swap similarity (5); record quality (17:
  token/char counts, DBA / legal / URL / native-script flags, IDF sums, source S2/S3); scale-free chain/mall rates
  `log1p(count per 100k)` (6); blocking provenance and competition (21: per-pass rank/score, reverse rank/score, P6, pass count,
  blocking score / rank / gap / reverse rank, list size, `rev_n`, `blk_rev_gap`); dense cosine / rank / reverse rank / in-lexical flag (4).
- **Context (L2, 16):** rank, gap to best, counts ≥ 0.5 / 0.9 and score sum within the S1's list; the best competing S1's score for the
  same record, the gap to it, this S1's rank for the record, S1s ≥ 0.5 for it; blocking context.
- **Band stacker (222):** L2 context + L1 score + two cross-encoder scores (V4.3 and V5 models) with presence / difference flags + 78
  evidence-pooling ("mate") features — the 75 content features comparing the candidate with its S1's strongest other candidate, the
  mate's score and the number of strong mates — + the 107 pair features + 14 Fellegi–Sunter features (12 per-field log-likelihood-ratio
  weights, total weight, EM posterior).

**Model type:**

| Layer | Model | Training |
|---|---|---|
| L1 pair model | XGBoost (`hist`, CUDA), monotone increasing in the core similarities (`n_tset`, `n_jw_core`, `n_c3`, `n_idf_ov`, `a_tset`, `a_c3`, `blk`, `dcos`) | 5-fold OOF grouped by S1 on mirror TRAIN; final = 1.1 × mean best iteration |
| L2 context re-ranker | XGBoost, monotone in the L1 score | OOF, same folds |
| Cross-encoder | `microsoft/mdeberta-v3-base` (MIT), pair of raw "name ; address" strings, bf16; V3 → retrained on mirror TRAIN band pairs (V4.3) → retrained on 1.09 M labelled band pairs, warm-started, OOF by halves (V5) | scores the uncertain edges (0.003 ≤ p ≤ 0.997; CE-v5: 0.01 ≤ p ≤ 0.99) |
| Fellegi–Sunter | 12-field comparison vectors; m/u initialised supervised on labelled folds, then **re-estimated by EM separately per country** on that country's own unlabelled band pairs (France adapts without labels) | GPU EM, 25 iterations max, floored m/u |
| Band stacker | XGBoost, 2 seeds averaged, monotone in the L2 and cross-encoder scores and the EM total | OOF over all labelled band rows (TRAIN + context S1s) |
| Singleton model | XGBoost on list statistics → P(S1 has ≥ 1 match), isotonic-calibrated | OOF |
| Learned policy | XGBoost regressor of the per-entity cost `1 − F0.5(top-k)` for k = 0…8 from 18 list features (incl. the expected-F0.5 of each k); choose argmin | 5-fold OOF on labelled S1s |

**Threshold selection method:** no single threshold. (1) isotonic calibration of the stacker's OOF scores (cross-fitted on TRAIN);
(2) exclusivity — each S2/S3 record keeps only its best S1, with an ambiguity margin ε / down-weight λ grid-searched;
(3) for each S1, choose k = 0…max_k maximising the **expected per-entity F0.5** of predicting its top-k, computed exactly with a
Poisson-binomial dynamic programme (k = 0 is valued by the calibrated singleton probability; unselected probability mass and the
measured blocking-miss rate count as expected false negatives); guardrails (probability floor, max list) are grid-searched on
mirror TRAIN; the rule is kept only if it beats a tuned global threshold. (4) The learned policy corrects the expected-F choice using
real per-entity outcomes. Every choice is fitted on mirror TRAIN and judged on HOLDOUT; components were adopted when a paired
bootstrap confidence interval on labelled data was above zero. One documented exception: in V5 the learned policy did not clear that
bar on the labelled S1s, but it scored higher on HOLDOUT in both countries (US 0.98969 vs 0.9895, India 0.99116 vs 0.9911), so the
submission switches it on for US/India — a choice that used HOLDOUT, so those two HOLDOUT numbers are slightly optimistic.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - V5 test-mirror HOLDOUT **0.99016** (US 0.9895, India 0.9911); +0.00063 over V4.3, paired bootstrap CI [+0.00053, +0.00074].
  - Submitted configuration on HOLDOUT (V5 + learned policy): **US 0.98969, India 0.99116** (France has no labels).
  - Public leaderboard: V4.3 0.984, V5 0.986 (from V4 on, the local HOLDOUT tracked the leaderboard to within 0.004–0.006).
  - Submitted file (validator PASS with `--check-ids`): 5,809,459 matches; 3.373 / 3.375 / 3.236 matches per S1 and 5.76 % predicted
    empty for US / India / France (the true singleton rate in training is 5.58 %).

- **Common false positives (wrong merges):**
  - *Shared premises, different business* — malls, business parks, co-working addresses: `preferred wisekey | 905 clark pl unit 1016
    nashville` vs `dovaveo | 905 clark pl nashville`; `bl lease` vs `kl service` at the same street address.
  - *Chains and near-identical names* at nearby addresses: `clarkstown community valley service | 10 hamden heights ct` vs
    `... service group 74191 | 13 hamden heights ct`.
  - *Name-only evidence with an empty address*: `ru xsolla llc` vs `ru xs0lla llc` (no address to disambiguate branches).
  - Mitigations: competition features and exclusivity, chain/mall frequency rates, name × address evidence features, evidence pooling
    across duplicates, and a decision layer that needs more confidence for each additional list item.

- **Common false negatives (missed matches):**
  - *Replaced or URL / hashtag names with little address overlap* — hard true matches in the training data such as
    `Integrated Telecom Technologies | 20 2115, Alpine, AZ → #integratedtelecom | 2115, ALPINE, AZ` and
    `New Life Catholic Church → Arctavo` (same address, name replaced by an unrelated word).
  - *Native-script names with truncated addresses* (missed by the V1 model: `Hotel Healthcare Private Limited` vs
    `ಹೋಟೆಲ್ ಹೆಲ್ತ್‌ಕೇರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | #02704/8, Mysore, ಕರ್ನಾಟಕ`).
  - *Blocking misses* before the dense pass (≈ 2.7 % of true pairs lexically; ≈ 0.08 % with dense retrieval on the sample).
  - *Deliberate misses*: the 4:1 cost of a false positive makes the policy stop early on crowded lists — the 4th/5th duplicate of an S1
    is often left out when a look-alike competitor exists.

- **Generalisation to an unseen country (France proxy):** leave-one-country-out on V2 features: train US → India 0.9413 (−0.038 vs
  in-country), train India → US 0.9787 (−0.007). Feature families that help LOCO most (ablation): frequency (−0.0034 without it),
  address components (−0.0028), provenance (−0.0015). This motivated scale-free rates, country-agnostic features, per-country EM,
  multilingual encoders and the France vote.

---

## 6. Conclusion
The decisive step was not a bigger model but an honest validation set: rebuilding everything at full scale with the test's S1 density
turned a 0.98 → 0.85 CV/leaderboard gap into a HOLDOUT that tracks the leaderboard, after which each added layer (dense retrieval,
cross-encoder, evidence pooling, EM, stacking, learned policy) was accepted only on paired-bootstrap evidence. Lessons: validate under
test conditions before tuning; make every feature scale-free; optimise the metric per entity rather than a global threshold; and for an
unseen domain, prefer agreement between diverse models over any single one.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — full instructions in its `README.md`, pinned versions in `requirements.txt`.

| Path (`src/`) | Role |
|---|---|
| `erpaths.py` | dataset (`ER_DATA`) and work-folder (`ER_WORK`) resolution |
| `v2/prepare.py` | V2 base layers (verbatim V2 notebook cells): samples, mined tables, parsed samples, anchor model |
| `v3/` | dense bi-encoder fine-tuning + GPU kNN (`dense.py`), cross-encoder (`ce.py`), Fellegi–Sunter EM (`fs_em.py`), expected-F / policy (`decision.py`), V3 training (`train.py`), V2 library (`v2_base.py`) |
| `er4/pipeline.py` | V4 / V4.1 stages: roles → mine → parse → blkfit → dense → block → trainfeat → l1 → score → l2 → decide → export |
| `er4/stack3.py`, `er4/final.py`, `er4/v5.py` | V4.2, V4.3, V5 stages |
| `er4_v5_policy_export.py`, `make_final.py` | V5 + policy export; final assembly (US/India V5+policy, France vote) + validator |
| `rebuild_candidates.py` | CPU-only regeneration of `candidate_pairs.tsv` (how the submitted file was produced) |
| `anchor_v2/` | V2's mined tables (used by every V4/V5 run) and V2's LightGBM (the mirror anchor) |

Entry points, run from `src/`: `python v2/prepare.py` → `V3_TIER=M python -m v3.train` →
`ER4_DENSE=1 python -m er4.pipeline --stage all` → `ER4_DENSE=1 python -m er4.stack3 --stage all` →
`ER4_DENSE=1 python -m er4.final --stage all` → `ER4_V5=1 python -m er4.v5 --stage all` →
`ER4_V5=1 python er4_v5_policy_export.py` → `python make_final.py --out <zip>/output`.

**Models and licences** (all ≤ 8 B parameters): `intfloat/multilingual-e5-small` (MIT, 118 M), `microsoft/mdeberta-v3-base`
(MIT, 278 M), XGBoost (Apache-2.0), LightGBM (MIT), own models (logistic blocking scorer, Fellegi–Sunter, policy, set transformer).
**Data:** only the provided files. No external data, APIs, geocoders, registries, gazetteers or address parsers; hand-written lists
are limited to generic normalisation seeds (street types, compass words, unit markers, legal-form seeds, stop words, landmark markers)
in `er4/normalize.py`. Unlabelled **test** records are used only unsupervised: per-country IDF and frequency statistics, legal-form
discovery and a self-mined French substitution table (124 address typo pairs, from high-precision seed pairs, no human review),
noise-variant positives for the dense encoder's self-supervised adaptation, and per-country Fellegi–Sunter EM. A strictly
train-only rerun is available (`ER4_STRICT=1`).

### B. Additional Results

**V2 decision-layer ledger (10 % sample OOF):** L1 + global threshold 0.98314 → isotonic 0.98313 → duplicate propagation 0.98337 →
exclusivity margin (ε 0.2, λ 0.75) 0.98348 → singleton model + expected-F0.5 0.98352. V2 L1 OOF AUC 0.99998; singleton-model AUC
0.9988; micro precision 0.9968, micro recall 0.9613.

**V3 ledger (10 % sample, OOF):** V2 top-50 completeness 0.97327 → + dense pool 0.99918 (64.8 pairs/S1), oracle 0.99976; distilled
student + exclusivity 0.99159; XGBoost level-1 0.99191; cross-encoder AUC in the uncertainty band 0.9786 (XGBoost 0.9774); set
transformer 0.99408; GBDT stacker 0.99419; learned policy 0.99436 (+0.00026 vs expected-F, CI [+0.00017, +0.00034]); Fellegi–Sunter
alone 0.9429.

**V2 feature importance (gain share by family):** cross 0.43, provenance 0.41, address similarity 0.09, name similarity 0.03 — the
concentration on `x_prod` and `blk_rev_rank` is what failed at full scale and motivated V4's scale-free redesign.

**Submission profile per country:**

| Country | Test S1 | Matches / S1 | Predicted empty | Source |
|---|---|---|---|---|
| US | 663,106 | 3.373 | 5.76 % | V5 + learned policy |
| India | 809,986 | 3.375 | 5.76 % | V5 + learned policy |
| France | 259,452 | 3.236 | 5.76 % | ≥ 2 of 4 versions (vote histogram 1/2/3/4: 58,536 / 22,490 / 18,037 / 798,981 pairs) |
