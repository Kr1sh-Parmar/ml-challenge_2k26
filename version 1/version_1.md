# Business Entity Resolution — Solution Design, Version 1

**Challenge:** Amazon ML Challenge — Business Entity Resolution
**Owner:** Krish
**Status:** V1 design, written before EDA
**Date:** 25 September 2026

---

## 0. How to read this document

This is the V1 blueprint for the full pipeline, from the raw TSV files to the two submission files. It names every component, model and technique we will use, and why. It contains no code.

Parameter values are marked as **starting values**. Every one of them is tuned on out-of-fold (OOF) validation.

Items marked **[Verify]** are assumptions that the EDA in Stage 1 must confirm before we depend on them.

---

## 1. Objective and scoring

**Task.** For every Source 1 entity (the deduplicated reference), return every Source 2 / Source 3 record that refers to the same real business.

- **Training data:** US and India.
- **Test data:** US, India, and France. France never appears in training.

**Metric.** F0.5 is computed per S1 entity, then macro-averaged over all S1 entities. In count form:

> **F0.5 = 5·TP / (5·TP + FN + 4·FP)**

| Property of the metric | Consequence for the design |
|---|---|
| A false positive costs 4× a false negative | Precision comes first. Every stage is judged by how many false merges it causes. |
| Macro-average over S1 entities | Each entity counts equally. An S1 with 10 matches weighs the same as an S1 with 1 match or a singleton. |
| Singletons score 1.0 if predicted empty, 0.0 otherwise | Empty vs. non-empty is a full-point decision for every entity, so it gets its own model (Stage 9). |
| Empty prediction on a non-singleton scores 0 (assumed) **[Verify]** | Predicting empty is not free for entities that do have matches. |
| With k correct predictions out of G true matches and no false positives, F = 5k / (4k + G) | The first correct match earns most of an entity's value. It can be included at lower confidence than later matches. |
| The private leaderboard decides the ranking, and France is unseen | Generalization matters more than fitting the public leaderboard. |

**Hard constraints**

- Every model must be MIT or Apache 2.0 licensed and have at most 8B parameters.
- No external data, APIs, lookups or geocoding.
- Final matches must be a subset of the submitted candidates.
- Both output files must follow the exact format rules and pass the provided validator.

---

## 2. Design principles

| # | Principle | What it means in practice |
|---|---|---|
| 1 | Precision first, recall second | A candidate is predicted only when its expected contribution to F0.5 is positive. There is no fixed global threshold. |
| 2 | Structure is a signal | Each S2/S3 record belongs to at most one S1 entity (many-to-one). This constraint is enforced explicitly after scoring. |
| 3 | Country-agnostic by construction | No feature, rule or filter assumes the set {US, India}. Country is never a model input. |
| 4 | The recall ceiling is decided in blocking | Blocking uses multiple passes in both directions, and its recall is measured before any modelling. |
| 5 | Everything is OOF and measured globally | Every learned component produces out-of-fold outputs. The score is computed over the whole training set at once. |
| 6 | Mine, don't hand-code | Variant and abbreviation tables are learned from the provided data. Hand-written lists are limited to small, documented seed lists. |
| 7 | Reproducible and auditable | The pipeline has one config and one entry point, with pinned versions, fixed seeds and a documented methodology. |

---

## 3. Pipeline at a glance

```
Raw TSV files (train + test)
        │
        ▼
[S0]  Ingestion & integrity checks
        │
        ▼
[S1]  EDA & assumption checks ─────────────► design decisions
        │
        ▼
[S2]  Normalization & field parsing
        │
        ▼
[S3]  Variant mining  (training ground truth + test self-mining)
        │
        ▼
[S4]  Blocking: 6 passes, both directions, union + cap ──► candidate_pairs.tsv
        │
        ▼
[S5]  Feature engineering  (~100 features)
        │
        ▼
[S6]  Pairwise models: LightGBM base → cross-encoder → LightGBM stacker (+ ensemble)
        │
        ▼
[S7]  Calibration (isotonic)
        │
        ▼
[S8]  Global constraints: exclusivity + duplicate-group propagation
        │
        ▼
[S9]  Decision layer: singleton model + expected-F0.5 subset selection
        │
        ▼
[S10] Export + validation ─────────────────► matching_results.tsv
```

| Stage | Purpose | Main techniques | Output |
|---|---|---|---|
| S0 | Load the data safely | String-only parsing, NA detection off, quoting off, integrity checks | Clean record tables and a ground-truth pair table |
| S1 | Confirm the assumptions | Structural checks on the ground truth, missingness and frequency profiles | Decision log |
| S2 | Make text comparable | Unicode folding, abbreviation contraction, name and address parsing | Normalized, structured fields |
| S3 | Learn noise equivalences | Token alignment on matched pairs, lift filtering, test self-mining | Substitution tables |
| S4 | Generate candidates | TF-IDF, rare-token index, postal block, dense ANN, reverse pass, transitive pass | Capped candidate set |
| S5 | Describe each pair | Similarity, 3-valued comparison, frequency, provenance and context features | Feature matrix |
| S6 | Score each pair | LightGBM, fine-tuned multilingual cross-encoder, stacked LightGBM, seed ensemble | Pair probability |
| S7 | Make probabilities trustworthy | Isotonic regression on OOF outputs | Calibrated probability |
| S8 | Enforce structure | Many-to-one exclusivity, duplicate groups | Constrained probabilities |
| S9 | Choose the final lists | Singleton model, expected-F0.5 optimization | Final matches |
| S10 | Ship | Format asserts, official validator | Two TSV files |

---

## 4. Stage 0 — Ingestion and integrity

**Reading rules**

- Read every column as a string.
- Turn off automatic NA detection, so values like "NA", "N/A", "None" and "NULL" stay as text.
- Turn off quote processing.
- Use tab as the separator.

**Integrity checks**

- The row count equals the line count minus 1.
- Every ID prefix matches its file (S1-, S2-, S3-).
- IDs are unique, and none are empty.
- Country labels are trimmed and case-folded, but kept as an open set of strings: never filtered and never one-hot encoded.

**Ground truth**

- The ground truth is expanded into a long pair table: (S1 id, S2/S3 id, label = 1).
- An empty match list sets a singleton flag on that S1.
- Each pair's source (S2 or S3) is derived from the ID prefix.

**Raw retention.** Raw strings are kept alongside every derived field, both for audit and as input to the cross-encoder.

---

## 5. Stage 1 — EDA and assumption checks

| Check | Decision it drives |
|---|---|
| Singleton rate, overall and per country | Score floor (the all-empty baseline) and the prior for the singleton model |
| Distribution of matches per S1; share from S2 vs. S3 | How much recall beyond top-1 matters; cap on list length |
| Does any S2/S3 ID appear under two S1s? | Validates the exclusivity constraint (S8) |
| Share of S2/S3 records matching no S1 | Distractor density; how much rejection matters |
| Near-duplicates within S2/S3 among matched records | Value of duplicate-group propagation |
| Do matched pairs always share a country label? Label spellings | Whether blocking can be restricted by country |
| Repeated normalized names and addresses within S1 | Chain and mall risk; weight of the frequency features |
| Missing rates for postal code, state, house number and unit, per source | Design of the 3-valued features and the parse heuristics |
| Record counts per source and country, train vs. test | Pool-density shift and France's share of the test set |
| Most frequent tokens per country and position | Seeds for legal-suffix discovery and stopwords |
| Most common token differences between matched pairs | Which noise types dominate; checks for the variant miner |
| Correlation of IDs or row order with matches | Leakage sanity check only. Never used as a feature. |

---

## 6. Stage 2 — Normalization and field parsing

Every record keeps its **raw view** and gains a set of **normalized, structured fields**. Each downstream component chooses the view it needs.

### 6.1 Character-level normalization (in order)

1. Unicode compatibility normalization.
2. Decomposition and removal of diacritics (é → e, ç → c).
3. An explicit map for characters that decomposition does not handle: œ → oe, æ → ae, ß → ss.
4. Case-folding.
5. Unification of quote, apostrophe and dash variants.
6. Ampersand canonicalization: "&" and "and" map to one token. French "et" is added only if variant mining confirms it.
7. Elision splitting: l', d', qu' are separated and the clitic is dropped.
8. **Numeric token protection before punctuation stripping.** Formats such as "12/3A", "4-5-678", "12 bis" and "B-204" are extracted intact first.
9. Remaining punctuation becomes whitespace, then whitespace is collapsed.

### 6.2 Token canonicalization

**Contract, never expand.** Every variant maps to one canonical short form: street → st, road → rd, avenue → ave, boulevard → blvd, saint → st. Contraction is many-to-one, so it never has to guess.

This also resolves the "St" collision: Street in "Main St", Saint in "St Louis" and "St-Denis".

**Seed dictionary categories.** These are kept small and all are documented:

- Street types.
- Compass directions.
- Ordinal words (fifth → 5th).
- Unit markers (suite, ste, #, unit, flat, floor).
- Indian address markers (plot no, shop no, h no).
- Common business words, folded from plural to singular (enterprises → enterprise, traders → trader, industries → industry).

**Guard rule.** Never contract tokens that are ambiguous inside names. For example, "MG" in "MG Road" must not be treated as an abbreviation of "marg".

**Extension.** The seed dictionary is extended automatically with the substitution tables from Stage 3.

**Stopwords.** A minimal set, derived from frequency and position (the, of, de, la, le, du, des).

### 6.3 Name parsing

| Field | How it is produced |
|---|---|
| `primary_name`, `trade_name` | Split on DBA markers: dba, d/b/a, t/a, trading as, aka, formerly |
| `legal_form` | Canonical legal-form class (e.g., limited, private limited, corporation, LLC, LLP, SARL-type), detected from a seed lexicon plus **frequency-based suffix discovery**. Discovery treats tokens in the top frequency band that appear in final position within a country as candidate legal forms, which finds SARL, SAS and SA on test without hard-coding. |
| `core_name` | The name with the legal form, DBA tail and stopwords removed. This is the main field for name comparison. |
| `acronym` | Initials of `core_name`, plus a flag when the name itself looks like an acronym (short all-caps or dotted letters) |
| `name_keys` | Sorted-token key; consonant-skeleton key; transliteration-folded key; Double Metaphone code (a weak, English-centric feature only) |

The transliteration-folded key applies light Indic folding rules: collapse double letters, ksh ↔ x, ee ↔ i, oo ↔ u, w ↔ v, ph ↔ f, drop a trailing "a". It is adopted only if it improves pair recall on training data.

### 6.4 Address parsing

The parser is heuristic and country-agnostic. No external address parser is used.

| Component | Extraction logic |
|---|---|
| `postal_code` | A standalone 4–6 digit token. US ZIP+4 is truncated to 5 digits. The Indian spaced form "400 001" is joined. The French CEDEX suffix is stripped. |
| `house_number` | The premise number, including suffixes (A, bis, ter) and municipal formats (4-5-678, 12/3A) |
| `unit` | Suite, ste, #, unit, flat, shop no, floor |
| `street_tokens` | The street segment after contraction |
| `locality` | Tokens that behave like a city, using a lexicon mined from the dataset itself |
| `region` | State or province token. Abbreviation pairs (e.g., long form ↔ 2-letter code) come from variant mining. |
| `landmark` | Phrases introduced by near, opp, opposite, behind, beside, next to, in front of, adjacent to. French markers are added via test mining. Landmarks are removed from the core address and stored separately. |
| `numeric_set` | Every number in the address |

Every component is either a value or **MISSING**, and each has a parse-confidence flag.

**Explicitly excluded:** libpostal and any pretrained address parser or gazetteer. They are built on external place-name data, which conflicts with the no-external-data rule.

### 6.5 Derived text views

| View | Composition | Used by |
|---|---|---|
| `name_norm` | The full normalized name | TF-IDF, similarity features |
| `core_name` | Name without legal form, DBA tail and stopwords | Most name features, blocking P1 and P2 |
| `addr_core` | Address without landmark and unit | Address features, blocking P5 |
| `serialized_record` | "[NAME] … [ADDR] … [PC] …", lightly normalized (case-folded, accents kept for the multilingual tokenizer) | Bi-encoder, cross-encoder |

---

## 7. Stage 3 — Variant mining

### 7.1 From the training ground truth

For every matched pair:

1. Align the tokens: remove exact shared tokens, then pair the remaining tokens by best character similarity (Jaro-Winkler above a threshold) or by a prefix relation (abbreviation).
2. Count each candidate substitution a ↔ b.
3. Keep substitutions with **support ≥ 5** and **match lift ≥ 10** (starting values). Match lift is how much more often the substitution appears in matched pairs than in non-matched candidate pairs.
4. Store separate tables for names and addresses, each per country and global.

The miner is expected to recover abbreviations (pvt ↔ private), transliterations (shri ↔ sri, laxmi ↔ lakshmi), city aliases, spelling variants and legal-form variants.

**Uses**

- Canonicalization: each variant maps to its most frequent form in a second normalization pass.
- A substitution-aware Jaccard feature.
- Noise augmentation for the encoders.

### 7.2 Test-time self-mining (France and unseen patterns)

1. **Seed pairs** come from high-precision rules on the test data: identical `core_name` plus identical `postal_code`, or name TF-IDF cosine ≥ 0.95 plus mutual best.
2. The same alignment runs on those pairs and produces a French substitution table (av ↔ avenue, bd ↔ boulevard, ste ↔ societe, ets ↔ etablissements, st ↔ saint).
3. The table is sanity-reviewed by hand, then applied.
4. Only one self-training iteration is run, to limit drift.

This uses only the provided test data, unsupervised, and is documented in the methodology.

### 7.3 Leakage control

During validation, substitution tables are mined only from the training folds.

---

## 8. Stage 4 — Blocking (candidate generation)

**Goal.** Pair completeness ≥ 99% on OOF data at no more than ~50 candidates per S1 (starting target; the final cap is taken from the knee of the recall-vs-size curve).

**Scope rule.** Block within the same country label only if the EDA confirms that matched pairs always share it **[Verify]**. Otherwise, country agreement becomes a soft score, not a filter.

### 8.1 Passes

| # | Pass | Representation | Retrieval | Starting params |
|---|---|---|---|---|
| P1 | Name character n-grams | TF-IDF on `core_name` (char 3–4 grams, word-boundary aware, sublinear TF). IDF is fit on the corpus being blocked, per country. | Sparse cosine top-K | K = 30 |
| P2 | Rare tokens | Inverted index over `core_name` tokens and `name_keys`. Tokens with posting lists above 500 are skipped as stop-like. | Pairs sharing ≥ 1 rare token, ranked by summed IDF of shared tokens | K = 20 |
| P3 | Postal code + name | Exact `postal_code` block | Within a block, keep pairs sharing any name token or with name char-similarity ≥ 0.3 | Cap 20 |
| P4 | Dense semantic | Fine-tuned multilingual bi-encoder on `serialized_record` | FAISS inner-product search (flat index; HNSW if the corpus is large) | K = 30 |
| P5 | Address character n-grams | TF-IDF on `addr_core` (char 3–4 grams) | Sparse cosine top-K | K = 15 |
| P6 | Transitive expansion | S2/S3 ↔ S2/S3 links from P1 + P4 (top-3, high threshold) | If x is a candidate of S1 *a* and x is strongly linked to y, add y to *a*'s list | ≤ 5 additions per S1 |

P5 catches records whose names are badly corrupted or replaced by a trade name.

### 8.2 Direction

P1–P5 run **S1 → S2/S3** (top-K per S1). P1 and P4 also run **S2/S3 → S1**, keeping the top-3 S1 per record.

Each S2/S3 record has at most one true parent, so the reverse direction recovers matches that per-S1 truncation drops for entities with many matches. The two directions are unioned.

### 8.3 Bi-encoder (used in P4 and as features)

| Aspect | Choice |
|---|---|
| Base model | `intfloat/multilingual-e5-base` (MIT, ~278M). Alternative: `BAAI/bge-m3` (MIT, ~568M). |
| Fine-tuning | Contrastive training on training pairs, with in-batch negatives plus hard negatives mined from P1 |
| Input convention | The model's query/passage prefixes are respected |
| Augmentation | Synthetic noise: drop the postal code, abbreviate with the mined tables, shuffle tokens, inject typos, strip or add accents |
| Leakage control | For OOF recall measurement, fine-tuning uses the training folds only. The final model trains on all training data. |

### 8.4 Union, cap and the candidate file

1. The union of all passes may exceed the target size. It is ranked by a lightweight **blocking scorer**: logistic regression on pass scores, pass ranks, number of passes that found the pair, and reverse rank.
2. The top N per S1 are kept (starting N = 50).
3. **The capped set is `candidate_pairs.tsv`, and it is exactly the set Stage 6 scores.**
4. Every ID that post-processing can add (propagation, transitivity) is already in this set, because P6 generated it. So matches are always a subset of candidates.

### 8.5 Blocking metrics

All metrics are tracked per country and per source.

| Metric | Meaning |
|---|---|
| Pair completeness | Share of true pairs present in the candidates (recall ceiling) |
| Reduction ratio | 1 − candidates / all possible pairs |
| Pairs quality | Share of candidates that are true matches |
| Average and p95 candidates per S1 | List size |
| Entity-level ceiling | Share of non-singleton S1s with *all* true matches present |
| Oracle macro F0.5 | The score a perfect classifier would reach on these candidates: the true upper bound |

---

## 9. Stage 5 — Feature engineering

**Rules**

- Every feature is country-agnostic.
- Missing information is always explicit and is never imputed as a mismatch.
- No country one-hot, no entity IDs, no row order.

### 9.1 Name features

| Feature | Captures |
|---|---|
| Token Jaccard on `core_name` | Plain token overlap |
| Substitution-aware Jaccard (Stage 3 tables) | Overlap after mined equivalences |
| IDF-weighted token overlap (shared IDF / union IDF) | Agreement on rare, informative tokens |
| TF-IDF char 3–4 gram cosine | Typos, spacing, partial words |
| Jaro-Winkler (full name and core name) | Typos near the start of the name |
| Normalized Levenshtein ratio | General edit distance |
| Token sort ratio, token set ratio, partial ratio | Word-order transpositions, subset names |
| Monge-Elkan (Jaro-Winkler inner) | Fuzzy token-to-token alignment |
| Longest common substring ratio | Shared stems |
| First-token match; last core-token match | Leading brand word; trailing descriptor |
| Acronym match (either direction) | "TCS" vs. the initials of the full name |
| Legal form: same / conflicting / one missing | Legal-suffix inconsistency vs. real conflict |
| Numeric tokens in names: all match / conflict / none | "Studio 54" vs. "Studio 45" |
| DBA max-similarity over (primary, trade) × (primary, trade) | Trade-name matches |
| Name-key equality: sorted, skeleton, transliteration-folded | Transpositions and transliteration |
| Double Metaphone equality (weak) | Phonetic similarity for English names |
| Length ratio; token-count difference | Truncation, extra words |
| Bi-encoder cosine (name only) | Semantic and multilingual similarity |

### 9.2 Address features

| Feature | Captures |
|---|---|
| Postal code: match / mismatch / missing | The strongest location signal, with missing kept separate |
| Postal prefix match (first 3 digits) | Postal-code typos, same area |
| House number: match / mismatch / missing | Premise identity |
| Unit: match / mismatch / missing | Same building, different business |
| Locality: match / mismatch / missing | City agreement |
| Region: match / mismatch / missing | State agreement |
| Numeric-set Jaccard | Municipal numbering reorderings |
| Street-token Jaccard (contracted; substitution-aware) | Street identity after abbreviation |
| `addr_core` TF-IDF char cosine; token set ratio | Overall address similarity, reordering |
| Landmark flags; landmark text similarity | Landmark-based references |
| Candidate's name appears inside the other record's landmark | Landmark contamination ("Near SBI ATM"), a negative signal |
| Missing-component counts, each side and difference | Address completeness and reliability |
| Address length ratio | Partial addresses |
| Bi-encoder cosine (address only) | Semantic similarity, transliterated place names |

### 9.3 Cross-field features

| Feature | Captures |
|---|---|
| Bi-encoder cosine on the full serialized record | Whole-record similarity |
| Name-similarity × address-similarity (product and min) | Requires both kinds of evidence |
| Name-in-address containment | Field swaps and landmark contamination |
| Field-swap similarity (name A vs. address B, and the reverse) | Values placed in the wrong field |

### 9.4 Record-quality features

| Feature | Captures |
|---|---|
| Token counts and character lengths of name and address, both sides | Information content |
| has_dba, has_landmark, has_legal_form, parse-confidence flags | Structure present in each record |
| Candidate source (S2 or S3) | Noise profiles that differ by source. The source is known at test time from the ID prefix. |

### 9.5 Frequency and ambiguity features

These are computed on the corpus of the split being scored.

| Feature | Captures |
|---|---|
| Number of S1 records sharing this `core_name` | Chains and franchises: name alone is not enough |
| Number of S1 records sharing postal code + street | Malls and business parks: address alone is not enough |
| Number of S2/S3 records sharing the candidate's `core_name` | Duplicate density on the candidate side |
| IDF sum of `core_name` tokens; maximum shared-token IDF | Name informativeness |

### 9.6 Blocking provenance features

| Feature | Captures |
|---|---|
| Per pass: found flag, rank within pass, pass score | Which retrieval route found the pair |
| Number of passes that found the pair | Agreement across routes, a cheap and strong signal |
| Found in reverse direction; reverse rank | Mutual retrieval |
| Found only via transitive expansion | A weaker route, flagged separately |

### 9.7 Context (relative) features

These are computed after the base model (see S6), from its OOF scores and from the cross-encoder scores.

| Feature | Captures |
|---|---|
| Rank of this pair's score within the S1's list; gap to best; ratio to best; z-score within list | Competition among candidates |
| Reverse rank: rank of this S1 among all S1s scored for this candidate; gap to that candidate's best S1 | Competition from the candidate's side |
| Mutual-best flag | The pair is each other's top choice |
| Number of candidates in the list with score ≥ 0.5 and ≥ 0.8; list size | How crowded the decision is |
| Best score among same-source candidates | Duplicate-aware competition |

**Size and tracking.** About 100 features in total. Importance and ablation results are tracked per family.

---

## 10. Stage 6 — Pairwise models

The models form a three-layer stack. Every layer is trained OOF with **5-fold GroupKFold grouped by S1 entity**, using the same folds everywhere.

| Layer | Model | Inputs | Output |
|---|---|---|---|
| L1 base | LightGBM binary classifier | Feature families 9.1–9.6 | `p_base` (OOF) |
| L2 cross-encoder | Fine-tuned `microsoft/mdeberta-v3-base` (MIT, ~278M). Alternative: `xlm-roberta-base` (MIT, ~278M). | Serialized record pair | `p_ce` (OOF) |
| L3 stacker | LightGBM binary classifier | All features + `p_base` + `p_ce` + context features (9.7) | `p_final` |

### 10.1 L1 — base LightGBM

| Aspect | Choice |
|---|---|
| Objective | Binary log-loss, with early stopping on each fold's validation part |
| Monotone constraints | Increasing on the core continuous similarities: name TF-IDF, Jaro-Winkler, token set ratio, IDF overlap, bi-encoder cosines. This stops the model from learning non-monotonic quirks that break under the France shift. |
| Capacity | Moderate: num_leaves 31–63, feature and bagging fractions below 1 |
| Class imbalance | No resampling and no reweighting, to keep probabilities calibratable |

### 10.2 L2 — cross-encoder

| Aspect | Choice |
|---|---|
| Serialization | Ditto-style: "[COL] name [VAL] … [COL] address [VAL] … [COL] postal [VAL] …" for both records, then a pair classification head |
| Training pairs | Blocking candidates (hard negatives come for free) plus augmented positives. Augmentation uses the mined abbreviations, token shuffles, dropped components, typos, and stripped or added accents. |
| Training setup | 2–3 epochs, max length 128–192 tokens, mixed precision |
| OOF scheme | 5 folds. If compute is short, 3 folds plus a full-train model for test inference. |
| Why | It learns transliteration and abbreviation equivalences that hand-built features miss, and multilingual pretraining gives a prior for French. |

### 10.3 L3 — stacker and ensemble

- **Configuration:** the same LightGBM setup as L1.
- **Role:** the context features let it reason across a whole candidate list.
- **Ensemble:** L3 is trained with 3 seeds, plus a CatBoost (Apache 2.0) variant. Probabilities are averaged and then recalibrated. This improves stability on the private leaderboard.

### 10.4 Optional V2 — LLM adjudicator

- **Model:** a Qwen-family 7–8B instruct model (Apache 2.0).
- **Scope:** applied only to the ambiguous band of `p_final` (starting band 0.35–0.75). It sees both records and returns a yes/no answer with confidence.
- **Use:** its output becomes an extra feature for a small re-stacker, adopted only if OOF macro F0.5 improves.
- **Why deferred:** it is excluded from V1 because of its compute and reproducibility cost.

---

## 11. Stage 7 — Calibration

- **Method:** isotonic regression on the OOF `p_final`, fit across all folds.
- **Checks:** reliability curves overall, per country, per source, and under leave-one-country-out (LOCO) validation.
- **Fallback:** Platt scaling if the isotonic fit is jagged.
- **Why it matters:** calibrated probabilities are required by Stages 8 and 9, because the expected-F0.5 computation assumes them.

---

## 12. Stage 8 — Global constraints

### 12.1 Duplicate-group propagation (runs first)

1. **Companion model.** A record-to-record model uses the same feature pipeline and LightGBM setup, trained on S2/S3 ↔ S2/S3 pairs.
   - Positives: two records that share the same S1 in the ground truth.
   - Negatives: blocking candidate pairs whose S1s differ.
2. **Groups.** Connected components on high-confidence links (starting threshold 0.9), with a **group-size cap** to prevent chaining.
3. **Group score.** For each S1, the group score is the max (or noisy-OR) of its members' probabilities for that S1.
4. **Inheritance.** Members inherit the group score only for S1s whose candidate lists already contain them. P6 guarantees this.

### 12.2 Exclusivity (many-to-one)

1. Each S2/S3 record (or group) keeps only its highest-probability S1. Every other pair for that record is set to 0.
2. **Ambiguity margin.** When the best and second-best S1 are within ε (starting 0.05), the best is kept but down-weighted. The size of the penalty is tuned OOF.

Depends on **[Verify]**: no S2/S3 ID appears under two S1s in the ground truth.

---

## 13. Stage 9 — Decision layer

### 13.1 Entity-level singleton model

| Aspect | Choice |
|---|---|
| Model | LightGBM, OOF, followed by isotonic calibration |
| Features per S1 | Max, 2nd and 3rd calibrated probabilities; sum of probabilities; counts above 0.5 and 0.8; candidate count; blocking-pass coverage; core-name frequency; name IDF sum; address completeness; source mix |
| Target | The S1 has at least one true match |
| Output | P(singleton) = 1 − P(has a match) |

This model also accounts for true matches that blocking missed.

### 13.2 Expected-F0.5 subset selection

For each S1:

1. Sort the candidates by calibrated probability.
2. For k = 0 … K_max (starting 10), compute the expected per-entity F0.5 of predicting the top-k:
   - **k = 0:** expected score = P(singleton).
   - **k ≥ 1:** take the expectation over the number of true positives among the chosen candidates (a Poisson-binomial computed with a small DP). Count unchosen true matches as FN: the sum of the unchosen probabilities, plus a small expected blocking-miss term estimated OOF. The case with no true matches scores 0.
3. Choose the k with the highest expected F0.5.

This automatically reproduces the threshold ladder and adapts it to each entity: about 0.50 for the first match, about 0.73 for the second, rising toward 0.80.

### 13.3 Guardrails

- A hard cap on list length, taken from the training p99 of match counts.
- A probability floor: never add a candidate below 0.3.
- Both are tuned OOF.

### 13.4 Independence caveat

The expected-F computation assumes candidates are independent, but duplicate records are correlated. The group treatment in 12.1 reduces this.

The expected-F selection must beat a tuned global threshold on OOF data. Whichever wins is kept.

---

## 14. Stage 10 — Export and validation

| File | Content | Rules |
|---|---|---|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | One row per test S1, in file order. Comma-joined S2/S3 IDs, no duplicates, empty string for no matches. |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | The capped blocking set from 8.4, which is exactly what the model scored |

**Checks before submission**

1. Internal asserts:
   - Every test S1 is present exactly once.
   - Every ID exists in the test files and has an S2 or S3 prefix.
   - No duplicates within a list.
   - Matches are a subset of candidates.
   - No literal "nan" strings.
2. Write with a tab separator, no quoting and no index. IDs within each list are in deterministic order.
3. Run the official `utils/validate_submission.py` and confirm PASS.
4. Cross-check row counts against the test S1 file.

---

## 15. Validation framework

### 15.1 Primary: global OOF

1. Split all training data with 5-fold GroupKFold by S1.
2. Every learned component produces OOF outputs: variant tables, bi-encoder, L1, L2, L3, calibrators and the singleton model.
3. Run Stages 8–9 **once over the full training set**.
4. Report macro F0.5 overall, per country, per source, and separately for singletons and non-singletons, along with precision and recall.

**Why global.** A naive holdout orphans S2/S3 records whose true S1 sits in another fold. That breaks the many-to-one structure and distorts the exclusivity step.

### 15.2 Leave-one-country-out (the France proxy)

- Train on US and evaluate on India, then the reverse.
- Track the gap against in-country OOF.
- Any component that widens the gap is suspect for France.

### 15.3 Blocking evaluation

Use the metrics in 8.5 at every change to Stages 2–4.

### 15.4 Ablation ledger

- Every component is toggled on and off, recording Δ macro F0.5 both in-country and under LOCO.
- A component is kept only if it helps both, or helps one without hurting the other.

### 15.5 Use of the public leaderboard

- End-to-end sanity checks.
- The **France differential probe**: one submission with France predictions, one with France forced empty. The difference measures France's net contribution on the public subset.
- No parameter tuning on the public leaderboard.

### 15.6 Leakage controls

| Risk | Control |
|---|---|
| Same entity in train and validation | Folds grouped by S1 |
| Corpus statistics seeing held-out data | IDF and frequency features computed per split corpus |
| Variant tables learned from held-out pairs | Mining restricted to the training folds |
| Bi-encoder seeing held-out pairs | Fine-tuned within folds for OOF recall |
| Stacker seeing in-fold predictions | L3 uses only OOF outputs from L1 and L2 |
| Calibrator overfitting | Fit on OOF outputs only |

---

## 16. France (unseen country) strategy

| Risk | Mitigation |
|---|---|
| Country-specific features carry no signal | Every feature has a country-agnostic definition (e.g., postal code = a 4–6 digit token) |
| France is an unseen category for the model | Country is never a model input |
| French text lowers raw similarity | Accent and ligature folding, elision splitting, contraction that resolves the St collision, French substitutions from test self-mining (7.2) |
| Unknown French legal forms | Frequency-based suffix discovery on the test corpus |
| IDF shift | IDF fit on the test corpus, per country |
| Tree quirks under distribution shift | Monotone constraints on the core similarities |
| Calibration shift | **Shift monitor:** compare per-country distributions of best-candidate score and predicted-empty rate on test. If France deviates strongly, fix normalization before submitting. |
| Unknown net effect of France predictions | Leaderboard differential probe (15.5) |
| Weak semantic signal for French | Multilingual bi-encoder and cross-encoder |

**Optional V2: France pseudo-labeling**

- **Method:** add very high-confidence France pairs (p ≥ 0.98 and mutual best) as extra training data for L1 and L3, then retrain.
- **Adoption test:** adopted only if the analogous experiment helps. That experiment trains on US, pseudo-labels India, and measures India.

---

## 17. Compliance

### 17.1 Models

| Component | Model | License | Parameters |
|---|---|---|---|
| Bi-encoder (blocking and features) | `intfloat/multilingual-e5-base` (alternative: `BAAI/bge-m3`) | MIT | ~278M (alternative: ~568M) |
| Cross-encoder | `microsoft/mdeberta-v3-base` (alternative: `xlm-roberta-base`) | MIT | ~278M |
| Gradient boosting | LightGBM; CatBoost | MIT; Apache 2.0 | Not applicable |
| Optional LLM (V2) | Qwen-family 7–8B instruct | Apache 2.0 | ≤ 8B |

- Each license is re-verified on the model card at the pinned revision.
- The size and license rule is applied conservatively, to **every** model in the pipeline, not only the final classifier.
- Not allowed: the Llama and Gemma families, which use custom licenses.

### 17.2 Data rules

- **Prohibited sources:** no external lookups, APIs, geocoders or business registries.
- **Prohibited tools:** no pretrained address parsers or gazetteers (libpostal is excluded), and no internet lists of cities, postal codes or companies.
- **Hand-written seed lists:** limited to generic linguistic normalization (legal-form and street-type abbreviations, stopwords, landmark markers). They are listed in full in the methodology document.
- **Everything else** is mined from the provided train and test data.
- **Test data** is used only unsupervised (IDF, frequencies, variant mining, optional pseudo-labels), and this is documented.
- **Pretrained weights** of licensed public models are used as-is, which is standard practice.

---

## 18. Reproducibility and packaging

### 18.1 Package layout (as required by the challenge)

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

### 18.2 Source module plan

| Module | Responsibility |
|---|---|
| config | Every parameter in one file |
| ingest | Stage 0 |
| eda | Stage 1 reports |
| normalize | Stage 2 |
| mine_variants | Stage 3 |
| block | Stage 4 and the candidate file |
| features | Stage 5 |
| train_biencoder / train_crossencoder | Encoder fine-tuning |
| train_gbdt | L1, L3, the companion model and the singleton model |
| calibrate | Stage 7 |
| postprocess | Stage 8 |
| decide | Stage 9 |
| export / validate | Stage 10 |
| run_pipeline | Single entry point: data → blocking → matching → outputs |

### 18.3 Reproducibility rules

- Fixed seeds for all libraries.
- Pinned package versions and pinned Hugging Face model revisions.
- Deterministic sorting everywhere.
- Cached intermediate artifacts with content hashes.
- The README states the environment and hardware: one GPU is recommended for encoder fine-tuning, with a CPU fallback that uses the untuned encoders.
- The README gives a runtime estimate and commands for two runs: the full train → test run, and inference only with saved models.

### 18.4 Mapping to the documentation template

| Template section | Source in this document |
|---|---|
| Methodology | §2–3 |
| Candidate generation / blocking | §8 |
| Model architecture and features | §9–10 |
| Other relevant information | §11–17 |

---

## 19. Roadmap

| Milestone | Scope | Exit criterion |
|---|---|---|
| M0 | Ingestion, EDA, validation harness, all-empty baseline | The harness reproduces all-empty score = singleton rate |
| M1 | Crude end-to-end: P1 blocking, a few features, logistic regression, one threshold | Validator PASS; first leaderboard submission |
| M2 | Full normalization, variant mining, all blocking passes | Pair completeness ≥ 99% at ≤ 50 candidates per S1 (OOF) |
| M3 | Full feature set and L1 LightGBM | OOF gain over M1 |
| M4 | Calibration, propagation, exclusivity, decision layer | OOF gain; expected-F selection beats a global threshold |
| M5 | France hardening and LOCO | Stable LOCO gap; clean shift monitor |
| M6 | Cross-encoder, stacker, ensemble | Gain on both OOF and LOCO |
| M7 | Packaging, documentation, clean-room rerun | A fresh environment reproduces the outputs (within GPU nondeterminism tolerance) |

---

## 20. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Chains and malls cause false merges | Precision | Frequency features, exclusivity, name × address evidence features |
| Heavily corrupted names escape blocking | Recall ceiling | P4 dense, P5 address, reverse direction, P6 transitive |
| Duplicate chaining over-propagates | Cascading false positives | High link threshold, group-size cap |
| Independence assumption in expected-F | Suboptimal lists | Group treatment; OOF comparison with a global threshold |
| Cross-encoder compute budget | Schedule slip | 3-fold fallback; smaller Apache-licensed multilingual MiniLM |
| Overfitting the public leaderboard | Private-leaderboard drop | OOF-only tuning |
| Parsing pitfalls in pandas | Silent data loss | String dtype, NA detection off, quoting off, row-count checks |
| License or external-data audit | Disqualification | §17 rules; documented seed lists |
| Reproduction failure | Review problems | Clean-room rerun at M7 |

---

## 21. Assumptions to verify and open questions

| ID | Item | Status |
|---|---|---|
| A1 | Each S2/S3 record matches at most one S1 | [Verify] in EDA |
| A2 | S2/S3 contain records matching no S1 | [Verify] in EDA |
| A3 | S2/S3 contain duplicates of the same entity | [Verify] in EDA |
| A4 | Matched pairs always share a country label | [Verify] in EDA |
| A5 | An empty prediction on a non-singleton scores 0 | Assumed; consistent with the formula |
| A6 | No ID or row-order leakage in train | Sanity check only |
| A7 | Train and test pools have similar density | [Verify] via record counts |
| Q1 | Contents of the challenge video | Open |
| Q2 | Contents of `Documentation_template.md` | Open; needed at M7 |
| Q3 | Daily submission limit on the portal | Open; affects the probe strategy |

---

## 22. Version history

| Version | Date | Notes |
|---|---|---|
| V1 | 25 Sep 2026 | Initial end-to-end design, pre-EDA |
