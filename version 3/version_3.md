# Business Entity Resolution — Solution Design, Version 3

**Challenge:** Amazon ML Challenge — Business Entity Resolution
**Owner:** Krish
**Status:** V3 design (major version V3.0.0 under the pipeline blueprint)
**Date:** 26 September 2026
**Goal:** The highest achievable macro F0.5 on the private leaderboard

| Reference | Score / role |
|---|---|
| Version 1 | ~95 |
| Team V2 | ~96 |
| V2 design | Parent design for V3 |
| Pipeline blueprint | Layer structure (L0–L17) that V3 follows exactly |

---

## 0. Scope and relationship to earlier versions

**What V3 keeps.** Everything that works from V1 and V2:

- The ingestion rules.
- The normalization and variant mining.
- The noise-operator model.
- The cascade architecture.
- The set transformer with a NULL slot (the model's "no match" output for an S1).
- Global assignment.
- The France program.
- OOF discipline, the error-budget workflow and compliance.

**What V3 adds.** It attacks the limits that remain *after* V2, each tied to an error bucket (§1). Every V3 component is specified as a **method variant of a blueprint layer**, so it can be adopted, ablated or rolled back independently.

**Entry condition.** The V2 error audit (V2 §2) is complete, so the bucket budget B1–B6 is known. V3 components are built in the order of points recoverable per hour (§13), not in document order.

**Honest ceiling note.** Label noise and genuinely ambiguous pairs set a hard limit. The ceiling estimate from V2 §2.4 decides how far the last gates are worth pushing.

---

## 1. What still limits V2, and V3's answer

| # | Remaining limitation after V2 | V3 answer | Layer | Buckets |
|---|---|---|---|---|
| 1 | Each record is judged alone, even when its duplicate carries the missing evidence (e.g., only one duplicate has the postal code) | **Evidence pooling:** resolve S2 ∪ S3 into record clusters first, build fused profiles, and match S1 ↔ cluster | L7, L9, L11, L14 | B2b, B4, B3 |
| 2 | Heavy models see only the top-M, so pruning errors are unrecoverable | **Distilled student pruner:** the heavy ensemble's knowledge is distilled into a fast model that scores the full pool | L8 | B2a, B2b |
| 3 | Subword encoders are brittle to character noise, transliteration and accents | **Domain-adaptive pretraining** (DAPT) on all provided text, plus a **byte-level encoder** (ByT5) for diversity | L10 | B2b, B3, B6 |
| 4 | The LLM scores pairs without seeing the competing candidates | **Listwise LLM reranker:** one prompt holds the S1 and all its candidates and scores every slot, including "none" | L11 | B1, B3, B5 |
| 5 | Retrieval uses a single dense mode | **Tri-modal retrieval** (dense + learned sparse + multi-vector) and cluster-aware retrieval | L6, L7 | B2a |
| 6 | The global solver is heuristic everywhere | **Component decomposition:** exact optimization on small connected components, heuristic only on large ones; learned edge-GNN as an alternative | L13 | B1, B3, B4 |
| 7 | Expected-F decisions rely on calibration and an independence assumption | **Learned cost-sensitive decision policy** trained directly on per-entity F0.5 costs, stacked with expected-F | L14 | B1, B5, B2b |
| 8 | France adaptation relies on weights transferred from US and India | **Unsupervised Fellegi–Sunter EM** re-estimated per country on test; DAPT including France text; co-training; optional domain-adversarial training | L5, L10, L12 | B6 |
| 9 | Ensemble weights and knobs are chosen by hand | **Greedy ensemble selection** + Bayesian hyperparameter search, both with a min(OOF, LOCO) objective | L10–L14 | All |
| 10 | Small score deltas are noisy under a single fold split | **Repeated CV** for the cheap downstream layers (L11–L14), which is valid because level-1 predictions are already OOF | L3, L16 | All |
| 11 | Hard negatives come only from retrieval | **Structured synthetic negatives** from the noise model: chain, mall and near-miss negatives | L5, L10 | B1, B3 |

---

## 2. V3 design principles

| # | Principle | Meaning |
|---|---|---|
| 1 | Pool evidence before judging | A record's verdict can use everything its confident duplicates say |
| 2 | Nothing heavy is a dead end | Any knowledge learned by the heavy models flows back into pruning through distillation |
| 3 | Joint at every scale | Candidates are judged jointly per S1 (set model, listwise LLM) and per connected component (exact solver) |
| 4 | Optimize the metric itself | The decision layer learns from per-entity F0.5 costs, not only from probabilities |
| 5 | Adapt without labels | Unseen countries get field weights, representations and priors estimated from unlabeled test data |
| 6 | Robust selection | Every choice maximizes min(in-country OOF, LOCO) under repeated CV |
| 7 | Tiered compute | A cheap tier, a medium tier and a full tier are each measured, so the final submission fits the budget |

---

## 3. V3 architecture

```
PHASE B  DATA & REPRESENTATION
  L4  Normalization v2 (from V2)
  L5  Resources v3: hierarchical noise model, structured synthetic negatives,
      Fellegi–Sunter m/u tables per country (EM on test)
  L6  Representation v3: DAPT corpora, tri-modal retrieval embeddings,
      self-supervised test adaptation
        │
PHASE C  CANDIDATES
  L7  Candidate generation v3:
        L7a record-side resolution (S2∪S3 clusters + fused profiles)
        L7b tri-modal + cluster-aware multi-pass blocking ─────────────► pool
  L8  Pruning v3: distilled student over the full pool, adaptive M ──► candidate_pairs.tsv
        │
PHASE D  SCORING
  L9  Features v3: + cluster/fused-profile, EM match weights, ensemble uncertainty
  L10 Level-1 scorers v3: GBDTs · Fellegi–Sunter · DAPT XLM-R-large · DAPT mDeBERTa ·
      ByT5 · Qwen3-Reranker (LoRA)
  L11 Joint v3: listwise LLM reranker → set transformer v2 (cluster + NULL tokens)
  L12 Calibration v3: uncertainty-aware monotone calibrator + prior-shift EM
        │
PHASE E  RESOLUTION
  L13 Structure v3: component decomposition, exact solve for small components,
      heuristic for large ones (edge-GNN alternative)
  L14 Decision v3: learned cost-sensitive policy stacked with expected-F0.5
        │
PHASE F  DELIVERY
  L15 Output · L16 Evaluation v3 (repeated CV, tier reports) · L17 Packaging (tiers)
```

### 3.1 V3.0.0 manifest (changes vs. the V2 design)

| Layer | V2 design | V3.0.0 |
|---|---|---|
| L0 Config | Manifest | + compute-tier switch (S / M / L) |
| L1 Ingestion | Standard | Standard |
| L2 Profiling | + error-audit inputs | + duplicate-cluster statistics |
| L3 Evaluation design | 5-fold frozen + LOCO | + repeated CV (3 × 5) for L11–L14 only |
| L4 Normalization | v2 | v2 (unchanged) |
| L5 Resources | Noise-operator model | Hierarchical noise model + structured negatives + Fellegi–Sunter m/u (EM per country) |
| L6 Representation | Field-specific e5/bge embeddings | DAPT corpora, tri-modal bge-m3 or Qwen3-Embedding, self-supervised test adaptation |
| L7 Blocking | V2 passes, adaptive K | + L7a record clusters and fused profiles; cluster-aware and tri-modal passes; BM25 field pass |
| L8 Pruning | GBDT fast scorer, top-10 | Distilled student cross-encoder + GBDT, adaptive M, force-keep clusters |
| L9 Features | + NOISE, RARE, SIB | + CLUS, EM, UNC |
| L10 Pair scorers | XLM-R-L, mDeBERTa, LLM-LoRA, GBDTs | DAPT XLM-R-L, DAPT mDeBERTa, ByT5-large, Qwen3-Reranker-LoRA, Fellegi–Sunter, GBDTs + HPO |
| L11 Joint | Set transformer + NULL | Listwise LLM reranker → set transformer v2 (cluster tokens, uncertainty inputs) |
| L12 Calibration | Bucket isotonic + prior EM | Monotone GBDT calibrator on (score, uncertainty, bucket) + prior EM |
| L13 Structure | Global assignment + re-scoring | Component decomposition + exact solve + heuristic; edge-GNN variant |
| L14 Decision | Expected-F joint Monte Carlo | Learned cost-sensitive policy stacked with expected-F |
| L15 Output | Standard | Cluster members expanded; subset rule asserted |
| L16 Evaluation | Buckets, bootstrap, noise floor | + repeated CV, cluster purity, component-size slices, tier comparison |
| L17 Packaging | Inference path, adapters | + three compute tiers, student-only fallback |

---

## 4. Phase B — Data and representation (L5, L6)

### 4.1 Hierarchical noise model (L5)

V2's operator model is extended in three ways:

- **Conditioning.** Each operator probability is conditioned on field, source, country group, token position and **token class** (legal form, street type, generic business word, proper noun, number).
- **Co-occurrence.** Operators that appear together in the same record are modelled jointly. For example, noisy records tend to carry several operators at once.
- **Per-record noise level.** A latent intensity controls how many operators a record receives, estimated from the training ground truth per source.

Result: synthetic variants match the real noise distribution closely. This matters for both augmentation and France transfer.

### 4.2 Structured synthetic negatives (L5)

Retrieval-mined negatives rarely cover the exact decision boundaries. The noise model now also **generates negatives** of three kinds:

| Type | Construction | Teaches |
|---|---|---|
| Chain negative | Same core name, different address sampled from the same locality distribution | Branches are distinct entities (B1, B3) |
| Mall negative | Same address, different business name drawn from the same area | Shared premises are not shared identity (B1) |
| Near-miss negative | One rare name token swapped for a similar-looking rare token (Sharma → Verma), or a house number changed by a non-typo amount | The exact boundary between noise and a different entity (B1, B3, B5) |

The negatives are used in training for L10 and L11, balanced so they do not distort calibration. Calibration is always refit on real OOF data only.

### 4.3 Fellegi–Sunter match weights (L5)

This is the classic probabilistic record-linkage model, used here as both a model and a feature source.

**Comparison vectors.** Each field comparison is discretized into levels, for example:
- Postal code: exact / edit-1 / mismatch / missing.
- Name: bands of Jaro-Winkler similarity.
- House number: exact / conflict / missing.

**m and u probabilities.**
- *m* is P(level | match); *u* is P(level | non-match).
- They are **initialized supervised** on the training folds.
- They are then **re-estimated by EM separately per country** on unlabeled data, including **France test data**.

**Outputs.** Per-pair log-likelihood-ratio match weights (a total and per field) become a feature family (F-EM) and a standalone level-1 model.

**Why.** France's fields may have different reliability, for example how often postal codes are dropped or how noisy names are. EM learns that from France's own data without labels.

### 4.4 Domain-adaptive pretraining corpora (L6)

- **Corpus:** every business name and address in all provided files (train and test, all sources, all countries, France included).
- **Encoders (XLM-R-large, mDeBERTa-v3):** continued masked-language-model pretraining with whole-word and span masking.
- **ByT5:** span corruption.
- **Order:** DAPT runs before any fine-tuning (L10).
- **Validation (LOCO analogue):** DAPT on US + India text, fine-tune on US labels only, measure on India. It is adopted if the India score improves.

### 4.5 Retrieval representations v3 (L6)

| Option | License | Parameters | Modes |
|---|---|---|---|
| `BAAI/bge-m3` | MIT | ~568M | Dense + learned sparse (lexical weights) + multi-vector late interaction, all from one model |
| `Qwen/Qwen3-Embedding-0.6B` / `-4B` | Apache 2.0 | 0.6B / 4B | Dense, strong multilingual retrieval |

**Selection.** The winner is chosen by the L6 gate (recall@K of true pairs, in-country and LOCO). Both options can coexist as separate passes.

**Fine-tuning recipe (either option):**
- Contrastive training with in-batch, retrieval-mined and structured synthetic negatives (§4.2).
- Noise-model positives.
- **Self-supervised test adaptation:** noise-model variants of *test* records serve as extra positives (SimCSE-style). This adapts the embedding space to France without labels.

---

## 5. Phase C — Candidates (L7, L8)

### 5.1 L7a — Record-side resolution (evidence pooling)

**Idea.** Resolve S2 ∪ S3 among themselves first, then link the resulting clusters to S1. Duplicates then share evidence instead of each fighting alone.

**Steps**

1. **Record-record candidates.** Blocking passes run among S2 ∪ S3 records (the same machinery as L7b).
2. **Scoring.** The companion duplicate model (V2) is upgraded to the V3 scorer family and trained on record pairs that share an S1 in the ground truth (positives) and on retrieval-mined negatives.
3. **High-precision clusters.**
   - Connected components at a strict threshold (starting 0.95), with a cluster-size cap taken from the training distribution of matches per S1.
   - A second, softer level (starting 0.8) is kept as *soft links* for features only.
4. **Fused profile per cluster:**
   - All name variants, with token vote counts.
   - The most complete address.
   - Every postal code, house number and numeric token observed, with counts.
   - Legal forms observed.
   - Source mix.
   - Member count.

**Uses downstream**

| Where | Use |
|---|---|
| L7b | A cluster is retrieved if its fused profile *or* any member is near the S1 |
| L9 | F-CLUS features (S1 vs. fused profile, member agreement) |
| L11 | Cluster tokens in the set transformer; cluster context in the listwise prompt |
| L13 / L14 | A cluster is decided as a unit, with a **member veto** when member-level evidence strongly contradicts the cluster |

**Gate.** On training data:
- **Cluster purity:** members share an S1.
- **Cluster completeness:** records sharing an S1 land in the same cluster.

Purity has priority, because an impure cluster spreads false positives.

**Candidate-file rule.** Clusters are expanded to member IDs in `candidate_pairs.tsv`, so every final match remains a listed candidate.

### 5.2 L7b — Blocking v3

**Target:** pair completeness **≥ 99.9%** on OOF.

| Pass | Status | Detail |
|---|---|---|
| V2 passes (char TF-IDF, rare-token, postal, postal-typo, trade-name, transliteration keys, reverse, transitive) | Kept | — |
| Tri-modal dense / sparse / multi-vector | New | From §4.5; three separate retrieval lists |
| Cluster-aware retrieval | New | Query S1 against fused profiles; expand to members |
| BM25 field pass | New | BM25 over name and address fields separately; a cheap lexical complement to TF-IDF |
| Pool ranker | Upgraded | Replaced by the distilled student (§5.3) |

**Pool size.** Adaptive, typically 50–100 per S1, bounded by a maximum.

### 5.3 L8 — Distilled student pruner

**Problem.** In V2, the strong models only ever see the top-M chosen by a GBDT. A true match ranked below M by the GBDT is lost for good.

**Teacher.** The full V3 scoring stack (L10 + L11) on the top-M. Its predictions are produced OOF.

**Student**
- A fast cross-encoder (DAPT mDeBERTa-v3-base) combined with the GBDT fast features, trained on the **full pool**.
- **Labels:** the ground-truth labels everywhere, plus the teacher's soft scores where they exist (top-M), using a blended distillation loss.

**Iteration.** One round:
1. Train the teacher on the V2 top-M.
2. Train the student.
3. The student re-selects the top-M.
4. Retrain the teacher on the new top-M.

Stop after this single round.

**Adaptive M.**
- The base M is 10.
- M grows for uncertain S1s (high entropy over the student scores, or a small gap between rank M and rank M+1), up to 20.
- It shrinks to 5 for decisive lists.

**Force-keep.** Whole clusters and duplicate-linked records are kept together.

**Gate.** Pruning recall relative to the pool ≥ 99.95%, at an average M ≤ 12.

**Output.** The pruned set is `candidate_pairs.tsv`.

---

## 6. Phase D — Scoring (L9–L12)

### 6.1 L9 — New feature families

| Family | Features | Buckets |
|---|---|---|
| **F-CLUS** | S1 vs. fused profile (name, address, postal, numeric agreement); cluster size; share of members that individually score high; member disagreement (spread of member scores); whether the cluster spans both S2 and S3 | B2b, B4, B3 |
| **F-EM** | Fellegi–Sunter total match weight and per-field weights, with country-specific m/u (EM-adapted on test) | B6, B2b |
| **F-UNC** | Mean, standard deviation, minimum and maximum across level-1 models; count of models above 0.5; disagreement between feature-view and text-view models | B1, B5 (used for decisions) |

F-UNC is computed only from OOF level-1 outputs, so it is available to L11, L12 and L14 but not to L10.

### 6.2 L10 — Level-1 scorers

| Model | License | Parameters | Input | New in V3 |
|---|---|---|---|---|
| LightGBM, CatBoost, XGBoost | MIT, Apache 2.0, Apache 2.0 | — | T8 features | Bayesian HPO (§9.2); structured negatives |
| Fellegi–Sunter | Our statistical model | — | Comparison vectors | New: unsupervised, country-adaptive |
| `FacebookAI/xlm-roberta-large` + DAPT | MIT | ~560M | Serialized pair | DAPT on provided text; structured negatives |
| `microsoft/mdeberta-v3-base` + DAPT | MIT | ~278M | Serialized pair | DAPT; also the student backbone (§5.3) |
| `google/byt5-large` (encoder + classification head) | Apache 2.0 | ~1.2B total | Serialized pair, raw bytes | New: byte-level robustness to typos, accents, transliteration |
| `Qwen/Qwen3-Reranker-4B` or `-8B` + LoRA | Apache 2.0 | 4B / 8B | Instruction + S1 record + candidate | New: a model pretrained for relevance judgment (yes/no logit), fine-tuned with LoRA on hard candidate pairs |

**Shared training recipe (all encoders)**

- **Positives:** ground-truth pairs plus noise-model synthetic variants.
- **Negatives:** top-M hard negatives, V2 false positives and structured synthetic negatives.
- **Stabilization:** multi-task field-agreement heads and consistency regularization.
- **Folds:** OOF by S1 group (5 folds for the encoders, 3 for the 4B/8B reranker).

**Serialization.**
- Field-tagged normalized text plus raw text.
- In V3 the serialization also carries the candidate's **fused-profile summary** (compact: best postal code, name variant count), so pooled evidence reaches the encoders.

**Retained from V2.** The generic Qwen classifier is kept only if it beats Qwen3-Reranker on OOF.

### 6.3 L11 — Joint scoring

The joint layer runs in three steps: J1 → J2, with J3 as the fallback.

#### J1 — Listwise LLM reranker

| Aspect | Choice |
|---|---|
| Base | `Qwen/Qwen3-8B` (Apache 2.0), or 4B under budget, with LoRA |
| Prompt | Instruction + S1 record + candidates labeled [1] … [M] (normalized and raw fields, fused-profile summary) + a final [NONE] slot |
| Target | One yes/no token per slot, in order (teacher-forced) |
| Score | P(yes) at each slot, read from the token logits (no free-text generation, so it is deterministic and calibratable) |
| Position bias | Candidate order is randomly permuted in training. At inference, scores are averaged over 3 permutations. |
| OOF | 3 folds by S1 group |
| Why | It combines world knowledge with **seeing all competitors at once**, which suits chains ("which branch?") and "none of these" |

#### J2 — Set transformer v2 (final learned combiner)

The V2 set transformer with four changes:

| Change | Detail |
|---|---|
| Richer candidate tokens | Every level-1 score, J1's slot score, F-UNC, F-CLUS |
| Cluster tokens | Cluster tokens alongside the member tokens, plus attention bias from soft duplicate links |
| NULL token input | Receives J1's [NONE] score |
| Loss | Per-candidate binary cross-entropy + NULL binary cross-entropy + soft macro-F0.5 surrogate (as in V2) |

**Repeated CV.** J2 is trained with repeated CV (3 × 5) on the OOF inputs, and its predictions are averaged across repeats and seeds.

#### J3 — Fallback

A GBDT stacker with context features, used if J2 loses on OOF.

### 6.4 L12 — Calibration v3

**Calibrator.** A monotone GBDT calibrator:
- **Inputs:** J2's score (monotone increasing), ensemble uncertainty (F-UNC), and bucket descriptors (list size, source mix, cluster size).
- **Fit:** on OOF only, with repeated CV.

**Why.** The same score deserves less trust when the models disagree. Plain isotonic regression cannot express that.

**Prior-shift EM.** Kept from V2 for France, applied after calibration.

**Gate.** Reliability and ECE per bucket, in-country and under LOCO.

---

## 7. Phase E — Resolution (L13, L14)

### 7.1 L13 — Component decomposition and exact solving

1. **Build the graph.**
   - Nodes: S1 entities and record clusters (or records).
   - Edges: calibrated S1 ↔ cluster probabilities, plus cluster ↔ cluster soft duplicate links.
2. **Decompose.** Drop edges below a floor (starting 0.02) and split the graph into **connected components**. Most components are small (one S1 with a few candidates, or a few S1s sharing candidates).
3. **Small components** (starting limit: at most 2^16 feasible configurations). **Exact maximization** of the sum of expected per-entity F0.5 over the component's S1s, subject to exclusivity (each cluster to at most one S1), via enumeration or branch-and-bound.
4. **Large components.** V2's iterated best response + min-cost flow.
5. **Collective re-scoring.** Kept from V2: round 2 on the affected S1s.

**Learned alternative: edge-GNN**
- **Architecture:** message passing over the S1–cluster–cluster graph, with edge features (calibrated probabilities, F-UNC, F-CLUS) and node features (list statistics, cluster statistics).
- **Output:** refined edge probabilities that feed step 3.
- **Training:** OOF on the training graph, with held-out fold labels masked.
- **Adoption:** only if it beats the solver-alone configuration under repeated CV.

### 7.2 L14 — Learned cost-sensitive decision policy

**Decision items.** Clusters, or single records: a cluster is chosen or rejected as one item, subject to the member veto from §5.1.

**Cost vectors (OOF).**
1. Sort each S1's items by their final score after L13.
2. For k = 0 … K (starting K = 8), the cost is c[k] = 1 − F0.5(top-k items vs. the ground truth).
3. c[0] = 0 for singletons and 1 otherwise. All costs are computed at the record level after expanding clusters.

**Policy model.**
- A GBDT predicts c[k] from entity features, with k as an input feature.
- **Entity features:** the top-k scores, gaps, P(null), F-UNC aggregates, cluster statistics, list size, the component size from L13, **the expected-F0.5 value of each k** (V2's estimate as a feature), and country-agnostic record descriptors.
- **Decision:** choose k = argmin of the predicted cost.

**Why.** Expected-F depends on calibration and independence. The learned policy corrects both biases using real outcomes. Stacking expected-F as a feature means the policy starts from the principled estimate and only learns corrections.

**Validation.**
- Repeated CV (3 × 5) at the entity level.
- Adopted only if it beats pure expected-F by a paired-bootstrap margin, in-country and under LOCO.

**Guardrails.** Kept from V2: a list cap, a probability floor, and a per-bucket minimum confidence for the first match.

**Consistency.** The chosen k per S1 is fed back into L13 step 3, so exclusivity still holds after the policy acts.

---

## 8. France program v3 (bucket B6)

| Component | Layer | Mechanism | Adoption test |
|---|---|---|---|
| Fellegi–Sunter EM per country | L5, L9, L10 | Unsupervised m/u re-estimation on France test data | LOCO analogue: EM on unlabeled India improves India |
| DAPT including France text | L6, L10 | Encoders learn French business and address vocabulary before fine-tuning | LOCO analogue with India text |
| Self-supervised test embeddings | L6 | Noise-variant positives from France records adapt the retrieval space | France recall proxy: mutual-best agreement rate rises |
| Synthetic French data (V2) + structured negatives | L5, L10 | Hierarchical noise model with the French lexicon | LOCO analogue |
| Co-training (2 rounds max) | L10 | Feature-view models (GBDT + Fellegi–Sunter) and text-view models (encoders) exchange confident France pseudo-labels (agreement ≥ 0.98, mutual best, NULL ≤ 0.02) | LOCO analogue with India as pseudo-domain |
| Domain-adversarial fine-tuning (optional) | L10 | A gradient-reversal country discriminator on the encoder representation, with domains US, India, France-unlabeled | Adopted only if the LOCO analogue improves *and* in-country OOF does not drop |
| Prior-shift EM (V2) | L12 | France priors estimated from unlabeled score distributions | LOCO analogue |
| Shift monitor + probes (V2) | X4, L16 | Distribution checks; three budgeted leaderboard probes | Noise-floor rule |

**Automated lexicon acceptance.** V2's hand review of the mined French table is replaced by automatic thresholds on support, lift and seed-pair precision. No human judgment touches test records (§12).

---

## 9. Ensembling, tuning and variance control

### 9.1 Greedy ensemble selection

- **Candidates:** every level-1 and joint model output.
- **Method:** forward selection with replacement (Caruana et al., 2004).
- **Objective:** min(in-country OOF macro F0.5, LOCO macro F0.5), measured **after the full L12–L14 chain**, so the choice reflects the metric and not log-loss.
- **Stopping:** when the objective stops improving by more than its bootstrap standard error.

### 9.2 Bayesian hyperparameter search

- **Tool:** Optuna with a TPE sampler (an MIT-licensed library).
- **Scope:** GBDT models, the calibrator, the L13 floors and limits, the L14 guardrails and policy parameters.
- **Objective:** the same min(OOF, LOCO) objective, with a trial budget per layer.
- **Anti-overfitting:** the best 5 trials are re-evaluated under repeated CV, and the one with the best mean (not the best single run) is chosen.

### 9.3 Repeated CV for the downstream layers

- Level-1 predictions are OOF for every entity, so any fresh fold split at L11–L14 is leakage-free.
- V3 runs **3 × 5-fold** at L11–L14 (cheap layers) while the heavy level-1 models keep the single frozen split.
- **Effect:** decisions on deltas of about 0.1 point become statistically meaningful.

### 9.4 Bagging

- Fold models, seeds (2–3 for encoders, 3–5 for cheap models) and inference permutations (J1) are averaged.
- Averages are always taken **before** calibration.

---

## 10. Evaluation v3 (L16 additions)

| Addition | Purpose |
|---|---|
| Repeated-CV reporting (mean ± SE) for every L11–L14 change | Stable adoption decisions |
| Cluster purity and completeness (L7a) | Controls error propagation from pooling |
| Pruning recall of the student vs. V2's GBDT pruner | Proves the value of distillation |
| Slices by component size (L13) | Where exact solving helps |
| Policy vs. expected-F head-to-head per bucket | Justifies the learned decision |
| France proxy triad: plain LOCO, LOCO + synthetic, LOCO + EM/DAPT | Isolates each adaptation mechanism |
| Tier report (§11) | Score vs. compute for S, M and L |

The adoption rule is unchanged: the OOF Δ confidence interval excludes 0, LOCO does not drop, and every gate holds.

---

## 11. Compute tiers and reproducibility

| Tier | Contents | Use |
|---|---|---|
| **S** | Normalization, blocking, student pruner, GBDTs, Fellegi–Sunter, set transformer, L13 / L14 | Guaranteed reproduction on modest hardware; fallback submission |
| **M** | S + DAPT XLM-R-large, DAPT mDeBERTa, ByT5-large | The main submission if the LLM budget is short |
| **L** | M + Qwen3-Reranker-LoRA + listwise Qwen3-8B | Full submission |

**Tier rules**

- Every tier is fully evaluated (OOF, LOCO, bucket table) and logged. The final submission uses the highest tier whose gain over the next tier clears the noise floor.
- **Distillation doubles as insurance.** The student (§5.3) carries much of the heavy ensemble's knowledge, so tier S degrades gracefully.

**Reproducibility**

- The V2 rules still apply.
- Every artifact is hashed and saved: DAPT checkpoints, LoRA adapters, the student, GBDTs, Fellegi–Sunter tables, the calibrator, the policy and the set transformer.
- The README documents three inference commands (one per tier) and the full retrain path.

---

## 12. Compliance (V3)

| Component | Model | License | Parameters |
|---|---|---|---|
| Retrieval | `BAAI/bge-m3` and/or `Qwen/Qwen3-Embedding-0.6B` / `-4B` | MIT / Apache 2.0 | ~568M / 0.6B / 4B |
| Cross-encoder | `FacebookAI/xlm-roberta-large` (+ DAPT) | MIT | ~560M |
| Cross-encoder / student | `microsoft/mdeberta-v3-base` (+ DAPT) | MIT | ~278M |
| Byte-level encoder | `google/byt5-large` | Apache 2.0 | ~1.2B |
| Pointwise reranker | `Qwen/Qwen3-Reranker-4B` / `-8B` + LoRA | Apache 2.0 | 4B / 8B |
| Listwise reranker | `Qwen/Qwen3-8B` (or 4B) + LoRA | Apache 2.0 | ≤ 8B |
| Own models | Set transformer, edge-GNN, Fellegi–Sunter, decision policy, calibrator | Ours | Small |
| Libraries | LightGBM, CatBoost, XGBoost, Optuna, OR-Tools | MIT / Apache 2.0 | — |

**Model rules**

- Every model's license and parameter count is re-verified on its model card at the pinned revision, then recorded in the X3 registry.
- No model exceeds 8B.
- The Llama and Gemma families are excluded.

**Data rules**

- **Provided data only.** DAPT, self-supervised adaptation, Fellegi–Sunter EM, synthetic data and pseudo-labels use only the provided train and test records. No external text, gazetteers, registries or APIs.
- **No manual labeling or hand review of test records.** All test-side mechanisms (lexicon mining, pseudo-labels, EM) are automatic and threshold-driven, and they are documented in the methodology.
- **LLMs are classifiers only.** They score provided records; they never generate or look up facts about businesses.
- **Hand review is limited to training data** (the V2 label-noise audit).

---

## 13. Roadmap with decision gates (ordered by expected points per hour)

| Gate | Scope | Target buckets | Pass condition |
|---|---|---|---|
| G0 | Re-baseline V3.0.0 on the V2 components; repeated-CV harness; tier switch | — | Reproduces the V2 OOF; repeated CV operational |
| G1 | Evidence pooling: L7a clusters, fused profiles, F-CLUS, cluster-level decisions | B2b, B4, B3 | Cluster purity ≥ 99%; OOF gain (CI excludes 0) |
| G2 | Decision v3 (learned policy) + L13 component decomposition and exact solve | B1, B5 | Beats V2's decision layer under repeated CV |
| G3 | France v3: Fellegi–Sunter EM, DAPT, co-training, automated lexicon | B6 | Every mechanism passes its LOCO analogue; France probe clears the noise floor |
| G4 | Retrieval v3 (tri-modal / Qwen3-Embedding) + distilled student pruner | B2a | Pair completeness ≥ 99.9%; pruning recall ≥ 99.95% |
| G5 | Encoder upgrades: DAPT XLM-R-L, ByT5, Qwen3-Reranker-LoRA, structured negatives | B2b, B3 | OOF and LOCO gains on the hard-case suites |
| G6 | Listwise LLM reranker + set transformer v2 | B1, B3, B5 | Beats the V2 set transformer under repeated CV |
| G7 | Ensemble selection, HPO, final calibration | All | Objective gain beyond its standard error |
| G8 | Tier packaging, documentation, clean-room reruns for S, M and L | — | All tiers reproduce; validator PASS |

**Order rationale**

- G1–G3 are relatively cheap and hit the buckets most likely to hold the remaining points: missing evidence, decision errors and France.
- G4–G6 are compute-heavy.
- **If the error budget says otherwise, reorder by the budget.**

---

## 14. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Impure clusters spread false positives | Precision loss across many S1s | Strict threshold, size cap, member veto, purity gate |
| Stacking depth causes leakage (L10 → J1 → J2 → policy) | Inflated OOF that fails on test | Every level trains only on OOF outputs of the level below; one frozen split for heavy models; the policy trained under separate repeated CV |
| Listwise LLM position bias | Unstable scores | Permutation augmentation and inference averaging |
| Fellegi–Sunter EM is unstable on France | Bad weights | Supervised initialization, bounded EM iterations, m/u constraints; used as features, not hard rules |
| Domain-adversarial training hurts in-domain accuracy | Score drop | Optional; strict dual adoption test |
| Synthetic negatives distort calibration | Miscalibrated decisions | Calibration refit on real OOF only; negative ratio tuned |
| Too many knobs overfit OOF | Private-leaderboard drop | min(OOF, LOCO) objective, repeated CV, re-evaluation of the best trials |
| Compute overrun | Missed deadline | Tiers S / M / L; student fallback; heavy gates last |
| Complexity makes reproduction fragile | Review issues | Blueprint contracts, lineage hashes, per-tier clean-room reruns |
| Ceiling below target | Diminishing returns | V2 ceiling estimate guides where to stop |

---

## 15. Version history

| Version | Date | Notes |
|---|---|---|
| V1 | — | Initial end-to-end design; leaderboard ~95 |
| Team V2 | — | Team iteration; leaderboard ~96 |
| V2 design | 25 Sep 2026 | Error budget, noise model, cascade, set transformer, global assignment, France program |
| Pipeline blueprint | 25 Sep 2026 | Layer structure L0–L17, contracts, versioning |
| **V3.0.0 (this document)** | 26 Sep 2026 | Evidence pooling, distilled pruner, DAPT + byte-level + Qwen3 rerankers, listwise LLM, exact component solving, learned decision policy, Fellegi–Sunter EM, ensemble selection, repeated CV, compute tiers |
