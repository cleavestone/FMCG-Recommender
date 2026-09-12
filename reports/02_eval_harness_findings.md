# Notebook 02 — Evaluation Harness & Baselines: Findings

All 13 code cells ran clean, 0 errors — outputs match the headline numbers
below exactly (verified against an independent run before handoff, so the
pipeline is reproducible, not a one-off fluke).

## 1. Data cleaning

2,595,732 → 2,581,266 transaction lines (**0.56% dropped**), entirely from
the `quantity > 0` filter (returns/corrections) — zero rows lost to orphan
`product_id`s, consistent with notebook 01's referential-integrity check.
Cleaned tables saved to `data/processed/{interactions,product_dim}.parquet`
so every later notebook reads the same source of truth instead of
re-deriving it.

## 2. Time-based split (weeks 1–88 / 89–95 / 96–102)

| split | weeks | rows | households | products |
|---|---|---|---|---|
| train | 1–88 | 2,181,391 | 2,499 | 85,037 |
| val | 89–95 | 202,051 | 2,197 | 29,957 |
| test | 96–102 | 197,824 | 2,168 | 29,054 |

2,197 of 2,499 training households (88%) have ≥1 purchase in the validation
window — a healthy evaluable population; the rest simply didn't shop in that
7-week slice, which is expected behaviour, not a data problem.

## 3. Candidate universe scoping (top 5,000 products)

- Top 5,000 products = **5.9%** of the 85,037 distinct products purchased in
  train, but cover **64.5%** of train transaction lines — the classic
  long-tail shape (most volume concentrated in a fraction of SKUs).
- **Reachable recall ceiling on validation: 58.5%**, averaged per household
  (not as one flat set across all households — see the note below, this
  distinction mattered).

> **A mistake worth knowing about, since it changes how to read this number:**
> the first version of this calculation measured "what fraction of *all
> distinct products* bought by *anyone* in validation are in the candidate
> set" and got **15.8%** — technically correct but the wrong question,
> because it weights every rare, one-off long-tail product as equally
> important regardless of how many households actually bought it. The
> harness itself measures recall **per household, then averages** — so the
> ceiling needs to match that, which is the corrected 58.5% figure. This is
> the kind of subtle mismatch that silently produces a wrong-but-plausible
> number if you don't sanity-check it against how the metric is actually
> used downstream — worth remembering as a general pattern for eval work.

**Implication:** no single model in this project can exceed ~58.5% recall
on average while restricted to the top-5,000 candidate universe. If a later
model's recall approaches that ceiling, the next lever is raising
`N_CANDIDATES`, not tuning the model further.

## 4. Metrics — hand-verified

Precision@5, Recall@5, NDCG@5, and MAP@5 were computed by hand on a toy
example (`recommended=[A,B,C,D,E]`, `relevant={B,D,F}`) and matched the
function outputs exactly (0.400 / 0.667 / 0.498 / 0.333). The metric
implementations can be trusted for everything that follows.

## 5. Baseline comparison (validation split)

| model | k | precision | recall | ndcg | map | repeat hits | novel hits |
|---|---|---|---|---|---|---|---|
| popularity | 10 | 0.189 | 0.037 | 0.232 | 0.124 | 1.72 | 0.16 |
| popularity | 20 | 0.149 | 0.052 | 0.192 | 0.081 | 2.65 | 0.33 |
| **buy_it_again** | **10** | **0.400** | **0.074** | **0.447** | **0.326** | 3.99 | 0.01 |
| **buy_it_again** | **20** | **0.319** | **0.107** | **0.382** | **0.241** | 6.36 | 0.01 |

**Reading this table:**

- `buy_it_again` beats `popularity` on every single metric, by a wide margin
  (MAP@10: 0.326 vs 0.124 — roughly **2.6x**). This is exactly the signal
  from notebook 01 (45.9% of purchases are repeats) showing up directly in
  model performance — a nice cross-notebook consistency check.
- Almost all of `buy_it_again`'s hits are repeat hits (3.99 repeat vs 0.01
  novel at k=10) — it is, definitionally, a repeat-purchase replay strategy
  with almost zero ability to surface anything new. That's fine for a
  baseline, but it sets the real bar for later notebooks: **a hybrid model
  earns its complexity only by beating `buy_it_again`'s repeat-hit
  performance while also generating meaningfully more novel hits than
  zero.**
- `popularity`'s novel-hit count (0.16–0.33) is low too, for a different
  reason: with no personalization, "novel" here just means "a popular item
  this household hasn't bought yet," which is a weak notion of discovery —
  not something to read as evidence popularity is a good discovery engine.

## Decisions this locks in for notebooks 03–05

- **Bar to beat:** MAP@10 ≥ 0.326 (buy_it_again), not just ≥ 0.124
  (popularity) — beating popularity alone is a low bar in repeat-dominated
  grocery data.
- **Recall ceiling:** 58.5% given `N_CANDIDATES = 5000`; revisit this
  constant, not the model, if later recall stalls near that number.
- **Shared interface:** `BaseRecommender.fit()/recommend()/similar_items()`
  is now the contract every remaining model (content, CF, hybrid) implements
  — the harness and, eventually, the serving API never need to change
  because of it.

## Next

Extract `evaluation/{splitting,metrics,harness}.py` and
`models/{base,popularity}.py` into `src/fmcg_reco/` with unit tests, then
draft `03_content_based_model.ipynb`.
