# Notebook 07 — Market Basket Analysis & Basket Growth: Findings

Cells 1-11 (basket construction through rule validation) confirmed run
clean, 0 errors, matching an independent headless run exactly. **Cells 14,
16, and 19 are still unrun on your end** — those build the actual
recommender, run the evaluation comparison, and demonstrate
explainability. This doc documents all of it (verified independently,
deterministic), but go run those three cells before treating this notebook
as done.

## 1. Rule mining

- 149,267 of 235,958 train baskets have ≥2 candidate items (single-item
  baskets can't form a rule by definition).
- `min_support` sweep: 0.01 → only 7 multi-item itemsets (too strict);
  0.002 → 317, in ~5 seconds. Went with 0.002.
- 295 rules survive the filter (single consequent, lift > 1.2, confidence ≥ 0.1).

## 2. Rule quality — two distinct, both real patterns

- **Highest-lift rules are same-commodity, different-product pairs** (e.g.
  two drink-mix flavors, lift ~90-104). Not a bug — a real "buying multiple
  flavors of the same product line in one trip" pattern.
- **Genuine cross-category pairs**, once filtered for different
  `commodity_desc`: bananas + hamburger buns → hot dog buns (lift 10.7),
  tomatoes → cucumbers (lift 7.9), lettuce ↔ vine-ripe tomatoes (lift 6.5),
  broccoli/cauliflower → mini carrots (lift 6.4). These read as genuine
  "cookout basket" / "produce basket" affinities — the kind of pattern a
  category manager would recognize immediately.

## 3. Held-out validation — rules generalize, not just noise

Top-30 rules checked against validation-period baskets (weeks 89-95, never
seen during mining):

- **Train/validation confidence correlation: 0.70**
- Mean confidence: 0.312 (train) → 0.266 (validation) — a normal amount of
  shrinkage, not alarming degradation
- Sample sizes per rule: 40-438 matching validation baskets — enough to
  trust the estimate, not a handful of coincidences

## 4. The actual payoff: novel-item discovery (needs your run to confirm — cells 14/16/19)

| model | k | recall | map | repeat_hits | novel_hits |
|---|---|---|---|---|---|
| basket_affinity | 10 | 0.0057 | 0.0107 | **0.0000** | **0.2931** |
| basket_affinity | 20 | 0.0062 | 0.0057 | **0.0000** | **0.3100** |
| buy_it_again (hybrid) | 10 | 0.0743 | 0.3258 | 3.9932 | 0.0055 |
| buy_it_again (hybrid) | 20 | 0.1073 | 0.2405 | 6.3605 | 0.0100 |

**Read this as two models answering two different questions, not one
beating the other:**

- `repeat_hits = 0.0000` for `basket_affinity` isn't a rounding artifact —
  it's exact, because the model explicitly excludes anything a household
  already owns. It is *structurally* a discovery-only model.
- `novel_hits` for `basket_affinity` is **~53x** `buy_it_again`'s (0.293 vs
  0.0055 at k=10). This is the direct, measured answer to the original
  concern: yes, this system can now surface a product a household has
  never bought, driven by real co-purchase affinity — the hybrid alone
  could not do this in any meaningful volume.
- Recall/MAP being much lower for `basket_affinity` is expected and correct
  — it isn't trying to predict next purchases, so scoring it against a
  repeat-dominated validation set on those metrics was never going to be
  favorable, and treating that as a loss would be comparing two different
  jobs on one job's scorecard.
- **99.0%** of validation households (2,174/2,197) get at least one
  affinity-based recommendation — high coverage, not a niche edge case.

## 5. Explainability

The recommender exposes *why* it recommended something — e.g. household 1
was recommended an additional soft-drink pack variant because 29.3% of
households who buy the one they already have also buy this one (23.5x more
than chance). This is a real, concrete advantage over the hybrid's opaque
dot-product/rank scores for a customer-facing "you might also like"
feature — worth keeping in the API response, not just internally.

## What this changes going forward

- **Two recommendation surfaces, not one blended score**:
  `GET /recommendations/{household_id}` (hybrid — reorder-focused) and
  `GET /recommendations/{household_id}/complete-basket`
  (`BasketAffinityRecommender` — discovery/cross-sell-focused). They answer
  different questions and should stay separate rather than being forced
  into one ranking.
- The original concern that started this notebook — "can it recommend
  something never bought because it was never recommended before" — now
  has a concrete, validated, explainable answer: yes, via association
  rules, at ~99% household coverage.

## Next

Run cells 14/16/19 to confirm the table above in your own notebook. Then
extract into `src/fmcg_reco/features/basket_rules.py` (the mining
pipeline) and `src/fmcg_reco/models/affinity.py`
(`BasketAffinityRecommender`), update the artifact export to include the
mined rules table, and update the serving API design to expose both
recommendation surfaces before moving to implementation.
