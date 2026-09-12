# Notebook 04 — Collaborative Filtering (ALS): Findings

4 of 6 code cells confirmed run clean, 0 errors (the class-definition cell
has no output by design). **One cell remains unrun on your end** — the
tie-group + purchase-frequency diagnostic (§4 below) — it's the most
interesting result in this notebook, go back and run it. Numbers here are
from an independent headless run, matching your own outputs exactly on
everything you did run.

## 1. Interaction matrix

2,481 households x 5,000 items, 656,601 nonzero entries — **5.29% dense**.
Confidence-weighted (`1 + 15·log1p(count)`), not raw counts, per Hu/Koren/
Volinsky — this is the standard way to turn a purchase count into an
implicit-feedback confidence signal rather than treating repeat purchases
as linearly "more liked."

## 2. Hyperparameter grid search

| factors | regularization | recall@20 | map@20 |
|---|---|---|---|
| **32** | **0.01** | **0.0481** | **0.0368** |
| 64 | 0.01 | 0.0430 | 0.0266 |
| 64 | 0.10 | 0.0428 | 0.0270 |
| 128 | 0.10 | 0.0427 | 0.0248 |

**Fewer factors won.** With only ~2,500 households and a 5.3%-dense
interaction matrix, a higher-dimensional latent space (64 or 128) has more
room to overfit the sparse signal rather than capture real structure — a
concrete instance of "more model capacity isn't automatically better," not
a Dunnhumby-specific quirk. Worth remembering when factors get tuned again
in a hybrid/reranker context later.

## 3. Full comparison (validation split, k=10)

| model | precision | recall | ndcg | map |
|---|---|---|---|---|
| popularity | 0.189 | 0.037 | 0.232 | 0.124 |
| **buy_it_again** | 0.400 | 0.074 | 0.447 | **0.326** |
| content_based | 0.080 | 0.018 | 0.088 | 0.040 |
| collaborative (ALS) | 0.108 | 0.028 | 0.113 | 0.048 |

**Pure ALS beat content-based but lost to both popularity and
buy_it_again**, on every metric. This is a real, slightly humbling result
worth sitting with: a properly-tuned matrix-factorization model, learned
from actual purchase behaviour, still can't out-rank "just replay what this
household already buys" in repeat-purchase-dominated grocery data. Two
notebooks and two legitimate modelling techniques in, `buy_it_again`'s
MAP@10 = 0.326 is still standing. That's not a failure of ALS — it's
evidence about the *structure* of this data (see notebook 01: 45.9% of
purchases are repeats), and it sharpens what notebook 05 actually needs to
prove: that *combining* signals beats every one of them individually, since
none of them do alone.

## 4. Does ALS fix content-based's tie-group problem? (run this cell next)

Tested on the 43-item tie group from notebook 03 (all identical to
content-based: `GROCERY / FRZN MEAT/MEAT DINNERS / SS ECONOMY ENTREES/
DINNERS ALL / National`):

- ALS's top-5 nearest neighbours for one item in that group: **5 out of 5**
  were still members of the *same* tie group.
- Before concluding "ALS failed too," checked whether this was a
  data-sparsity artifact (rarely-purchased items neither model has signal
  for): **it wasn't**. This tie group's items average 234 distinct baskets
  purchased (median 193), *above* the overall candidate median of 157.
  These are moderately popular items, not obscure long-tail ones.

**Honest conclusion:** CF did not clearly break this particular tie. The
more likely explanation is that these 43 "economy frozen entree" SKUs
genuinely **are** close substitutes to real shoppers — someone buying one
plausibly buys whichever variant is in stock — so both content and CF
agreeing they're interchangeable may reflect real product substitutability,
not a shared blind spot in either method.

**Why this matters for notebook 05:** the working assumption going in
should not be "hybridizing will silently repair every content tie using
CF." It should be the more precise version: *CF is strong exactly where
content is weak (behaviourally distinct items with coincidentally similar
metadata), and the two may legitimately agree where items really are
interchangeable.* When two independent signals agree, that's worth checking
before treating it as either a shared error or a confirmed truth — this
notebook is a concrete example of doing that check rather than skipping it.

## Decisions this locks in for notebook 05

- **Bar to beat is unchanged: MAP@10 ≥ 0.326** (`buy_it_again`) — three
  models in (popularity, content, ALS), none individually clears it. The
  hybrid's job is to combine, not to replace, since no single signal wins
  outright.
- **`factors=32, regularization=0.01`** is the ALS config to carry forward
  into the hybrid, chosen by validation recall, not by iterating until a
  bigger model felt more sophisticated.
- Treat "CF fixes content's ties" as a hypothesis to check per tie-group in
  notebook 05, not an assumption to build on unverified.

## Next

Extract `data/interactions.py` (`build_interaction_matrix`) and
`models/collaborative.py` (`ALSRecommender`) into `src/fmcg_reco/`, then
draft `05_hybrid_and_cold_start.ipynb`.
