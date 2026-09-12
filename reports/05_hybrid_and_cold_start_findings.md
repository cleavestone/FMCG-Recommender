# Notebook 05 — Hybrid Recommender & Cold Start: Findings

All 8 code cells ran clean, 0 errors, fully confirmed on your run (matches
an independent headless run exactly on every number).

## The headline result: the hybrid did not beat `buy_it_again`

The weight grid search picked **`content=0, cf=0, repeat=1.0`** — and the
resulting "hybrid" row in the comparison table is identical to
`buy_it_again` to four decimal places (MAP@10 = 0.3258 either way). Three
notebooks and three modelling techniques in, nothing has moved the needle
past repeat-purchase replay.

**This is not a failed notebook — it's the most informative result in the
project so far**, and here's the reasoning to hold onto:

Linear rank-fusion combines signals by weighted sum. When one signal
(`buy_it_again`) is much stronger than the others in exactly the region
that Recall@K rewards (correctly ranking a household's own repeat
purchases near the top), blending in a weaker, weakly-correlated signal can
only dilute that ranking — it never adds information the strong signal
didn't already have for those items. So the optimizer correctly concludes
weight 1 on the strong signal is best. This is a well-documented property
of fixed linear blending, not a bug in the grid search or the
implementation.

**What this concretely argues for:** the project roadmap's planned P4
learning-to-rank reranker (LightGBM), not a bigger grid or more exotic
blend formula here. A global weight has to be a compromise applied
identically to every household. A learned reranker, trained on per-item
features (recency, frequency, content similarity score, CF score, category
affinity), could in principle learn *conditional* rules a single weight
can't express — e.g. "trust content more for thin-history households,"
"trust CF more for high-frequency shoppers." The `get_candidates`-style
retrieve-then-score structure built in this notebook is deliberately the
same shape an LTR reranker needs, so that upgrade is additive later, not a
rewrite.

## What content and CF still contribute, despite losing the blend-weight vote

- **Cold-start detection**: content and ALS agree exactly on which 18 of
  2,499 households have no candidate-item history (verified via an
  assertion in the notebook, not assumed) — that agreement is what makes
  the segment-popularity fallback trigger correctly.
- **Demographic-segment popularity, used for the first time**: `hh_comp_desc`
  segments (`2 Adults No Kids`, `Single Female`, etc.) built from
  `hh_demographic.csv`, unused since notebook 01, now backstop the 18 cold
  households (801 households have a known segment; the rest fall through to
  global popularity).
- **Item cold-start, actually demonstrated**: a real product excluded from
  the top-5,000 candidates (a stand-in for "just added to the catalog")
  scored at **0.609 similarity** against existing frozen-novelty items
  using metadata alone — confirmed absent from the ALS interaction matrix
  entirely (`False`), so CF and repeat-purchase have *zero* signal for it.
  This is something the Recall@K harness structurally cannot measure (the
  candidate universe excludes zero-history items by construction), but it's
  the actual product justification for keeping content-based in the system.

**A transparency note worth remembering**: the first candidate picked for
that demo (a hair-spray product) had **zero** candidate items sharing its
commodity at all, so it only matched at a weak department-level score
(0.40, tied among 89 unrelated candy products). Rather than quietly
swapping to a better-looking example, the notebook documents this as a
real limitation — content-based similarity is only as good as what exists
in the candidate universe to match against — and then deliberately selects
an item whose commodity *does* have coverage for the main demo.

## Decisions this locks in for notebook 06 and the serving API

- `HybridRecommender` ships with the found weights (effectively
  `buy_it_again` for warm households, segment popularity for cold ones) —
  an honest reflection of what the evidence supports, not a forced "hybrid"
  narrative.
- Content-based stays in the system and gets exposed via `similar_items()`
  specifically for cold-start/discovery use cases, not because it improved
  the leaderboard.
- The P4 learning-to-rank reranker is now a concretely motivated next step
  (backed by this notebook's evidence), not just a roadmap line item — worth
  remembering if/when that phase gets picked up later.

## Next

Extract `HybridRecommender` into `src/fmcg_reco/models/hybrid.py` and
`SegmentPopularityRecommender` into `src/fmcg_reco/models/popularity.py`,
then draft `06_export_artifacts.ipynb` — the last modelling notebook before
the serving microservice build begins.
