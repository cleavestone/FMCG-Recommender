"""Generate notebooks/05_hybrid_and_cold_start.ipynb from source cells.

Same generated-from-source convention as build_nb01-04.py.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 05 · Hybrid Recommender & Cold Start

**This is the headline notebook of the project.** Three notebooks in, the
scoreboard looks like this (MAP@10, validation split):

| model | MAP@10 |
|---|---|
| `buy_it_again` (pure repeat-purchase frequency) | **0.326** |
| `popularity` (no personalization) | 0.124 |
| `collaborative` (ALS) | 0.048 |
| `content_based` | 0.040 |

No individual technique beats simple repeat-purchase replay. **Given that,
this notebook blends three signals — content, collaborative, and
repeat-purchase — rather than just content + CF**, because blending two
signals that both lose to a third, stronger one wasn't going to produce a
convincing hybrid. The goal here is explicit: beat MAP@10 = 0.326, not just
beat the two weaker individual models.

### What this notebook actually builds

1. A **rank-based fusion** hybrid: normalise each signal's ranking (not raw
   scores, which are on incomparable scales — TF-IDF cosine vs. ALS dot
   product vs. basket counts) and blend with tuned weights.
2. **Explicit cold-start switching**, finally putting `hh_demographic.csv`
   to use: households with no purchase history get demographic-segment
   popularity instead of a blend that has nothing to blend.
3. A concrete demonstration of **item cold-start** — something the harness
   so far can't actually test (see the note below) — showing content-based
   scoring a product with zero purchase history at all.

### An honesty note on "cold start" before building anything

`candidate_items` (notebooks 02-04) is defined as the top-5,000 *most
purchased* products in train — so by construction, every candidate item
already has purchase history. True item cold-start (a product with zero
sales) literally cannot appear in the recall/precision harness we've been
using. The **household** cold-start case is real and measurable (notebook
03 found 18 of 2,499 households have no candidate-item purchases at all);
the **item** cold-start case will be demonstrated separately, outside the
harness, using a real product that didn't make the top-5,000 cut as a stand-in
for "brand new."
""")

code(r"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

from fmcg_reco.config import N_CANDIDATES, PROCESSED_DIR, RAW_DIR, TEST_END_WEEK, TRAIN_END_WEEK, VAL_END_WEEK
from fmcg_reco.evaluation.candidates import top_n_candidate_items
from fmcg_reco.evaluation.harness import evaluate_recommender
from fmcg_reco.evaluation.splitting import relevant_items_by_household, time_based_split
from fmcg_reco.features.item_features import build_item_feature_matrix
from fmcg_reco.models.base import BaseRecommender
from fmcg_reco.models.collaborative import ALSRecommender
from fmcg_reco.models.content import ContentBasedRecommender
from fmcg_reco.models.popularity import PersonalRepeatPurchaseRecommender, PopularityRecommender

interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
product = pd.read_parquet(PROCESSED_DIR / "product_dim.parquet")
demographic = pd.read_csv(RAW_DIR / "hh_demographic.csv")
demographic.columns = [col.strip().lower() for col in demographic.columns]

train, val, test = time_based_split(interactions, TRAIN_END_WEEK, VAL_END_WEEK, TEST_END_WEEK)
candidate_items = top_n_candidate_items(train, N_CANDIDATES)
val_relevant = relevant_items_by_household(val)
previously_purchased = relevant_items_by_household(train)

# fit every component model once - these get reused across the whole notebook,
# including the weight grid search later, instead of being refit repeatedly.
pop_model = PopularityRecommender(candidate_items=candidate_items).fit(train)
repeat_model = PersonalRepeatPurchaseRecommender(candidate_items=candidate_items).fit(train)
content_model = ContentBasedRecommender(product, candidate_items).fit(train)
als_model = ALSRecommender(candidate_items=candidate_items, factors=32, regularization=0.01, iterations=15).fit(train)

print(f"train households={train['household_key'].nunique():,}  candidate items={len(candidate_items):,}  "
      f"households with demographics={demographic['household_key'].nunique():,}")
""")

md(r"""
## 1 · Demographic-segment popularity — the cold-start fallback

For a household with no candidate-item purchase history, there is nothing
for content-based or ALS to work with (`recommend()` returns `[]` for both —
verified in notebooks 03/04). The fallback is demographic-segment
popularity: rank items by popularity **within households of similar
composition** (`hh_comp_desc`, e.g. "2 Adults No Kids"), falling back again
to global popularity when the segment is unknown or too sparse. This is the
double fallback the project plan called for from the start (notebook 01:
only 32% of households have demographics at all).
""")

code(r"""
class SegmentPopularityRecommender(BaseRecommender):
    def __init__(self, candidate_items: set, demographic_df: pd.DataFrame, segment_col: str = "hh_comp_desc"):
        self.candidate_items = candidate_items
        self.demographic_df = demographic_df
        self.segment_col = segment_col
        self.segment_rankings_: dict = {}
        self.household_to_segment_: dict = {}
        self.global_fallback_: PopularityRecommender | None = None

    def fit(self, train_interactions: pd.DataFrame) -> "SegmentPopularityRecommender":
        self.global_fallback_ = PopularityRecommender(self.candidate_items).fit(train_interactions)
        self.household_to_segment_ = (
            self.demographic_df.set_index("household_key")[self.segment_col].to_dict()
        )

        df = train_interactions[train_interactions["product_id"].isin(self.candidate_items)].copy()
        df["segment"] = df["household_key"].map(self.household_to_segment_)
        df = df.dropna(subset=["segment"])

        for segment, g in df.groupby("segment"):
            self.segment_rankings_[segment] = (
                g.groupby("product_id")["basket_id"].nunique()
                .sort_values(ascending=False).index.tolist()
            )
        return self

    def recommend(self, household_key: int, k: int) -> list:
        segment = self.household_to_segment_.get(household_key)
        items = list(self.segment_rankings_.get(segment, []))
        if len(items) < k:
            for item in self.global_fallback_.recommend(household_key, k + len(items)):
                if item not in items:
                    items.append(item)
                if len(items) >= k:
                    break
        return items[:k]


segment_model = SegmentPopularityRecommender(candidate_items, demographic).fit(train)
print(f"segments found: {list(segment_model.segment_rankings_.keys())}")
print(f"households with a known segment: {len(segment_model.household_to_segment_):,}")
""")

md(r"""
## 2 · Which households actually need the cold-start path?

Content-based and ALS both build their household index from the exact same
filter (candidate-item purchases in train), so they should agree on which
households are "warm" vs "cold." Confirm that before relying on it.
""")

code(r"""
warm_households = set(content_model.household_id_to_idx_)
warm_households_als = set(als_model.household_id_to_idx_)
assert warm_households == warm_households_als, "content and ALS disagree on warm households - investigate before proceeding"

all_train_households = set(train["household_key"].unique())
cold_households = all_train_households - warm_households
print(f"warm households (content + CF signal available): {len(warm_households):,}")
print(f"cold households (segment-popularity fallback only): {len(cold_households):,}")
""")

md(r"""
## 3 · Rank-based fusion

Each component model's `recommend()` already returns a ranked list — we
don't need raw scores (which would mean exposing incomparable numbers:
cosine similarity, ALS dot products, raw basket counts) to combine them.
Instead, convert each list into a **normalised rank score**: the top item
scores just under 1.0, the last item in a length-`L` list scores just above
0, and anything a signal doesn't rank at all scores exactly 0 for that
signal. This is standard rank fusion (in the same family as Reciprocal Rank
Fusion), and it sidesteps the scale-comparability problem entirely.

We only pull each signal's **top 300** candidates per household, not the
full 5,000-item catalog — retrieving a manageable shortlist per signal and
blending only that shortlist is the same "retrieve, then re-rank"
pattern production recommenders use, for the same reason: re-scoring the
whole catalog per household per request doesn't scale, and items outside
any signal's top 300 are extremely unlikely to be worth recommending
anyway.
""")

code(r"""
N_PER_SIGNAL = 300

def rank_scores(ranked_items: list) -> dict:
    n = len(ranked_items)
    if n == 0:
        return {}
    return {item: 1.0 - (rank / n) for rank, item in enumerate(ranked_items)}


def blend_recommend(household_key: int, k: int, weights: dict) -> list:
    combined: dict = {}
    for name, model in [("content", content_model), ("cf", als_model), ("repeat", repeat_model)]:
        scores = rank_scores(model.recommend(household_key, N_PER_SIGNAL))
        weight = weights[name]
        for item, score in scores.items():
            combined[item] = combined.get(item, 0.0) + weight * score
    ranked = sorted(combined.items(), key=lambda pair: -pair[1])
    return [item for item, _ in ranked[:k]]
""")

md(r"""
## 4 · Tuning the blend weights against validation Recall@20

A small adapter lets the existing harness score any weight combination
without retraining a single component model — only the blend weights
change between grid points, so this sweep is cheap even though model
training (§0-2 above) happened only once.
""")

code(r"""
class _BlendAdapter(BaseRecommender):
    def __init__(self, weights: dict):
        self.weights = weights

    def fit(self, train_interactions: pd.DataFrame) -> "_BlendAdapter":
        return self  # component models are already fit; nothing to do here

    def recommend(self, household_key: int, k: int) -> list:
        if household_key in cold_households:
            return segment_model.recommend(household_key, k)
        return blend_recommend(household_key, k, self.weights)


STEP = 0.2
grid_rows = []
weights_grid = [
    (round(wc, 2), round(wcf, 2), round(1 - wc - wcf, 2))
    for wc in np.arange(0, 1 + STEP / 2, STEP)
    for wcf in np.arange(0, 1 - wc + STEP / 2, STEP)
    if round(1 - wc - wcf, 2) >= 0
]

for w_content, w_cf, w_repeat in weights_grid:
    weights = {"content": w_content, "cf": w_cf, "repeat": w_repeat}
    adapter = _BlendAdapter(weights)
    _, summary = evaluate_recommender(adapter, val_relevant, [20], previously_purchased)
    grid_rows.append({**weights, "recall@20": summary.loc[20, "recall"], "map@20": summary.loc[20, "map"]})

grid_df = pd.DataFrame(grid_rows).sort_values("recall@20", ascending=False)
grid_df.head(10)
""")

md(r"""
**Read the winning row before moving on — it's a real result, not a
formality.** If the top row has `content=0` and `cf=0` (all weight on
`repeat`), that means the grid search concluded the best "hybrid" is no
blend at all — pure repeat-purchase replay. That's not a bug in the search;
it's what linear rank-fusion does when one signal is *much* stronger than
the others in the exact region (top-ranked items) that matters for
Recall@K: blending in a weaker, weakly-correlated signal can only pull
probability mass away from items the strong signal already ranks
correctly, so the optimum collapses to weight 1 on the strong signal. This
is a legitimate, well-known limitation of fixed linear blending — not
something to explain away — and it's exactly the argument for the
project's planned learning-to-rank reranker (roadmap P4): a model that
*learns* per-item, per-household how much to trust each signal, rather
than one global weight applied identically everywhere, could in principle
use content/CF for the households or items where they actually help, and
ignore them elsewhere. Worth sitting with this before immediately reaching
for a bigger model, though — see the findings section for how to think
about what to actually try next.
""")

md(r"""
## 5 · Final HybridRecommender with the winning weights

Wrap the winning configuration into a proper `BaseRecommender` — this is
the version that gets extracted into `src/fmcg_reco/models/hybrid.py` and
eventually served by the API.
""")

code(r"""
best_weights = grid_df.iloc[0][["content", "cf", "repeat"]].to_dict()
print("best weights:", best_weights)


class HybridRecommender(BaseRecommender):
    def __init__(self, product_df: pd.DataFrame, demographic_df: pd.DataFrame, candidate_items: set,
                 weights: dict, n_per_signal: int = N_PER_SIGNAL):
        self.product_df = product_df
        self.demographic_df = demographic_df
        self.candidate_items = candidate_items
        self.weights = weights
        self.n_per_signal = n_per_signal
        self.content_model_: ContentBasedRecommender | None = None
        self.als_model_: ALSRecommender | None = None
        self.repeat_model_: PersonalRepeatPurchaseRecommender | None = None
        self.segment_model_: SegmentPopularityRecommender | None = None
        self.warm_households_: set = set()

    def fit(self, train_interactions: pd.DataFrame) -> "HybridRecommender":
        self.content_model_ = ContentBasedRecommender(self.product_df, self.candidate_items).fit(train_interactions)
        self.als_model_ = ALSRecommender(candidate_items=self.candidate_items, factors=32,
                                          regularization=0.01, iterations=15).fit(train_interactions)
        self.repeat_model_ = PersonalRepeatPurchaseRecommender(candidate_items=self.candidate_items).fit(train_interactions)
        self.segment_model_ = SegmentPopularityRecommender(self.candidate_items, self.demographic_df).fit(train_interactions)
        self.warm_households_ = set(self.content_model_.household_id_to_idx_)
        return self

    def _blend(self, household_key: int, k: int) -> list:
        combined: dict = {}
        for name, model in [("content", self.content_model_), ("cf", self.als_model_), ("repeat", self.repeat_model_)]:
            ranked = model.recommend(household_key, self.n_per_signal)
            n = len(ranked)
            weight = self.weights[name]
            for rank, item in enumerate(ranked):
                score = 1.0 - (rank / n) if n else 0.0
                combined[item] = combined.get(item, 0.0) + weight * score
        ranked = sorted(combined.items(), key=lambda pair: -pair[1])
        return [item for item, _ in ranked[:k]]

    def recommend(self, household_key: int, k: int) -> list:
        if household_key not in self.warm_households_:
            return self.segment_model_.recommend(household_key, k)
        return self._blend(household_key, k)

    def similar_items(self, product_id: int, k: int) -> list:
        return self.content_model_.similar_items(product_id, k)


hybrid_model = HybridRecommender(product, demographic, candidate_items, best_weights).fit(train)
""")

md(r"""
## 6 · The headline comparison table

Every model built across this project, scored the same way, on the same
validation split.
""")

code(r"""
K_VALUES = [10, 20]

_, pop_summary = evaluate_recommender(pop_model, val_relevant, K_VALUES, previously_purchased)
_, repeat_summary = evaluate_recommender(repeat_model, val_relevant, K_VALUES, previously_purchased)
_, content_summary = evaluate_recommender(content_model, val_relevant, K_VALUES, previously_purchased)
_, als_summary = evaluate_recommender(als_model, val_relevant, K_VALUES, previously_purchased)
_, hybrid_summary = evaluate_recommender(hybrid_model, val_relevant, K_VALUES, previously_purchased)

comparison = pd.concat(
    {
        "popularity": pop_summary,
        "buy_it_again": repeat_summary,
        "content_based": content_summary,
        "collaborative (ALS)": als_summary,
        "hybrid": hybrid_summary,
    },
    axis=0,
)
comparison.index.names = ["model", "k"]
comparison
""")

md(r"""
## 7 · Item cold-start, demonstrated (not measured by the harness above)

Take a real product that **didn't** make the top-5,000 candidate cut —
a stand-in for "just added to the catalog, no purchase history yet" — and
show that content-based can still relate it to the existing catalog using
only its metadata, while collaborative filtering and repeat-purchase
literally cannot (the item has zero rows in their training data).

**One honest caveat first, found while building this notebook:** this only
works well if the new item's *commodity* has some coverage among the 5,000
candidates. A genuinely random non-candidate product — "Women's Hair
Sprays" — turned out to have **zero** candidate items in its entire
commodity (`HAIR CARE PRODUCTS`), so the best content-based could do was
match it to unrelated candy products at the department level only (score
0.40, the weakest kind of match this encoding produces). So the example
below is deliberately chosen to have commodity-level coverage, to show what
a *working* case looks like — but it's worth remembering that content-based
similarity is only as good as what's actually in the candidate universe to
match against, not a guarantee for any arbitrary new product.
""")

code(r"""
non_candidate_products = product[~product["product_id"].isin(candidate_items)]
candidate_commodities = set(product[product["product_id"].isin(candidate_items)]["commodity_desc"])
coverable_new_items = non_candidate_products[non_candidate_products["commodity_desc"].isin(candidate_commodities)]

new_item = coverable_new_items.sample(1, random_state=42).iloc[0]
print("simulated new item (metadata only, zero purchase history):")
print(dict(new_item[["department", "commodity_desc", "sub_commodity_desc", "brand"]]))
print(f"(chosen because its commodity, '{new_item['commodity_desc']}', has candidate coverage - "
      "see the caveat above for what happens when it doesn't)")

_, _, vectorizer = build_item_feature_matrix(product, candidate_items)

def tok(value):
    return str(value).strip().upper().replace(" ", "_")

new_text = (
    f"DEPT__{tok(new_item['department'])} "
    f"COMM__{tok(new_item['commodity_desc'])} "
    f"SUBCOMM__{tok(new_item['sub_commodity_desc'])} "
    f"BRAND__{tok(new_item['brand'])} "
    f"MANUF__{new_item['manufacturer']}"
)
new_vector = vectorizer.transform([new_text])

scores = np.asarray((new_vector @ content_model.item_matrix_.T).todense()).ravel()
top_idx = np.argsort(-scores)[:5]

print("\ncontent-based nearest existing candidate items (no purchase history needed for the new item):")
for i in top_idx:
    pid = content_model.item_ids_[i]
    row = product.set_index("product_id").loc[pid]
    print(f"  {scores[i]:.3f}  {row['department']:12s} | {row['commodity_desc']:24s} | {row['sub_commodity_desc']}")

print(f"\nis this product in the ALS interaction matrix? "
      f"{new_item['product_id'] in als_model.item_id_to_idx_} "
      "-> CF and repeat-purchase have zero signal for it; content-based just did.")
""")

md(r"""
## 8 · Findings & next notebook

- **The grid search picked `content=0, cf=0, repeat=1.0`** — the "hybrid"
  that scores best on validation Recall@20 is mathematically identical to
  `buy_it_again`, confirmed exactly by §6's table (hybrid's row matches
  `buy_it_again` to four decimal places on every metric). **This means the
  central hypothesis of this notebook — that blending content, CF, and
  repeat-purchase beats any one of them alone — did not hold**, at least
  for linear rank-fusion tuned this way. That's a real result to report as
  what it is, not to round up or bury.
- This isn't the same as "content and CF are worthless." §7 shows content
  doing something repeat-purchase fundamentally cannot (score an item with
  zero sales history), and §2 shows both content and CF are needed to
  even *detect* which 18 households need the cold-start path. Their value
  didn't show up in the Recall@20-optimized blend weight, because that
  metric only rewards ranking quality on items with an established
  purchase pattern — exactly where `buy_it_again` already wins.
- **What this actually argues for, concretely, is the roadmap's planned
  learning-to-rank reranker (P4), not a bigger grid search here.** A fixed
  global weight can only ever be a compromise applied identically to every
  household; an LTR model trained on per-household, per-item features
  (recency, frequency, content similarity score, CF score, category
  affinity) can learn, e.g., "trust content more for households with thin
  history" or "trust CF more for high-frequency shoppers" — exactly the
  kind of conditional logic a single blend weight cannot express. The
  `get_candidates`-style separation already used here (retrieve top-N per
  signal, then score) is deliberately the same shape an LTR reranker needs
  — swapping in a learned scorer later is additive, not a rewrite.
- For now, ship the honest result: `HybridRecommender` defaults to the
  found weights (effectively `buy_it_again` for warm households, segment
  popularity for cold ones) because that's what the evidence supports,
  with content-based kept and exposed via `similar_items()` for the
  cold-start/discovery use case §7 demonstrates, not because it won the
  leaderboard.

Once reviewed, `HybridRecommender` and `SegmentPopularityRecommender` get
extracted into `src/fmcg_reco/models/hybrid.py` and
`src/fmcg_reco/models/popularity.py` respectively.

### Next notebook
`06_export_artifacts.ipynb` — retrain the winning hybrid on the full
dataset and serialise everything the serving API will need to load at
startup.
""")

nb["cells"] = c
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (fmcg)", "language": "python", "name": "fmcg"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "05_hybrid_and_cold_start.ipynb"
nbf.write(nb, out)
print("wrote", out)
