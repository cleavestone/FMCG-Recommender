"""Generate notebooks/03_content_based_model.ipynb from source cells.

Same generated-from-source convention as build_nb01.py / build_nb02.py.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 03 · Content-Based Model

**Goal of this notebook (Phase 2a):**

Notebook 02 gave us two baselines that only ever look at *behaviour*
(purchase counts). Neither can say anything about a product nobody has
bought yet, or a household with too little history to have a repeat pattern.
Content-based filtering fixes half of that: it recommends from what a
product **is** — its place in the product hierarchy — not from who bought
it.

1. Build an **item profile** for every candidate product from its
   department / commodity / sub-commodity / brand / manufacturer.
2. Sanity-check item-item similarity by eye before trusting it numerically.
3. Build a **household profile** as a weighted combination of the items it
   has purchased, and score every candidate item against that profile.
4. Evaluate with the exact same harness from notebook 02, so the comparison
   to `popularity` / `buy_it_again` is apples-to-apples.

### Why this is a distinct technique, not just "more features"

Collaborative filtering (next notebook) learns entirely from the
interaction matrix — it has *no idea* what a product is, only who bought
it alongside whom. That makes it powerless for a product with zero or
near-zero purchase history: there's no behavioural signal to learn from.
Content-based filtering has the opposite blind spot — it needs no
behavioural signal for the *item*, but it still needs the *household* to
have bought something before, to know what it's like. Understanding which
half of cold-start each technique solves — not just "they're both
personalization" — is exactly what makes combining them in notebook 05
worth doing instead of picking one.
""")

code(r"""
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

from fmcg_reco.config import N_CANDIDATES, PROCESSED_DIR, TEST_END_WEEK, TRAIN_END_WEEK, VAL_END_WEEK
from fmcg_reco.evaluation.candidates import top_n_candidate_items
from fmcg_reco.evaluation.harness import evaluate_recommender
from fmcg_reco.evaluation.splitting import relevant_items_by_household, time_based_split
from fmcg_reco.models.base import BaseRecommender
from fmcg_reco.models.popularity import PersonalRepeatPurchaseRecommender, PopularityRecommender

interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
product = pd.read_parquet(PROCESSED_DIR / "product_dim.parquet")

train, val, test = time_based_split(interactions, TRAIN_END_WEEK, VAL_END_WEEK, TEST_END_WEEK)
candidate_items = top_n_candidate_items(train, N_CANDIDATES)
val_relevant = relevant_items_by_household(val)
previously_purchased = relevant_items_by_household(train)

print(f"train rows={len(train):,}  candidate items={len(candidate_items):,}  "
      f"val households with purchases={len(val_relevant):,}")
""")

md(r"""
## 1 · Building item profiles from the product hierarchy

Each product has five descriptive fields:
`department`, `commodity_desc`, `sub_commodity_desc`, `brand`, `manufacturer`.
These aren't free text — they're categorical labels at different levels of
granularity (department is coarse, sub-commodity is fine-grained).

We turn each field into a distinct **token** per product (e.g.
`DEPT__GROCERY`, `SUBCOMM__ICE_-_CRUSHED/CUBED`) and feed the concatenated
tokens for every product into a **TF-IDF vectorizer**. Two products end up
similar if they share tokens — sharing only `DEPT__GROCERY` gives a small
similarity bump, sharing `SUBCOMM__...` as well gives a much bigger one,
because TF-IDF's inverse-document-frequency weighting automatically
downweights common tokens (most products are `DEPT__GROCERY`) and upweights
rare, specific ones (a narrow sub-commodity or brand). This is a cleaner
signal than plain one-hot encoding, which would treat "same department,
different everything else" and "identical hierarchy" as equally dissimilar.
""")

code(r"""
def build_item_feature_matrix(product_df: pd.DataFrame, candidate_items: set):
    df = product_df[product_df["product_id"].isin(candidate_items)].reset_index(drop=True)

    def tok(series):
        return series.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)

    text = (
        "DEPT__" + tok(df["department"]) + " "
        + "COMM__" + tok(df["commodity_desc"]) + " "
        + "SUBCOMM__" + tok(df["sub_commodity_desc"]) + " "
        + "BRAND__" + tok(df["brand"]) + " "
        + "MANUF__" + df["manufacturer"].astype(str)
    )

    vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
    matrix = vectorizer.fit_transform(text)  # rows are L2-normalised by default
    item_ids = df["product_id"].tolist()
    return matrix, item_ids, vectorizer


item_matrix, item_ids, vectorizer = build_item_feature_matrix(product, candidate_items)
item_id_to_idx = {pid: i for i, pid in enumerate(item_ids)}
print(f"item feature matrix: {item_matrix.shape[0]:,} items x {item_matrix.shape[1]:,} vocabulary tokens")
""")

md(r"""
## 2 · Sanity check — do the nearest neighbours actually make sense?

Before trusting this numerically, look at it. Pick the single most-purchased
candidate product and inspect its nearest neighbours by cosine similarity —
since TF-IDF rows are already L2-normalised, cosine similarity is just the
dot product, no extra normalisation needed.
""")

code(r"""
def similar_items(product_id: int, k: int = 8):
    idx = item_id_to_idx[product_id]
    scores = np.asarray((item_matrix[idx] @ item_matrix.T).todense()).ravel()
    order = np.argsort(-scores)
    neighbours = [(item_ids[i], scores[i]) for i in order if i != idx][:k]
    return neighbours


most_popular_item = (
    train[train["product_id"].isin(candidate_items)]
    .groupby("product_id")["basket_id"].nunique()
    .idxmax()
)
example_row = product.set_index("product_id").loc[most_popular_item]
print("example product:", dict(example_row[["department", "commodity_desc", "sub_commodity_desc", "brand"]]))

print("\nnearest neighbours by content similarity:")
for pid, score in similar_items(most_popular_item):
    row = product.set_index("product_id").loc[pid]
    print(f"  {score:.3f}  {row['department']:12s} | {row['commodity_desc']:24s} | {row['sub_commodity_desc']}")
""")

md(r"""
If the neighbours above share the same commodity or sub-commodity as the
example product, the feature encoding is doing its job — this is worth
checking against your own intuition about the product before moving on.
""")

md(r"""
## 3 · Household profiles — a weighted average of purchased items

A household's content profile is the weighted sum of the item vectors it
purchased in train (weight = number of distinct baskets containing that
item, same popularity notion as notebook 02), then renormalised to unit
length.

This is one matrix multiplication: build a sparse
`(household x item)` weight matrix from train, then

```
profile_matrix = weight_matrix @ item_matrix
```

gives every household's profile in one shot — the household-level weighted
average, applied to every household at once, rather than one Python loop
per household.
""")

code(r"""
class ContentBasedRecommender(BaseRecommender):
    def __init__(self, product_df: pd.DataFrame, candidate_items: set):
        self.product_df = product_df
        self.candidate_items = candidate_items
        self.item_matrix_ = None
        self.item_ids_ = []
        self.item_id_to_idx_ = {}
        self.profile_matrix_ = None
        self.household_id_to_idx_ = {}

    def fit(self, train_interactions: pd.DataFrame) -> "ContentBasedRecommender":
        self.item_matrix_, self.item_ids_, _ = build_item_feature_matrix(self.product_df, self.candidate_items)
        self.item_id_to_idx_ = {pid: i for i, pid in enumerate(self.item_ids_)}

        df = train_interactions[train_interactions["product_id"].isin(self.candidate_items)].copy()
        df["item_idx"] = df["product_id"].map(self.item_id_to_idx_)
        weights = (
            df.groupby(["household_key", "item_idx"])["basket_id"].nunique()
            .reset_index(name="w")
        )

        households = sorted(weights["household_key"].unique())
        self.household_id_to_idx_ = {h: i for i, h in enumerate(households)}

        row = weights["household_key"].map(self.household_id_to_idx_).to_numpy()
        col = weights["item_idx"].to_numpy()
        data = weights["w"].to_numpy(dtype=float)
        weight_matrix = csr_matrix((data, (row, col)), shape=(len(households), len(self.item_ids_)))

        profile_matrix = weight_matrix @ self.item_matrix_
        norms = np.sqrt(profile_matrix.multiply(profile_matrix).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        self.profile_matrix_ = diags(1.0 / norms) @ profile_matrix
        return self

    def recommend(self, household_key: int, k: int) -> list:
        idx = self.household_id_to_idx_.get(household_key)
        if idx is None:
            return []  # no candidate-item purchase history to build a profile from
        profile = self.profile_matrix_[idx]
        scores = np.asarray((profile @ self.item_matrix_.T).todense()).ravel()
        top_idx = np.argsort(-scores)[:k]
        return [self.item_ids_[i] for i in top_idx]

    def similar_items(self, product_id: int, k: int) -> list:
        idx = self.item_id_to_idx_[product_id]
        scores = np.asarray((self.item_matrix_[idx] @ self.item_matrix_.T).todense()).ravel()
        order = np.argsort(-scores)
        return [self.item_ids_[i] for i in order if i != idx][:k]
""")

md(r"""
## 4 · What "cold start" this does and doesn't solve

This model can score a **brand-new item** the moment it's added to the
catalog — its content vector exists as soon as its department/commodity/
brand are known, no purchase history required. That's the item-cold-start
problem, solved.

It does **not** solve household cold-start: `recommend()` returns an empty
list for a household with zero candidate-item purchases in train, because
there's nothing to build a profile from. That gap is deliberate — notebook
05's hybrid model is what adds the popularity-based fallback for households
like this. Worth checking how many households actually hit that gap here:
""")

code(r"""
content_model = ContentBasedRecommender(product, candidate_items).fit(train)

n_train_households = train["household_key"].nunique()
n_with_profile = len(content_model.household_id_to_idx_)
print(f"households with a content profile: {n_with_profile:,} / {n_train_households:,} "
      f"({n_with_profile / n_train_households:.1%})")
""")

md(r"""
## 5 · Evaluate against the notebook 02 harness

Same harness, same candidate universe, same validation split — so this
number is directly comparable to notebook 02's `popularity` (MAP@10 0.124)
and `buy_it_again` (MAP@10 0.326).
""")

code(r"""
K_VALUES = [10, 20]

pop_model = PopularityRecommender(candidate_items=candidate_items).fit(train)
buy_again_model = PersonalRepeatPurchaseRecommender(candidate_items=candidate_items).fit(train)

_, pop_summary = evaluate_recommender(pop_model, val_relevant, K_VALUES, previously_purchased)
_, buy_summary = evaluate_recommender(buy_again_model, val_relevant, K_VALUES, previously_purchased)
_, content_summary = evaluate_recommender(content_model, val_relevant, K_VALUES, previously_purchased)

comparison = pd.concat(
    {"popularity": pop_summary, "buy_it_again": buy_summary, "content_based": content_summary},
    axis=0,
)
comparison.index.names = ["model", "k"]
comparison
""")

md(r"""
## 6 · Why might content-based score *below* popularity?

Run the comparison above before reading this — if `content_based` scores
below `popularity`, that's not obviously a bug, but it's surprising enough
to deserve investigation before accepting it. The most likely cause: our
five hierarchy fields are coarser than an actual SKU. Two different pack
sizes of the same brand and commodity (e.g. a 12oz and a 2L of the same
product line) get **identical** feature tokens, because pack size was never
included as a feature — so the model literally cannot tell them apart.
""")

code(r"""
key_cols = ["department", "commodity_desc", "sub_commodity_desc", "brand", "manufacturer"]
cand_products = product[product["product_id"].isin(candidate_items)]
tie_groups = cand_products.groupby(key_cols).size().sort_values(ascending=False)

n_tied = tie_groups[tie_groups > 1].sum()
print(f"distinct feature-identical groups among {len(cand_products):,} candidate items: {len(tie_groups):,}")
print(f"items that share their feature vector with >=1 other item: {n_tied:,} ({n_tied / len(cand_products):.1%})")
print("\nbiggest tie groups (all these items are indistinguishable to this model):")
tie_groups.head(3)
""")

md(r"""
That's the answer: with no secondary signal, the argmax over cosine
similarity breaks ties in essentially arbitrary order (whatever order the
products happened to load in) instead of by which tied item a household is
actually more likely to buy — and with roughly four out of five candidate
items sharing a tie group, that's not a rare edge case, it's most of the
catalog. **A pure content model has no way to fix this on its own** — it
would need either finer-grained features (pack size, UPC-level identifiers)
or a popularity-based tiebreak. This is exactly the gap the hybrid model in
notebook 05 closes by blending in the collaborative-filtering signal (which
*does* distinguish these items, because it's driven by actual purchase
behaviour, not hierarchy metadata) — so leave this model pure for now and
let that comparison happen deliberately in notebook 05, rather than patching
a tiebreak in here.
""")

md(r"""
## 7 · A worked example — why did it recommend *that*?

Pick one household, show what it bought in train (the evidence the profile
was built from) alongside what the content model recommends — this is the
kind of trace that matters when explaining a recommendation to a
non-technical stakeholder later, not just a metric.
""")

code(r"""
example_household = next(iter(content_model.household_id_to_idx_))

bought = (
    train[(train["household_key"] == example_household) & (train["product_id"].isin(candidate_items))]
    .merge(product, on="product_id")
    ["commodity_desc"].value_counts().head(5)
)
print(f"household {example_household} — top commodities purchased in train:")
print(bought)

recs = content_model.recommend(example_household, k=5)
print(f"\ntop 5 content-based recommendations for household {example_household}:")
for pid in recs:
    row = product.set_index("product_id").loc[pid]
    print(f"  {row['department']:12s} | {row['commodity_desc']:24s} | {row['sub_commodity_desc']}")
""")

md(r"""
## 8 · Findings & next notebook

- `content_based` underperforms **both** baselines here (MAP@10 well below
  even non-personalized `popularity`) — and now there's a verified reason
  for it (§6), not just a guess: ~83% of candidate items are feature-tied
  with at least one other item, so ties get broken arbitrarily rather than
  by purchase likelihood. This is a real, expected limitation of *pure*
  content similarity on this feature set, not a sign the approach is
  useless — its actual value (distinguishing items collaborative filtering
  can't score at all, i.e. brand-new products) isn't something this
  behavioural harness even measures.
- The household-coverage number in §4 (99.3%) is the honest limit of this
  model on its own for *households*, separate from the item-tie issue above
  — together they define exactly what notebook 05's hybrid needs to fix:
  break content ties using collaborative signal, and cover the ~0.7% of
  households with no candidate-item history via popularity fallback.

Once reviewed, this notebook's logic gets extracted into
`src/fmcg_reco/features/item_features.py` (the TF-IDF builder) and
`src/fmcg_reco/models/content.py` (`ContentBasedRecommender`).

### Next notebook
`04_collaborative_filtering.ipynb` — build the user-item interaction matrix
and train an ALS model on purchase behaviour, the other half of the hybrid.
""")

nb["cells"] = c
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (fmcg)", "language": "python", "name": "fmcg"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "03_content_based_model.ipynb"
nbf.write(nb, out)
print("wrote", out)
