"""Generate notebooks/04_collaborative_filtering.ipynb from source cells.

Same generated-from-source convention as build_nb01/02/03.py.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 04 · Collaborative Filtering (ALS)

**Goal of this notebook (Phase 2b):**

Notebook 03 showed content-based filtering's blind spot clearly: it can't
tell two items apart if they share the same hierarchy metadata, no matter
how differently households actually buy them. Collaborative filtering (CF)
has exactly the opposite strength — it learns purely from **who bought what
alongside whom**, so two items end up "similar" only if real purchase
behaviour says so, regardless of what their metadata looks like.

1. Build the implicit-feedback interaction matrix with a **confidence**
   weighting (not raw counts) — this is a real, well-known technique, not
   an implementation detail to skim past.
2. Train **ALS** (Alternating Least Squares) via the `implicit` library and
   explain why matrix factorization fits this problem.
3. Tune the model against the notebook 02 harness (not by eyeballing loss).
4. Evaluate against every prior baseline — including whether ALS actually
   fixes notebook 03's tie problem.

### Why ALS specifically, and why this is a distinct skill from notebook 03

Content-based similarity was hand-defined (TF-IDF over metadata we chose).
ALS instead **learns** a small set of latent factors per household and per
item such that a household's affinity for an item is approximated by the
dot product of their factor vectors — nobody tells it what the factors
mean. That's the core idea behind essentially all modern matrix-
factorization recommenders, and it comes with a real cost worth naming
up front: it can say nothing about an item with no purchase history at all
(the mirror image of content-based's item-tie problem), which is exactly
the gap notebook 05's hybrid needs both techniques to jointly close.
""")

code(r"""
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

from implicit.als import AlternatingLeastSquares

from fmcg_reco.config import N_CANDIDATES, PROCESSED_DIR, TEST_END_WEEK, TRAIN_END_WEEK, VAL_END_WEEK
from fmcg_reco.evaluation.candidates import top_n_candidate_items
from fmcg_reco.evaluation.harness import evaluate_recommender
from fmcg_reco.evaluation.splitting import relevant_items_by_household, time_based_split
from fmcg_reco.models.base import BaseRecommender
from fmcg_reco.models.content import ContentBasedRecommender
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
## 1 · From purchase counts to confidence

Naively, we could feed ALS the raw number of times each household bought
each item. Two problems with that:

- The range is huge and skewed (notebook 01: baskets/household from 1 to
  1,300) — a household that bought one item 40 times would swamp the
  optimisation compared to one bought twice, even though "bought it twice"
  is already a meaningful repeat-purchase signal.
- Implicit feedback isn't a rating — buying something 5 times doesn't mean
  "5/5 stars," it means **stronger confidence** the household likes it, with
  diminishing returns per additional purchase.

The standard fix (Hu, Koren & Volinsky, 2008 — the paper the `implicit`
library implements) is to turn a raw count `r` into a **confidence**:

```
confidence = 1 + alpha * log1p(r)
```

The `log1p` compresses the long tail (40 purchases isn't "20x more
confident" than 2), and `alpha` controls how much purchase frequency should
matter at all relative to the baseline confidence of 1. We reuse the same
distinct-basket-count definition of `r` as every earlier notebook, for
consistency.
""")

code(r"""
ALPHA = 15.0

def build_interaction_matrix(train_interactions: pd.DataFrame, candidate_items: set, alpha: float = ALPHA):
    df = train_interactions[train_interactions["product_id"].isin(candidate_items)]
    counts = (
        df.groupby(["household_key", "product_id"])["basket_id"].nunique()
        .reset_index(name="r")
    )

    households = sorted(counts["household_key"].unique())
    items = sorted(candidate_items)
    household_to_idx = {h: i for i, h in enumerate(households)}
    item_to_idx = {p: i for i, p in enumerate(items)}

    confidence = (1.0 + alpha * np.log1p(counts["r"].to_numpy())).astype(np.float32)
    row = counts["household_key"].map(household_to_idx).to_numpy()
    col = counts["product_id"].map(item_to_idx).to_numpy()
    matrix = csr_matrix((confidence, (row, col)), shape=(len(households), len(items)))
    return matrix, households, items, household_to_idx, item_to_idx


user_items, households, items, household_to_idx, item_to_idx = build_interaction_matrix(train, candidate_items)
print(f"interaction matrix: {user_items.shape[0]:,} households x {user_items.shape[1]:,} items, "
      f"{user_items.nnz:,} nonzero entries ({user_items.nnz / (user_items.shape[0] * user_items.shape[1]):.2%} dense)")
""")

md(r"""
## 2 · Training ALS and tuning against the harness

ALS alternates between fixing item factors and solving for the best
household factors (then vice versa) — hence "alternating least squares."
Three hyperparameters matter most:

- **`factors`** — dimensionality of the learned taste space. Too few and it
  can't capture real variety in preference; too many and it overfits the
  sparse signal we have.
- **`regularization`** — standard L2 penalty against overfitting.
- **`iterations`** — how many alternating passes to run.

We don't guess these — we grid-search a small set and pick whichever
scores highest on **validation Recall@20**, exactly the discipline notebook
02 was built to enforce.
""")

code(r"""
class ALSRecommender(BaseRecommender):
    def __init__(self, candidate_items: set, factors: int = 64, regularization: float = 0.05,
                 iterations: int = 15, alpha: float = ALPHA, random_state: int = 42):
        self.candidate_items = candidate_items
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self.user_items_ = None
        self.household_ids_: list = []
        self.item_ids_: list = []
        self.household_id_to_idx_: dict = {}
        self.item_id_to_idx_: dict = {}
        self.model_: AlternatingLeastSquares | None = None

    def fit(self, train_interactions: pd.DataFrame) -> "ALSRecommender":
        (self.user_items_, self.household_ids_, self.item_ids_,
         self.household_id_to_idx_, self.item_id_to_idx_) = build_interaction_matrix(
            train_interactions, self.candidate_items, alpha=self.alpha
        )
        self.model_ = AlternatingLeastSquares(
            factors=self.factors, regularization=self.regularization,
            iterations=self.iterations, random_state=self.random_state,
        )
        self.model_.fit(self.user_items_, show_progress=False)
        return self

    def recommend(self, household_key: int, k: int) -> list:
        idx = self.household_id_to_idx_.get(household_key)
        if idx is None:
            return []  # ALS has no factors for a household absent from train
        ids, _ = self.model_.recommend(
            idx, self.user_items_[idx], N=k, filter_already_liked_items=False,
        )
        return [self.item_ids_[i] for i in ids]

    def similar_items(self, product_id: int, k: int) -> list:
        idx = self.item_id_to_idx_.get(product_id)
        if idx is None:
            raise KeyError(f"product_id {product_id} has no ALS factors (unseen in train)")
        ids, _ = self.model_.similar_items(idx, N=k + 1)
        return [self.item_ids_[i] for i in ids if i != idx][:k]
""")

md(r"""
**A detail that matters more than it looks like:** `implicit`'s `.recommend()`
defaults to `filter_already_liked_items=True` — it hides anything the
household already interacted with, because in most CF use cases (movies,
articles) re-recommending something already consumed is a bad experience.
Grocery is the opposite: 45.9% of transaction lines (notebook 01) are
*repeats*, and `buy_it_again` is the baseline to beat. We explicitly pass
`filter_already_liked_items=False` so ALS is judged on the same terms as
every other model in this project.
""")

code(r"""
K_VALUES = [10, 20]
GRID = [
    {"factors": 32, "regularization": 0.01},
    {"factors": 64, "regularization": 0.01},
    {"factors": 64, "regularization": 0.10},
    {"factors": 128, "regularization": 0.10},
]

grid_results = []
for params in GRID:
    model = ALSRecommender(candidate_items=candidate_items, iterations=15, **params).fit(train)
    _, summary = evaluate_recommender(model, val_relevant, [20], previously_purchased)
    grid_results.append({**params, "recall@20": summary.loc[20, "recall"], "map@20": summary.loc[20, "map"]})

grid_df = pd.DataFrame(grid_results).sort_values("recall@20", ascending=False)
grid_df
""")

md(r"""
## 3 · Fit the winning configuration and evaluate

Refit with the best row from the grid above, then run it through the exact
same harness call every other model in this project has used — so this
table is directly comparable to notebooks 02 and 03.
""")

code(r"""
best_params = grid_df.iloc[0][["factors", "regularization"]].to_dict()
best_params["factors"] = int(best_params["factors"])
print("best config:", best_params)

als_model = ALSRecommender(candidate_items=candidate_items, iterations=15, **best_params).fit(train)

pop_model = PopularityRecommender(candidate_items=candidate_items).fit(train)
buy_again_model = PersonalRepeatPurchaseRecommender(candidate_items=candidate_items).fit(train)
content_model = ContentBasedRecommender(product, candidate_items).fit(train)

_, pop_summary = evaluate_recommender(pop_model, val_relevant, K_VALUES, previously_purchased)
_, buy_summary = evaluate_recommender(buy_again_model, val_relevant, K_VALUES, previously_purchased)
_, content_summary = evaluate_recommender(content_model, val_relevant, K_VALUES, previously_purchased)
_, als_summary = evaluate_recommender(als_model, val_relevant, K_VALUES, previously_purchased)

comparison = pd.concat(
    {"popularity": pop_summary, "buy_it_again": buy_summary,
     "content_based": content_summary, "collaborative (ALS)": als_summary},
    axis=0,
)
comparison.index.names = ["model", "k"]
comparison
""")

md(r"""
## 4 · Does ALS actually solve notebook 03's tie problem?

Take the exact same tie group that beat content-based in notebook 03 (43
items tied on identical hierarchy metadata) and check whether ALS's
learned item factors — which come from real purchase behaviour, not
metadata — actually separate them.
""")

code(r"""
key_cols = ["department", "commodity_desc", "sub_commodity_desc", "brand", "manufacturer"]
cand_products = product[product["product_id"].isin(candidate_items)]
tie_groups = cand_products.groupby(key_cols).size().sort_values(ascending=False)
biggest_group_key = tie_groups.index[0]
tied_ids = cand_products.set_index(key_cols).loc[[biggest_group_key], "product_id"].tolist()
print(f"tie group ({biggest_group_key}): {len(tied_ids)} items, all identical to content-based")

available = [pid for pid in tied_ids if pid in als_model.item_id_to_idx_]
example_id = available[0]
neighbours = als_model.similar_items(example_id, k=5)
overlap_with_same_tie_group = len(set(neighbours) & set(tied_ids))

pop_in_train = train[train["product_id"].isin(candidate_items)].groupby("product_id")["basket_id"].nunique()
tie_group_freq = pop_in_train.reindex(tied_ids).describe()

print(f"ALS similar_items({example_id}) -> {neighbours}")
print(f"of those, {overlap_with_same_tie_group}/5 are still in the same content tie-group\n")
print("purchase frequency (distinct baskets) within this tie group:")
print(tie_group_freq[["mean", "50%", "min", "max"]])
print(f"\noverall candidate item median basket count: {pop_in_train.median():.0f}")
""")

md(r"""
**Read this result carefully before assuming CF "fixes" content-based's
ties** — it may not, and checking why matters more than the headline number.

If `similar_items` still returns mostly items from the same content tie
group, the first hypothesis to rule out is that these are just obscure,
rarely-purchased items ALS also has no signal for. The frequency check
above rules that out here: this tie group's items are purchased about as
often as, or more often than, the median candidate item (233 vs the
candidate-wide median around 157 baskets) — the overlap isn't a data-
sparsity artifact.

The more likely explanation: these 43 items genuinely **are** close
substitutes for real shoppers, not just for content metadata — a household
buying a "frozen economy entree" plausibly buys whichever brand/variant is
in stock, so *both* signals agreeing they're interchangeable may be the
correct answer, not a shared blind spot. That reframes the goal for
notebook 05: hybridization isn't "CF will silently repair every content
tie," it's "CF and content each catch different, real structure — CF is
strong exactly where content is weak (arbitrary metadata boundaries between
genuinely different purchase patterns), and agrees with content exactly
where the items really are interchangeable." Worth remembering as a general
lesson: when two independent signals agree, check whether that's shared
error or shared truth before drawing a conclusion either way.
""")

md(r"""
## 5 · Findings & next notebook

- The grid search picked **fewer factors (32), not more** — `factors=64/128`
  scored *worse* on validation recall than `factors=32`. With only ~2,500
  households and a 5.3%-dense interaction matrix, a higher-dimensional
  factor space has more room to overfit sparse signal rather than capture
  real structure. Bigger isn't automatically better once data is this
  limited — a concrete instance of a general ML lesson, not just a Dunnhumby
  quirk.
- **ALS beat `content_based` but lost to both `popularity` and
  `buy_it_again`** on every k. On its own, a purely behavioural
  matrix-factorization model doesn't out-rank "just replay what this
  household already buys" in repeat-purchase-dominated grocery data — the
  same bar notebook 02 set is still standing after two more modelling
  attempts.
- §4 is the more interesting result: CF didn't clearly "solve" content's
  tie-group problem for the one example checked — it agreed with content
  instead, and the frequency check shows that agreement probably reflects
  genuine product substitutability, not a shared data-sparsity failure.
  That's a more honest and more useful takeaway for notebook 05 than a
  simple "combining signals fixes everything" story would have been.

Once reviewed, this notebook's logic gets extracted into
`src/fmcg_reco/data/interactions.py` (`build_interaction_matrix`) and
`src/fmcg_reco/models/collaborative.py` (`ALSRecommender`).

### Next notebook
`05_hybrid_and_cold_start.ipynb` — combine content and collaborative scores,
handle cold-start explicitly, and produce the project's headline comparison
table.
""")

nb["cells"] = c
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (fmcg)", "language": "python", "name": "fmcg"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "04_collaborative_filtering.ipynb"
nbf.write(nb, out)
print("wrote", out)
