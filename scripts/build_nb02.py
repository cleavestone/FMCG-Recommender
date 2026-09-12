"""Generate notebooks/02_eval_harness_and_baselines.ipynb from source cells.

Same generated-from-source convention as build_nb01.py: keeps notebook diffs
clean while we iterate.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 02 · Evaluation Harness & Baselines

**Goal of this notebook (Phase 1):**

Before we build a single recommender, we need a way to know whether it is any
good. Get this wrong and every later notebook (content-based, collaborative,
hybrid) is comparing models against a broken yardstick — so this is the
notebook everything else depends on.

1. Turn raw transactions into a clean, reusable **interactions** table.
2. Split time correctly (no peeking into the future).
3. Decide what a model is even allowed to recommend (the *candidate universe*).
4. Implement and hand-verify the ranking metrics we'll use everywhere.
5. Define a common `Recommender` interface every model in this project will
   share — popularity and content-based and collaborative and hybrid.
6. Build two **baselines** and score them with the harness. Nothing beats a
   baseline in this project without a number to prove it.

### Why this is a distinct skill from "building a model"

It's tempting to jump straight to a fancy model. But an implicit-feedback
recommender (we only observe *purchased / not purchased*, never a 1–5 star
rating) is graded entirely by how well it ranks things — and ranking
evaluation has sharp edges (leakage, undefined metrics for users with no
holdout activity, an unbounded candidate space) that will silently produce
misleading numbers if you don't handle them deliberately. Getting comfortable
with *why* each guard rail exists is the actual transferable skill here, not
the baseline models themselves.
""")

code(r"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
print("raw dir:", RAW)
print("processed dir:", PROC)
""")

md(r"""
## 1 · Build the interactions table

Every notebook from here on (content-based, CF, hybrid) needs the same clean
`(household_key, product_id, week_no, ...)` interaction records. We build it
**once, here**, and save it to `data/processed/` so we're not re-deriving it
five times with five subtly different cleaning rules.

Cleaning decisions, made explicit (this is the kind of thing that quietly
breaks a project if left implicit):

- Drop transaction lines for a `product_id` that doesn't resolve in the
  product catalog (orphans — small in this dataset, per notebook 01).
- Drop lines with `quantity <= 0`. These are returns/corrections, not
  purchase signal — for an *implicit feedback* model we want "this household
  chose to acquire this product," and a return is the opposite of that.
- Keep `sales_value` as-is (it can be small/zero from full coupon coverage —
  that's a promotions question, not a data error) but we won't use it as the
  feedback strength; quantity/frequency drives that later.
""")

code(r"""
product = pd.read_csv(RAW / "product.csv")
product.columns = [c.strip().lower() for c in product.columns]
product = product.drop_duplicates(subset="product_id").reset_index(drop=True)

txn = pd.read_csv(RAW / "transaction_data.csv")
txn.columns = [c.strip().lower() for c in txn.columns]

n_before = len(txn)
txn = txn[txn["product_id"].isin(product["product_id"])]
txn = txn[txn["quantity"] > 0]
n_after = len(txn)
print(f"transaction lines: {n_before:,} -> {n_after:,} after cleaning "
      f"({n_before - n_after:,} dropped, {(n_before - n_after) / n_before:.2%})")

interactions = txn[[
    "household_key", "basket_id", "day", "week_no", "product_id",
    "quantity", "sales_value",
]].reset_index(drop=True)

interactions.to_parquet(PROC / "interactions.parquet", index=False)
product.to_parquet(PROC / "product_dim.parquet", index=False)
print(interactions.shape)
interactions.head()
""")

md(r"""
## 2 · Time-based split — why not a random split

A random train/test split on rows would let a model see a household's
**future** purchases while "predicting" its past ones — the model would look
great and be useless. Grocery recommendation is deployed as: *given
everything we know up to today, what will this household want next* — so the
split has to mirror that: **one global cutoff in time**, train on everything
before it, evaluate on everything after.

There's no calendar date in this dataset, only `week_no` (1–102), so the
cutoff is a week number. We use:

| split | weeks | purpose |
|---|---|---|
| train | 1–88 | fit every model |
| validation | 89–95 | pick hyperparameters / blend weights (later notebooks) |
| test | 96–102 | final, only-looked-at-once comparison |

88/7/7 keeps most of the ~2 years of history for training (recommenders need
depth of purchase history to say anything about repeat behaviour) while still
giving val/test windows wide enough to contain a meaningful number of
purchase events per household.
""")

code(r"""
def time_based_split(df: pd.DataFrame, train_end: int, val_end: int, test_end: int):
    train = df[df["week_no"] <= train_end]
    val = df[(df["week_no"] > train_end) & (df["week_no"] <= val_end)]
    test = df[(df["week_no"] > val_end) & (df["week_no"] <= test_end)]
    return train, val, test

TRAIN_END, VAL_END, TEST_END = 88, 95, 102
train, val, test = time_based_split(interactions, TRAIN_END, VAL_END, TEST_END)

for name, split in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:5s} weeks {split['week_no'].min():>3}-{split['week_no'].max():<3}  "
          f"rows={len(split):>8,}  households={split['household_key'].nunique():>5}  "
          f"products={split['product_id'].nunique():>6}")
""")

md(r"""
## 3 · The candidate universe — what a model is *allowed* to recommend

The catalog has tens of thousands of products. Scoring every model against
the *entire* catalog for every household is wasteful, and it's not even the
real problem: the long tail of one-off/rarely-stocked products has almost no
purchase signal to learn from, so no model can say anything reliable about
them anyway.

We restrict the recommendable universe to the **top-N most-purchased
products in train** (by number of distinct baskets they appeared in — more
robust than raw quantity, which one bulk-buying household could dominate).

This is a real scoping decision, not a detail to gloss over: any product a
household buys in val/test that falls **outside** this candidate set is an
automatic miss for every model, no matter how good. So we report the
**reachable recall ceiling** — the best recall@K any model could possibly get
given this scoping — right after choosing N, to be honest about the limit
we've imposed.
""")

code(r"""
N_CANDIDATES = 5000

item_popularity = (
    train.groupby("product_id")["basket_id"].nunique()
    .sort_values(ascending=False)
)
candidate_items = set(item_popularity.head(N_CANDIDATES).index)

coverage_of_train_lines = train["product_id"].isin(candidate_items).mean()
print(f"top {N_CANDIDATES} products = "
      f"{N_CANDIDATES / item_popularity.shape[0]:.1%} of the {item_popularity.shape[0]:,} "
      f"products purchased in train, but cover {coverage_of_train_lines:.1%} of train transaction lines")
""")

code(r"""
def relevant_items_by_household(df: pd.DataFrame) -> dict[int, set[int]]:
    return df.groupby("household_key")["product_id"].apply(set).to_dict()

val_relevant = relevant_items_by_household(val)

# The ceiling has to be measured the same way recall itself will be measured
# later: averaged per household, not as one flat set across all households.
# A flat "distinct products in candidates / distinct products bought by
# anyone" ratio looks far worse than this (~16% at N=5000) because it treats
# every long-tail product as equally important regardless of how many
# households actually bought it — but almost no single household's own
# purchases are dominated by that long tail, so it understates what's
# actually reachable per household. The per-household average is the number
# that's honest about the harness we're actually going to use.
per_household_ceiling = [
    len(relevant & candidate_items) / len(relevant)
    for relevant in val_relevant.values() if relevant
]
print(f"households with >=1 purchase in val window: {len(val_relevant):,}")
print(f"reachable recall ceiling on val, given top-{N_CANDIDATES} candidates "
      f"(mean over households): {np.mean(per_household_ceiling):.1%}")
""")

md(r"""
## 4 · Ranking metrics — definitions and a hand-worked check

Relevance here is **binary**: a household either bought a product in the
eval window or it didn't (no star ratings to weight by). For each household
we produce a ranked list of `k` recommended products and compare it to the
set of products it actually bought.

- **Precision@K** — of the K items we showed, what fraction were actually
  bought? (*Are we wasting the household's attention?*)
- **Recall@K** — of everything the household actually bought, what fraction
  did we surface? (*Are we finding all the things they wanted?*)
- **NDCG@K** — like recall, but a hit ranked #1 counts more than a hit ranked
  #10 (`1/log2(rank+1)` discount). (*Are the good hits also near the top?*)
- **MAP@K** (mean average precision) — rewards getting hits *and* getting
  them early, averaged over all relevant items, normalised by
  `min(|relevant|, K)`.

**Why this combination fits grocery, specifically:** in something like movie
recommendation, almost every relevant test item is something the user has
never seen before, so hit-rate against random negatives is basically enough.
Grocery is **repeat-purchase dominated** — most of what a household buys next
week is something it has bought before — so a model that just replays known
history already scores deceptively well. Recall@K and MAP@K (which reward
surfacing *many* of the correct repeat items, ranked well across the whole
relevant set) matter more here than in movie-style eval, and in notebook 05
we'll explicitly separate hits into "repeat" vs "genuinely novel" so a
model can't hide behind repeat-purchase alone.
""")

code(r"""
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k <= 0:
        return np.nan
    topk = recommended[:k]
    hits = sum(1 for item in topk if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    topk = recommended[:k]
    hits = sum(1 for item in topk if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    topk = recommended[:k]
    dcg = sum((1.0 if item in relevant else 0.0) / np.log2(i + 2) for i, item in enumerate(topk))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else np.nan


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    topk = recommended[:k]
    hits, sum_precision = 0, 0.0
    for i, item in enumerate(topk, start=1):
        if item in relevant:
            hits += 1
            sum_precision += hits / i
    denom = min(len(relevant), k)
    return sum_precision / denom if denom > 0 else np.nan
""")

code(r"""
# Hand-worked example so we trust the functions before trusting any model score.
# recommended (ranked, best first): A B C D E
# actually relevant (held-out purchases): B, D, F  (F was never recommended at all)
recommended = ["A", "B", "C", "D", "E"]
relevant = {"B", "D", "F"}

# by hand:
#   hits in top 5: B (rank 2), D (rank 4)  -> 2 hits
#   precision@5 = 2/5 = 0.400
#   recall@5    = 2/3 = 0.667   (F was never reachable)
#   ndcg@5:  dcg = 1/log2(3) + 1/log2(5) = 0.6309 + 0.4307 = 1.0616
#            idcg (3 relevant, but only room for k=5 -> ideal 3 hits at ranks 1,2,3)
#                 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309 + 0.5 = 2.1309
#            ndcg = 1.0616 / 2.1309 = 0.498
#   map@5:   hit at rank2 -> precision@2 = 1/2 = 0.5
#            hit at rank4 -> precision@4 = 2/4 = 0.5
#            sum = 1.0, denom = min(3, 5) = 3  -> map@5 = 0.333

expected = {"precision": 0.400, "recall": 0.667, "ndcg": 0.498, "map": 0.333}
actual = {
    "precision": round(precision_at_k(recommended, relevant, 5), 3),
    "recall": round(recall_at_k(recommended, relevant, 5), 3),
    "ndcg": round(ndcg_at_k(recommended, relevant, 5), 3),
    "map": round(average_precision_at_k(recommended, relevant, 5), 3),
}
print("expected:", expected)
print("actual:  ", actual)
assert all(abs(actual[m] - expected[m]) < 0.01 for m in expected), "metric mismatch — fix before trusting anything below"
print("\nmetrics verified against hand calculation.")
""")

md(r"""
## 5 · Repeat vs. novel hits

A model that only ever recommends a household's own purchase history can
still post a strong recall@K in grocery data, because so much of grocery
demand *is* repeat purchase. That's a legitimate strategy (it's literally our
second baseline below) — but it's not the same achievement as surfacing
something a household hasn't bought before yet would want. We track both so
later notebooks can't accidentally win on repeat purchases alone and call it
"personalization."
""")

code(r"""
def repeat_vs_novel_hits(recommended: list, relevant: set, previously_purchased: set, k: int) -> tuple[int, int]:
    hits = [item for item in recommended[:k] if item in relevant]
    repeat_hits = sum(1 for item in hits if item in previously_purchased)
    novel_hits = len(hits) - repeat_hits
    return repeat_hits, novel_hits
""")

md(r"""
## 6 · A common `Recommender` interface

Every model this project builds — popularity, content-based, collaborative,
hybrid, and eventually a learning-to-rank reranker — will share the same
shape: `fit(train)` once, then `recommend(household_key, k)` many times (and
later, `similar_items(product_id, k)` for item-to-item lookups). Defining
that interface now means the evaluation harness below never has to change as
models get more sophisticated, and it's exactly the interface the serving
API will call later.
""")

code(r"""
from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    # Common interface for every recommender in this project.

    @abstractmethod
    def fit(self, train_interactions: pd.DataFrame) -> "BaseRecommender":
        ...

    @abstractmethod
    def recommend(self, household_key: int, k: int) -> list:
        ...

    def similar_items(self, product_id: int, k: int) -> list:
        raise NotImplementedError(f"{type(self).__name__} does not support item-item similarity")
""")

md(r"""
## 7 · Baseline 1 — Popularity

The reference bar every later model must beat. No personalization at all:
rank the candidate items by how many distinct baskets they appeared in
during train, and hand every household the same list.
""")

code(r"""
class PopularityRecommender(BaseRecommender):
    def __init__(self, candidate_items: set | None = None):
        self.candidate_items = candidate_items
        self.ranking_: list = []

    def fit(self, train_interactions: pd.DataFrame) -> "PopularityRecommender":
        df = train_interactions
        if self.candidate_items is not None:
            df = df[df["product_id"].isin(self.candidate_items)]
        self.ranking_ = (
            df.groupby("product_id")["basket_id"].nunique()
            .sort_values(ascending=False)
            .index.tolist()
        )
        return self

    def recommend(self, household_key: int, k: int) -> list:
        return self.ranking_[:k]
""")

md(r"""
## 8 · Baseline 2 — "Buy it again"

Grocery-specific and deceptively hard to beat: rank each household's *own*
previously purchased products by how often they recur, and only fall back to
global popularity to fill remaining slots (e.g. for a household with fewer
than `k` distinct past products).
""")

code(r"""
class PersonalRepeatPurchaseRecommender(BaseRecommender):
    def __init__(self, candidate_items: set | None = None):
        self.candidate_items = candidate_items
        self.personal_map_: dict[int, list] = {}
        self.fallback_: PopularityRecommender | None = None

    def fit(self, train_interactions: pd.DataFrame) -> "PersonalRepeatPurchaseRecommender":
        df = train_interactions
        if self.candidate_items is not None:
            df = df[df["product_id"].isin(self.candidate_items)]

        counts = (
            df.groupby(["household_key", "product_id"])["basket_id"].nunique()
            .reset_index(name="n_baskets")
            .sort_values(["household_key", "n_baskets"], ascending=[True, False])
        )
        self.personal_map_ = counts.groupby("household_key")["product_id"].apply(list).to_dict()

        self.fallback_ = PopularityRecommender(self.candidate_items).fit(train_interactions)
        return self

    def recommend(self, household_key: int, k: int) -> list:
        items = list(self.personal_map_.get(household_key, []))
        if len(items) < k:
            for item in self.fallback_.recommend(household_key, k + len(items)):
                if item not in items:
                    items.append(item)
                if len(items) >= k:
                    break
        return items[:k]
""")

md(r"""
## 9 · The evaluation harness

Runs any `BaseRecommender` against a `{household_key: relevant_items}` map
for a list of `k` values, and returns both the per-household results (useful
for later error analysis) and a mean summary table. Households with **no**
purchases in the eval window are skipped — recall/precision are undefined
for them, not zero, so including them would silently bias every metric.
""")

code(r"""
def evaluate_recommender(
    model: BaseRecommender,
    relevant_map: dict[int, set],
    k_values: list[int],
    previously_purchased_map: dict[int, set] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    max_k = max(k_values)
    for household_key, relevant in relevant_map.items():
        if not relevant:
            continue
        recs = model.recommend(household_key, max_k)
        prev = previously_purchased_map.get(household_key, set()) if previously_purchased_map else set()
        for k in k_values:
            row = {
                "household_key": household_key,
                "k": k,
                "precision": precision_at_k(recs, relevant, k),
                "recall": recall_at_k(recs, relevant, k),
                "ndcg": ndcg_at_k(recs, relevant, k),
                "map": average_precision_at_k(recs, relevant, k),
            }
            if previously_purchased_map is not None:
                repeat_hits, novel_hits = repeat_vs_novel_hits(recs, relevant, prev, k)
                row["repeat_hits"], row["novel_hits"] = repeat_hits, novel_hits
            rows.append(row)

    results = pd.DataFrame(rows)
    metric_cols = [col for col in ["precision", "recall", "ndcg", "map", "repeat_hits", "novel_hits"] if col in results]
    summary = results.groupby("k")[metric_cols].mean().round(4)
    return results, summary
""")

md(r"""
## 10 · Run it — popularity vs. buy-it-again on the validation split
""")

code(r"""
previously_purchased = relevant_items_by_household(train)

pop_model = PopularityRecommender(candidate_items=candidate_items).fit(train)
buy_again_model = PersonalRepeatPurchaseRecommender(candidate_items=candidate_items).fit(train)

K_VALUES = [10, 20]
pop_results, pop_summary = evaluate_recommender(pop_model, val_relevant, K_VALUES, previously_purchased)
buy_results, buy_summary = evaluate_recommender(buy_again_model, val_relevant, K_VALUES, previously_purchased)

comparison = pd.concat({"popularity": pop_summary, "buy_it_again": buy_summary}, axis=0)
comparison.index.names = ["model", "k"]
comparison
""")

md(r"""
## 11 · Findings & next notebook

- The interactions/product tables are now materialised at
  `data/processed/{interactions,product_dim}.parquet` — every later notebook
  reads these instead of re-cleaning raw CSVs.
- The reachable recall ceiling printed in §3 is the honest upper bound every
  later model is subject to, given the top-N candidate scoping — worth
  re-checking if `N_CANDIDATES` ever changes.
- **`buy_it_again` should outperform plain `popularity`** on recall/MAP,
  because it's exploiting repeat-purchase structure the popularity model
  ignores entirely — that gap *is* the signal that personalization is worth
  building further. If it doesn't outperform popularity, that's a red flag
  to debug before moving on, not a result to accept.
- These numbers are the yardstick: notebook 05's hybrid model has to beat
  `buy_it_again`, not just beat popularity, to justify combining content and
  collaborative signals at all.

Once reviewed, this notebook's logic gets extracted into
`src/fmcg_reco/evaluation/{splitting,metrics,harness}.py` and
`src/fmcg_reco/models/{base,popularity}.py`, with unit tests around the
metric functions and the split boundaries.

### Next notebook
`03_content_based_model.ipynb` — build item profiles from the product
hierarchy and a first personalized, non-collaborative recommender.
""")

nb["cells"] = c
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (fmcg)", "language": "python", "name": "fmcg"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_eval_harness_and_baselines.ipynb"
nbf.write(nb, out)
print("wrote", out)
