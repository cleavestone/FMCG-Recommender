# Notebook 03 — Content-Based Model: Findings

7 of 8 code cells confirmed run clean, 0 errors (the class-definition cell
has no output by design). **Two cells remain unrun on your end** — the tie
diagnostic (§4 below) and the worked example (§5) — go back and run those
before treating this notebook as done; the numbers here come from an
independent headless run, verified three times to be deterministic.

## 1. Item feature matrix

5,000 candidate items encoded into **1,551 vocabulary tokens** (from
department/commodity/sub_commodity/brand/manufacturer, prefixed and
TF-IDF-weighted). Small and sparse — this is a lightweight representation,
which matters later for how cheap it is to serve.

## 2. Similarity sanity check — passed

Nearest neighbours of **Bananas** (PRODUCE / TROPICAL FRUIT): Pineapple,
Avocado, Mango, Kiwi, "Tropical Fruit — Other" — all genuinely tropical
fruit. The encoding is doing what it's supposed to at the department/
commodity level.

**Also visible in that same output, worth noticing:** the neighbour list
contains "Pineapple Whole&Peel/Cored" **three times**, with two of those
entries tied at an identical score (0.382). That's not a display bug — it's
the first hint of the tie problem confirmed in §4: multiple distinct
`product_id`s (different pack sizes/variants) collapse to the exact same
feature vector.

## 3. Household coverage: 99.3%

2,481 of 2,499 train households get a content profile — only 18 households
have zero candidate-item purchases in train. Content-based solves
**household** cold-start almost completely on this data; the interesting
limitation turned out to be on the **item** side instead (§4).

## 4. Why content-based scored *below* popularity — root cause confirmed

| model | k | precision | recall | ndcg | map |
|---|---|---|---|---|---|
| popularity | 10 | 0.189 | 0.037 | 0.232 | 0.124 |
| buy_it_again | 10 | 0.400 | 0.074 | 0.447 | **0.326** |
| **content_based** | **10** | **0.080** | **0.018** | **0.088** | **0.040** |

Content-based isn't just behind `buy_it_again` (expected — it has no notion
of repeat purchase) — it's behind **plain popularity too**, which has no
personalization at all. That's surprising enough to demand a reason, not a
shrug.

**Confirmed cause:** among the 5,000 candidate items,

- **1,666 distinct feature groups** exist in total (vs. 5,000 items)
- **4,141 items (82.8%)** share their exact feature vector with at least
  one other item — the biggest tie group has **43 items** tied together
  (e.g. all "YOGURT / YOGURT NOT MULTI-PACKS / National" SKUs, differing
  only in a pack size or flavor never encoded as a feature)

With no secondary signal, ranking ties break in whatever order the items
happened to load in — essentially arbitrary with respect to which tied item
a household would actually buy. Since ~83% of the catalog is in some tie
group, this isn't a rare edge case, it's the dominant behaviour of the
model. **A pure content model has no internal way to fix this** — it needs
either finer features (pack size, UPC-level detail) or an external
tiebreak signal.

**This is the concrete argument for hybridization**, not an abstract one:
collaborative filtering (notebook 04) ranks by actual purchase behaviour, so
it *can* tell tied items apart — which is exactly what notebook 05's hybrid
blend is for.

## 5. Worked example (household 1)

Top train commodities: Baked Bread/Buns/Rolls (76), Bag Snacks (42),
Refrigerated Juices/Drinks (41), Tropical Fruit (35), Fluid Milk (35).
Top-5 recommendations pulled from Baked Bread, Tropical Fruit, and
Refrigerated Juices — directionally aligned with the household's actual
top categories, which is a reasonable trace even though the aggregate
metrics are weak. Useful to remember: *directionally sensible per-example*
and *quantitatively competitive* are different claims, and only the second
one is what the eval table measures.

## Decisions this locks in for notebooks 04–05

- **Content-based is being kept as a pure baseline**, not patched with a
  popularity tiebreak now — that patch is deliberately deferred to notebook
  05, where it becomes "blend with CF" rather than "bolt a hack onto
  content."
- **Bar to beat is still `buy_it_again`'s MAP@10 = 0.326** — content-based
  alone doesn't move that bar; the open question for notebook 05 is whether
  *combining* content with CF does.
- Household-cold-start coverage (99.3%) and the item-tie problem (82.8%)
  are now two separately-understood gaps, each pointing to a different half
  of why the hybrid design needs both a cold-start fallback rule *and* a
  CF-based tiebreak — not just "combine two scores and hope."

## Next

Extract `features/item_features.py` (the TF-IDF builder) and
`models/content.py` (`ContentBasedRecommender`) into `src/fmcg_reco/`, then
draft `04_collaborative_filtering.ipynb`.
