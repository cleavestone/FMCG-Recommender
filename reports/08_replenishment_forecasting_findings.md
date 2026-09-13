# Notebook 08 — Replenishment Forecasting: Findings

All 10 code cells confirmed run clean, 0 errors, matching an independent
headless run exactly on every number.

## The core distinction: this notebook answers a different question than 02-07

Every prior model answers *"is this item relevant"* (a ranking question).
This one answers *"when will this household need it again, and how much
will they buy"* — forecasting, not ranking. It needed its own evaluation
methodology (MAE against held-out actual purchases) rather than reusing
`Recall@K`.

## 1. Two real bugs caught before they became false findings

- **Quantity scoping bug**: the first version of the quantity validation
  produced an MAE of ~198 units — traced to gasoline and gift-card-
  activation SKUs (`KIOSK-GAS`, `MISC SALES TRAN`/`COUPON/MISC ITEMS`
  departments) recording "quantity" in units that don't mean "count of
  items" at all (one row had quantity=36,231). Fixed by excluding 134 such
  SKUs from the forecastable universe entirely — a scoping fix, not a
  metric patch, since "typical purchase quantity" doesn't apply to
  gasoline in the first place.
- **"Most overdue" UX bug**: the first version of the demo sorted purely
  by `days_until_due`, surfacing items last purchased 400+ days ago as the
  "most due." 46% of eligible pairs hadn't been repurchased in 90+ days as
  of end of train — most of that list was abandoned habits, not upcoming
  needs. Fixed with a 2x-typical-gap "still active" filter in
  `forecast()`. After the fix, the example household's due-soon list runs
  -83 to -8 days (mildly overdue to nearly due) instead of -446 to -201.

## 2. Scoping: only ~8% of pairs are even forecastable

1,211,935 (household, product) pairs after excluding non-grocery SKUs;
only **95,964 (7.9%)** have the minimum 4 purchases needed to estimate a
gap and its spread. This mirrors every prior scoping decision in this
project (candidate universe, tie groups, etc.) — most of the catalog
simply has no repeat pattern to forecast, and pretending otherwise would
produce garbage estimates from 1-2 data points.

## 3. Quantity forecasting works

| model | MAE (units) |
|---|---|
| per-pair median quantity | **0.28** |
| naive "always predict 1" | 0.39 |

A real, meaningful ~28% error reduction — confirms the earlier CV finding
(quantity CV median 0.19, 51% of pairs always buy the exact same amount).
This is the sub-problem worth shipping with confidence.

## 4. Timing forecasting did **not** earn its complexity

| model | MAE (days) |
|---|---|
| per-product median gap | 56.3 |
| household-average gap (ignores which product) | **52.1** |

The simpler baseline won. Not a rounding difference. Likely cause: a
per-pair median comes from a median of only 3 gaps (`MIN_PURCHASES=4`),
a genuinely noisy small-sample estimate, while the household-average
pools across every eligible product a household has, trading away
product-specificity for far more data to average out noise. Flagged
**shrinkage** (blend the noisy per-product estimate toward the household
average, weighted by how much history each product has) as the concrete
next step — not built now, since assuming it would help without checking
would repeat the same mistake this notebook is built around avoiding.

57.9% of actual gaps fell within the predicted median ± 1 std window —
usable for a coarse "roughly due" signal, not for a precise date.

## Net assessment

Quantity: ship it, it works. Timing: usable for a coarse "probably due
soon" list (via the 2x-gap active-habit filter) but the exact day
shouldn't be oversold — "due in ~5 days" is honest, a specific calendar
date is not.

## Next

Extract `ReplenishmentForecaster` into `src/fmcg_reco/models/replenishment.py`
with unit tests, add a third API surface
(`GET /households/{household_id}/replenishment`), then commit.
