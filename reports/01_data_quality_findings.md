# Notebook 01 — Data Audit & EDA: Findings

**Dataset:** Dunnhumby *The Complete Journey* · 8 raw tables, no calendar dates
(only `day` 1–711 and `week_no` 1–102). All cells ran clean, 0 errors.

## Scale

| | value |
|---|---|
| households (transacting) | 2,500 |
| households with demographics | 801 (**32%**) |
| distinct products purchased | 92,339 (catalog: 92,353) |
| baskets | 276,484 |
| transaction lines | 2,595,732 |
| stores | 582 |
| total sales value | $8,057,463 |
| total quantity | 260,685,622 units |

No missing values anywhere across any of the 8 tables — a clean dataset by
Kaggle standards.

## Findings that will directly shape modelling decisions

1. **Demographics cover only 32% of households.** Any model that leans on
   `hh_demographic` (segment-based popularity, cold-start fallbacks) needs a
   *second* fallback for the 68% of households it can't reach. → drove the
   double-fallback design (segment popularity → global popularity) planned
   for the hybrid model in notebook 05.
2. **Repeat purchase dominates grocery behaviour.** 45.9% of all transaction
   lines are repurchases of something the household already bought before;
   26.9% of (household, product) pairs recur across multiple baskets. →
   explains why the `buy_it_again` baseline in notebook 02 crushed plain
   `popularity`, and why the eval harness tracks a repeat-vs-novel hit
   breakdown rather than trusting recall alone.
3. **Shopping frequency is extremely uneven.** Baskets per household: median
   79, mean 110.6, but max 1,300 (a 16x spread from median). → a naive
   quantity-weighted popularity ranking would let a handful of heavy
   shoppers dominate; ranking by *distinct-basket count* instead (as
   notebook 02 does) is materially more robust.
4. **Zero orphan product IDs** — every `product_id` in `transactions`
   resolves in the `product` dimension. Cleaning losses in later notebooks
   come entirely from behavioural filters (e.g. dropping `quantity <= 0`
   returns), not referential-integrity gaps.
5. **`coupon.csv` has 5,164 exact duplicate rows** (on `coupon_upc` +
   `product_id` + `campaign`) and **`causal_data.csv` has 15,245 duplicate
   rows** (on `product_id` + `store_id` + `week_no`). Neither table is used
   before Phase 5 (uplift / coupon targeting) or promo-exposure features, so
   this is a **TODO for that phase**, not a blocker now: dedupe before
   building any coupon-response or promo-lift features.
6. **Coupon redemption is sparse relative to campaign exposure**: 63% of
   households received ≥1 campaign, but only 434 households (17%) ever
   redeemed a coupon, across just 2,318 events total. Flags a real
   signal-scarcity constraint for the eventual uplift-modelling phase.
7. **Category mix is grocery-dominated** — `GROCERY` alone is 50.8% of sales;
   the top 5 departments (`GROCERY`, `DRUG GM`, `PRODUCE`, `MEAT`,
   `KIOSK-GAS`) cover ~84%. Reasonable expectation for FMCG data, worth
   knowing when picking which categories to spot-check model output against
   later.

## Not yet actioned (left as-is deliberately)

- Duplicate rows in `coupon`/`causal_data` are **not deduped yet** — no
  current notebook depends on either table being unique.
- No calendar dates exist, so all temporal work is anchored to `week_no`.

## Feeds into

Notebook 02's cleaning (`quantity > 0` filter), split design (time-based on
`week_no`), and candidate-popularity ranking (basket-count based) all build
directly on findings 3 and 4 above.
