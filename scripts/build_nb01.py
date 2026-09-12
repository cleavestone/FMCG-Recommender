"""Generate notebooks/01_data_audit_and_eda.ipynb from source cells.

Keeping the notebook under version control as generated-from-source avoids the
usual notebook diff noise while we iterate on Phase 0.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 01 · Data Audit & EDA — Dunnhumby *The Complete Journey*

**Goal of this notebook (Phase 0):**

1. Load all 8 raw tables and record their shape / schema.
2. Audit each table: missing values, duplicates, key cardinality, value ranges.
3. Check referential integrity between tables (the implied star schema).
4. Establish the core facts we will build on: time span, households, products,
   baskets, spend.
5. First-look EDA: spend, basket size, category mix, repeat-purchase behaviour,
   campaign / coupon coverage.
6. Write a short data-quality findings list to `reports/`.

Nothing here is modelling — this is the "know your data before you touch it" pass.

### The tables

| file | grain | role |
|---|---|---|
| `transaction_data` | one product line within a basket | fact |
| `product` | one product | dimension |
| `hh_demographic` | one household (subset only) | dimension |
| `campaign_desc` | one campaign | dimension |
| `campaign_table` | household × campaign received | bridge |
| `coupon` | coupon × product × campaign | dimension |
| `coupon_redempt` | household × coupon redemption event | fact |
| `causal_data` | product × store × week promo exposure | fact |
""")

code(r"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")

# repo root = parent of notebooks/
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW = ROOT / "data" / "raw"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
print("raw dir:", RAW)
print(sorted(p.name for p in RAW.glob("*.csv")))
""")

code(r"""
# Load everything. All small except causal_data (~36.7M rows / 664 MB) which we
# read with narrow dtypes so it stays ~700 MB in RAM instead of several GB.
FILES = {
    "transactions": "transaction_data.csv",
    "product": "product.csv",
    "demographic": "hh_demographic.csv",
    "campaign_desc": "campaign_desc.csv",
    "campaign_table": "campaign_table.csv",
    "coupon": "coupon.csv",
    "coupon_redempt": "coupon_redempt.csv",
}
raw = {}
for name, fn in FILES.items():
    raw[name] = pd.read_csv(RAW / fn)
    print(f"{name:16s} {str(raw[name].shape):>20s}   {fn}")

raw["causal"] = pd.read_csv(
    RAW / "causal_data.csv",
    dtype={"PRODUCT_ID": "int32", "STORE_ID": "int32", "WEEK_NO": "int16",
           "display": "category", "mailer": "category"},
)
print(f"{'causal':16s} {str(raw['causal'].shape):>20s}   causal_data.csv")
""")

code(r"""
# Normalise column names to snake_case lower for our sanity.
for name, df in raw.items():
    df.columns = [c.strip().lower() for c in df.columns]
raw["transactions"].head()
""")

md("## 1 · Per-table schema & head")

code(r"""
def schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_null": df.isna().sum(),
        "pct_null": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    out["example"] = [df[col].dropna().iloc[0] if df[col].notna().any() else None
                      for col in df.columns]
    return out

for name, df in raw.items():
    print(f"\n=== {name}  {df.shape} ===")
    display(schema(df))
""")

md("## 2 · Duplicates & key candidates")

code(r"""
key_guesses = {
    "transactions": ["household_key", "basket_id", "product_id"],
    "product": ["product_id"],
    "demographic": ["household_key"],
    "campaign_desc": ["campaign"],
    "campaign_table": ["household_key", "campaign"],
    "coupon": ["coupon_upc", "product_id", "campaign"],
    "coupon_redempt": ["household_key", "coupon_upc", "campaign", "day"],
    "causal": ["product_id", "store_id", "week_no"],
}
for name, keys in key_guesses.items():
    df = raw[name]
    keys = [k for k in keys if k in df.columns]
    dup_rows = df.duplicated().sum()
    dup_key = df.duplicated(subset=keys).sum() if keys else np.nan
    print(f"{name:16s} full-row dups={dup_rows:>7d}   dups on {keys} = {dup_key}")
""")

md("""
## 3 · Referential integrity

Every `product_id` / `household_key` that appears in a fact table should resolve
to its dimension — except `hh_demographic`, which Dunnhumby only provides for a
subset of households (this is expected and matters for cold-start framing).
""")

code(r"""
def coverage(child, child_col, parent, parent_col):
    c, p = raw[child][child_col], raw[parent][parent_col]
    missing = set(c.dropna().unique()) - set(p.dropna().unique())
    n_child = c.nunique()
    print(f"{child}.{child_col} -> {parent}.{parent_col}: "
          f"{n_child - len(missing)}/{n_child} resolve "
          f"({len(missing)} orphan ids)")

coverage("transactions", "product_id", "product", "product_id")
coverage("transactions", "household_key", "demographic", "household_key")
coverage("causal", "product_id", "product", "product_id")
coverage("coupon", "product_id", "product", "product_id")
coverage("coupon_redempt", "household_key", "campaign_table", "household_key")
coverage("campaign_table", "campaign", "campaign_desc", "campaign")

n_hh_txn = raw["transactions"]["household_key"].nunique()
n_hh_demo = raw["demographic"]["household_key"].nunique()
print(f"\nhouseholds with transactions : {n_hh_txn}")
print(f"households with demographics : {n_hh_demo} "
      f"({n_hh_demo / n_hh_txn:.0%} of transacting households)")
""")

md("## 4 · Core facts we build on")

code(r"""
t = raw["transactions"]
# DAY is an integer day index (1..711); WEEK_NO is 1..102. No calendar dates given.
facts = {
    "date span (day index)": (int(t["day"].min()), int(t["day"].max())),
    "date span (week_no)": (int(t["week_no"].min()), int(t["week_no"].max())),
    "n households": t["household_key"].nunique(),
    "n baskets": t["basket_id"].nunique(),
    "n products purchased": t["product_id"].nunique(),
    "n stores": t["store_id"].nunique(),
    "n transaction lines": len(t),
    "total sales_value": round(t["sales_value"].sum(), 2),
    "total quantity": int(t["quantity"].sum()),
}
for k, v in facts.items():
    print(f"{k:26s} {v}")

# sanity: lines per basket, baskets per household
lpb = t.groupby("basket_id").size()
bph = t.groupby("household_key")["basket_id"].nunique()
print("\nlines per basket   ", lpb.describe(percentiles=[.5, .9, .99]).round(1).to_dict())
print("baskets per household", bph.describe(percentiles=[.5, .9, .99]).round(1).to_dict())
""")

md("## 5 · First-look EDA")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(16, 4))

t.groupby("basket_id")["sales_value"].sum().clip(upper=200).hist(bins=60, ax=ax[0])
ax[0].set_title("Basket value ($, clipped at 200)")

lpb.clip(upper=40).hist(bins=40, ax=ax[1])
ax[1].set_title("Distinct products per basket (clipped at 40)")

weekly = t.groupby("week_no")["sales_value"].sum()
weekly.plot(ax=ax[2])
ax[2].set_title("Total weekly sales")
plt.tight_layout()
plt.savefig(FIG / "01_basket_and_weekly.png", dpi=110)
plt.show()
""")

code(r"""
# Category mix via the product hierarchy
p = raw["product"]
tp = t.merge(p[["product_id", "department", "commodity_desc"]], on="product_id", how="left")

dept = (tp.groupby("department")
          .agg(sales=("sales_value", "sum"), lines=("product_id", "size"))
          .sort_values("sales", ascending=False))
dept["sales_share"] = (dept["sales"] / dept["sales"].sum() * 100).round(1)
display(dept.head(15))

ax = dept["sales"].head(15).iloc[::-1].plot.barh(figsize=(8, 6))
ax.set_title("Top departments by sales")
plt.tight_layout()
plt.savefig(FIG / "01_department_sales.png", dpi=110)
plt.show()
""")

code(r"""
# Repeat-purchase behaviour — the thing that makes grocery recommendation different.
# For each (household, product): how many separate baskets contained it?
hp = (t.groupby(["household_key", "product_id"])["basket_id"]
        .nunique().rename("n_baskets").reset_index())
share_repeat = (hp["n_baskets"] > 1).mean()
print(f"share of (household, product) pairs bought in >1 basket: {share_repeat:.1%}")

# Of all purchase events, how many are of a product the household already bought before?
t_sorted = t.sort_values(["household_key", "day"])
first_seen = t_sorted.groupby(["household_key", "product_id"])["day"].transform("min")
repeat_line_share = (t_sorted["day"] > first_seen).mean()
print(f"share of transaction lines that are repeat purchases: {repeat_line_share:.1%}")

hp["n_baskets"].clip(upper=20).hist(bins=20)
plt.title("How many baskets contained a given household-product pair (clipped 20)")
plt.savefig(FIG / "01_repeat_purchase.png", dpi=110)
plt.show()
""")

code(r"""
# Campaign / coupon coverage — the raw material for Phase 5 (uplift).
ct, cd = raw["campaign_table"], raw["campaign_desc"]
cr = raw["coupon_redempt"]

print("distinct campaigns              :", cd["campaign"].nunique())
print("campaign types                  :", cd["description"].value_counts().to_dict())
print("households that received >=1 campaign:",
      ct["household_key"].nunique(),
      f"({ct['household_key'].nunique() / n_hh_txn:.0%})")
print("campaigns received per household :",
      ct.groupby("household_key")["campaign"].nunique()
        .describe(percentiles=[.5, .9]).round(1).to_dict())
print("households with >=1 coupon redemption:", cr["household_key"].nunique())
print("total coupon redemption events   :", len(cr))
""")

md(r"""
## 6 · Data-quality findings

_Fill this in as we go, then mirror the final list into `reports/data_quality.md`._

- **No calendar dates** — only `day` (1–711) and `week_no` (1–102) integer indices.
  All temporal splits use `week_no`.
- **Demographics cover only a subset of households** — cold-start / content-based
  models must not assume demographics exist.
- `sales_value` is *net* of `retail_disc` / `coupon_disc`; watch for zero and
  negative values (returns / full-coupon items).
- Products can appear in `transactions` without a `product` row (orphans) — decide
  keep vs drop.
- Campaigns were **not randomly assigned** — critical caveat for uplift modelling.

### Next notebook
`02_eval_harness_and_baselines.ipynb` — build the cleaned `interactions` /
`product_dim` tables, define the time-based train/val/test split, implement
the ranking metrics, and score the first baselines against them.
""")

nb["cells"] = c
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (fmcg)", "language": "python", "name": "fmcg"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_audit_and_eda.ipynb"
nbf.write(nb, out)
print("wrote", out)
