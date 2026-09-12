"""Generate notebooks/06_export_artifacts.ipynb from source cells.

Same generated-from-source convention as build_nb01-05.py.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 06 · Export Artifacts

**This is the last modelling notebook.** Everything from here on is
implementation code (the serving API), not further experimentation. The
job of this notebook is narrow but important: turn the validated model from
notebook 05 into files a running service can load at startup, with no
dependency on the raw 2.6M-row transaction history or a live Jupyter kernel.

1. Retrain the winning hybrid configuration on **all available data**, not
   just the notebook 02-05 train split.
2. Export it via `fmcg_reco.artifacts.save_artifacts`.
3. Reload it via `load_artifacts` and **prove** the reload is faithful —
   not just "it didn't crash," but that it produces identical
   recommendations to the model that was just fit, for both a warm and a
   cold household.

### Why retrain on all the data now

Notebooks 02-05 held out weeks 89-102 specifically so evaluation wouldn't
be contaminated by data the model was trained on. That constraint no longer
applies to the artifact that actually gets served — there's no "future" left
to leak from once we're not scoring against it. The deployed model should
reflect the most complete picture of household behaviour and product
popularity available, so both the **candidate universe** and the **hybrid's
component models** get refit on the full `interactions` table.
""")

code(r"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

from fmcg_reco.artifacts import load_artifacts, save_artifacts
from fmcg_reco.config import N_CANDIDATES, PROCESSED_DIR, RAW_DIR, ROOT
from fmcg_reco.evaluation.candidates import top_n_candidate_items
from fmcg_reco.models.hybrid import DEFAULT_WEIGHTS, HybridRecommender

interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
product = pd.read_parquet(PROCESSED_DIR / "product_dim.parquet")
demographic = pd.read_csv(RAW_DIR / "hh_demographic.csv")
demographic.columns = [col.strip().lower() for col in demographic.columns]

# recompute the candidate universe on ALL weeks, not just weeks 1-88 -
# the deployed model should reflect current popularity, not a stale window.
candidate_items = top_n_candidate_items(interactions, N_CANDIDATES)
print(f"households={interactions['household_key'].nunique():,}  "
      f"candidate items={len(candidate_items):,}  weeks covered={interactions['week_no'].nunique()}")
print(f"hybrid weights being deployed: {DEFAULT_WEIGHTS} (notebook 05's finding: "
      "collapses to pure repeat-purchase - see that notebook for why)")
""")

md(r"""
## 1 · Fit the final model

Same `HybridRecommender`, same weights notebook 05 found — the only
difference from notebook 05 is the data it's trained on (all of
`interactions`, not the weeks-1-88 train split).
""")

code(r"""
final_model = HybridRecommender(product, demographic, candidate_items, weights=DEFAULT_WEIGHTS).fit(interactions)

print(f"warm households: {len(final_model.warm_households_):,}")
print(f"sample recommendation (household {next(iter(final_model.warm_households_))}): "
      f"{final_model.recommend(next(iter(final_model.warm_households_)), k=5)}")
""")

md(r"""
## 2 · Export

Writes everything a fresh process needs to serve recommendations, without
ever seeing `interactions.parquet` again. See `fmcg_reco/artifacts.py`'s
module docstring for exactly what gets persisted and why (raw ALS factor
arrays rather than a pickled model object, JSON for the plain-dict pieces,
sparse `.npz` for the content vectors).
""")

code(r"""
ARTIFACT_DIR = ROOT / "artifacts" / "latest"
save_artifacts(final_model, ARTIFACT_DIR)

written = sorted(ARTIFACT_DIR.iterdir())
for f in written:
    print(f"{f.name:36s} {f.stat().st_size / 1024:8.1f} KB")
print(f"\ntotal: {sum(f.stat().st_size for f in written) / 1024:.1f} KB across {len(written)} files")
""")

md(r"""
## 3 · Reload and prove it's faithful

Loading the exact same files a freshly-started API process would load, on
a completely independent `HybridRecommender` instance that never called
`.fit()`. If this doesn't match the original bit-for-bit, the artifact
format has a bug that would silently serve wrong recommendations in
production — this check exists specifically to catch that class of error
before it ships.
""")

code(r"""
reloaded_model = load_artifacts(ARTIFACT_DIR, product, demographic)

sample_warm = list(final_model.warm_households_)[:5]
sample_cold = list(set(interactions["household_key"].unique()) - final_model.warm_households_)[:5]

all_match = True
for household_key in sample_warm + sample_cold:
    original = final_model.recommend(household_key, k=10)
    reloaded = reloaded_model.recommend(household_key, k=10)
    match = original == reloaded
    all_match &= match
    kind = "warm" if household_key in sample_warm else "cold"
    print(f"household {household_key} ({kind}): {'match' if match else 'MISMATCH'}")

example_item = next(iter(candidate_items))
similar_match = final_model.similar_items(example_item, k=5) == reloaded_model.similar_items(example_item, k=5)
print(f"\nsimilar_items({example_item}) match: {similar_match}")

assert all_match and similar_match, "reload did not reproduce the original model's output - fix before shipping"
print("\nall checks passed: the reloaded artifact is behaviourally identical to the freshly-fit model.")
""")

md(r"""
## 4 · Findings & what's next

- Confirm after running: the artifact bundle size (§2) and the reload check
  (§3) should both look unremarkable — small files, exact matches. That's
  the point; a boring reload check is a passing one.
- From here, the project moves from notebooks into regular application
  code: a FastAPI service that calls `fmcg_reco.artifacts.load_artifacts`
  once at startup and serves `recommend()` / `similar_items()` behind
  authenticated endpoints. There's no notebook 07 — the remaining work
  (OAuth2 password-grant auth, routers, Dockerfile, CI) is implementation,
  not experimentation, per the project plan.
- `scripts/train.py` wraps this notebook's §0-2 logic as a CLI
  (`uv run python scripts/train.py`) so artifacts can be regenerated
  without opening Jupyter — useful once this becomes something CI or a
  scheduled job runs, not just something triggered by hand.
""")

nb["cells"] = c
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (fmcg)", "language": "python", "name": "fmcg"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "06_export_artifacts.ipynb"
nbf.write(nb, out)
print("wrote", out)
