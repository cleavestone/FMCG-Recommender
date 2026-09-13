"""CLI wrapper around notebooks 06/07/08's export logic: fit every
recommender on all available interaction data and write servable
artifacts to disk.

    uv run python scripts/train.py [--artifact-dir DIR]

This is a thin wrapper, not a second implementation: the actual training
and export logic lives in fmcg_reco.models.{hybrid,affinity,replenishment}
and fmcg_reco.artifacts. Useful once retraining needs to run outside a
notebook (CI, a scheduled job) without duplicating the logic.

Writes three independent artifact bundles under --artifact-dir, one per
recommendation surface (see notebook 07/08 findings for why these are
kept separate rather than blended into one model):
    <artifact-dir>/hybrid/        - reorder-focused (HybridRecommender)
    <artifact-dir>/affinity/      - discovery/cross-sell-focused (BasketAffinityRecommender)
    <artifact-dir>/replenishment/ - "what's due soon" (ReplenishmentForecaster)
"""
import argparse

import pandas as pd

from fmcg_reco.artifacts import (
    save_affinity_artifacts,
    save_artifacts,
    save_replenishment_artifacts,
)
from fmcg_reco.config import N_CANDIDATES, PROCESSED_DIR, RAW_DIR, ROOT
from fmcg_reco.evaluation.candidates import top_n_candidate_items
from fmcg_reco.models.affinity import BasketAffinityRecommender
from fmcg_reco.models.hybrid import DEFAULT_WEIGHTS, HybridRecommender
from fmcg_reco.models.replenishment import ReplenishmentForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", type=str, default=str(ROOT / "artifacts" / "latest"),
        help="directory to write artifacts to (default: artifacts/latest)",
    )
    args = parser.parse_args()

    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    product = pd.read_parquet(PROCESSED_DIR / "product_dim.parquet")
    demographic = pd.read_csv(RAW_DIR / "hh_demographic.csv")
    demographic.columns = [col.strip().lower() for col in demographic.columns]

    candidate_items = top_n_candidate_items(interactions, N_CANDIDATES)
    print(f"training on {interactions['household_key'].nunique():,} households, "
          f"{len(candidate_items):,} candidate items")

    hybrid = HybridRecommender(product, demographic, candidate_items, weights=DEFAULT_WEIGHTS).fit(interactions)
    save_artifacts(hybrid, f"{args.artifact_dir}/hybrid")
    print(f"wrote hybrid artifacts to {args.artifact_dir}/hybrid")

    affinity = BasketAffinityRecommender(candidate_items).fit(interactions)
    save_affinity_artifacts(affinity, f"{args.artifact_dir}/affinity")
    print(f"wrote affinity artifacts to {args.artifact_dir}/affinity")

    replenishment = ReplenishmentForecaster(product).fit(interactions)
    save_replenishment_artifacts(replenishment, f"{args.artifact_dir}/replenishment")
    print(f"wrote replenishment artifacts to {args.artifact_dir}/replenishment")


if __name__ == "__main__":
    main()
