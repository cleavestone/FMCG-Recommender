"""Serialize a fitted HybridRecommender or BasketAffinityRecommender to disk
and reload without retraining — the train/serve boundary for both
recommenders that make up this project's two recommendation surfaces
(reorder-focused hybrid, and discovery-focused basket affinity).

Extracted from notebooks/06_export_artifacts.ipynb and 07_market_basket_analysis.ipynb.

Design choices, and why:

- ALS is persisted as raw `user_factors`/`item_factors` numpy arrays, not a
  pickled `implicit.als.AlternatingLeastSquares` object. Arrays are stable
  across library versions; a pickled model object is tied to that exact
  class layout and can break on an `implicit` upgrade. Verified empirically
  (see notebook 06) that `implicit`'s `.recommend()` output is identical
  regardless of the `user_items` row passed in, as long as
  `filter_already_liked_items=False` and `recalculate_user=False` (both
  true here) — so the training interaction matrix itself does NOT need to
  be persisted at all, only a zero-shaped placeholder at load time.
- Content-based state (item feature matrix, household profiles) is
  persisted as sparse arrays (`.npz`) plus id-order lists (`.json`) —
  small, human-inspectable, and not tied to a specific TfidfVectorizer
  pickle version.
- Popularity-derived rankings (repeat-purchase personal map, segment
  rankings, global fallback) are plain JSON — no numpy/scipy dependency to
  read them back, and they're small enough that this costs nothing.
- Loading reconstructs each sub-model's fitted attributes directly
  (bypassing `.fit()` entirely) rather than retraining, which is the whole
  point of the train/serve split: the service should never need the raw
  2.6M-row transaction history to start up.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, load_npz, save_npz

from fmcg_reco.models.affinity import BasketAffinityRecommender
from fmcg_reco.models.collaborative import ALSRecommender
from fmcg_reco.models.content import ContentBasedRecommender
from fmcg_reco.models.hybrid import HybridRecommender
from fmcg_reco.models.popularity import (
    PersonalRepeatPurchaseRecommender,
    PopularityRecommender,
    SegmentPopularityRecommender,
)
from fmcg_reco.models.replenishment import ReplenishmentForecaster


def _json_default(obj):
    """Groupby/pandas keys and values often come back as numpy scalar types
    (np.int64 etc.), which the stdlib json module doesn't know how to encode.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dumps(obj) -> str:
    return json.dumps(obj, default=_json_default)


def _ids_by_index(id_to_idx: dict) -> list:
    """Invert an {id: row_index} map back into a list ordered by row index,
    so a saved sparse matrix's rows can be matched back to their ids exactly.
    """
    ids = [None] * len(id_to_idx)
    for item_id, idx in id_to_idx.items():
        ids[idx] = item_id
    return ids


def save_artifacts(hybrid: HybridRecommender, artifact_dir: Path) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    content = hybrid.content_model_
    als = hybrid.als_model_
    repeat = hybrid.repeat_model_
    segment = hybrid.segment_model_

    save_npz(artifact_dir / "content_item_features.npz", content.item_matrix_)
    (artifact_dir / "content_item_ids.json").write_text(_dumps(content.item_ids_))

    save_npz(artifact_dir / "content_profiles.npz", content.profile_matrix_.tocsr())
    (artifact_dir / "content_household_ids.json").write_text(
        _dumps(_ids_by_index(content.household_id_to_idx_))
    )

    np.savez(
        artifact_dir / "als_factors.npz",
        user_factors=als.model_.user_factors,
        item_factors=als.model_.item_factors,
    )
    (artifact_dir / "als_household_ids.json").write_text(_dumps(als.household_ids_))
    (artifact_dir / "als_item_ids.json").write_text(_dumps(als.item_ids_))

    (artifact_dir / "repeat_personal_map.json").write_text(
        _dumps({str(h): items for h, items in repeat.personal_map_.items()})
    )

    (artifact_dir / "segment_rankings.json").write_text(_dumps(segment.segment_rankings_))
    (artifact_dir / "segment_household_map.json").write_text(
        _dumps({str(h): s for h, s in segment.household_to_segment_.items()})
    )

    # repeat.fallback_ and segment.global_fallback_ are both a PopularityRecommender
    # fit on the identical (train_interactions, candidate_items) - genuinely the
    # same ranking computed twice inside HybridRecommender.fit(). Store it once.
    (artifact_dir / "global_popularity_ranking.json").write_text(
        _dumps(repeat.fallback_.ranking_)
    )

    meta = {
        "candidate_items": sorted(hybrid.candidate_items),
        "weights": hybrid.weights,
        "n_per_signal": hybrid.n_per_signal,
        "als_config": {
            "factors": als.factors,
            "regularization": als.regularization,
            "iterations": als.iterations,
            "alpha": als.alpha,
        },
        "segment_col": segment.segment_col,
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))


def load_artifacts(artifact_dir: Path, product_df: pd.DataFrame, demographic_df: pd.DataFrame) -> HybridRecommender:
    artifact_dir = Path(artifact_dir)
    meta = json.loads((artifact_dir / "meta.json").read_text())
    candidate_items = set(meta["candidate_items"])

    hybrid = HybridRecommender(product_df, demographic_df, candidate_items, weights=meta["weights"],
                                n_per_signal=meta["n_per_signal"])

    content = ContentBasedRecommender(product_df, candidate_items)
    content.item_matrix_ = load_npz(artifact_dir / "content_item_features.npz")
    content.item_ids_ = json.loads((artifact_dir / "content_item_ids.json").read_text())
    content.item_id_to_idx_ = {pid: i for i, pid in enumerate(content.item_ids_)}
    content.profile_matrix_ = load_npz(artifact_dir / "content_profiles.npz")
    content_household_ids = json.loads((artifact_dir / "content_household_ids.json").read_text())
    content.household_id_to_idx_ = {h: i for i, h in enumerate(content_household_ids)}

    als_config = meta["als_config"]
    als = ALSRecommender(candidate_items=candidate_items, **als_config)
    als.household_ids_ = json.loads((artifact_dir / "als_household_ids.json").read_text())
    als.item_ids_ = json.loads((artifact_dir / "als_item_ids.json").read_text())
    als.household_id_to_idx_ = {h: i for i, h in enumerate(als.household_ids_)}
    als.item_id_to_idx_ = {p: i for i, p in enumerate(als.item_ids_)}
    factors = np.load(artifact_dir / "als_factors.npz")
    als.model_ = AlternatingLeastSquares(factors=als_config["factors"])
    als.model_.user_factors = factors["user_factors"]
    als.model_.item_factors = factors["item_factors"]
    # filter_already_liked_items=False + recalculate_user=False (both defaults used at
    # serve time) mean the row's actual content never affects the score - see module
    # docstring - so a zero placeholder of the right shape is sufficient here.
    als.user_items_ = csr_matrix((len(als.household_ids_), len(als.item_ids_)), dtype=np.float32)

    global_ranking = json.loads((artifact_dir / "global_popularity_ranking.json").read_text())
    global_fallback = PopularityRecommender(candidate_items)
    global_fallback.ranking_ = global_ranking

    repeat = PersonalRepeatPurchaseRecommender(candidate_items=candidate_items)
    personal_map = json.loads((artifact_dir / "repeat_personal_map.json").read_text())
    repeat.personal_map_ = {int(h): items for h, items in personal_map.items()}
    repeat.fallback_ = global_fallback

    segment = SegmentPopularityRecommender(candidate_items, demographic_df, segment_col=meta["segment_col"])
    segment.segment_rankings_ = json.loads((artifact_dir / "segment_rankings.json").read_text())
    household_map = json.loads((artifact_dir / "segment_household_map.json").read_text())
    segment.household_to_segment_ = {int(h): s for h, s in household_map.items()}
    segment.global_fallback_ = global_fallback

    hybrid.content_model_ = content
    hybrid.als_model_ = als
    hybrid.repeat_model_ = repeat
    hybrid.segment_model_ = segment
    hybrid.warm_households_ = set(content_household_ids)
    return hybrid


def save_affinity_artifacts(model: BasketAffinityRecommender, artifact_dir: Path) -> None:
    """BasketAffinityRecommender's fitted state is much simpler than the
    hybrid's (a small rules table + a household->purchased-items map), so
    plain parquet/JSON is enough - no sparse arrays or ML-library objects
    to work around here.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rules_out = model.rules_.copy()
    rules_out["antecedents"] = rules_out["antecedents"].apply(lambda s: sorted(s))
    rules_out["consequents"] = rules_out["consequents"].apply(lambda s: sorted(s))
    rules_out.to_parquet(artifact_dir / "affinity_rules.parquet", index=False)

    (artifact_dir / "affinity_household_items.json").write_text(
        _dumps({str(h): sorted(items) for h, items in model.household_items_.items()})
    )

    meta = {
        "candidate_items": sorted(model.candidate_items),
        "min_support": model.min_support,
        "min_lift": model.min_lift,
        "min_confidence": model.min_confidence,
        "max_len": model.max_len,
    }
    (artifact_dir / "affinity_meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))


def load_affinity_artifacts(artifact_dir: Path) -> BasketAffinityRecommender:
    artifact_dir = Path(artifact_dir)
    meta = json.loads((artifact_dir / "affinity_meta.json").read_text())
    candidate_items = set(meta["candidate_items"])

    model = BasketAffinityRecommender(
        candidate_items, min_support=meta["min_support"], min_lift=meta["min_lift"],
        min_confidence=meta["min_confidence"], max_len=meta["max_len"],
    )

    rules = pd.read_parquet(artifact_dir / "affinity_rules.parquet")
    rules["antecedents"] = rules["antecedents"].apply(lambda ids: frozenset(int(i) for i in ids))
    rules["consequents"] = rules["consequents"].apply(lambda ids: frozenset(int(i) for i in ids))
    model.rules_ = rules

    household_items = json.loads((artifact_dir / "affinity_household_items.json").read_text())
    model.household_items_ = {int(h): set(items) for h, items in household_items.items()}
    return model


def save_replenishment_artifacts(model: ReplenishmentForecaster, artifact_dir: Path) -> None:
    """ReplenishmentForecaster's fitted state is a single small, plain
    DataFrame (no sparse arrays, no per-household dicts) - parquet alone
    is enough.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model.pair_stats_.to_parquet(artifact_dir / "pair_stats.parquet", index=False)

    meta = {
        "min_purchases": model.min_purchases,
        "quantity_cap_quantile": model.quantity_cap_quantile,
        "reference_day": model.reference_day_,
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))


def load_replenishment_artifacts(artifact_dir: Path, product_df: pd.DataFrame) -> ReplenishmentForecaster:
    artifact_dir = Path(artifact_dir)
    meta = json.loads((artifact_dir / "meta.json").read_text())

    model = ReplenishmentForecaster(
        product_df, min_purchases=meta["min_purchases"], quantity_cap_quantile=meta["quantity_cap_quantile"]
    )
    model.pair_stats_ = pd.read_parquet(artifact_dir / "pair_stats.parquet")
    model.reference_day_ = meta["reference_day"]
    return model
