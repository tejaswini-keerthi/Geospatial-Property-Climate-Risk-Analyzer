"""Feature engineering for the property value model and risk scoring.

This module is the single source of truth for *which* columns feed the model and
*how* categorical variables are encoded. It deliberately excludes identifiers,
raw coordinates, county, and H3 columns from the model matrix: county is held out
whole during spatial cross-validation, so leaking it (or coordinates that proxy it)
as a feature would defeat the point of testing cross-parish generalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

TARGET: str = "property_value_usd"
CURRENT_YEAR: int = 2026

#: Numeric features that generalize across parishes.
NUMERIC_FEATURES: list[str] = [
    "elevation_ft",
    "base_flood_elevation_ft",
    "distance_to_water_miles",
    "distance_to_coast_miles",
    "square_footage",
    "lot_size_acres",
    "num_bedrooms",
    "num_bathrooms",
    "stories",
    "year_built",
    "roof_age_years",
    "census_median_income",
    "storm_surge_risk_score",
    "annual_rainfall_inches",
    "hurricane_history_count",
    "drainage_quality_score",
    "impervious_surface_pct",
    "insurance_premium_usd",
    "has_flood_insurance",
]

#: Engineered numeric features added by :func:`build_features`.
DERIVED_FEATURES: list[str] = [
    "property_age",
    "freeboard_ft",
    "total_rooms",
    "sqft_per_bedroom",
    "is_high_hazard_zone",
]

#: Categorical features and their fixed category vocabularies (for stable one-hot).
CATEGORICAL_FEATURES: dict[str, list[str]] = {
    "fema_flood_zone": ["X", "A", "AE", "VE"],
    "foundation_type": ["slab", "pier_and_beam", "raised"],
}

#: Ordinal severity used by the composite climate-risk score.
_ZONE_SEVERITY: dict[str, float] = {"X": 0.0, "A": 0.45, "AE": 0.7, "VE": 1.0}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered numeric features to a copy of the input frame.

    Args:
        df: Raw (or ingested) property frame.

    Returns:
        A copy with :data:`DERIVED_FEATURES` columns added.
    """
    out = df.copy()
    out["property_age"] = (CURRENT_YEAR - out["year_built"]).clip(lower=0)
    # Freeboard: how far the ground sits above the base flood elevation (can be negative).
    out["freeboard_ft"] = out["elevation_ft"] - out["base_flood_elevation_ft"]
    out["total_rooms"] = out["num_bedrooms"] + out["num_bathrooms"]
    out["sqft_per_bedroom"] = out["square_footage"] / out["num_bedrooms"].clip(lower=1)
    out["is_high_hazard_zone"] = out["fema_flood_zone"].isin(["AE", "VE"]).astype(int)
    return out


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build the model design matrix ``X`` with one-hot encoded categoricals.

    Args:
        df: Frame that has already passed through :func:`build_features`.

    Returns:
        A tuple ``(X, feature_names)`` where ``X`` is fully numeric and
        column-stable regardless of which categories appear in ``df``.
    """
    numeric_cols = NUMERIC_FEATURES + DERIVED_FEATURES
    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing engineered feature columns: {missing}. Run build_features first.")

    parts: list[pd.DataFrame] = [df[numeric_cols].astype(float).reset_index(drop=True)]

    for col, categories in CATEGORICAL_FEATURES.items():
        cat = pd.Categorical(df[col], categories=categories)
        dummies = pd.get_dummies(cat, prefix=col).astype(float).reset_index(drop=True)
        # Guarantee every category column exists even if unobserved in this slice.
        for category in categories:
            expected = f"{col}_{category}"
            if expected not in dummies.columns:
                dummies[expected] = 0.0
        dummies = dummies[[f"{col}_{c}" for c in categories]]
        parts.append(dummies)

    x = pd.concat(parts, axis=1)
    feature_names = list(x.columns)
    logger.info("Built feature matrix: %d rows x %d features", x.shape[0], x.shape[1])
    return x, feature_names


def compute_climate_risk_score(df: pd.DataFrame) -> pd.Series:
    """Compute a 0-100 composite climate-risk score for each parcel.

    The score blends FEMA flood-zone severity, storm-surge exposure, elevation
    deficit relative to base flood elevation, and drainage quality. It is a
    domain-driven index (independent of the value model) used to colour the map.

    Args:
        df: Frame with the required hazard columns.

    Returns:
        A float Series in [0, 100] aligned to ``df``'s index.
    """
    zone_component = df["fema_flood_zone"].map(_ZONE_SEVERITY).fillna(0.0) * 100.0
    surge_component = df["storm_surge_risk_score"].clip(0, 100)
    # Negative freeboard (below base flood elevation) drives risk up sharply.
    freeboard = (df["elevation_ft"] - df["base_flood_elevation_ft"]).clip(-15, 15)
    elevation_component = ((15 - freeboard) / 30.0 * 100.0).clip(0, 100)
    drainage_component = 100.0 - df["drainage_quality_score"].clip(0, 100)

    score = (
        0.40 * zone_component
        + 0.28 * surge_component
        + 0.20 * elevation_component
        + 0.12 * drainage_component
    )
    return np.clip(score, 0, 100).round(2)


def risk_band(score: float) -> str:
    """Bucket a numeric risk score into a human-readable band.

    Args:
        score: Composite climate-risk score in [0, 100].

    Returns:
        One of ``"Low"``, ``"Moderate"``, ``"High"``, or ``"Severe"``.
    """
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Severe"
