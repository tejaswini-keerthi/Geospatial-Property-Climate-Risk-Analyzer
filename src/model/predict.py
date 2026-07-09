"""Model loading, prediction, and scored-dataset assembly for the dashboard.

The dashboard does not train anything at runtime; instead it consumes a single
pre-computed artifact (``scored_properties.parquet``) produced by
:func:`build_scored_dataset`, which contains predicted values, the composite
climate-risk score, H3 cells, and each parcel's top SHAP features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import get_logger, settings
from src.model import evaluate, train
from src.pipeline import features as feat

logger = get_logger(__name__)


def load_model(model_path: Path | None = None) -> tuple[XGBRegressor, dict]:
    """Load a persisted XGBoost model and its metadata sidecar.

    Args:
        model_path: Path to ``model.json``; defaults to settings.

    Returns:
        Tuple of (loaded model, metadata dict).

    Raises:
        FileNotFoundError: If the model artifact is missing.
    """
    model_path = model_path or settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m src.model.train` first."
        )
    model = XGBRegressor()
    model.load_model(model_path)
    meta_path = model_path.with_name("model_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    logger.info(
        "Loaded model from %s (%d features)", model_path, len(meta.get("feature_names", []))
    )
    return model, meta


def predict_values(model: XGBRegressor, df: pd.DataFrame) -> np.ndarray:
    """Predict property values for a feature-engineered frame.

    Args:
        model: Fitted XGBoost regressor.
        df: Frame that has passed through :func:`~src.pipeline.features.build_features`.

    Returns:
        Array of predicted property values in USD.
    """
    x, _ = feat.get_feature_matrix(df)
    return model.predict(x)


def _serialize_top_features(
    top_features: list[list[tuple[str, float]]],
) -> list[str]:
    """Serialize per-row top SHAP features to JSON strings for parquet storage."""
    return [
        json.dumps([{"feature": name, "shap": round(val, 2)} for name, val in row])
        for row in top_features
    ]


def build_scored_dataset(save: bool = True, output_path: Path | None = None) -> pd.DataFrame:
    """Assemble the fully-scored dataset consumed by the dashboard.

    Steps: prepare features -> load model -> predict -> compute per-parcel top SHAP
    features -> attach risk score/band -> persist to parquet.

    Args:
        save: Whether to write the parquet artifact.
        output_path: Destination parquet path; defaults to settings.

    Returns:
        The scored DataFrame.
    """
    output_path = output_path or settings.scored_properties_path
    df = train.prepare_dataset()
    model, _ = load_model()

    x, _ = feat.get_feature_matrix(df)
    df = df.reset_index(drop=True)
    df["predicted_value_usd"] = model.predict(x).round(2)
    df["valuation_delta_usd"] = (df["predicted_value_usd"] - df[feat.TARGET]).round(2)

    logger.info("Computing per-parcel top SHAP features (n=%d)", len(df))
    top_feats = evaluate.top_shap_features(model, x, top_n=5)
    df["top_shap_features"] = _serialize_top_features(top_feats)

    df["climate_risk_score"] = feat.compute_climate_risk_score(df)
    df["risk_band"] = df["climate_risk_score"].apply(feat.risk_band)

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Wrote scored dataset (%d rows) to %s", len(df), output_path)
    return df


if __name__ == "__main__":
    scored = build_scored_dataset()
    logger.info("Scored dataset columns: %s", list(scored.columns))
