"""Train an XGBoost property-value regressor with spatial block cross-validation.

Two evaluation regimes are reported:

1. **Spatial block CV** — :class:`~sklearn.model_selection.GroupKFold` folds keyed on
   H3 resolution-5 cells, so neighbouring parcels never appear in both train and test.
   This estimates local generalization without geographic leakage.
2. **Held-out parish** — train on two parishes, test on a third entirely unseen
   parish. This is the headline metric and targets an R^2 of ~0.72-0.76.

The final production model is refit on the full dataset and saved to ``model.json``
alongside a sidecar ``model_meta.json`` capturing feature order and metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.config import get_logger, settings
from src.pipeline import features as feat
from src.pipeline import h3_encoder, ingest

logger = get_logger(__name__)

#: Hyperparameters tuned for a stable, well-regularized fit on ~18k rows.
XGB_PARAMS: dict[str, object] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.045,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_lambda": 1.5,
    "reg_alpha": 0.1,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

#: Smallest parish, held out entirely for the headline generalization metric.
HOLDOUT_COUNTY: str = "Delacroix Parish"


@dataclass
class Metrics:
    """Container for regression evaluation metrics."""

    r2: float
    rmse: float
    mae: float
    n: int

    def pretty(self) -> str:
        """Return a formatted one-line summary."""
        return f"R2={self.r2:.4f} | RMSE=${self.rmse:,.0f} | MAE=${self.mae:,.0f} | n={self.n:,}"


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Compute R^2, RMSE, and MAE for a set of predictions."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return Metrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=rmse,
        mae=float(mean_absolute_error(y_true, y_pred)),
        n=len(y_true),
    )


def prepare_dataset(force_regenerate: bool = False) -> pd.DataFrame:
    """Load, H3-encode, and feature-engineer the full property dataset.

    Args:
        force_regenerate: Regenerate the mock CSV before loading (mock mode only).

    Returns:
        A model-ready DataFrame with engineered features, H3 columns, and risk score.
    """
    raw = ingest.load_data(force_regenerate=force_regenerate)
    encoded = h3_encoder.encode(raw)
    engineered = feat.build_features(encoded)
    engineered["climate_risk_score"] = feat.compute_climate_risk_score(engineered)
    return engineered


def spatial_block_cv(
    df: pd.DataFrame, n_splits: int = 5, params: dict[str, object] | None = None
) -> Metrics:
    """Run GroupKFold CV with H3 res-5 cells as spatial blocks.

    Args:
        df: Model-ready dataset (must contain ``h3_res5``).
        n_splits: Number of spatial folds.
        params: Optional XGBoost parameter override.

    Returns:
        Aggregate out-of-fold :class:`Metrics` across all folds.
    """
    params = params or XGB_PARAMS
    x, _ = feat.get_feature_matrix(df)
    y = df[feat.TARGET].to_numpy()
    groups = df["h3_res5"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(df))
    oof_mask = np.zeros(len(df), dtype=bool)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups), start=1):
        model = XGBRegressor(**params)
        model.fit(x.iloc[train_idx], y[train_idx])
        oof_pred[test_idx] = model.predict(x.iloc[test_idx])
        oof_mask[test_idx] = True
        fold_metrics = _score(y[test_idx], oof_pred[test_idx])
        logger.info("Spatial fold %d/%d -> %s", fold, n_splits, fold_metrics.pretty())

    return _score(y[oof_mask], oof_pred[oof_mask])


def evaluate_holdout_county(
    df: pd.DataFrame, holdout: str = HOLDOUT_COUNTY, params: dict[str, object] | None = None
) -> tuple[Metrics, XGBRegressor, pd.DataFrame, pd.Series]:
    """Train on all-but-one parish and evaluate on the held-out parish.

    Args:
        df: Model-ready dataset.
        holdout: Parish name to hold out for testing.
        params: Optional XGBoost parameter override.

    Returns:
        Tuple of (metrics, fitted model, held-out feature matrix, held-out target).
    """
    params = params or XGB_PARAMS
    train_df = df[df["county"] != holdout]
    test_df = df[df["county"] == holdout]

    x_train, _ = feat.get_feature_matrix(train_df)
    x_test, _ = feat.get_feature_matrix(test_df)
    y_train = train_df[feat.TARGET].to_numpy()
    y_test = test_df[feat.TARGET].to_numpy()

    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    metrics = _score(y_test, preds)
    logger.info("Held-out parish '%s' -> %s", holdout, metrics.pretty())
    return metrics, model, x_test, test_df[feat.TARGET].reset_index(drop=True)


def train_final_model(
    df: pd.DataFrame, params: dict[str, object] | None = None
) -> tuple[XGBRegressor, list[str]]:
    """Fit the production model on the entire dataset.

    Args:
        df: Model-ready dataset.
        params: Optional XGBoost parameter override.

    Returns:
        Tuple of (fitted model, ordered feature-name list).
    """
    params = params or XGB_PARAMS
    x, feature_names = feat.get_feature_matrix(df)
    y = df[feat.TARGET].to_numpy()
    model = XGBRegressor(**params)
    model.fit(x, y)
    return model, feature_names


def save_model(
    model: XGBRegressor,
    feature_names: list[str],
    holdout_metrics: Metrics,
    cv_metrics: Metrics,
    model_path: Path | None = None,
) -> Path:
    """Persist the model and a metadata sidecar.

    Args:
        model: Fitted XGBoost regressor.
        feature_names: Ordered feature columns used at fit time.
        holdout_metrics: Held-out parish metrics.
        cv_metrics: Spatial CV metrics.
        model_path: Destination for ``model.json``; defaults to settings.

    Returns:
        The path the model was written to.
    """
    model_path = model_path or settings.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    meta = {
        "feature_names": feature_names,
        "target": feat.TARGET,
        "holdout_county": HOLDOUT_COUNTY,
        "holdout_metrics": asdict(holdout_metrics),
        "spatial_cv_metrics": asdict(cv_metrics),
        "params": XGB_PARAMS,
    }
    meta_path = model_path.with_name("model_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved model to %s and metadata to %s", model_path, meta_path)
    return model_path


def run_training(make_shap: bool = True) -> dict[str, Metrics]:
    """Full training entry point: CV, held-out eval, SHAP, and model persistence.

    Args:
        make_shap: Whether to generate and save SHAP plots.

    Returns:
        Dict with ``"cv"`` and ``"holdout"`` metrics.
    """
    df = prepare_dataset()

    print("\n" + "=" * 68)
    print(" GEOSPATIAL PROPERTY CLIMATE RISK ANALYZER - TRAINING REPORT")
    print("=" * 68)
    print(f" Records: {len(df):,}  |  Parishes: {df['county'].nunique()}")
    print(f" Features: {len(feat.get_feature_matrix(df)[1])}")
    print("-" * 68)

    cv_metrics = spatial_block_cv(df)
    print(f" Spatial block CV (H3 res-5 folds): {cv_metrics.pretty()}")

    holdout_metrics, holdout_model, x_test, _y_test = evaluate_holdout_county(df)
    print(f" Held-out parish ({HOLDOUT_COUNTY}):    {holdout_metrics.pretty()}")
    in_band = 0.72 <= holdout_metrics.r2 <= 0.76
    print(f" Held-out R^2 in target band [0.72, 0.76]: {'YES' if in_band else 'NO'}")
    print("-" * 68)

    final_model, feature_names = train_final_model(df)
    save_model(final_model, feature_names, holdout_metrics, cv_metrics)

    if make_shap:
        # Imported here to keep training importable without a display backend.
        from src.model import evaluate

        evaluate.generate_shap_plots(holdout_model, x_test)
        print(f" SHAP summary plot -> {settings.shap_summary_path}")
        print(f" SHAP bar plot     -> {settings.shap_bar_path}")

    print(f" Model artifact     -> {settings.model_path}")
    print("=" * 68 + "\n")
    return {"cv": cv_metrics, "holdout": holdout_metrics}


if __name__ == "__main__":
    run_training()
