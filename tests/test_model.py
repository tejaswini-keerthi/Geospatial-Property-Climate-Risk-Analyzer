"""Tests for model training, spatial CV, evaluation, and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.model import evaluate, train
from src.pipeline import features as feat
from src.pipeline import h3_encoder, ingest


@pytest.fixture(scope="module")
def prepared_df() -> pd.DataFrame:
    """A modest H3-encoded, feature-engineered dataset for model tests."""
    raw = ingest.generate_mock_dataset(n_records=3000, seed=11)
    encoded = h3_encoder.encode(raw)
    return feat.build_features(encoded)


def test_score_metrics_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = train._score(y, y)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.n == 4


def test_spatial_block_cv_uses_groups_and_learns(prepared_df: pd.DataFrame) -> None:
    # Use a light-weight parameter set for test speed.
    params = {**train.XGB_PARAMS, "n_estimators": 60, "max_depth": 4}
    metrics = train.spatial_block_cv(prepared_df, n_splits=3, params=params)
    # The model should learn a real signal well above chance (R^2 > 0).
    assert metrics.r2 > 0.4
    assert metrics.n == len(prepared_df)


def test_holdout_county_split_is_disjoint(prepared_df: pd.DataFrame) -> None:
    params = {**train.XGB_PARAMS, "n_estimators": 60, "max_depth": 4}
    metrics, model, x_test, _y_test = train.evaluate_holdout_county(
        prepared_df, holdout=train.HOLDOUT_COUNTY, params=params
    )
    # Held-out parish must not appear in training, and predictions must be finite.
    assert len(x_test) == (prepared_df["county"] == train.HOLDOUT_COUNTY).sum()
    preds = model.predict(x_test)
    assert np.isfinite(preds).all()
    assert metrics.r2 > 0.4


def test_train_final_model_feature_order(prepared_df: pd.DataFrame) -> None:
    params = {**train.XGB_PARAMS, "n_estimators": 40, "max_depth": 4}
    model, names = train.train_final_model(prepared_df, params=params)
    _, expected_names = feat.get_feature_matrix(prepared_df)
    assert names == expected_names
    assert model.n_features_in_ == len(names)


def test_shap_top_features_shape(prepared_df: pd.DataFrame) -> None:
    params = {**train.XGB_PARAMS, "n_estimators": 40, "max_depth": 4}
    model, _ = train.train_final_model(prepared_df, params=params)
    x, _ = feat.get_feature_matrix(prepared_df)
    top = evaluate.top_shap_features(model, x.head(20), top_n=5)
    assert len(top) == 20
    assert all(len(row) == 5 for row in top)
    # Each entry is a (feature_name, float_impact) pair.
    name, impact = top[0][0]
    assert isinstance(name, str)
    assert isinstance(impact, float)


def test_global_feature_importance_is_ranked(prepared_df: pd.DataFrame) -> None:
    params = {**train.XGB_PARAMS, "n_estimators": 40, "max_depth": 4}
    model, _ = train.train_final_model(prepared_df, params=params)
    x, _ = feat.get_feature_matrix(prepared_df)
    imp = evaluate.global_feature_importance(model, x.head(200))
    assert list(imp.columns) == ["feature", "mean_abs_shap"]
    assert imp["mean_abs_shap"].is_monotonic_decreasing
