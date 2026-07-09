"""Tests for the ingestion, H3 encoding, and feature-engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.pipeline import features as feat
from src.pipeline import h3_encoder, ingest


@pytest.fixture(scope="module")
def mock_df() -> pd.DataFrame:
    """A small deterministic mock dataset for fast tests."""
    return ingest.generate_mock_dataset(n_records=500, seed=7)


def test_mock_dataset_schema_and_size(mock_df: pd.DataFrame) -> None:
    assert len(mock_df) == 500
    assert list(mock_df.columns) == ingest.SCHEMA_COLUMNS
    assert mock_df["parcel_id"].is_unique


def test_mock_dataset_value_ranges(mock_df: pd.DataFrame) -> None:
    # Flood zones are restricted to the known vocabulary.
    assert set(mock_df["fema_flood_zone"]).issubset(set(ingest.FLOOD_ZONES))
    # Coordinates fall within the configured Louisiana bounding boxes.
    assert mock_df["latitude"].between(29.0, 30.5).all()
    assert mock_df["longitude"].between(-91.5, -89.5).all()
    # Target is strictly positive and clipped to the configured ceiling.
    assert (mock_df["property_value_usd"] > 0).all()
    assert mock_df["property_value_usd"].max() <= 2_500_000


def test_mock_dataset_is_reproducible() -> None:
    a = ingest.generate_mock_dataset(n_records=200, seed=123)
    b = ingest.generate_mock_dataset(n_records=200, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_h3_encoding_adds_all_resolutions(mock_df: pd.DataFrame) -> None:
    encoded = h3_encoder.encode(mock_df)
    for col in h3_encoder.H3_COLUMNS.values():
        assert col in encoded.columns
        assert encoded[col].notna().all()
    # Finer resolution (res8) must have at least as many unique cells as coarser (res5).
    assert encoded["h3_res8"].nunique() >= encoded["h3_res5"].nunique()


def test_h3_encode_missing_columns_raises() -> None:
    with pytest.raises(KeyError):
        h3_encoder.encode(pd.DataFrame({"foo": [1, 2]}))


def test_h3_roundtrip_polygon_is_closed(mock_df: pd.DataFrame) -> None:
    cell = h3_encoder.latlon_to_cell(29.7, -90.7, 7)
    poly = h3_encoder.cell_to_polygon(cell)
    assert len(poly) >= 6
    assert all(len(vertex) == 2 for vertex in poly)


def test_build_features_adds_derived_columns(mock_df: pd.DataFrame) -> None:
    engineered = feat.build_features(mock_df)
    for col in feat.DERIVED_FEATURES:
        assert col in engineered.columns
    assert (engineered["property_age"] >= 0).all()
    assert engineered["is_high_hazard_zone"].isin([0, 1]).all()


def test_feature_matrix_is_numeric_and_stable(mock_df: pd.DataFrame) -> None:
    engineered = feat.build_features(mock_df)
    x, names = feat.get_feature_matrix(engineered)
    assert x.shape[0] == len(mock_df)
    assert all(np.issubdtype(dtype, np.number) for dtype in x.dtypes)
    # One-hot columns for every configured category must be present and stable.
    for col, cats in feat.CATEGORICAL_FEATURES.items():
        for cat in cats:
            assert f"{col}_{cat}" in names


def test_climate_risk_score_bounds_and_band(mock_df: pd.DataFrame) -> None:
    engineered = feat.build_features(mock_df)
    score = feat.compute_climate_risk_score(engineered)
    assert score.between(0, 100).all()
    assert feat.risk_band(10) == "Low"
    assert feat.risk_band(60) == "High"
    assert feat.risk_band(90) == "Severe"
