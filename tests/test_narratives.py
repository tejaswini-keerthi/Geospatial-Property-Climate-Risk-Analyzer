"""Tests for the Claude narrative generator (offline / cache behaviour)."""

from __future__ import annotations

import json

import pytest
from src.narratives import claude_narrator as cn


def _ctx(**overrides: object) -> cn.PropertyContext:
    base = {
        "parcel_id": "LA-99-000001",
        "county": "Delacroix Parish",
        "fema_flood_zone": "VE",
        "elevation_ft": 3.2,
        "distance_to_water_miles": 0.4,
        "predicted_value_usd": 210000.0,
        "climate_risk_score": 82.0,
        "risk_band": "Severe",
        "top_features": [("storm_surge_risk_score", -21000.0), ("elevation_ft", -8000.0)],
    }
    base.update(overrides)
    return cn.PropertyContext(**base)  # type: ignore[arg-type]


def test_cache_key_is_stable_and_content_sensitive() -> None:
    a = _ctx()
    b = _ctx()
    assert a.cache_key() == b.cache_key()
    c = _ctx(fema_flood_zone="X", climate_risk_score=10.0)
    assert a.cache_key() != c.cache_key()


def test_fallback_narrative_is_three_sentences() -> None:
    text = cn._fallback_narrative(_ctx())
    # Three sentences -> roughly three terminal periods.
    assert text.count(".") >= 3
    assert "Delacroix Parish" in text
    assert len(text) > 80


def test_build_prompt_includes_key_fields() -> None:
    prompt = cn.build_prompt(_ctx())
    assert "flood zone: VE" in prompt
    assert "Delacroix Parish" in prompt
    assert "storm_surge_risk_score" in prompt


def test_generate_for_property_offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Force offline mode regardless of local environment (settings is frozen).
    monkeypatch.setattr(cn, "_make_client", lambda: None)
    cache = cn.NarrativeCache(path=tmp_path / "cache.json")
    text = cn.generate_for_property(_ctx(), cache=cache)
    assert isinstance(text, str) and len(text) > 0
    # Result must be cached and persisted to disk.
    assert len(cache) == 1
    assert (tmp_path / "cache.json").exists()


def test_cache_roundtrip_persists(tmp_path) -> None:
    path = tmp_path / "cache.json"
    cache = cn.NarrativeCache(path=path)
    cache.set("abc", "hello world.")
    cache.flush()
    reloaded = cn.NarrativeCache(path=path)
    assert reloaded.get("abc") == "hello world."


def test_context_from_row_parses_shap_json() -> None:
    row = {
        "parcel_id": "LA-01-000123",
        "county": "Bayou Cane Parish",
        "fema_flood_zone": "AE",
        "elevation_ft": 5.0,
        "distance_to_water_miles": 1.1,
        "predicted_value_usd": 180000.0,
        "climate_risk_score": 55.0,
        "risk_band": "High",
        "top_shap_features": json.dumps(
            [
                {"feature": "square_footage", "shap": 42000.0},
                {"feature": "elevation_ft", "shap": -3000.0},
            ]
        ),
    }
    ctx = cn.context_from_row(row)
    assert ctx.parcel_id == "LA-01-000123"
    assert ctx.top_features[0] == ("square_footage", 42000.0)
