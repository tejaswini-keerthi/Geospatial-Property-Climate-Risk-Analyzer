"""PyDeck 3D hexagon choropleth for the climate-risk dashboard.

Parcels are aggregated into H3 cells (resolution 7 by default) and rendered as an
extruded :class:`pydeck.Layer` where both the fill colour and the extrusion height
encode the mean composite climate-risk score of the cell.
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk

from src.config import get_logger
from src.pipeline import h3_encoder

logger = get_logger(__name__)

#: Colour stops (risk_threshold, [r, g, b]) from low to severe risk.
_COLOR_STOPS: list[tuple[float, list[int]]] = [
    (0.0, [46, 139, 87]),  # sea green
    (25.0, [154, 205, 50]),  # yellow-green
    (50.0, [255, 193, 7]),  # amber
    (75.0, [255, 87, 34]),  # deep orange
    (100.0, [183, 28, 28]),  # dark red
]

DARK_MAP_STYLE: str = "mapbox://styles/mapbox/dark-v11"
DEFAULT_ELEVATION_SCALE: float = 45.0


def risk_to_color(score: float, alpha: int = 200) -> list[int]:
    """Map a 0-100 risk score to an RGBA colour via linear interpolation.

    Args:
        score: Composite climate-risk score.
        alpha: Opacity (0-255).

    Returns:
        ``[r, g, b, a]`` colour list.
    """
    score = max(0.0, min(100.0, score))
    for i in range(len(_COLOR_STOPS) - 1):
        low_t, low_c = _COLOR_STOPS[i]
        high_t, high_c = _COLOR_STOPS[i + 1]
        if low_t <= score <= high_t:
            frac = 0.0 if high_t == low_t else (score - low_t) / (high_t - low_t)
            rgb = [round(low_c[j] + frac * (high_c[j] - low_c[j])) for j in range(3)]
            return [*rgb, alpha]
    return [*_COLOR_STOPS[-1][1], alpha]


def aggregate_hexes(df: pd.DataFrame, resolution: int = 7) -> pd.DataFrame:
    """Aggregate scored parcels into H3 cells for choropleth rendering.

    Args:
        df: Scored property frame with H3 columns and risk/value fields.
        resolution: H3 resolution to aggregate at (7 or 8).

    Returns:
        One row per hex cell with aggregate statistics, a boundary polygon, and colour.
    """
    col = h3_encoder.H3_COLUMNS.get(resolution)
    if col is None or col not in df.columns:
        raise KeyError(f"No H3 column for resolution {resolution}; expected '{col}'.")

    grouped = (
        df.groupby(col)
        .agg(
            parcel_count=("parcel_id", "count"),
            avg_risk_score=("climate_risk_score", "mean"),
            avg_predicted_value=("predicted_value_usd", "mean"),
            avg_elevation_ft=("elevation_ft", "mean"),
            pct_high_hazard=("is_high_hazard_zone", "mean"),
            county=("county", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        )
        .reset_index()
        .rename(columns={col: "h3_cell"})
    )
    grouped["avg_risk_score"] = grouped["avg_risk_score"].round(1)
    grouped["avg_predicted_value"] = grouped["avg_predicted_value"].round(0)
    grouped["avg_elevation_ft"] = grouped["avg_elevation_ft"].round(1)
    grouped["pct_high_hazard"] = (grouped["pct_high_hazard"] * 100).round(1)
    grouped["polygon"] = grouped["h3_cell"].apply(h3_encoder.cell_to_polygon)
    grouped["fill_color"] = grouped["avg_risk_score"].apply(risk_to_color)
    logger.info(
        "Aggregated %d parcels into %d hex cells (res %d)", len(df), len(grouped), resolution
    )
    return grouped


def build_hex_layer(
    agg: pd.DataFrame, elevation_scale: float = DEFAULT_ELEVATION_SCALE
) -> pdk.Layer:
    """Build the extruded PolygonLayer for the hex choropleth.

    Args:
        agg: Aggregated hex frame from :func:`aggregate_hexes`.
        elevation_scale: Multiplier converting risk score to extrusion height.

    Returns:
        A configured :class:`pydeck.Layer`.
    """
    return pdk.Layer(
        "PolygonLayer",
        data=agg,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_elevation="avg_risk_score",
        elevation_scale=elevation_scale,
        extruded=True,
        wireframe=True,
        pickable=True,
        auto_highlight=True,
        get_line_color=[255, 255, 255, 40],
        line_width_min_pixels=1,
    )


def build_view_state(agg: pd.DataFrame) -> pdk.ViewState:
    """Compute an initial camera view centred on the data.

    Args:
        agg: Aggregated hex frame (needs ``polygon`` column).

    Returns:
        A :class:`pydeck.ViewState`.
    """
    lons: list[float] = []
    lats: list[float] = []
    for poly in agg["polygon"]:
        for lon, lat in poly:
            lons.append(lon)
            lats.append(lat)
    center_lon = sum(lons) / len(lons) if lons else -90.5
    center_lat = sum(lats) / len(lats) if lats else 29.7
    return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=9.2, pitch=45, bearing=0)


def build_deck(agg: pd.DataFrame, elevation_scale: float = DEFAULT_ELEVATION_SCALE) -> pdk.Deck:
    """Assemble the full PyDeck deck with a dark theme and tooltip.

    Args:
        agg: Aggregated hex frame from :func:`aggregate_hexes`.
        elevation_scale: Extrusion multiplier.

    Returns:
        A ready-to-render :class:`pydeck.Deck`.
    """
    tooltip = {
        "html": (
            "<b>Risk score:</b> {avg_risk_score}/100<br/>"
            "<b>Parcels:</b> {parcel_count}<br/>"
            "<b>Avg value:</b> ${avg_predicted_value}<br/>"
            "<b>High-hazard %:</b> {pct_high_hazard}%<br/>"
            "<b>Parish:</b> {county}"
        ),
        "style": {"backgroundColor": "#111827", "color": "#F9FAFB", "fontSize": "12px"},
    }
    return pdk.Deck(
        layers=[build_hex_layer(agg, elevation_scale)],
        initial_view_state=build_view_state(agg),
        map_style=DARK_MAP_STYLE,
        tooltip=tooltip,
    )
