"""Streamlit entry point for the Geospatial Property Climate Risk Analyzer.

Run with:

    streamlit run src/dashboard/app.py

The app reads the pre-computed ``scored_properties.parquet`` artifact, renders a
full-width 3D PyDeck hex choropleth coloured by composite climate-risk score, and
lets the user drill into any hex cell to see aggregate stats, a representative
parcel's top SHAP drivers, and a Claude-generated plain-language risk narrative.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import settings
from src.dashboard import map_layer
from src.narratives import claude_narrator
from src.pipeline import features as feat

st.set_page_config(
    page_title="Geospatial Property Climate Risk Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
    .stApp { background-color: #0b1120; color: #e5e7eb; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    [data-testid="stMetricValue"] { color: #38bdf8; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #94a3b8; }
    h1, h2, h3 { color: #f1f5f9; }
    .risk-narrative {
        background: #111827; border-left: 4px solid #38bdf8;
        padding: 1rem 1.25rem; border-radius: 6px; font-size: 1.02rem; line-height: 1.55;
    }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading scored property data...")
def load_scored_data() -> pd.DataFrame:
    """Load the pre-computed scored dataset (cached across reruns).

    Returns:
        The scored property DataFrame.

    Raises:
        FileNotFoundError: If the scored artifact is missing.
    """
    path = settings.scored_properties_path
    if not path.exists():
        raise FileNotFoundError(
            f"Scored dataset not found at {path}. Run `python -m src.model.predict` first."
        )
    return pd.read_parquet(path)


@st.cache_resource
def get_narrative_cache() -> claude_narrator.NarrativeCache:
    """Return a process-wide narrative cache."""
    return claude_narrator.NarrativeCache()


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered frame.

    Args:
        df: The full scored dataset.

    Returns:
        The filtered DataFrame.
    """
    st.sidebar.header("Filters")

    counties = sorted(df["county"].unique())
    selected_counties = st.sidebar.multiselect("Parish", counties, default=counties)

    zones = ["X", "A", "AE", "VE"]
    selected_zones = st.sidebar.multiselect("FEMA flood zone", zones, default=zones)

    value_min = int(df["predicted_value_usd"].min())
    value_max = int(df["predicted_value_usd"].max())
    low, high = st.sidebar.slider(
        "Predicted value (USD)",
        min_value=value_min,
        max_value=value_max,
        value=(value_min, value_max),
        step=5000,
        format="$%d",
    )

    resolution = st.sidebar.radio(
        "Hex resolution", options=[7, 8], index=0, help="H3 aggregation level (7 = coarser)."
    )
    st.session_state["resolution"] = resolution

    mask = (
        df["county"].isin(selected_counties)
        & df["fema_flood_zone"].isin(selected_zones)
        & df["predicted_value_usd"].between(low, high)
    )
    filtered = df[mask]
    st.sidebar.markdown(f"**{len(filtered):,}** parcels match filters.")
    return filtered


def render_metrics(df: pd.DataFrame) -> None:
    """Render the top-of-page aggregate metric row.

    Args:
        df: The filtered scored dataset.
    """
    total = len(df)
    avg_risk = df["climate_risk_score"].mean() if total else 0.0
    pct_high = (df["is_high_hazard_zone"].mean() * 100) if total else 0.0
    avg_value = df["predicted_value_usd"].mean() if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total parcels", f"{total:,}")
    c2.metric("Avg risk score", f"{avg_risk:.1f}/100")
    c3.metric("% in high-hazard zones", f"{pct_high:.1f}%")
    c4.metric("Avg predicted value", f"${avg_value:,.0f}")


def render_map(df: pd.DataFrame, resolution: int) -> str | None:
    """Render the 3D hex map and return the clicked hex cell id, if any.

    Args:
        df: The filtered scored dataset.
        resolution: H3 resolution to aggregate at.

    Returns:
        The selected ``h3_cell`` id, or None.
    """
    agg = map_layer.aggregate_hexes(df, resolution=resolution)
    deck = map_layer.build_deck(agg)
    event = st.pydeck_chart(
        deck,
        use_container_width=True,
        height=560,
        on_select="rerun",
        selection_mode="single-object",
    )
    try:
        objects = event.selection.get("objects", {})  # type: ignore[union-attr]
        for _layer, rows in objects.items():
            if rows:
                return rows[0].get("h3_cell")
    except (AttributeError, KeyError, TypeError):
        return None
    return None


def render_detail(df: pd.DataFrame, hex_cell: str, resolution: int) -> None:
    """Render the drill-down panel for a selected hex cell.

    Args:
        df: The filtered scored dataset.
        hex_cell: The selected H3 cell id.
        resolution: The active H3 resolution.
    """
    col = f"h3_res{resolution}"
    cell_df = df[df[col] == hex_cell]
    if cell_df.empty:
        st.info("No parcels in the selected cell for the current filters.")
        return

    st.subheader(f"Hex cell {hex_cell}  ·  {len(cell_df):,} parcels")
    a, b, c = st.columns(3)
    a.metric("Avg risk score", f"{cell_df['climate_risk_score'].mean():.1f}/100")
    b.metric("Avg predicted value", f"${cell_df['predicted_value_usd'].mean():,.0f}")
    c.metric("Dominant flood zone", cell_df["fema_flood_zone"].mode().iat[0])

    # Representative parcel = highest risk in the cell.
    parcel = cell_df.sort_values("climate_risk_score", ascending=False).iloc[0]
    ctx = claude_narrator.context_from_row(parcel.to_dict())

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Representative parcel")
        st.write(
            {
                "Parcel ID": ctx.parcel_id,
                "Parish": ctx.county,
                "Flood zone": ctx.fema_flood_zone,
                "Elevation (ft)": round(ctx.elevation_ft, 1),
                "Distance to water (mi)": round(ctx.distance_to_water_miles, 2),
                "Risk score": f"{ctx.climate_risk_score:.0f}/100 ({ctx.risk_band})",
                "Predicted value": f"${ctx.predicted_value_usd:,.0f}",
                "Square footage": int(parcel["square_footage"]),
                "Year built": int(parcel["year_built"]),
            }
        )
        st.markdown("#### Top 5 SHAP value drivers")
        if ctx.top_features:
            shap_df = pd.DataFrame(ctx.top_features, columns=["Feature", "SHAP impact ($)"])
            st.dataframe(shap_df, hide_index=True, use_container_width=True)
        else:
            st.caption("No SHAP data available for this parcel.")

    with right:
        st.markdown("#### Climate risk narrative")
        with st.spinner("Generating narrative..."):
            narrative = claude_narrator.generate_for_property(ctx, cache=get_narrative_cache())
        st.markdown(f"<div class='risk-narrative'>{narrative}</div>", unsafe_allow_html=True)
        source = "Claude" if settings.anthropic_api_key else "offline template"
        st.caption(f"Narrative source: {source} ({settings.claude_model}).")


def _ensure_risk_band(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill the ``risk_band`` column if an older artifact lacks it.

    Args:
        df: The loaded scored dataset.

    Returns:
        A frame guaranteed to have a ``risk_band`` column.
    """
    if "risk_band" not in df.columns:
        df = df.copy()
        df["risk_band"] = df["climate_risk_score"].apply(feat.risk_band)
    return df


def main() -> None:
    """Compose and render the full dashboard."""
    st.title("🌊 Geospatial Property Climate Risk Analyzer")
    st.caption(
        "XGBoost valuation + SHAP explainability + H3 spatial aggregation + Claude narratives, "
        "over three fictional flood-prone Louisiana parishes."
    )

    try:
        df = load_scored_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    df = _ensure_risk_band(df)
    filtered = sidebar_filters(df)
    render_metrics(filtered)

    if filtered.empty:
        st.warning("No parcels match the current filters. Widen your selection.")
        return

    resolution = st.session_state.get("resolution", 7)
    st.markdown("### Climate-risk hex choropleth")
    st.caption(
        "Colour and height encode mean composite climate-risk score. Click a hex to drill in."
    )
    selected_hex = render_map(filtered, resolution)

    st.divider()
    if selected_hex:
        render_detail(filtered, selected_hex, resolution)
    else:
        st.info("👆 Click a hex cell on the map to see its SHAP drivers and Claude risk narrative.")


main()
