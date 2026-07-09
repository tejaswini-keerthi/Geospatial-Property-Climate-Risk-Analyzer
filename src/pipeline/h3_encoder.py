"""H3 hexagonal spatial encoding.

Uber's H3 grid is used at three resolutions:

* **res 5** — coarse cells used as folds for *spatial block* cross-validation so
  that geographically-adjacent parcels never straddle the train/test boundary.
* **res 7** — neighbourhood-scale aggregation for the dashboard choropleth.
* **res 8** — block-scale aggregation for fine-grained hover detail.

All functions operate on a DataFrame carrying ``latitude`` / ``longitude`` columns.
"""

from __future__ import annotations

import h3
import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

#: Resolutions attached to every record.
SPATIAL_CV_RESOLUTION: int = 5
RES7: int = 7
RES8: int = 8

H3_COLUMNS: dict[int, str] = {SPATIAL_CV_RESOLUTION: "h3_res5", RES7: "h3_res7", RES8: "h3_res8"}


def latlon_to_cell(latitude: float, longitude: float, resolution: int) -> str:
    """Convert a single lat/lon pair to an H3 cell index string.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        resolution: H3 resolution (0-15).

    Returns:
        The H3 cell index as a hex string.
    """
    return h3.latlng_to_cell(latitude, longitude, resolution)


def cell_to_latlon(cell: str) -> tuple[float, float]:
    """Return the (lat, lon) centroid of an H3 cell.

    Args:
        cell: H3 cell index string.

    Returns:
        Tuple of (latitude, longitude) for the cell centroid.
    """
    lat, lon = h3.cell_to_latlng(cell)
    return lat, lon


def cell_to_polygon(cell: str) -> list[list[float]]:
    """Return the boundary of an H3 cell as ``[[lon, lat], ...]`` for PyDeck.

    PyDeck's ``PolygonLayer`` expects ``[longitude, latitude]`` vertex order, which
    is the reverse of H3's native ``(lat, lng)`` output.

    Args:
        cell: H3 cell index string.

    Returns:
        A closed ring of ``[lon, lat]`` vertices.
    """
    boundary = h3.cell_to_boundary(cell)
    return [[lon, lat] for lat, lon in boundary]


def encode(
    df: pd.DataFrame, *, lat_col: str = "latitude", lon_col: str = "longitude"
) -> pd.DataFrame:
    """Attach H3 cell columns at resolutions 5, 7, and 8.

    Args:
        df: Input frame with latitude/longitude columns.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.

    Returns:
        A copy of ``df`` with ``h3_res5``, ``h3_res7``, and ``h3_res8`` columns added.

    Raises:
        KeyError: If the latitude/longitude columns are missing.
    """
    for col in (lat_col, lon_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found for H3 encoding.")

    out = df.copy()
    lats = out[lat_col].to_numpy()
    lons = out[lon_col].to_numpy()
    for resolution, col_name in H3_COLUMNS.items():
        out[col_name] = [
            h3.latlng_to_cell(float(lat), float(lon), resolution)
            for lat, lon in zip(lats, lons, strict=True)
        ]
    logger.info(
        "Encoded %d rows into H3 cells (res5=%d, res7=%d, res8=%d unique)",
        len(out),
        out["h3_res5"].nunique(),
        out["h3_res7"].nunique(),
        out["h3_res8"].nunique(),
    )
    return out
