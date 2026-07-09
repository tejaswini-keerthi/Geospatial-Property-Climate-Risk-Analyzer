"""Data ingestion for the climate-risk analyzer.

By default (``MOCK_DATA=true``) this module loads a synthetic but realistic dataset
of property + climate records for three fictional flood-prone parishes in coastal
Louisiana. When ``MOCK_DATA=false`` the same public API surface calls out to real
providers (NOAA, FEMA NFHL, Zillow, US Census). The real-provider functions are
implemented as faithful, documented stubs that show exactly how each request would
be constructed; they raise if the corresponding credential is missing so that a
misconfigured live run fails loudly rather than silently returning mock data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import get_logger, settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: Canonical column order for the property dataset.
SCHEMA_COLUMNS: list[str] = [
    "parcel_id",
    "county",
    "latitude",
    "longitude",
    "elevation_ft",
    "fema_flood_zone",
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
    "foundation_type",
    "census_median_income",
    "storm_surge_risk_score",
    "annual_rainfall_inches",
    "hurricane_history_count",
    "drainage_quality_score",
    "impervious_surface_pct",
    "insurance_premium_usd",
    "has_flood_insurance",
    "property_value_usd",
]

#: FEMA flood-zone codes present in the dataset (ordered low -> high hazard).
FLOOD_ZONES: list[str] = ["X", "A", "AE", "VE"]

#: Three fictional Louisiana parishes with plausible coastal bounding boxes.
COUNTIES: dict[str, dict[str, float]] = {
    "Bayou Cane Parish": {
        "lat_min": 29.55,
        "lat_max": 29.80,
        "lon_min": -90.90,
        "lon_max": -90.60,
        "base_value": 15000.0,
        "income_center": 52000.0,
    },
    "Cypress Hollow Parish": {
        "lat_min": 30.05,
        "lat_max": 30.35,
        "lon_min": -91.30,
        "lon_max": -90.95,
        "base_value": 42000.0,
        "income_center": 61000.0,
    },
    "Delacroix Parish": {
        "lat_min": 29.20,
        "lat_max": 29.55,
        "lon_min": -89.95,
        "lon_max": -89.60,
        "base_value": 4000.0,
        "income_center": 46000.0,
    },
}

#: Flood-zone additive effect on property value (USD).
_ZONE_VALUE_EFFECT: dict[str, float] = {"X": 0.0, "A": -14000.0, "AE": -24000.0, "VE": -46000.0}

#: Foundation-type additive effect on property value (USD).
_FOUNDATION_VALUE_EFFECT: dict[str, float] = {
    "slab": 0.0,
    "pier_and_beam": 9000.0,
    "raised": 18000.0,
}

DEFAULT_N_RECORDS: int = 18_000
RANDOM_SEED: int = 42
_CURRENT_YEAR: int = 2026


# ---------------------------------------------------------------------------
# Mock-data generation
# ---------------------------------------------------------------------------


def _assign_flood_zones(
    elevation: np.ndarray, dist_water: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Assign FEMA flood zones as a stochastic function of elevation and water distance.

    Lower, closer-to-water parcels are far more likely to fall in high-hazard zones.

    Args:
        elevation: Per-parcel elevation in feet.
        dist_water: Per-parcel distance to nearest water body in miles.
        rng: Seeded NumPy random generator.

    Returns:
        Array of flood-zone code strings.
    """
    # Hazard score in [0, 1]: high when elevation is low and water is close.
    hazard = (
        np.clip(1.0 - (elevation / 25.0), 0.0, 1.0) * 0.6
        + np.clip(1.0 - (dist_water / 5.0), 0.0, 1.0) * 0.4
    )
    zones = np.empty(elevation.shape[0], dtype=object)
    draw = rng.random(elevation.shape[0])
    for i in range(elevation.shape[0]):
        h = hazard[i]
        # Probability weights shift toward hazardous zones as h -> 1.
        p_ve = 0.02 + 0.35 * h**2
        p_ae = 0.05 + 0.40 * h
        p_a = 0.08 + 0.20 * h
        p_x = max(1.0 - (p_ve + p_ae + p_a), 0.01)
        total = p_x + p_a + p_ae + p_ve
        cum = np.cumsum([p_x, p_a, p_ae, p_ve]) / total
        d = draw[i]
        if d < cum[0]:
            zones[i] = "X"
        elif d < cum[1]:
            zones[i] = "A"
        elif d < cum[2]:
            zones[i] = "AE"
        else:
            zones[i] = "VE"
    return zones


def generate_mock_dataset(
    n_records: int = DEFAULT_N_RECORDS, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate a synthetic, internally-consistent property + climate dataset.

    The target (``property_value_usd``) is a deterministic function of the features
    plus heteroscedastic noise, so that a model can learn a genuine — but not
    perfect — signal that generalizes across parishes.

    Args:
        n_records: Number of property records to generate.
        seed: Random seed for full reproducibility.

    Returns:
        A DataFrame with columns in :data:`SCHEMA_COLUMNS` order.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d mock property records across %d parishes", n_records, len(COUNTIES))

    county_names = list(COUNTIES.keys())
    # Roughly balanced parishes with slight imbalance for realism.
    weights = np.array([0.38, 0.34, 0.28])
    county_idx = rng.choice(len(county_names), size=n_records, p=weights)
    county = np.array(county_names)[county_idx]

    lat = np.empty(n_records)
    lon = np.empty(n_records)
    base_value = np.empty(n_records)
    income_center = np.empty(n_records)
    for i, name in enumerate(county_names):
        mask = county_idx == i
        c = COUNTIES[name]
        lat[mask] = rng.uniform(c["lat_min"], c["lat_max"], mask.sum())
        lon[mask] = rng.uniform(c["lon_min"], c["lon_max"], mask.sum())
        base_value[mask] = c["base_value"]
        income_center[mask] = c["income_center"]

    # Physical / geographic features.
    elevation_ft = np.clip(rng.gamma(shape=2.2, scale=3.2, size=n_records), 0.2, 45.0)
    distance_to_water_miles = np.clip(rng.gamma(shape=1.8, scale=1.1, size=n_records), 0.02, 12.0)
    distance_to_coast_miles = np.clip(
        distance_to_water_miles + rng.gamma(1.5, 3.5, n_records), 0.1, 40.0
    )
    fema_flood_zone = _assign_flood_zones(elevation_ft, distance_to_water_miles, rng)
    base_flood_elevation_ft = np.clip(
        elevation_ft + rng.normal(1.5, 1.2, n_records), 0.0, 40.0
    ).round(1)

    # Structure features.
    square_footage = np.clip(rng.normal(1950, 620, n_records), 620, 6500).round().astype(int)
    lot_size_acres = np.clip(rng.gamma(2.0, 0.18, n_records), 0.05, 4.0).round(3)
    num_bedrooms = np.clip(
        np.round(square_footage / 620 + rng.normal(0, 0.6, n_records)), 1, 7
    ).astype(int)
    num_bathrooms = np.clip(
        np.round(num_bedrooms * 0.7 + rng.normal(0, 0.4, n_records)), 1, 6
    ).astype(int)
    stories = rng.choice([1, 2, 3], size=n_records, p=[0.62, 0.33, 0.05])
    year_built = np.clip(rng.normal(1988, 22, n_records), 1920, _CURRENT_YEAR).round().astype(int)
    roof_age_years = (
        np.clip(np.minimum(_CURRENT_YEAR - year_built, rng.gamma(2.5, 5.0, n_records)), 0, 60)
        .round()
        .astype(int)
    )
    foundation_type = rng.choice(
        ["slab", "pier_and_beam", "raised"], size=n_records, p=[0.55, 0.30, 0.15]
    )

    # Socioeconomic + climate features.
    census_median_income = (
        np.clip(rng.normal(income_center, 12000, n_records), 22000, 140000).round().astype(int)
    )
    storm_surge_risk_score = np.clip(
        70 * np.clip(1 - elevation_ft / 25.0, 0, 1)
        + 30 * np.clip(1 - distance_to_coast_miles / 40.0, 0, 1)
        + rng.normal(0, 6, n_records),
        0,
        100,
    ).round(1)
    annual_rainfall_inches = np.clip(rng.normal(64, 6, n_records), 45, 88).round(1)
    hurricane_history_count = rng.poisson(
        2.0 + 2.5 * np.clip(1 - distance_to_coast_miles / 40.0, 0, 1), n_records
    ).astype(int)
    drainage_quality_score = np.clip(
        60 + elevation_ft * 1.2 - storm_surge_risk_score * 0.3 + rng.normal(0, 8, n_records), 0, 100
    ).round(1)
    impervious_surface_pct = np.clip(rng.normal(42, 15, n_records), 5, 92).round(1)

    # Insurance features derived from hazard.
    insurance_premium_usd = np.clip(
        700
        + storm_surge_risk_score * 22
        + np.isin(fema_flood_zone, ["AE", "VE"]) * 1400
        + rng.normal(0, 250, n_records),
        350,
        12000,
    ).round(2)
    has_flood_insurance = (
        rng.random(n_records)
        < np.clip(0.35 + 0.45 * np.isin(fema_flood_zone, ["AE", "VE", "A"]), 0, 0.95)
    ).astype(int)

    # ------------------------------------------------------------------
    # Target: property_value_usd
    # ------------------------------------------------------------------
    zone_effect = np.array([_ZONE_VALUE_EFFECT[z] for z in fema_flood_zone])
    foundation_effect = np.array([_FOUNDATION_VALUE_EFFECT[f] for f in foundation_type])

    value = (
        88.0 * square_footage
        + 1.55 * (census_median_income - 45000)
        + 820.0 * elevation_ft
        + 3800.0 * distance_to_water_miles
        + 9000.0 * num_bedrooms
        + 12500.0 * num_bathrooms
        + 26000.0 * lot_size_acres
        - 480.0 * (_CURRENT_YEAR - year_built)
        - 650.0 * roof_age_years
        - 300.0 * storm_surge_risk_score
        + 210.0 * drainage_quality_score
        + zone_effect
        + foundation_effect
        + base_value
    )
    # Heteroscedastic multiplicative + additive noise keeps held-out R^2 realistic.
    value *= rng.normal(1.0, 0.155, n_records)
    value += rng.normal(0, 16000, n_records)
    property_value_usd = np.clip(value, 45000, 2_500_000).round(2)

    parcel_id = np.array([f"LA-{seed:02d}-{i:06d}" for i in range(n_records)])

    frame = pd.DataFrame(
        {
            "parcel_id": parcel_id,
            "county": county,
            "latitude": lat.round(6),
            "longitude": lon.round(6),
            "elevation_ft": elevation_ft.round(2),
            "fema_flood_zone": fema_flood_zone,
            "base_flood_elevation_ft": base_flood_elevation_ft,
            "distance_to_water_miles": distance_to_water_miles.round(3),
            "distance_to_coast_miles": distance_to_coast_miles.round(3),
            "square_footage": square_footage,
            "lot_size_acres": lot_size_acres,
            "num_bedrooms": num_bedrooms,
            "num_bathrooms": num_bathrooms,
            "stories": stories,
            "year_built": year_built,
            "roof_age_years": roof_age_years,
            "foundation_type": foundation_type,
            "census_median_income": census_median_income,
            "storm_surge_risk_score": storm_surge_risk_score,
            "annual_rainfall_inches": annual_rainfall_inches,
            "hurricane_history_count": hurricane_history_count,
            "drainage_quality_score": drainage_quality_score,
            "impervious_surface_pct": impervious_surface_pct,
            "insurance_premium_usd": insurance_premium_usd,
            "has_flood_insurance": has_flood_insurance,
            "property_value_usd": property_value_usd,
        }
    )
    return frame[SCHEMA_COLUMNS]


def save_mock_dataset(path: Path | None = None, n_records: int = DEFAULT_N_RECORDS) -> Path:
    """Generate and persist the mock dataset to CSV.

    Args:
        path: Destination path; defaults to :attr:`Settings.mock_data_path`.
        n_records: Number of records to generate.

    Returns:
        The path the dataset was written to.
    """
    path = path or settings.mock_data_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_mock_dataset(n_records=n_records)
    frame.to_csv(path, index=False)
    logger.info("Wrote %d records to %s", len(frame), path)
    return path


# ---------------------------------------------------------------------------
# Real-provider ingestion (stubs that show real request construction)
# ---------------------------------------------------------------------------


def fetch_noaa_climate(latitude: float, longitude: float, *, timeout: int = 30) -> dict:
    """Fetch climate normals (rainfall, storm history) from NOAA Climate Data Online.

    Args:
        latitude: Property latitude in decimal degrees.
        longitude: Property longitude in decimal degrees.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON payload from NOAA CDO.

    Raises:
        RuntimeError: If ``NOAA_API_TOKEN`` is not configured.
    """
    if not settings.noaa_api_token:
        raise RuntimeError("NOAA_API_TOKEN is required for live NOAA ingestion.")
    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    params = {
        "datasetid": "NORMAL_ANN",
        "datatypeid": "ANN-PRCP-NORMAL",
        "extent": f"{latitude - 0.05},{longitude - 0.05},{latitude + 0.05},{longitude + 0.05}",
        "units": "standard",
        "limit": 1000,
    }
    headers = {"token": settings.noaa_api_token}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_fema_flood_zone(latitude: float, longitude: float, *, timeout: int = 30) -> dict:
    """Query the FEMA National Flood Hazard Layer (NFHL) for a point's flood zone.

    Args:
        latitude: Property latitude in decimal degrees.
        longitude: Property longitude in decimal degrees.
        timeout: Request timeout in seconds.

    Returns:
        Parsed Esri FeatureService JSON containing the flood-zone attributes.
    """
    url = f"{settings.fema_nfhl_base_url}/public/NFHL/MapServer/28/query"
    params = {
        "geometry": f"{longitude},{latitude}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FLD_ZONE,STATIC_BFE,ZONE_SUBTY",
        "returnGeometry": "false",
        "f": "json",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_zillow_valuation(address: str, *, timeout: int = 30) -> dict:
    """Fetch a Zestimate-style valuation from the Zillow/Bridge Interactive API.

    Args:
        address: Full street address to resolve.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON payload with valuation fields.

    Raises:
        RuntimeError: If ``ZILLOW_API_KEY`` is not configured.
    """
    if not settings.zillow_api_key:
        raise RuntimeError("ZILLOW_API_KEY is required for live Zillow ingestion.")
    url = "https://api.bridgedataoutput.com/api/v2/zestimates"
    params = {"access_token": settings.zillow_api_key, "address": address}
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_census_income(state_fips: str, county_fips: str, *, timeout: int = 30) -> dict:
    """Fetch median household income from the US Census ACS 5-year API.

    Args:
        state_fips: Two-digit state FIPS code (e.g. ``"22"`` for Louisiana).
        county_fips: Three-digit county FIPS code.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON payload with ACS variable ``B19013_001E`` (median income).
    """
    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "NAME,B19013_001E",
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    if settings.census_api_key:
        params["key"] = settings.census_api_key
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _ingest_from_live_apis() -> pd.DataFrame:
    """Assemble a property dataset from live provider APIs.

    This orchestrator shows the intended join order: FEMA (hazard) + NOAA (climate)
    + Census (socioeconomic) + Zillow (valuation). It is intentionally not exercised
    in mock mode; wiring a real parcel source is deployment-specific.

    Raises:
        NotImplementedError: Always, until a concrete parcel source is supplied.
    """
    raise NotImplementedError(
        "Live ingestion requires a parcel source (county assessor extract). "
        "Set MOCK_DATA=true to use the bundled synthetic dataset, or implement the "
        "parcel loader and combine it with fetch_fema_flood_zone / fetch_noaa_climate / "
        "fetch_census_income / fetch_zillow_valuation."
    )


def load_data(force_regenerate: bool = False) -> pd.DataFrame:
    """Load the property dataset, honoring the ``MOCK_DATA`` toggle.

    In mock mode the bundled CSV is read (and generated on first use). In live mode
    the provider APIs are orchestrated via :func:`_ingest_from_live_apis`.

    Args:
        force_regenerate: If True (mock mode only), regenerate the CSV from scratch.

    Returns:
        The property dataset as a DataFrame.
    """
    if not settings.mock_data:
        logger.info("MOCK_DATA=false -> ingesting from live provider APIs")
        return _ingest_from_live_apis()

    path = settings.mock_data_path
    if force_regenerate or not path.exists():
        logger.info("Mock dataset missing or regeneration forced; generating at %s", path)
        save_mock_dataset(path)
    logger.info("Loading mock dataset from %s", path)
    return pd.read_csv(path)


if __name__ == "__main__":
    out = save_mock_dataset()
    df = pd.read_csv(out)
    logger.info("Generated dataset shape: %s", df.shape)
    logger.info("Parishes: %s", df["county"].value_counts().to_dict())
    logger.info("Median property value: $%s", f"{df['property_value_usd'].median():,.0f}")
