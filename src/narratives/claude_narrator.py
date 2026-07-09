"""Generate plain-language climate-risk narratives with the Claude API.

Design goals:

* **Homebuyer-friendly** — three short sentences, no jargon.
* **Grounded** — the prompt is built only from the property's real attributes and
  the model's top SHAP drivers, so the narrative reflects the data.
* **Efficient** — requests run concurrently (bounded semaphore) and results are
  cached on disk keyed by a content hash, so re-runs cost nothing.
* **Offline-safe** — with no ``ANTHROPIC_API_KEY`` the module returns a deterministic
  templated narrative, so the dashboard and tests work without network access.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_logger, settings

logger = get_logger(__name__)

MAX_TOKENS: int = 300
SYSTEM_PROMPT: str = (
    "You are a residential climate-risk advisor. You explain a property's flood and "
    "storm exposure to a prospective homebuyer in plain, reassuring-but-honest language. "
    "Always respond with exactly three sentences and no preamble, bullet points, or headings."
)


@dataclass
class PropertyContext:
    """Salient inputs for a single property's risk narrative."""

    parcel_id: str
    county: str
    fema_flood_zone: str
    elevation_ft: float
    distance_to_water_miles: float
    predicted_value_usd: float
    climate_risk_score: float
    risk_band: str
    top_features: list[tuple[str, float]] = field(default_factory=list)

    def cache_key(self) -> str:
        """Return a stable content hash used as the cache key."""
        payload = json.dumps(
            {
                "zone": self.fema_flood_zone,
                "elev": round(self.elevation_ft, 1),
                "water": round(self.distance_to_water_miles, 2),
                "value": round(self.predicted_value_usd, -3),
                "risk": round(self.climate_risk_score, 0),
                "feats": [f for f, _ in self.top_features[:5]],
                "model": settings.claude_model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_prompt(ctx: PropertyContext) -> str:
    """Construct the user prompt for a single property.

    Args:
        ctx: The property context.

    Returns:
        A fully-rendered prompt string.
    """
    feature_lines = ", ".join(
        f"{name} (impact {shap:+,.0f})" for name, shap in ctx.top_features[:5]
    )
    return (
        f"Property in {ctx.county}, Louisiana.\n"
        f"FEMA flood zone: {ctx.fema_flood_zone}.\n"
        f"Ground elevation: {ctx.elevation_ft:.1f} ft.\n"
        f"Distance to nearest water: {ctx.distance_to_water_miles:.2f} miles.\n"
        f"Estimated value: ${ctx.predicted_value_usd:,.0f}.\n"
        f"Composite climate-risk score: {ctx.climate_risk_score:.0f}/100 ({ctx.risk_band}).\n"
        f"Top value drivers (SHAP): {feature_lines or 'n/a'}.\n\n"
        "Write a three-sentence climate-risk summary a homebuyer would understand: "
        "what the flood/storm exposure is, what it means practically (e.g. insurance, "
        "mitigation), and an honest overall takeaway."
    )


def _fallback_narrative(ctx: PropertyContext) -> str:
    """Return a deterministic templated narrative (used when no API key is set).

    Args:
        ctx: The property context.

    Returns:
        A three-sentence narrative string.
    """
    zone_desc = {
        "X": "a minimal-hazard flood zone",
        "A": "a high-risk flood zone without detailed base-flood elevations",
        "AE": "a high-risk flood zone with mapped base-flood elevations",
        "VE": "a high-risk coastal zone exposed to storm-driven waves",
    }.get(ctx.fema_flood_zone, "an unclassified flood zone")
    high_risk = ctx.climate_risk_score >= 50
    insurance_advice = "strongly advised" if high_risk else "worth budgeting for"
    takeaway = (
        "a meaningful climate exposure to weigh carefully"
        if high_risk
        else "a manageable but real climate consideration"
    )
    return (
        f"This {ctx.county} property sits in {zone_desc} at about "
        f"{ctx.elevation_ft:.0f} ft elevation, roughly {ctx.distance_to_water_miles:.1f} miles "
        f"from open water. Its composite climate-risk score is {ctx.climate_risk_score:.0f}/100 "
        f"({ctx.risk_band.lower()} risk), so flood insurance and drainage upgrades are "
        f"{insurance_advice}. "
        f"Overall, at an estimated ${ctx.predicted_value_usd:,.0f} the location is {takeaway}."
    )


class NarrativeCache:
    """Simple JSON-file cache of narratives keyed by content hash."""

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the cache, loading any existing entries from disk.

        Args:
            path: Cache file path; defaults to settings.
        """
        self.path = path or settings.narrative_cache_path
        self._data: dict[str, str] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Corrupt narrative cache at %s; starting fresh", self.path)

    def get(self, key: str) -> str | None:
        """Return a cached narrative, or None if absent."""
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        """Store a narrative in memory (call :meth:`flush` to persist)."""
        self._data[key] = value

    def flush(self) -> None:
        """Persist the in-memory cache to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        """Return the number of cached narratives."""
        return len(self._data)


async def _generate_one(
    client: Any,
    semaphore: asyncio.Semaphore,
    ctx: PropertyContext,
    cache: NarrativeCache,
) -> str:
    """Generate (or fetch from cache) a single narrative.

    Args:
        client: An ``AsyncAnthropic`` client, or None for offline mode.
        semaphore: Concurrency limiter.
        ctx: The property context.
        cache: The narrative cache.

    Returns:
        The narrative text.
    """
    key = ctx.cache_key()
    cached = cache.get(key)
    if cached is not None:
        return cached

    if client is None:
        text = _fallback_narrative(ctx)
        cache.set(key, text)
        return text

    async with semaphore:
        try:
            response = await client.messages.create(
                model=settings.claude_model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(ctx)}],
            )
            text = response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Claude call failed for %s (%s); using fallback", ctx.parcel_id, exc)
            text = _fallback_narrative(ctx)

    cache.set(key, text)
    return text


async def generate_batch(
    contexts: list[PropertyContext], cache: NarrativeCache | None = None
) -> list[str]:
    """Generate narratives for many properties concurrently.

    Args:
        contexts: Property contexts to summarize.
        cache: Optional shared cache; a new one is created if omitted.

    Returns:
        Narratives in the same order as ``contexts``.
    """
    if cache is None:
        cache = NarrativeCache()
    client = _make_client()
    semaphore = asyncio.Semaphore(settings.claude_max_concurrency)
    tasks = [_generate_one(client, semaphore, ctx, cache) for ctx in contexts]
    results = await asyncio.gather(*tasks)
    cache.flush()
    if client is not None:
        await client.close()
    logger.info("Generated %d narratives (cache size now %d)", len(results), len(cache))
    return list(results)


def _make_client() -> Any:
    """Create an AsyncAnthropic client, or None when offline.

    Returns:
        An ``AsyncAnthropic`` instance, or None if the SDK/key is unavailable.
    """
    if not settings.anthropic_api_key:
        logger.info("ANTHROPIC_API_KEY not set; narratives will use offline templates.")
        return None
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        logger.warning("anthropic SDK not installed; using offline templates.")
        return None
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


def generate_for_property(ctx: PropertyContext, cache: NarrativeCache | None = None) -> str:
    """Synchronous convenience wrapper for a single property.

    Args:
        ctx: The property context.
        cache: Optional shared cache.

    Returns:
        The narrative text.
    """
    return asyncio.run(generate_batch([ctx], cache=cache))[0]


def context_from_row(row: dict[str, Any]) -> PropertyContext:
    """Build a :class:`PropertyContext` from a scored-dataset row.

    Args:
        row: A mapping with the scored-dataset columns.

    Returns:
        A populated :class:`PropertyContext`.
    """
    raw_feats = row.get("top_shap_features")
    top_features: list[tuple[str, float]] = []
    if isinstance(raw_feats, str) and raw_feats:
        try:
            top_features = [(d["feature"], float(d["shap"])) for d in json.loads(raw_feats)]
        except (json.JSONDecodeError, KeyError, TypeError):
            top_features = []
    return PropertyContext(
        parcel_id=str(row.get("parcel_id", "unknown")),
        county=str(row.get("county", "")),
        fema_flood_zone=str(row.get("fema_flood_zone", "X")),
        elevation_ft=float(row.get("elevation_ft", 0.0)),
        distance_to_water_miles=float(row.get("distance_to_water_miles", 0.0)),
        predicted_value_usd=float(
            row.get("predicted_value_usd", row.get("property_value_usd", 0.0))
        ),
        climate_risk_score=float(row.get("climate_risk_score", 0.0)),
        risk_band=str(row.get("risk_band", "Moderate")),
        top_features=top_features,
    )
