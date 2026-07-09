"""Centralized configuration loaded from environment variables.

All modules import from here so that credentials and tunable paths live in exactly
one place. Values are read from a `.env` file (via python-dotenv) with sensible
defaults so the project runs out-of-the-box in mock mode.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repository root = two levels up from this file (src/config.py -> src -> root).
ROOT_DIR: Path = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable in a forgiving way."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    path = Path(path_str)
    return path if path.is_absolute() else (ROOT_DIR / path)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings sourced from the environment."""

    mock_data: bool = field(default_factory=lambda: _env_bool("MOCK_DATA", True))
    mock_data_path: Path = field(
        default_factory=lambda: _resolve(os.getenv("MOCK_DATA_PATH", "data/mock/properties.csv"))
    )
    processed_dir: Path = field(
        default_factory=lambda: _resolve(os.getenv("PROCESSED_DIR", "data/processed"))
    )
    model_path: Path = field(
        default_factory=lambda: _resolve(os.getenv("MODEL_PATH", "data/processed/model.json"))
    )

    # Claude / Anthropic
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    )
    claude_max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("CLAUDE_MAX_CONCURRENCY", "5"))
    )

    # Real-provider credentials (only used when mock_data is False)
    noaa_api_token: str | None = field(default_factory=lambda: os.getenv("NOAA_API_TOKEN"))
    fema_nfhl_base_url: str = field(
        default_factory=lambda: os.getenv(
            "FEMA_NFHL_BASE_URL", "https://hazards.fema.gov/gis/nfhl/rest/services"
        )
    )
    zillow_api_key: str | None = field(default_factory=lambda: os.getenv("ZILLOW_API_KEY"))
    census_api_key: str | None = field(default_factory=lambda: os.getenv("CENSUS_API_KEY"))

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def shap_summary_path(self) -> Path:
        """Path to the SHAP beeswarm summary plot."""
        return self.processed_dir / "shap_summary.png"

    @property
    def shap_bar_path(self) -> Path:
        """Path to the SHAP mean-absolute bar plot."""
        return self.processed_dir / "shap_bar.png"

    @property
    def narrative_cache_path(self) -> Path:
        """Path to the on-disk Claude narrative cache."""
        return self.processed_dir / "narrative_cache.json"

    @property
    def scored_properties_path(self) -> Path:
        """Path to the model-scored, feature-engineered property table."""
        return self.processed_dir / "scored_properties.parquet"


settings = Settings()


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once, using the level from settings by default.

    Args:
        level: Optional log level name overriding the configured default.
    """
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
