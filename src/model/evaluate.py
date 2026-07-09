"""Model evaluation utilities: metrics and SHAP explainability plots.

SHAP (SHapley Additive exPlanations) is used to explain the XGBoost model both
globally (which features drive value across the market) and locally (why an
individual parcel is valued the way it is — surfaced in the dashboard).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend: safe for CI and servers.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from src.config import get_logger, settings

logger = get_logger(__name__)


def compute_shap_values(model: XGBRegressor, x: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for a fitted tree model.

    Args:
        model: Fitted XGBoost regressor.
        x: Feature matrix to explain.

    Returns:
        A SHAP :class:`~shap.Explanation` object.
    """
    explainer = shap.TreeExplainer(model)
    return explainer(x)


def generate_shap_plots(
    model: XGBRegressor,
    x: pd.DataFrame,
    summary_path: Path | None = None,
    bar_path: Path | None = None,
    max_display: int = 15,
    sample_size: int = 2000,
) -> tuple[Path, Path]:
    """Generate and save SHAP beeswarm and bar plots.

    Args:
        model: Fitted XGBoost regressor.
        x: Feature matrix to explain (a random sample is used for speed).
        summary_path: Destination for the beeswarm plot.
        bar_path: Destination for the mean-|SHAP| bar plot.
        max_display: Max features to show in each plot.
        sample_size: Cap on rows explained (for plotting performance).

    Returns:
        Tuple of (summary_path, bar_path).
    """
    summary_path = summary_path or settings.shap_summary_path
    bar_path = bar_path or settings.shap_bar_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if len(x) > sample_size:
        x = x.sample(sample_size, random_state=42)
    shap_values = compute_shap_values(model, x)

    plt.figure()
    shap.summary_plot(shap_values, x, max_display=max_display, show=False)
    plt.title("SHAP feature impact on predicted property value")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, x, plot_type="bar", max_display=max_display, show=False)
    plt.title("Mean absolute SHAP value by feature")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP plots: %s, %s", summary_path, bar_path)
    return summary_path, bar_path


def top_shap_features(
    model: XGBRegressor, x: pd.DataFrame, top_n: int = 5
) -> list[list[tuple[str, float]]]:
    """Return the top-N most influential features per row, with signed SHAP values.

    Args:
        model: Fitted XGBoost regressor.
        x: Feature matrix.
        top_n: Number of features to return per row.

    Returns:
        For each row, a list of ``(feature_name, shap_value)`` tuples ordered by
        descending absolute impact.
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    feature_names = list(x.columns)
    results: list[list[tuple[str, float]]] = []
    for row in values:
        order = np.argsort(np.abs(row))[::-1][:top_n]
        results.append([(feature_names[i], float(row[i])) for i in order])
    return results


def global_feature_importance(model: XGBRegressor, x: pd.DataFrame) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value across all rows.

    Args:
        model: Fitted XGBoost regressor.
        x: Feature matrix.

    Returns:
        DataFrame with columns ``feature`` and ``mean_abs_shap``, sorted descending.
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    importance = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": list(x.columns), "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
