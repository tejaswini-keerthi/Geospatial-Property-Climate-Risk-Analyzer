"""Geospatial Property Climate Risk Analyzer.

A production-grade, end-to-end pipeline that ingests property + climate data,
encodes it into H3 hexagons, trains an XGBoost model with spatial cross-validation,
explains predictions with SHAP, generates plain-language risk narratives with the
Claude API, and serves everything through an interactive Streamlit + PyDeck dashboard.
"""

__version__ = "0.1.0"
