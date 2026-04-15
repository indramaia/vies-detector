"""
aggregation
───────────
Camada 3 — Agregação de classificações sentence-level em BiasScore por
artigo e índice editorial por veículo.
"""

from .bias_score import compute_article_bias, ArticleBiasResult, CLASS_WEIGHTS, BIAS_BANDS
from .window_aggregator import aggregate_by_vehicle, VehicleIndex, compute_trend

__all__ = [
    "compute_article_bias",
    "ArticleBiasResult",
    "CLASS_WEIGHTS",
    "BIAS_BANDS",
    "aggregate_by_vehicle",
    "VehicleIndex",
    "compute_trend",
]
