"""
aggregation/window_aggregator.py
─────────────────────────────────
Agrega BiasScores de artigos individuais em um índice por veículo,
calculado sobre uma janela temporal configurável.

Índice do veículo:
    vehicle_bias_index = mean(BiasScore_i) para i em [t - window, t]

Também calcula:
    - Mediana (mais robusta a outliers)
    - Desvio padrão (dispersão editorial)
    - Número de artigos na janela
    - Tendência (comparação com janela anterior)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from .bias_score import ArticleBiasResult


@dataclass
class VehicleIndex:
    """Índice editorial de um veículo em uma janela temporal."""
    source_name: str
    ideology_id: str
    window_days: int
    reference_date: datetime       # Fim da janela
    article_count: int
    mean_bias: float               # Índice principal
    median_bias: float
    std_bias: float
    min_bias: float
    max_bias: float
    trend: float | None            # Diferença em relação à janela anterior (None se indisponível)
    window_start: datetime
    window_end: datetime


def aggregate_by_vehicle(
    articles: Sequence[ArticleBiasResult],
    window_days: int = 30,
    reference_date: datetime | None = None,
) -> dict[str, VehicleIndex]:
    """
    Calcula o VehicleIndex para cada veículo presente nos resultados.

    Args:
        articles        : resultados de ArticleBiasResult já computados
        window_days     : tamanho da janela temporal em dias
        reference_date  : data de referência (fim da janela); usa now() se None

    Returns:
        Dicionário {ideology_id: VehicleIndex}
    """
    now = reference_date or datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)

    # Agrupa artigos por veículo, filtrando pela janela temporal
    # Nota: ArticleBiasResult não carrega published_at diretamente;
    # a filtragem temporal é feita no pipeline que chama esta função.
    # Aqui agrupamos apenas pelo ideology_id.
    groups: dict[str, list[float]] = {}
    meta: dict[str, tuple[str, str]] = {}  # ideology_id → (source_name, ideology_id)

    for art in articles:
        groups.setdefault(art.ideology_id, []).append(art.bias_score)
        meta[art.ideology_id] = (art.source_name, art.ideology_id)

    result: dict[str, VehicleIndex] = {}

    for ideology_id, scores in groups.items():
        source_name, _ = meta[ideology_id]
        n = len(scores)
        mean_b = round(statistics.mean(scores), 4)
        median_b = round(statistics.median(scores), 4)
        std_b = round(statistics.stdev(scores) if n > 1 else 0.0, 4)

        result[ideology_id] = VehicleIndex(
            source_name=source_name,
            ideology_id=ideology_id,
            window_days=window_days,
            reference_date=now,
            article_count=n,
            mean_bias=mean_b,
            median_bias=median_b,
            std_bias=std_b,
            min_bias=round(min(scores), 4),
            max_bias=round(max(scores), 4),
            trend=None,  # Calculado externamente ao comparar janelas
            window_start=window_start,
            window_end=now,
        )

    return result


def compute_trend(
    current: VehicleIndex,
    previous: VehicleIndex,
) -> float:
    """
    Calcula a tendência de viés entre duas janelas temporais.

    Positivo = viés aumentou; negativo = viés diminuiu.
    """
    return round(current.mean_bias - previous.mean_bias, 4)
