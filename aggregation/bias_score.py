"""
aggregation/bias_score.py
─────────────────────────
Cálculo do BiasScore por artigo a partir das classificações sentence-level.

Fórmula (definida no TCC):
    BiasScore = (1 × n_enviesada + 2 × n_fortemente_enviesada) / n_total

Intervalo: [0.0, 2.0]
    0.0 → todas as sentenças factuais
    2.0 → todas as sentenças fortemente enviesadas

Tabela de referência:
    [0.0, 0.4) → Predominantemente factual
    [0.4, 0.8) → Viés moderado
    [0.8, 1.4) → Viés elevado
    [1.4, 2.0] → Linguagem fortemente enviesada
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from classifier.sentence_classifier import SentenceResult


# ── Pesos por classe ──────────────────────────────────────────────────────────
CLASS_WEIGHTS: dict[str, float] = {
    "factual": 0.0,
    "enviesada": 1.0,
    "fortemente_enviesada": 2.0,
}

# ── Faixas interpretativas ────────────────────────────────────────────────────
BIAS_BANDS: list[tuple[float, float, str]] = [
    (0.0, 0.4,  "Predominantemente factual"),
    (0.4, 0.8,  "Viés moderado"),
    (0.8, 1.4,  "Viés elevado"),
    (1.4, 2.01, "Linguagem fortemente enviesada"),
]


@dataclass
class ArticleBiasResult:
    """Resultado do BiasScore para um artigo."""
    url_hash: str
    source_name: str
    ideology_id: str
    bias_score: float            # [0.0, 2.0]
    interpretation: str          # Faixa interpretativa
    sentence_count: int
    n_factual: int
    n_biased: int
    n_strongly_biased: int
    sentence_results: list[SentenceResult]


def _interpret(score: float) -> str:
    for low, high, label in BIAS_BANDS:
        if low <= score < high:
            return label
    return "Linguagem fortemente enviesada"


def compute_article_bias(
    url_hash: str,
    source_name: str,
    ideology_id: str,
    sentence_results: list[SentenceResult],
) -> ArticleBiasResult:
    """
    Calcula o BiasScore de um artigo.

    Args:
        url_hash         : hash SHA-256 do artigo (identificador único)
        source_name      : nome canônico do veículo
        ideology_id      : chave do mapeamento ideológico
        sentence_results : saída do SentenceClassifier para cada sentença

    Returns:
        ArticleBiasResult com score, interpretação e estatísticas por classe.
    """
    if not sentence_results:
        return ArticleBiasResult(
            url_hash=url_hash,
            source_name=source_name,
            ideology_id=ideology_id,
            bias_score=0.0,
            interpretation="Sem sentenças classificáveis",
            sentence_count=0,
            n_factual=0,
            n_biased=0,
            n_strongly_biased=0,
            sentence_results=[],
        )

    n_total = len(sentence_results)
    n_factual = sum(1 for r in sentence_results if r.label == "factual")
    n_biased = sum(1 for r in sentence_results if r.label == "enviesada")
    n_strongly = sum(1 for r in sentence_results if r.label == "fortemente_enviesada")

    # Fórmula do BiasScore
    numerator = (CLASS_WEIGHTS["enviesada"] * n_biased +
                 CLASS_WEIGHTS["fortemente_enviesada"] * n_strongly)
    bias_score = round(numerator / n_total, 4)

    return ArticleBiasResult(
        url_hash=url_hash,
        source_name=source_name,
        ideology_id=ideology_id,
        bias_score=bias_score,
        interpretation=_interpret(bias_score),
        sentence_count=n_total,
        n_factual=n_factual,
        n_biased=n_biased,
        n_strongly_biased=n_strongly,
        sentence_results=sentence_results,
    )
