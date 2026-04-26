"""
classifier/types.py
───────────────────
Dataclasses compartilhados entre o classifier e o aggregation.
Sem dependências pesadas (torch/transformers) — seguro para importar na API.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SentenceResult:
    """Resultado da classificação de uma única sentença."""
    sentence: str
    label: str          # "factual" | "enviesada" | "fortemente_enviesada"
    label_id: int       # 0 | 1 | 2
    confidence: float   # probabilidade da classe predita
    scores: dict[str, float]  # probabilidades para todas as classes
