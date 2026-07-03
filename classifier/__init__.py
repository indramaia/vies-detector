"""
classifier
──────────
Camada 2 — Classificação sentence-level de viés editorial via BERTimbau.
"""

from .sentence_classifier import SentenceClassifier
from aggregation.bias_score import SentenceResult
from .model_loader import load_model, ID2LABEL, LABEL2ID, NUM_LABELS

__all__ = [
    "SentenceClassifier",
    "SentenceResult",
    "load_model",
    "ID2LABEL",
    "LABEL2ID",
    "NUM_LABELS",
]
