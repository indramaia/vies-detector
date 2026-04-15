"""
classifier/sentence_classifier.py
──────────────────────────────────
Classificação de viés editorial em nível de sentença usando BERTimbau.

Para cada sentença, o modelo retorna:
    label     : "factual" | "enviesada" | "fortemente_enviesada"
    label_id  : 0 | 1 | 2
    scores    : probabilidades softmax para cada classe (dict)
    confidence: probabilidade da classe predita

Arquitetura:
    Token [CLS] → camada linear → softmax → 3 classes
    (DEVLIN et al., 2019; SOUZA et al., 2020)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from loguru import logger

from .model_loader import load_model, ID2LABEL, LABEL2ID, MAX_LENGTH


@dataclass
class SentenceResult:
    """Resultado da classificação de uma única sentença."""
    sentence: str
    label: str          # "factual" | "enviesada" | "fortemente_enviesada"
    label_id: int       # 0 | 1 | 2
    confidence: float   # probabilidade da classe predita
    scores: dict[str, float]  # probabilidades para todas as classes


class SentenceClassifier:
    """
    Interface de alto nível para classificação de sentenças.

    Suporta classificação individual e em batch (mais eficiente para
    artigos longos).

    Exemplo de uso:
        clf = SentenceClassifier()
        results = clf.classify_batch(["O ministro afirmou...", "O corrupto governo..."])
    """

    def __init__(self, model_path: str | None = None):
        from .model_loader import DEFAULT_MODEL_PATH
        _path = model_path or DEFAULT_MODEL_PATH
        self._model, self._tokenizer = load_model(_path)
        self._device = next(self._model.parameters()).device

    # ── Classificação de sentença única ──────────────────────────────────────

    def classify(self, sentence: str) -> SentenceResult:
        """Classifica uma única sentença."""
        return self.classify_batch([sentence])[0]

    # ── Classificação em batch ────────────────────────────────────────────────

    def classify_batch(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
    ) -> list[SentenceResult]:
        """
        Classifica uma lista de sentenças em batches.

        Args:
            sentences  : lista de strings (sentenças limpas)
            batch_size : tamanho do mini-batch para inferência

        Returns:
            Lista de SentenceResult na mesma ordem das sentenças de entrada.
        """
        if not sentences:
            return []

        results: list[SentenceResult] = []

        for i in range(0, len(sentences), batch_size):
            batch = list(sentences[i : i + batch_size])
            batch_results = self._run_batch(batch)
            results.extend(batch_results)

        return results

    # ── Internos ──────────────────────────────────────────────────────────────

    def _run_batch(self, sentences: list[str]) -> list[SentenceResult]:
        encoding = self._tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)

        probs = F.softmax(outputs.logits, dim=-1).cpu()  # (batch, 3)

        batch_results: list[SentenceResult] = []
        for idx, sentence in enumerate(sentences):
            prob_vec = probs[idx]
            label_id = int(prob_vec.argmax().item())
            label = ID2LABEL[label_id]
            confidence = float(prob_vec[label_id].item())
            scores = {ID2LABEL[j]: float(prob_vec[j].item()) for j in range(len(ID2LABEL))}

            batch_results.append(
                SentenceResult(
                    sentence=sentence,
                    label=label,
                    label_id=label_id,
                    confidence=confidence,
                    scores=scores,
                )
            )

        return batch_results
