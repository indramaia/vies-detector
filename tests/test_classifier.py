"""
tests/test_classifier.py
─────────────────────────
Testes unitários para a Camada 2 (classifier).

Nota: os testes de inferência usam mock do modelo para evitar dependência
de GPU e do modelo fine-tuned. Testes de integração com o modelo real
devem ser executados separadamente com pytest -m integration.
"""

import pytest
from unittest.mock import MagicMock, patch
import torch

from classifier.sentence_classifier import SentenceResult
from classifier.model_loader import ID2LABEL, LABEL2ID


class TestSentenceResult:
    def test_fields_present(self):
        result = SentenceResult(
            sentence="O governo anunciou medidas.",
            label="factual",
            label_id=0,
            confidence=0.92,
            scores={"factual": 0.92, "enviesada": 0.05, "fortemente_enviesada": 0.03},
        )
        assert result.label == "factual"
        assert result.label_id == 0
        assert result.confidence == pytest.approx(0.92)
        assert sum(result.scores.values()) == pytest.approx(1.0, abs=0.01)


class TestSentenceClassifierMocked:
    """
    Testes com modelo mockado — não requerem GPU nem modelo fine-tuned.
    """

    @pytest.fixture
    def mock_classifier(self):
        """Cria um SentenceClassifier com modelo e tokenizador mockados."""
        with patch("classifier.sentence_classifier.load_model") as mock_load:
            # Mock do tokenizador
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {
                "input_ids": torch.zeros(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }

            # Mock do modelo: retorna logits que favorecem "factual" (classe 0)
            mock_model = MagicMock()
            mock_model.parameters.return_value = iter([torch.zeros(1)])
            mock_output = MagicMock()
            mock_output.logits = torch.tensor([[5.0, 1.0, 0.5]])  # factual dominante
            mock_model.return_value = mock_output

            mock_load.return_value = (mock_model, mock_tokenizer)

            from classifier.sentence_classifier import SentenceClassifier
            clf = SentenceClassifier.__new__(SentenceClassifier)
            clf._model = mock_model
            clf._tokenizer = mock_tokenizer
            clf._device = torch.device("cpu")
            return clf

    def test_classify_returns_sentence_result(self, mock_classifier):
        result = mock_classifier.classify("Texto de teste para classificação.")
        assert isinstance(result, SentenceResult)
        assert result.label in ID2LABEL.values()
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_batch_length_matches(self, mock_classifier):
        sentences = [
            "O ministro anunciou novas medidas fiscais.",
            "O governo corrupto destruiu o país.",
            "A taxa de desemprego atingiu 8,5% em janeiro.",
        ]

        # Ajusta o mock para retornar logits para 3 sentenças
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([
            [5.0, 1.0, 0.5],
            [0.5, 1.0, 5.0],
            [5.0, 0.5, 0.5],
        ])
        mock_classifier._model.return_value = mock_output

        results = mock_classifier.classify_batch(sentences)
        assert len(results) == 3

    def test_empty_batch_returns_empty(self, mock_classifier):
        results = mock_classifier.classify_batch([])
        assert results == []


class TestLabelMapping:
    def test_id2label_coverage(self):
        assert set(ID2LABEL.values()) == {"factual", "enviesada", "fortemente_enviesada"}

    def test_label2id_inverse_of_id2label(self):
        for label_id, label in ID2LABEL.items():
            assert LABEL2ID[label] == label_id

    def test_num_labels_is_three(self):
        from classifier.model_loader import NUM_LABELS
        assert NUM_LABELS == 3
