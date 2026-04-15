"""
tests/test_aggregation.py
──────────────────────────
Testes unitários para a Camada 3 (aggregation).
"""

import pytest
from classifier.sentence_classifier import SentenceResult
from aggregation.bias_score import compute_article_bias, CLASS_WEIGHTS
from aggregation.window_aggregator import aggregate_by_vehicle, VehicleIndex
from aggregation import ArticleBiasResult


def _make_result(label: str, confidence: float = 0.9) -> SentenceResult:
    scores = {"factual": 0.0, "enviesada": 0.0, "fortemente_enviesada": 0.0}
    scores[label] = confidence
    label_map = {"factual": 0, "enviesada": 1, "fortemente_enviesada": 2}
    return SentenceResult(
        sentence=f"Sentença classificada como {label}.",
        label=label,
        label_id=label_map[label],
        confidence=confidence,
        scores=scores,
    )


class TestComputeArticleBias:
    def test_all_factual_gives_zero(self):
        sentences = [_make_result("factual")] * 5
        result = compute_article_bias("hash1", "Folha", "folha", sentences)
        assert result.bias_score == 0.0

    def test_all_strongly_biased_gives_two(self):
        sentences = [_make_result("fortemente_enviesada")] * 4
        result = compute_article_bias("hash2", "Folha", "folha", sentences)
        assert result.bias_score == 2.0

    def test_mixed_calculates_correctly(self):
        # 2 factuais + 2 enviesadas + 1 fortemente → (0+2+2)/5 = 0.8
        sentences = (
            [_make_result("factual")] * 2 +
            [_make_result("enviesada")] * 2 +
            [_make_result("fortemente_enviesada")] * 1
        )
        result = compute_article_bias("hash3", "Estadão", "estadao", sentences)
        assert result.bias_score == pytest.approx(0.8, abs=0.001)

    def test_empty_sentences_returns_zero(self):
        result = compute_article_bias("hash4", "Globo", "oglobo", [])
        assert result.bias_score == 0.0
        assert result.sentence_count == 0

    def test_counts_are_correct(self):
        sentences = (
            [_make_result("factual")] * 3 +
            [_make_result("enviesada")] * 2 +
            [_make_result("fortemente_enviesada")] * 1
        )
        result = compute_article_bias("hash5", "Carta", "cartacapital", sentences)
        assert result.n_factual == 3
        assert result.n_biased == 2
        assert result.n_strongly_biased == 1
        assert result.sentence_count == 6

    def test_interpretation_band(self):
        sentences = [_make_result("fortemente_enviesada")] * 10
        result = compute_article_bias("hash6", "Gazeta", "gazetadopovo", sentences)
        assert "fortemente" in result.interpretation.lower()


class TestAggregateByVehicle:
    def _make_article_result(self, ideology_id: str, score: float) -> ArticleBiasResult:
        return ArticleBiasResult(
            url_hash=f"hash_{ideology_id}_{score}",
            source_name=ideology_id.capitalize(),
            ideology_id=ideology_id,
            bias_score=score,
            interpretation="test",
            sentence_count=5,
            n_factual=3,
            n_biased=1,
            n_strongly_biased=1,
            sentence_results=[],
        )

    def test_mean_is_correct(self):
        arts = [
            self._make_article_result("folha", 0.2),
            self._make_article_result("folha", 0.4),
            self._make_article_result("folha", 0.6),
        ]
        indices = aggregate_by_vehicle(arts)
        assert "folha" in indices
        assert indices["folha"].mean_bias == pytest.approx(0.4, abs=0.001)

    def test_groups_by_ideology_id(self):
        arts = [
            self._make_article_result("folha", 0.3),
            self._make_article_result("estadao", 0.5),
            self._make_article_result("folha", 0.5),
        ]
        indices = aggregate_by_vehicle(arts)
        assert set(indices.keys()) == {"folha", "estadao"}
        assert indices["folha"].article_count == 2

    def test_empty_returns_empty_dict(self):
        assert aggregate_by_vehicle([]) == {}
