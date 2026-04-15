"""
tests/test_ideological.py
──────────────────────────
Testes unitários para a Camada 4 (ideological).
"""

import pytest
from datetime import datetime, timezone
from aggregation.window_aggregator import VehicleIndex
from ideological.reference_map import load_reference_map, get_profile
from ideological.spectrum import contextualize, contextualize_all, get_spectrum_summary


def _make_vehicle_index(ideology_id: str, mean_bias: float = 0.5) -> VehicleIndex:
    now = datetime.now(timezone.utc)
    return VehicleIndex(
        source_name=ideology_id.capitalize(),
        ideology_id=ideology_id,
        window_days=30,
        reference_date=now,
        article_count=10,
        mean_bias=mean_bias,
        median_bias=mean_bias,
        std_bias=0.1,
        min_bias=0.0,
        max_bias=2.0,
        trend=None,
        window_start=now,
        window_end=now,
    )


class TestReferenceMap:
    def test_loads_known_vehicles(self):
        ref = load_reference_map()
        assert "folha" in ref
        assert "estadao" in ref
        assert "cartacapital" in ref

    def test_ideology_scores_in_range(self):
        ref = load_reference_map()
        for vid, profile in ref.items():
            assert -1.0 <= profile.ideology_score <= 1.0, f"{vid} fora do intervalo"

    def test_uncertainty_positive(self):
        ref = load_reference_map()
        for vid, profile in ref.items():
            assert profile.uncertainty > 0, f"{vid} sem incerteza"

    def test_get_profile_returns_none_for_unknown(self):
        assert get_profile("veiculo_inexistente") is None

    def test_cartacapital_is_left(self):
        profile = get_profile("cartacapital")
        assert profile.ideology_score < -0.5

    def test_gazetadopovo_is_right(self):
        profile = get_profile("gazetadopovo")
        assert profile.ideology_score > 0.5


class TestContextualize:
    def test_known_vehicle_has_ideology_score(self):
        vi = _make_vehicle_index("folha", 0.3)
        ctx = contextualize(vi)
        assert ctx.ideology_score is not None
        assert ctx.position_label is not None

    def test_unknown_vehicle_has_none_ideology(self):
        vi = _make_vehicle_index("veiculo_desconhecido", 0.5)
        ctx = contextualize(vi)
        assert ctx.ideology_score is None
        assert ctx.position_label is None

    def test_caveat_always_present(self):
        vi = _make_vehicle_index("folha", 0.5)
        ctx = contextualize(vi)
        assert len(ctx.caveat) > 20

    def test_contextualization_text_not_empty(self):
        vi = _make_vehicle_index("estadao", 1.2)
        ctx = contextualize(vi)
        assert len(ctx.contextualization) > 10

    def test_bias_score_preserved(self):
        vi = _make_vehicle_index("oglobo", 1.5)
        ctx = contextualize(vi)
        assert ctx.bias_score == pytest.approx(1.5)


class TestGetSpectrumSummary:
    def test_ordered_left_to_right(self):
        indices = {
            "cartacapital": _make_vehicle_index("cartacapital", 1.0),
            "gazetadopovo": _make_vehicle_index("gazetadopovo", 0.8),
            "folha": _make_vehicle_index("folha", 0.3),
        }
        from ideological.spectrum import contextualize_all
        contexts = contextualize_all(indices)
        summary = get_spectrum_summary(contexts)
        scores = [v["ideology_score"] for v in summary if v["ideology_score"] is not None]
        assert scores == sorted(scores), "Espectro não está ordenado da esquerda para direita"

    def test_returns_list_of_dicts(self):
        indices = {"folha": _make_vehicle_index("folha")}
        from ideological.spectrum import contextualize_all
        contexts = contextualize_all(indices)
        summary = get_spectrum_summary(contexts)
        assert isinstance(summary, list)
        assert "source_name" in summary[0]
        assert "ideology_score" in summary[0]
        assert "bias_score" in summary[0]
