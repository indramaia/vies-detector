"""
ideological/spectrum.py
───────────────────────
Camada 4 — Contextualização ideológica do BiasScore.

Combina o BiasScore (Camada 3) com o posicionamento acadêmico do veículo
para gerar um contexto interpretativo ao leitor.

Saída por veículo:
    - ideology_score    : posição no espectro [-1, +1]
    - position_label    : rótulo textual (ex: "Centro-direita")
    - bias_score        : índice quantitativo de viés linguístico [0, 2]
    - contextualization : texto explicativo para o usuário final
    - caveat            : aviso epistêmico obrigatório

AVISO ÉTICO:
    O posicionamento ideológico é baseado em análises acadêmicas de períodos
    específicos, não em julgamento do sistema. Ver Mitchell et al. (2019)
    sobre documentação responsável de modelos.
"""

from __future__ import annotations

from dataclasses import dataclass

from .reference_map import load_reference_map, IdeologicalProfile, get_profile
from aggregation.window_aggregator import VehicleIndex


@dataclass
class IdeologicalContext:
    """Resultado completo da contextualização ideológica para um veículo."""
    source_name: str
    ideology_id: str
    # Camada 3
    bias_score: float
    bias_interpretation: str
    article_count: int
    window_days: int
    # Camada 4
    ideology_score: float | None        # None se veículo desconhecido
    uncertainty: float | None
    position_label: str | None
    position_description: str | None
    academic_refs: list[str]
    # Texto gerado para a interface
    contextualization: str
    caveat: str


# ── Caveat epistêmico padrão ──────────────────────────────────────────────────
_CAVEAT = (
    "⚠️  O BiasScore é uma estimativa probabilística gerada por modelo de linguagem "
    "treinado sobre o corpus FactNews (VARGAS et al., 2023). O posicionamento ideológico "
    "baseia-se em análises acadêmicas e pode não refletir mudanças editoriais recentes. "
    "Esta ferramenta apoia a leitura crítica — não substitui julgamento humano."
)


def _build_contextualization(
    vehicle_index: VehicleIndex,
    profile: IdeologicalProfile | None,
) -> str:
    """Gera o texto de contextualização para a interface do usuário."""
    bias_desc = _bias_narrative(vehicle_index.mean_bias)

    if profile is None:
        return (
            f"'{vehicle_index.source_name}' apresentou BiasScore médio de "
            f"{vehicle_index.mean_bias:.2f} nos últimos {vehicle_index.window_days} dias "
            f"({vehicle_index.article_count} artigos analisados). {bias_desc} "
            f"Não há mapeamento ideológico acadêmico disponível para este veículo."
        )

    direction = _ideology_direction(profile.ideology_score)
    return (
        f"'{vehicle_index.source_name}' é classificado como '{profile.position_label}' "
        f"por estudos acadêmicos ({', '.join(profile.academic_refs[:1])}). "
        f"No período analisado ({vehicle_index.window_days} dias, "
        f"{vehicle_index.article_count} artigos), o BiasScore médio foi "
        f"{vehicle_index.mean_bias:.2f} (σ={vehicle_index.std_bias:.2f}). "
        f"{bias_desc} "
        f"O veículo tende a produzir enquadramentos {direction} segundo a literatura de referência."
    )


def _bias_narrative(score: float) -> str:
    if score < 0.4:
        return "A linguagem predominante nas matérias analisadas é factual."
    if score < 0.8:
        return "As matérias analisadas apresentam viés linguístico moderado."
    if score < 1.4:
        return "As matérias analisadas apresentam viés linguístico elevado."
    return "As matérias analisadas apresentam linguagem fortemente enviesada."


def _ideology_direction(score: float) -> str:
    if score <= -0.5:
        return "progressistas/de esquerda"
    if score <= -0.1:
        return "centro-progressistas"
    if score < 0.1:
        return "de centro"
    if score < 0.5:
        return "centro-conservadores"
    return "conservadores/de direita"


# ── Interface principal ───────────────────────────────────────────────────────

def contextualize(vehicle_index: VehicleIndex) -> IdeologicalContext:
    """
    Gera o contexto ideológico para um VehicleIndex.

    Args:
        vehicle_index : resultado da Camada 3 para um veículo

    Returns:
        IdeologicalContext com todos os campos preenchidos.
    """
    profile = get_profile(vehicle_index.ideology_id)
    text = _build_contextualization(vehicle_index, profile)

    return IdeologicalContext(
        source_name=vehicle_index.source_name,
        ideology_id=vehicle_index.ideology_id,
        bias_score=vehicle_index.mean_bias,
        bias_interpretation=_bias_narrative(vehicle_index.mean_bias),
        article_count=vehicle_index.article_count,
        window_days=vehicle_index.window_days,
        ideology_score=profile.ideology_score if profile else None,
        uncertainty=profile.uncertainty if profile else None,
        position_label=profile.position_label if profile else None,
        position_description=profile.description if profile else None,
        academic_refs=profile.academic_refs if profile else [],
        contextualization=text,
        caveat=_CAVEAT,
    )


def contextualize_all(
    vehicle_indices: dict[str, VehicleIndex],
) -> dict[str, IdeologicalContext]:
    """
    Aplica contextualização ideológica a todos os veículos.

    Args:
        vehicle_indices : saída de aggregation.aggregate_by_vehicle()

    Returns:
        Dicionário {ideology_id: IdeologicalContext}
    """
    return {vid: contextualize(vi) for vid, vi in vehicle_indices.items()}


def get_spectrum_summary(
    contexts: dict[str, IdeologicalContext],
) -> list[dict]:
    """
    Retorna lista ordenada de veículos no espectro ideológico,
    do mais progressista ao mais conservador.

    Útil para renderizar o gráfico de espectro na interface.
    """
    summary = []
    for ctx in contexts.values():
        summary.append({
            "source_name": ctx.source_name,
            "ideology_id": ctx.ideology_id,
            "ideology_score": ctx.ideology_score,
            "uncertainty": ctx.uncertainty,
            "position_label": ctx.position_label,
            "bias_score": ctx.bias_score,
            "article_count": ctx.article_count,
        })
    return sorted(
        summary,
        key=lambda x: (x["ideology_score"] if x["ideology_score"] is not None else 0),
    )
