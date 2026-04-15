"""
collector/sources.py
────────────────────
Catálogo de feeds RSS dos veículos monitorados.

Estrutura de cada entrada:
    name        : nome canônico do veículo
    url         : URL do feed RSS
    ideology_id : chave que referencia ideological/data/ideological_references.json
    active      : se False, o coletor ignora o veículo
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NewsSource:
    name: str
    url: str
    ideology_id: str
    active: bool = True


# ── Veículos monitorados ──────────────────────────────────────────────────────
SOURCES: list[NewsSource] = [
    NewsSource(
        name="Folha de S.Paulo",
        url="https://feeds.folha.uol.com.br/emcimadahora/rss091.xml",
        ideology_id="folha",
    ),
    NewsSource(
        name="O Estado de S. Paulo",
        url="https://www.estadao.com.br/rss/ultimas.xml",
        ideology_id="estadao",
    ),
    NewsSource(
        name="O Globo",
        url="https://oglobo.globo.com/rss.xml",
        ideology_id="oglobo",
    ),
    NewsSource(
        name="Gazeta do Povo",
        url="https://www.gazetadopovo.com.br/feed/rss/ultimas-noticias.xml",
        ideology_id="gazetadopovo",
    ),
    NewsSource(
        name="Carta Capital",
        url="https://www.cartacapital.com.br/feed/",
        ideology_id="cartacapital",
    ),
]

ACTIVE_SOURCES: list[NewsSource] = [s for s in SOURCES if s.active]


def get_source_by_ideology_id(ideology_id: str) -> NewsSource | None:
    return next((s for s in SOURCES if s.ideology_id == ideology_id), None)
