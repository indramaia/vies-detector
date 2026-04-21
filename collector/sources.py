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
    NewsSource(
        name="G1",
        url="https://g1.globo.com/rss/g1/",
        ideology_id="g1",
    ),
    NewsSource(
        name="UOL Notícias",
        url="https://rss.uol.com.br/feed/noticias.xml",
        ideology_id="uol",
    ),
    NewsSource(
        name="CNN Brasil",
        url="https://www.cnnbrasil.com.br/feed/",
        ideology_id="cnnbrasil",
    ),
    NewsSource(
        name="Veja",
        url="https://veja.abril.com.br/feed/",
        ideology_id="veja",
    ),
    NewsSource(
        name="Agência Brasil",
        url="https://agenciabrasil.ebc.com.br/rss/ultimasnoticias/feed.xml",
        ideology_id="agenciabrasil",
    ),
    NewsSource(
        name="R7",
        url="https://noticias.r7.com/feed.xml",
        ideology_id="r7",
    ),
    NewsSource(
        name="El País Brasil",
        url="https://brasil.elpais.com/rss/brasil/portada_completa.xml",
        ideology_id="elpais",
    ),
    NewsSource(
        name="Metrópoles",
        url="https://www.metropoles.com/feed",
        ideology_id="metropoles",
    ),
    NewsSource(
        name="Brasil de Fato",
        url="https://www.brasildefato.com.br/rss.xml",
        ideology_id="brasildefato",
    ),
    NewsSource(
        name="The Intercept Brasil",
        url="https://theintercept.com/brasil/feed/",
        ideology_id="intercept",
    ),
    NewsSource(
        name="Agência Pública",
        url="https://apublica.org/feed/",
        ideology_id="agenciapublica",
    ),
    NewsSource(
        name="Le Monde Diplomatique Brasil",
        url="https://diplomatique.org.br/feed/",
        ideology_id="lemonde",
    ),
    NewsSource(
        name="Outras Palavras",
        url="https://outraspalavras.net/feed/",
        ideology_id="outraspalavras",
    ),
    NewsSource(
        name="Jovem Pan News",
        url="https://jovempan.com.br/feed/",
        ideology_id="jovempan",
    ),
]

ACTIVE_SOURCES: list[NewsSource] = [s for s in SOURCES if s.active]


def get_source_by_ideology_id(ideology_id: str) -> NewsSource | None:
    return next((s for s in SOURCES if s.ideology_id == ideology_id), None)
