"""
collector/rss_fetcher.py
────────────────────────
Coleta de artigos via feeds RSS com deduplicação e pré-processamento.

Fluxo por veículo:
    1. Parseia o feed RSS com feedparser
    2. Para cada entrada, verifica deduplicação por SHA-256
    3. Extrai metadados (título, URL, data, veículo)
    4. Pré-processa o snippet de texto
    5. Retorna lista de ArticleData prontos para o classificador

LGPD:
    Apenas metadados e snippet (≤ 500 chars) são armazenados.
    O texto completo do artigo NUNCA é persistido.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import feedparser
from loguru import logger

from .deduplicator import Deduplicator, compute_hash
from .preprocessor import preprocess_article
from .sources import NewsSource, ACTIVE_SOURCES


@dataclass
class ArticleData:
    """Representa um artigo coletado, pronto para classificação."""
    url_hash: str
    url: str
    title: str
    source_name: str
    ideology_id: str
    published_at: datetime
    snippet: str
    sentences: list[str]
    sentence_count: int
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _parse_date(entry: feedparser.FeedParserDict) -> datetime:
    """Tenta extrair a data de publicação de uma entrada RSS."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _extract_text(entry: feedparser.FeedParserDict) -> str:
    """Extrai o melhor trecho de texto disponível na entrada RSS."""
    # Prioridade: content > summary > title
    if hasattr(entry, "content") and entry.content:
        return entry.content[0].get("value", "")
    if hasattr(entry, "summary") and entry.summary:
        return entry.summary
    if hasattr(entry, "title") and entry.title:
        return entry.title
    return ""


def fetch_feed(
    source: NewsSource,
    deduplicator: Deduplicator,
    request_delay: float = 1.0,
) -> list[ArticleData]:
    """
    Coleta e pré-processa artigos de um único feed RSS.

    Args:
        source          : NewsSource com URL do feed
        deduplicator    : instância Deduplicator para verificar duplicatas
        request_delay   : pausa em segundos após a requisição (cortesia ao servidor)

    Returns:
        Lista de ArticleData novos (não duplicados).
    """
    logger.info(f"[{source.name}] Coletando feed: {source.url}")

    try:
        feed = feedparser.parse(source.url)
    except Exception as exc:
        logger.error(f"[{source.name}] Falha ao parsear feed: {exc}")
        return []

    if feed.bozo:
        logger.warning(f"[{source.name}] Feed malformado: {feed.bozo_exception}")

    articles: list[ArticleData] = []

    for entry in feed.entries:
        url = getattr(entry, "link", None)
        if not url:
            continue

        if deduplicator.is_duplicate(url):
            logger.debug(f"[{source.name}] Duplicata ignorada: {url}")
            continue

        url_hash = deduplicator.register(url)
        raw_text = _extract_text(entry)
        processed = preprocess_article(raw_text)

        if processed["sentence_count"] == 0:
            logger.debug(f"[{source.name}] Artigo sem sentenças válidas: {url}")
            continue

        article = ArticleData(
            url_hash=url_hash,
            url=url,
            title=getattr(entry, "title", "Sem título"),
            source_name=source.name,
            ideology_id=source.ideology_id,
            published_at=_parse_date(entry),
            snippet=processed["snippet"],
            sentences=processed["sentences"],
            sentence_count=processed["sentence_count"],
        )
        articles.append(article)

    time.sleep(request_delay)
    logger.info(f"[{source.name}] {len(articles)} artigos novos coletados.")
    return articles


def fetch_all_feeds(
    deduplicator: Deduplicator,
    sources: list[NewsSource] | None = None,
    request_delay: float = 1.0,
) -> list[ArticleData]:
    """
    Coleta feeds de todos os veículos ativos.

    Args:
        deduplicator  : instância compartilhada para deduplicação global
        sources       : lista personalizada; usa ACTIVE_SOURCES se None
        request_delay : pausa entre requisições

    Returns:
        Lista concatenada de ArticleData de todos os veículos.
    """
    sources = sources or ACTIVE_SOURCES
    all_articles: list[ArticleData] = []

    for source in sources:
        try:
            articles = fetch_feed(source, deduplicator, request_delay)
            all_articles.extend(articles)
        except Exception as exc:
            logger.error(f"Erro inesperado no feed '{source.name}': {exc}")

    logger.info(f"Total coletado: {len(all_articles)} artigos novos.")
    return all_articles
