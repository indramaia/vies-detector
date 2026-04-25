"""
collector/rss_fetcher.py
────────────────────────
Coleta de artigos via feeds RSS com deduplicação, pré-processamento
e extração de imagem (RSS nativo + fallback og:image via scraping).

Fluxo por veículo:
    1. Parseia o feed RSS com feedparser
    2. Para cada entrada, verifica deduplicação por SHA-256
    3. Extrai metadados (título, URL, data, veículo, imagem)
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

import feedparser
from loguru import logger

from .article_scraper import scrape_article
from .deduplicator import Deduplicator, compute_hash
from .preprocessor import preprocess_article
from .sources import NewsSource, ACTIVE_SOURCES


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────
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
    image_url: str | None = None
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ──────────────────────────────────────────────────────────────────
# Image extraction
# ──────────────────────────────────────────────────────────────────
def _extract_image_from_entry(entry: feedparser.FeedParserDict) -> str | None:
    """Extrai URL da imagem direto do RSS (sem fazer requisição extra)."""
    # 1. media:content (Media RSS)
    if hasattr(entry, "media_content") and entry.media_content:
        for m in entry.media_content:
            if m.get("url"):
                return m["url"]

    # 2. media:thumbnail
    if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
        for m in entry.media_thumbnail:
            if m.get("url"):
                return m["url"]

    # 3. enclosures (imagem anexada)
    if hasattr(entry, "enclosures") and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get("type", "").startswith("image/"):
                return enc.get("href") or enc.get("url")

    # 4. links com rel="enclosure"
    if hasattr(entry, "links"):
        for link in entry.links:
            if (
                link.get("rel") == "enclosure"
                and link.get("type", "").startswith("image/")
            ):
                return link.get("href")

    # 5. <img> embutida no summary/content HTML
    html_blob = ""
    if hasattr(entry, "content") and entry.content:
        html_blob = entry.content[0].get("value", "")
    elif hasattr(entry, "summary"):
        html_blob = entry.summary or ""

    if html_blob and "<img" in html_blob.lower():
        try:
            soup = BeautifulSoup(html_blob, "html.parser")
            img = soup.find("img")
            if img and img.get("src"):
                return img["src"]
        except Exception:
            pass

    return None


# ──────────────────────────────────────────────────────────────────
# RSS helpers
# ──────────────────────────────────────────────────────────────────
def _parse_date(entry: feedparser.FeedParserDict) -> datetime:
    """Tenta extrair a data de publicação de uma entrada RSS."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _extract_text(entry: feedparser.FeedParserDict) -> str:
    """Extrai o melhor trecho de texto disponível na entrada RSS."""
    if hasattr(entry, "content") and entry.content:
        return entry.content[0].get("value", "")
    if hasattr(entry, "summary") and entry.summary:
        return entry.summary
    if hasattr(entry, "title") and entry.title:
        return entry.title
    return ""


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────
def fetch_feed(
    source: NewsSource,
    deduplicator: Deduplicator,
    request_delay: float = 1.0,
    enable_scraping: bool = True,
) -> list[ArticleData]:
    """
    Coleta artigos de um único feed RSS e raspa o corpo completo de cada artigo.

    Args:
        source          : NewsSource com URL do feed
        deduplicator    : instância Deduplicator para verificar duplicatas
        request_delay   : pausa em segundos entre feeds (cortesia ao servidor)
        enable_scraping : se True, faz GET na página do artigo para extrair
                          corpo completo + imagem numa única requisição.
                          Desative em testes ou quando a rede não estiver disponível.

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
        bozo_msg = str(feed.bozo_exception)
        if "declared as" in bozo_msg and "parsed as" in bozo_msg:
            logger.debug(f"[{source.name}] Encoding declarado diferente do real (inofensivo): {bozo_msg}")
        else:
            logger.warning(f"[{source.name}] Feed malformado: {bozo_msg}")

    articles: list[ArticleData] = []
    scraped_count = 0
    duplicate_count = 0

    for entry in feed.entries:
        url = getattr(entry, "link", None)
        if not url:
            continue

        if deduplicator.is_duplicate(url):
            duplicate_count += 1
            continue

        url_hash  = deduplicator.register(url)
        rss_text  = _extract_text(entry)
        image_url = _extract_image_from_entry(entry)
        full_text = ""

        if enable_scraping and source.scraping:
            scraped = scrape_article(url)
            if scraped["ok"]:
                full_text = scraped["full_text"]   # em memória, não persistido
                scraped_count += 1
            # imagem do scraper só se o RSS não trouxe
            if not image_url and scraped.get("image_url"):
                image_url = scraped["image_url"]
            time.sleep(0.5)  # cortesia entre requisições do mesmo veículo

        processed = preprocess_article(rss_text, full_text)

        if processed["sentence_count"] == 0:
            logger.debug(f"[{source.name}] Artigo sem sentenças válidas: {url}")
            continue

        articles.append(ArticleData(
            url_hash=url_hash,
            url=url,
            title=getattr(entry, "title", "Sem título"),
            source_name=source.name,
            ideology_id=source.ideology_id,
            published_at=_parse_date(entry),
            snippet=processed["snippet"],
            sentences=processed["sentences"],
            sentence_count=processed["sentence_count"],
            image_url=image_url,
        ))

    time.sleep(request_delay)
    if duplicate_count:
        logger.debug(f"[{source.name}] {duplicate_count} duplicatas ignoradas.")
    logger.info(
        f"[{source.name}] {len(articles)} artigos novos "
        f"({scraped_count} com corpo completo raspado)."
    )
    return articles


def fetch_all_feeds(
    deduplicator: Deduplicator,
    sources: list[NewsSource] | None = None,
    request_delay: float = 1.0,
    enable_scraping: bool = True,
) -> list[ArticleData]:
    """
    Coleta feeds de todos os veículos ativos e raspa o corpo dos artigos.

    Args:
        deduplicator    : instância compartilhada para deduplicação global
        sources         : lista personalizada; usa ACTIVE_SOURCES se None
        request_delay   : pausa entre requisições de feed
        enable_scraping : ativa raspagem do corpo completo de cada artigo

    Returns:
        Lista concatenada de ArticleData de todos os veículos.
    """
    sources = sources or ACTIVE_SOURCES
    all_articles: list[ArticleData] = []

    for source in sources:
        try:
            articles = fetch_feed(
                source,
                deduplicator,
                request_delay,
                enable_scraping=enable_scraping,
            )
            all_articles.extend(articles)
        except Exception as exc:
            logger.error(f"Erro inesperado no feed '{source.name}': {exc}")

    logger.info(f"Total coletado: {len(all_articles)} artigos novos.")
    return all_articles
