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
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from loguru import logger

from .deduplicator import Deduplicator, compute_hash
from .preprocessor import preprocess_article
from .sources import NewsSource, ACTIVE_SOURCES


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────
USER_AGENT = (
    "Mozilla/5.0 (compatible; BiasRadarBot/1.0; "
    "+https://biasradar.lovable.app)"
)
SCRAPE_TIMEOUT = 8  # segundos


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


def _extract_og_image(soup: BeautifulSoup, base_url: str) -> str | None:
    """Extrai a imagem destacada da página HTML."""
    # 1. og:image (Open Graph)
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return urljoin(base_url, og["content"])

    # 2. twitter:image
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if tw and tw.get("content"):
        return urljoin(base_url, tw["content"])

    # 3. primeira <img> dentro de <article>
    article_tag = soup.find("article") or soup
    img = article_tag.find("img")
    if img and img.get("src"):
        return urljoin(base_url, img["src"])

    return None


def _scrape_image_fallback(url: str) -> str | None:
    """
    Faz GET na URL do artigo e tenta extrair og:image / twitter:image.
    Usado APENAS quando o RSS não traz imagem.
    """
    try:
        resp = requests.get(
            url,
            timeout=SCRAPE_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.debug(f"Fallback og:image falhou para {url}: {exc}")
        return None

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        return _extract_og_image(soup, url)
    except Exception as exc:
        logger.debug(f"Parse HTML falhou para {url}: {exc}")
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
    enable_image_fallback: bool = True,
) -> list[ArticleData]:
    """
    Coleta e pré-processa artigos de um único feed RSS.

    Args:
        source                : NewsSource com URL do feed
        deduplicator          : instância Deduplicator para verificar duplicatas
        request_delay         : pausa em segundos após o feed (cortesia ao servidor)
        enable_image_fallback : se True, faz GET extra pra pegar og:image quando
                                o RSS não traz imagem. Custo: 1 request por artigo
                                novo sem imagem.

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
    fallback_count = 0

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

        # 1) tenta imagem direto do RSS (grátis)
        image_url = _extract_image_from_entry(entry)

        # 2) fallback: scraping da página pra pegar og:image
        if not image_url and enable_image_fallback:
            image_url = _scrape_image_fallback(url)
            if image_url:
                fallback_count += 1
            # cortesia: pequena pausa entre scrapes do mesmo veículo
            time.sleep(0.5)

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
            image_url=image_url,
        )
        articles.append(article)

    time.sleep(request_delay)
    logger.info(
        f"[{source.name}] {len(articles)} artigos novos coletados "
        f"({fallback_count} via fallback og:image)."
    )
    return articles


def fetch_all_feeds(
    deduplicator: Deduplicator,
    sources: list[NewsSource] | None = None,
    request_delay: float = 1.0,
    enable_image_fallback: bool = True,
) -> list[ArticleData]:
    """
    Coleta feeds de todos os veículos ativos.

    Args:
        deduplicator          : instância compartilhada para deduplicação global
        sources               : lista personalizada; usa ACTIVE_SOURCES se None
        request_delay         : pausa entre requisições de feed
        enable_image_fallback : ativa scraping de og:image quando RSS não traz

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
                enable_image_fallback=enable_image_fallback,
            )
            all_articles.extend(articles)
        except Exception as exc:
            logger.error(f"Erro inesperado no feed '{source.name}': {exc}")

    logger.info(f"Total coletado: {len(all_articles)} artigos novos.")
    return all_articles
