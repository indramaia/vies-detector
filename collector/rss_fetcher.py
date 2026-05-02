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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

import re as _re

import feedparser
from bs4 import BeautifulSoup as _BeautifulSoup
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
    scraped: bool = False  # True se o corpo completo foi extraído com sucesso
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

import re as _re
import requests as _requests

_FEED_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; feedparser/6.0)",
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
}
_FEED_TIMEOUT = 15


def _fetch_feed_robust(source: NewsSource):
    """
    Busca o feed em duas tentativas:
      1. feedparser direto (caminho normal)
      2. Se bozo E sem entradas: busca raw via requests, limpa bytes
         ilegais e repassa ao feedparser — recupera feeds com encoding
         quebrado ou caracteres de controle (R7, El País, Le Monde etc.)
    """
    try:
        feed = feedparser.parse(source.url)
    except Exception as exc:
        logger.error(f"[{source.name}] Falha ao parsear feed: {exc}")
        return None

    # Feed OK ou bozo mas com entradas → usa direto
    if not feed.bozo or feed.entries:
        return feed

    # Bozo sem entradas → tenta recuperação via raw fetch
    logger.debug(f"[{source.name}] Tentando recuperação de feed malformado…")
    try:
        resp = _requests.get(source.url, headers=_FEED_HEADERS, timeout=_FEED_TIMEOUT)
        resp.raise_for_status()
        raw = resp.content

        # Remove bytes de controle ilegais em XML (exceto \t \n \r)
        raw_clean = _re.sub(rb"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", b"", raw)

        # Tenta forçar UTF-8 se a declaração de encoding estiver errada
        try:
            raw_clean = raw_clean.decode("utf-8", errors="replace").encode("utf-8")
        except Exception:
            pass

        recovered = feedparser.parse(raw_clean)
        if recovered.entries:
            logger.info(
                f"[{source.name}] Feed recuperado: {len(recovered.entries)} entradas "
                f"(encoding/caracteres ilegais corrigidos)."
            )
            return recovered
        else:
            logger.warning(f"[{source.name}] Feed malformado sem recuperação possível.")
            return feed  # devolve o bozo original (sem entradas)
    except Exception as exc:
        logger.warning(f"[{source.name}] Falha na recuperação do feed: {exc}")
        return feed


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
# Homepage-based collection (fontes sem RSS)
# ──────────────────────────────────────────────────────────────────

def _slug_to_title(url: str) -> str:
    """Converte o slug da URL em título legível como fallback."""
    slug = url.rstrip("/").split("/")[-1]
    slug = _re.sub(r"-\d{8}$", "", slug)   # remove data DDMMYYYY do final
    return slug.replace("-", " ").capitalize()


def _fetch_homepage_source(
    source: NewsSource,
    deduplicator: Deduplicator,
    enable_scraping: bool = True,
) -> list[ArticleData]:
    """
    Coleta artigos de um veículo sem RSS raspando links da homepage.

    Fluxo:
        1. GET na homepage → extrai todos os <a href>
        2. Filtra URLs que batem com source.article_url_re
        3. Deduplicação por hash
        4. Scraping paralelo de cada artigo (mesmo pipeline do RSS)
        5. Retorna lista de ArticleData prontos para classificação
    """
    logger.info(f"[{source.name}] Coletando via homepage: {source.homepage_url}")

    try:
        resp = _requests.get(source.homepage_url, headers=_FEED_HEADERS, timeout=_FEED_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.error(f"[{source.name}] Falha ao buscar homepage: {exc}")
        return []

    soup = _BeautifulSoup(resp.text, "html.parser")
    pattern = _re.compile(source.article_url_re) if source.article_url_re else None

    seen: set[str] = set()
    candidates: list[tuple[str, str, datetime]] = []   # (url, título, pub)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pattern and not pattern.search(href):
            continue
        if href in seen:
            continue
        seen.add(href)

        # Título: texto do link; fallback = slug legível
        title = a.get_text(" ", strip=True)
        if len(title) < 15:
            title = _slug_to_title(href)

        # Data: extrai DDMMYYYY do final da URL
        pub = datetime.now(timezone.utc)
        if dm := _re.search(r"(\d{2})(\d{2})(\d{4})/?$", href):
            try:
                pub = datetime(
                    int(dm.group(3)), int(dm.group(2)), int(dm.group(1)),
                    tzinfo=timezone.utc,
                )
            except ValueError:
                pass

        candidates.append((href, title, pub))

    if not candidates:
        logger.warning(f"[{source.name}] Nenhum artigo encontrado na homepage.")
        return []

    # Deduplicação
    pending: list[tuple[str, str, str, datetime]] = []  # (url_hash, url, título, pub)
    for url, title, pub in candidates:
        if deduplicator.is_duplicate(url):
            continue
        url_hash = deduplicator.register(url)
        pending.append((url_hash, url, title, pub))

    duplicate_count = len(candidates) - len(pending)
    if not pending:
        logger.info(f"[{source.name}] Todos os artigos já registrados.")
        return []

    # Scraping paralelo — idêntico ao fluxo RSS
    scrape_map: dict[str, dict] = {}
    scraped_count = 0
    if enable_scraping:
        with ThreadPoolExecutor(max_workers=3) as pool:
            future_to_url = {
                pool.submit(scrape_article, url): url
                for _, url, _, _ in pending
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                except Exception:
                    result = {"full_text": "", "image_url": None, "ok": False}
                scrape_map[url] = result
                if result["ok"]:
                    scraped_count += 1

    # Construção dos ArticleData
    articles: list[ArticleData] = []
    for url_hash, url, title, pub in pending:
        scrape_result = scrape_map.get(url, {})
        ok        = scrape_result.get("ok", False)
        full_text = scrape_result.get("full_text", "") if ok else ""
        image_url = scrape_result.get("image_url")

        processed = preprocess_article("", full_text)
        if processed["sentence_count"] == 0:
            logger.debug(f"[{source.name}] Artigo sem sentenças válidas: {url}")
            continue

        articles.append(ArticleData(
            url_hash=url_hash,
            url=url,
            title=title,
            source_name=source.name,
            ideology_id=source.ideology_id,
            published_at=pub,
            snippet=processed["snippet"],
            sentences=processed["sentences"],
            sentence_count=processed["sentence_count"],
            image_url=image_url,
            scraped=ok,
        ))

    if duplicate_count:
        logger.debug(f"[{source.name}] {duplicate_count} duplicatas ignoradas.")
    logger.info(
        f"[{source.name}] {len(articles)} artigos novos "
        f"({scraped_count} com corpo completo raspado)."
    )
    return articles


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
    if source.homepage_url:
        return _fetch_homepage_source(source, deduplicator, enable_scraping)

    logger.info(f"[{source.name}] Coletando feed: {source.url}")

    feed = _fetch_feed_robust(source)
    if feed is None:
        return []

    if feed.bozo:
        bozo_msg = str(feed.bozo_exception)
        if "declared as" in bozo_msg and "parsed as" in bozo_msg:
            logger.debug(f"[{source.name}] Encoding declarado diferente do real (inofensivo): {bozo_msg}")
        else:
            logger.warning(f"[{source.name}] Feed bozo (recuperado): {bozo_msg}")

    duplicate_count = 0

    # Fase 1 — deduplicação sequencial (necessário para thread safety do Deduplicator)
    pending: list[tuple] = []   # (url_hash, url, rss_text, image_rss, entry)
    for entry in feed.entries:
        url = getattr(entry, "link", None)
        if not url:
            continue
        if deduplicator.is_duplicate(url):
            duplicate_count += 1
            continue
        url_hash  = deduplicator.register(url)
        rss_text  = _extract_text(entry)
        image_rss = _extract_image_from_entry(entry)
        pending.append((url_hash, url, rss_text, image_rss, entry))

    # Fase 2 — scraping paralelo (I/O-bound; 3 workers por veículo)
    scrape_map: dict[str, dict] = {}
    scraped_count = 0
    if enable_scraping and source.scraping and pending:
        with ThreadPoolExecutor(max_workers=3) as pool:
            future_to_url = {
                pool.submit(scrape_article, url): url
                for _, url, _, _, _ in pending
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                except Exception:
                    result = {"full_text": "", "image_url": None, "ok": False}
                scrape_map[url] = result
                if result["ok"]:
                    scraped_count += 1

    # Fase 3 — construção dos ArticleData (preserva ordem original do feed)
    articles: list[ArticleData] = []
    for url_hash, url, rss_text, image_rss, entry in pending:
        scrape_result = scrape_map.get(url, {})
        ok        = scrape_result.get("ok", False)
        full_text = scrape_result.get("full_text", "") if ok else ""
        image_url = image_rss or scrape_result.get("image_url")

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
            scraped=ok,
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
