"""
collector/article_scraper.py
────────────────────────────
Extrai o corpo completo de artigos jornalísticos a partir do HTML da página.

Estratégia em cascata:
    1. <article> semântico
    2. Atributos itemprop / data-component / role
    3. Seletores CSS comuns de portais brasileiros
    4. Todos os <p> com ≥ MIN_PARAGRAPH_CHARS (fallback genérico)

O texto completo é retornado APENAS em memória — nunca persistido.
O banco armazena somente snippet ≤ MAX_SNIPPET_CHARS (conformidade LGPD).
"""

from __future__ import annotations

from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

SCRAPE_TIMEOUT   = 10   # segundos por requisição
MIN_PARA_CHARS   = 40   # <p> com menos caracteres são descartados (artefatos)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

# Seletores em ordem de prioridade — mais semântico → mais genérico.
# Cobre os principais portais brasileiros monitorados.
_BODY_SELECTORS = [
    "article",
    "[itemprop='articleBody']",
    "[data-component='article-body']",
    "[role='main']",
    # portais nacionais
    ".article-body",
    ".article__body",
    ".article-content",
    ".article__content",
    ".article__text",
    ".content-text",           # G1
    ".c-news__body",           # Folha
    ".materia-conteudo",       # portais Globo legados
    ".news-full-body",
    ".post-content",
    ".entry-content",          # WordPress genérico (CartaCapital, etc.)
    ".td-post-content",
    "#article-body",
    "#materia",
    "main",
]


def _extract_body(soup: BeautifulSoup) -> str:
    """
    Percorre os seletores em cascata e retorna o texto dos parágrafos
    do primeiro container que tiver conteúdo suficiente.
    """
    for selector in _BODY_SELECTORS:
        container = soup.select_one(selector)
        if not container:
            continue
        paragraphs = [
            p.get_text(separator=" ", strip=True)
            for p in container.find_all("p")
            if len(p.get_text(strip=True)) >= MIN_PARA_CHARS
        ]
        if len(paragraphs) >= 2:          # pelo menos 2 parágrafos úteis
            return " ".join(paragraphs)

    # Fallback: todos os <p> da página com conteúdo mínimo
    paragraphs = [
        p.get_text(separator=" ", strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) >= MIN_PARA_CHARS
    ]
    return " ".join(paragraphs)


def _extract_image(soup: BeautifulSoup, base_url: str) -> str | None:
    """Extrai a imagem destacada do artigo (og:image → twitter:image → <img>)."""
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return urljoin(base_url, og["content"])

    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if tw and tw.get("content"):
        return urljoin(base_url, tw["content"])

    container = soup.find("article") or soup
    img = container.find("img")
    if img and img.get("src"):
        return urljoin(base_url, img["src"])

    return None


def scrape_article(url: str) -> dict:
    """
    Faz GET no URL do artigo e extrai corpo completo + imagem numa única
    requisição — evitando o segundo GET que o rss_fetcher fazia só para imagem.

    Returns:
        {
            "full_text" : str,        # corpo completo — NÃO persistir
            "image_url" : str | None,
            "ok"        : bool,       # False se a requisição falhou
        }

    O chamador é responsável por descartar `full_text` após a classificação.
    """
    try:
        resp = requests.get(
            url,
            timeout=SCRAPE_TIMEOUT,
            headers=_HEADERS,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.debug(f"Scrape falhou [{url}]: {exc}")
        return {"full_text": "", "image_url": None, "ok": False}

    try:
        soup      = BeautifulSoup(resp.text, "html.parser")
        full_text = _extract_body(soup)
        image_url = _extract_image(soup, url)
        ok        = bool(full_text)
        if not ok:
            logger.debug(f"Scrape sem corpo extraível [{url}]")
        return {"full_text": full_text, "image_url": image_url, "ok": ok}
    except Exception as exc:
        logger.debug(f"Parse HTML falhou [{url}]: {exc}")
        return {"full_text": "", "image_url": None, "ok": False}
