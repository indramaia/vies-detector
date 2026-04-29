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
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import HTTPError, Timeout, TooManyRedirects
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
    ".content__article-body",  # CNN Brasil
    ".content-article__body",  # CNN Brasil (variante)
    ".article__content-text",  # CNN Brasil (variante 2)
    "main",
]

# Sinais textuais de paywall / login obrigatório
_PAYWALL_SIGNALS = [
    "assine agora", "assine já", "seja assinante",
    "conteúdo exclusivo para assinantes", "conteúdo para assinantes",
    "subscribe", "subscription required", "paywall", "premium content",
    "faça login", "fazer login", "login to read", "sign in to",
    "registre-se para ler", "cadastre-se para ler",
]


def _detect_no_body_reason(soup: BeautifulSoup) -> str:
    """Classifica o motivo de não ter extraído corpo do artigo."""
    text_lower = soup.get_text(" ", strip=True).lower()

    for signal in _PAYWALL_SIGNALS:
        if signal in text_lower:
            return f"paywall/login (\"{signal}\")"

    scripts    = soup.find_all("script")
    paragraphs = soup.find_all("p")
    if len(scripts) > 10 and len(paragraphs) < 3:
        return "SPA/JS-only (sem parágrafos, muitos scripts)"

    body      = soup.find("body")
    body_text = body.get_text(strip=True) if body else ""
    if len(body_text) < 200:
        return f"página quase vazia ({len(body_text)} chars)"

    return "nenhum seletor CSS correspondeu ao corpo"


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
            "reason"    : str | None, # motivo da falha (None se ok=True)
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

    except Timeout:
        reason = f"timeout (>{SCRAPE_TIMEOUT}s)"
        logger.debug(f"Scrape falhou [{url}]: {reason}")
        return {"full_text": "", "image_url": None, "ok": False, "reason": reason}

    except TooManyRedirects:
        reason = "too many redirects"
        logger.debug(f"Scrape falhou [{url}]: {reason}")
        return {"full_text": "", "image_url": None, "ok": False, "reason": reason}

    except ReqConnectionError as exc:
        reason = f"connection error: {type(exc).__name__}"
        logger.debug(f"Scrape falhou [{url}]: {reason}")
        return {"full_text": "", "image_url": None, "ok": False, "reason": reason}

    except HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status in (401, 403):
            reason = f"acesso negado ({status})"
        elif status == 404:
            reason = "não encontrado (404)"
        elif status == 429:
            reason = "rate limit (429)"
        elif isinstance(status, int) and status >= 500:
            reason = f"erro do servidor ({status})"
        else:
            reason = f"HTTP {status}"
        logger.debug(f"Scrape falhou [{url}]: {reason}")
        return {"full_text": "", "image_url": None, "ok": False, "reason": reason}

    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.debug(f"Scrape falhou [{url}]: {reason}")
        return {"full_text": "", "image_url": None, "ok": False, "reason": reason}

    try:
        soup      = BeautifulSoup(resp.text, "html.parser")
        full_text = _extract_body(soup)
        image_url = _extract_image(soup, url)

        if full_text:
            return {"full_text": full_text, "image_url": image_url, "ok": True, "reason": None}

        reason = _detect_no_body_reason(soup)
        logger.debug(f"Scrape sem corpo [{url}]: {reason}")
        return {"full_text": "", "image_url": image_url, "ok": False, "reason": reason}

    except Exception as exc:
        reason = f"parse HTML: {type(exc).__name__}"
        logger.debug(f"Parse HTML falhou [{url}]: {reason}")
        return {"full_text": "", "image_url": None, "ok": False, "reason": reason}
