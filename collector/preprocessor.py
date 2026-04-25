"""
collector/preprocessor.py
─────────────────────────
Limpeza de texto e tokenização de sentenças para o classificador.

Etapas:
    1. Remoção de HTML residual (tags, entidades)
    2. Normalização de espaços e pontuação
    3. Truncamento para MAX_SNIPPET_CHARS (conformidade LGPD)
    4. Tokenização em sentenças via NLTK punkt (português)

Referência:
    LOPER, E.; BIRD, S. NLTK: The Natural Language Toolkit. 2002.
"""

from __future__ import annotations

import os
import re
import unicodedata
from html.parser import HTMLParser

import nltk
from loguru import logger

# Garante que o tokenizador de sentenças em pt-BR esteja disponível
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    logger.info("Baixando tokenizador NLTK punkt_tab…")
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


MAX_SNIPPET_CHARS: int = int(os.getenv("MAX_SNIPPET_CHARS", "500"))
MIN_SENTENCE_TOKENS: int = 5   # Sentenças muito curtas são descartadas
MIN_SENTENCE_CHARS: int = 20


class _HTMLStripper(HTMLParser):
    """Remove todas as tags HTML de uma string."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(text: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text)


def clean_text(raw: str) -> str:
    """Limpeza sem truncamento — para classificação do texto completo."""
    text = strip_html(raw)
    text = remove_urls(text)
    return normalize_whitespace(text)


def make_snippet(text: str) -> str:
    """Trunca para MAX_SNIPPET_CHARS (conformidade LGPD) para armazenamento."""
    return text[:MAX_SNIPPET_CHARS]


def tokenize_sentences(text: str) -> list[str]:
    """
    Divide o texto em sentenças usando o tokenizador NLTK para português.

    Sentenças com menos de MIN_SENTENCE_CHARS caracteres são descartadas,
    pois geralmente correspondem a artefatos de formatação ou chamadas
    editoriais sem conteúdo analítico relevante.
    """
    try:
        sentences = nltk.sent_tokenize(text, language="portuguese")
    except Exception as exc:
        logger.warning(f"NLTK falhou ({exc}), usando fallback por ponto final.")
        sentences = [s.strip() for s in text.split(".") if s.strip()]

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) >= MIN_SENTENCE_CHARS
    ]


def preprocess_article(rss_text: str, full_text: str = "") -> dict:
    """
    Pré-processa um artigo combinando texto RSS e corpo completo raspado.

    Args:
        rss_text  : texto extraído do feed RSS (summary/content)
        full_text : corpo completo raspado da página (opcional).
                    Usado para classificação — NUNCA persistido integralmente.

    Retorna:
        snippet        : trecho limpo (≤ MAX_SNIPPET_CHARS) — armazenado no DB
        sentences      : sentenças do texto completo — para o classificador
        sentence_count : número de sentenças
    """
    rss_clean  = clean_text(rss_text)
    snippet    = make_snippet(rss_clean)

    # Sentenças para classificação: texto completo se disponível, senão snippet
    classify_source = clean_text(full_text) if full_text.strip() else rss_clean
    sentences = tokenize_sentences(classify_source)

    return {
        "snippet": snippet,
        "sentences": sentences,
        "sentence_count": len(sentences),
    }
