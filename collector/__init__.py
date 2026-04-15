"""
collector
─────────
Camada 1 — Coleta e pré-processamento de notícias via RSS.

Exporta a interface pública mínima para uso pelo pipeline.
"""

from .rss_fetcher import fetch_all_feeds, fetch_feed, ArticleData
from .deduplicator import Deduplicator, compute_hash
from .preprocessor import preprocess_article, clean_text, tokenize_sentences
from .sources import SOURCES, ACTIVE_SOURCES, NewsSource

__all__ = [
    "fetch_all_feeds",
    "fetch_feed",
    "ArticleData",
    "Deduplicator",
    "compute_hash",
    "preprocess_article",
    "clean_text",
    "tokenize_sentences",
    "SOURCES",
    "ACTIVE_SOURCES",
    "NewsSource",
]
