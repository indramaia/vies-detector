"""
collector/deduplicator.py
─────────────────────────
Geração e verificação de hashes SHA-256 para deduplicação de artigos.

Princípio (SCULLEY et al., 2015):
    Cada artigo é identificado de forma única pelo hash SHA-256 de seu URL
    canônico (sem parâmetros de tracking). O sistema nunca reprocessa um
    artigo cujo hash já esteja registrado no banco de dados.

LGPD: nenhum conteúdo do artigo é persistido — apenas o hash derivado
do URL (metadado público) e os resultados da análise.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse


# Parâmetros de tracking que devem ser removidos antes do hash
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "origem", "from",
})


def canonicalize_url(url: str) -> str:
    """Remove parâmetros de tracking e normaliza o URL para hashing."""
    parsed = urlparse(url.strip())
    params = {
        k: v for k, v in parse_qs(parsed.query).items()
        if k.lower() not in _TRACKING_PARAMS
    }
    clean_query = urlencode(
        {k: v[0] for k, v in sorted(params.items())},
        doseq=False,
    )
    canonical = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/") or "/",
        parsed.params,
        clean_query,
        "",  # Remove fragment
    ))
    return canonical


def compute_hash(url: str) -> str:
    """Retorna o hash SHA-256 hexadecimal do URL canônico."""
    canonical = canonicalize_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Deduplicator:
    """
    Verifica e registra hashes de artigos já processados.

    Em produção, `seen_hashes` é persistido via SQLAlchemy (ver setup_db.py).
    Esta classe aceita qualquer objeto com métodos `contains(hash)` e
    `add(hash)` como backend, facilitando testes com sets em memória.
    """

    def __init__(self, backend=None):
        # Backend padrão: set em memória (adequado para testes)
        self._backend = backend if backend is not None else set()

    def is_duplicate(self, url: str) -> bool:
        h = compute_hash(url)
        if hasattr(self._backend, "contains"):
            return self._backend.contains(h)
        return h in self._backend

    def register(self, url: str) -> str:
        """Registra o URL e retorna seu hash."""
        h = compute_hash(url)
        if hasattr(self._backend, "add"):
            self._backend.add(h)
        return h
