"""
scripts/analyze_single_url.py
──────────────────────────────
Analisa um único artigo por URL: scraping → classificação → persistência.
Executado pelo GitHub Actions workflow (analyze_url.yml) via workflow_dispatch.

Uso:
    python scripts/analyze_single_url.py --url <URL> --url_hash <SHA256>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

from scripts.setup_db import (
    get_session, init_db,
    ArticleRecord, SentenceRecord, OnDemandRequest,
)
from collector.article_scraper import scrape_article
from collector.preprocessor import clean_text, make_snippet, tokenize_sentences
from classifier import SentenceClassifier
from aggregation import compute_article_bias

_MAX_SENTENCES = 100  # consistente com pipeline/main_flow.py

# Mapa de domínio → (source_name, ideology_id) para as fontes monitoradas.
# Cobre as variações de subdomínio mais comuns dos portais brasileiros.
_DOMAIN_MAP: dict[str, tuple[str, str]] = {
    "folha.uol.com.br":         ("Folha de S.Paulo",          "folha"),
    "estadao.com.br":           ("O Estado de S. Paulo",       "estadao"),
    "oglobo.globo.com":         ("O Globo",                    "oglobo"),
    "g1.globo.com":             ("G1",                         "g1"),
    "uol.com.br":               ("UOL Notícias",               "uol"),
    "cnnbrasil.com.br":         ("CNN Brasil",                 "cnnbrasil"),
    "veja.abril.com.br":        ("Veja",                       "veja"),
    "agenciabrasil.ebc.com.br": ("Agência Brasil",             "agenciabrasil"),
    "noticias.r7.com":          ("R7",                         "r7"),
    "gazetadopovo.com.br":      ("Gazeta do Povo",             "gazetadopovo"),
    "cartacapital.com.br":      ("Carta Capital",              "cartacapital"),
    "metropoles.com":           ("Metrópoles",                 "metropoles"),
    "brasildefato.com.br":      ("Brasil de Fato",             "brasildefato"),
    "theintercept.com":         ("The Intercept Brasil",       "intercept"),
    "apublica.org":             ("Agência Pública",            "agenciapublica"),
    "diplomatique.org.br":      ("Le Monde Diplomatique",      "lemonde"),
    "outraspalavras.net":       ("Outras Palavras",            "outraspalavras"),
    "jovempan.com.br":          ("Jovem Pan News",             "jovempan"),
}


def detect_source(url: str) -> tuple[str, str]:
    """Retorna (source_name, ideology_id) para a URL, ou (domínio, 'unknown')."""
    host = urlparse(url).netloc.lower().removeprefix("www.")
    for domain, info in _DOMAIN_MAP.items():
        if host == domain or host.endswith("." + domain):
            return info
    return (host or "desconhecido", "unknown")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace("\x00", "")


def analyze_url(url: str, url_hash: str) -> dict:
    """
    Scrape, classifica e persiste um artigo avulso.

    Segue o mesmo fluxo do pipeline regular mas para uma única URL:
      1. scrape_article  → HTML → texto completo + título + imagem
      2. tokenize_sentences → sentenças para o classificador
      3. SentenceClassifier.classify_batch → scores por sentença
      4. compute_article_bias → BiasScore
      5. Persistência em articles + sentences + on_demand_requests
    """
    source_name, ideology_id = detect_source(url)
    logger.info(f"Fonte detectada: {source_name} ({ideology_id})")

    # ── Scraping ─────────────────────────────────────────────────────────────
    scraped   = scrape_article(url)
    full_text = scraped["full_text"] if scraped["ok"] else ""
    image_url = scraped.get("image_url")
    title     = scraped.get("title")

    if not scraped["ok"]:
        logger.warning(f"Scraping falhou: {scraped['reason']}")

    # ── Pré-processamento ─────────────────────────────────────────────────────
    clean   = clean_text(full_text) if full_text else ""
    snippet = make_snippet(clean) if clean else make_snippet(url)
    sentences = tokenize_sentences(clean) if clean else []

    if not sentences:
        logger.warning("Nenhuma sentença extraída — possível paywall ou JS-only.")

    # ── Classificação ─────────────────────────────────────────────────────────
    sentence_results = []
    if sentences:
        clf = SentenceClassifier()
        sentence_results = clf.classify_batch(sentences[:_MAX_SENTENCES])

    bias_result = compute_article_bias(
        url_hash=url_hash,
        source_name=source_name,
        ideology_id=ideology_id,
        sentence_results=sentence_results,
    )

    # ── Persistência ──────────────────────────────────────────────────────────
    with get_session() as session:
        existing = session.get(ArticleRecord, url_hash)
        if existing is None:
            session.bulk_insert_mappings(ArticleRecord, [dict(
                url_hash=url_hash,
                url=_clean(url),
                title=_clean(title),
                source_name=source_name,
                ideology_id=ideology_id,
                published_at=None,
                collected_at=datetime.now(timezone.utc),
                snippet=_clean(snippet),
                sentence_count=bias_result.sentence_count,
                bias_score=bias_result.bias_score,
                bias_interpretation=bias_result.interpretation,
                n_factual=bias_result.n_factual,
                n_biased=bias_result.n_biased,
                n_strongly_biased=bias_result.n_strongly_biased,
                image_url=_clean(image_url),
            )])

            if bias_result.sentence_results:
                session.bulk_insert_mappings(SentenceRecord, [
                    dict(
                        url_hash=url_hash,
                        sentence=_clean(sr.sentence),
                        label=sr.label,
                        label_id=sr.label_id,
                        confidence=sr.confidence,
                        score_factual=sr.scores.get("factual", 0.0),
                        score_biased=sr.scores.get("enviesada", 0.0),
                        score_strongly_biased=sr.scores.get("fortemente_enviesada", 0.0),
                    )
                    for sr in bias_result.sentence_results
                ])
        else:
            logger.info("Artigo já existia no banco — ignorando insert.")

        req = session.get(OnDemandRequest, url_hash)
        if req:
            req.status = "completed"
            req.completed_at = datetime.now(timezone.utc)

    logger.info(
        f"BiasScore={bias_result.bias_score:.4f} "
        f"({bias_result.interpretation}) | "
        f"sentenças={bias_result.sentence_count}"
    )
    return {
        "url_hash": url_hash,
        "source_name": source_name,
        "ideology_id": ideology_id,
        "bias_score": bias_result.bias_score,
        "interpretation": bias_result.interpretation,
        "sentence_count": bias_result.sentence_count,
        "n_factual": bias_result.n_factual,
        "n_biased": bias_result.n_biased,
        "n_strongly_biased": bias_result.n_strongly_biased,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisa um artigo avulso por URL")
    parser.add_argument("--url",      required=True, help="URL do artigo")
    parser.add_argument("--url_hash", required=True, help="SHA-256 hex da URL")
    args = parser.parse_args()

    logger.info("Inicializando banco de dados…")
    init_db()

    # Marca como em processamento antes de começar
    with get_session() as session:
        req = session.get(OnDemandRequest, args.url_hash)
        if req and req.status == "pending":
            req.status = "processing"

    try:
        analyze_url(args.url, args.url_hash)
        logger.info("Análise avulsa concluída com sucesso.")
    except Exception:
        logger.exception("Análise avulsa falhou")
        with get_session() as session:
            req = session.get(OnDemandRequest, args.url_hash)
            if req:
                req.status = "failed"
                req.error = "Erro interno durante análise — consulte os logs do workflow."
        sys.exit(1)


if __name__ == "__main__":
    main()
