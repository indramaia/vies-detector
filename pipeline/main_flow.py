"""
pipeline/main_flow.py
─────────────────────

Fluxo:
    1. [collector]    Coleta RSS de todos os veículos ativos
    2. [classifier]   Classifica sentenças de cada artigo
    3. [aggregation]  Calcula BiasScore por artigo e índice por veículo
    4. [ideological]  Gera contextualização ideológica

Referência:
    SCULLEY et al. Hidden Technical Debt in ML Systems (2015).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from loguru import logger
from collector import fetch_all_feeds, Deduplicator, ArticleData
from classifier import SentenceClassifier
from aggregation import compute_article_bias, aggregate_by_vehicle, ArticleBiasResult
from ideological import contextualize_all, get_spectrum_summary

from scripts.setup_db import (
    get_session,
    ArticleRecord,
    SentenceRecord,
    VehicleIndexRecord,
)


# ══════════════════════════════════════════════════════════════════════════════
#  TASKS — cada camada é uma task Prefect com retry automático
# ══════════════════════════════════════════════════════════════════════════════

def task_collect(db_session) -> list[ArticleData]:
    """Camada 1: coleta RSS com deduplicação baseada no banco de dados."""

    # Backend de deduplicação: hashes já presentes no banco
    existing_hashes = {row.url_hash for row in db_session.query(ArticleRecord.url_hash).all()}
    logger.info(f"Hashes já registrados no banco: {len(existing_hashes)}")

    dedup = Deduplicator(backend=existing_hashes)
    articles = fetch_all_feeds(dedup)
    logger.info(f"Artigos novos coletados: {len(articles)}")
    return articles


def task_classify(articles: list[ArticleData]) -> list[ArticleBiasResult]:
    """Camada 2 + 3: classifica sentenças e calcula BiasScore por artigo."""
    
    if not articles:
        logger.info("Nenhum artigo novo para classificar.")
        return []

    clf = SentenceClassifier()
    results: list[ArticleBiasResult] = []

    for art in articles:
        if not art.sentences:
            continue
        sentence_results = clf.classify_batch(art.sentences)
        bias_result = compute_article_bias(
            url_hash=art.url_hash,
            source_name=art.source_name,
            ideology_id=art.ideology_id,
            sentence_results=sentence_results,
        )
        results.append(bias_result)

    logger.info(f"Artigos classificados: {len(results)}")
    return results


def task_persist(
    articles: list[ArticleData],
    bias_results: list[ArticleBiasResult],
    db_session,
) -> None:
    """Persiste metadados, resultados de artigos e sentenças no banco."""

    bias_map = {r.url_hash: r for r in bias_results}

    for art in articles:
        bias = bias_map.get(art.url_hash)

        article_rec = ArticleRecord(
            url_hash=art.url_hash,
            url=art.url,
            title=art.title,
            source_name=art.source_name,
            ideology_id=art.ideology_id,
            published_at=art.published_at,
            collected_at=art.collected_at,
            snippet=art.snippet,
            sentence_count=art.sentence_count,
            bias_score=bias.bias_score if bias else None,
            bias_interpretation=bias.interpretation if bias else None,
            n_factual=bias.n_factual if bias else None,
            n_biased=bias.n_biased if bias else None,
            n_strongly_biased=bias.n_strongly_biased if bias else None,
        )
        db_session.add(article_rec)

        if bias:
            for sr in bias.sentence_results:
                sent_rec = SentenceRecord(
                    url_hash=art.url_hash,
                    sentence=sr.sentence,
                    label=sr.label,
                    label_id=sr.label_id,
                    confidence=sr.confidence,
                    score_factual=sr.scores.get("factual", 0.0),
                    score_biased=sr.scores.get("enviesada", 0.0),
                    score_strongly_biased=sr.scores.get("fortemente_enviesada", 0.0),
                )
                db_session.add(sent_rec)

    db_session.commit()
    logger.info(f"Persistidos {len(articles)} artigos no banco.")


def task_aggregate_contextualize(
    bias_results: list[ArticleBiasResult],
    db_session,
    window_days: int = 30,
) -> None:
    """Camadas 3 + 4: agrega por veículo e salva índice com contexto ideológico."""
  
    if not bias_results:
        logger.info("Nenhum resultado para agregar.")
        return

    vehicle_indices = aggregate_by_vehicle(bias_results, window_days=window_days)
    contexts = contextualize_all(vehicle_indices)
    spectrum = get_spectrum_summary(contexts)

    now = datetime.now(timezone.utc)
    for ctx in contexts.values():
        rec = VehicleIndexRecord(
            ideology_id=ctx.ideology_id,
            source_name=ctx.source_name,
            computed_at=now,
            window_days=window_days,
            article_count=ctx.article_count,
            mean_bias=ctx.bias_score,
            ideology_score=ctx.ideology_score,
            uncertainty=ctx.uncertainty,
            position_label=ctx.position_label,
            contextualization=ctx.contextualization,
        )
        db_session.merge(rec)  # upsert por ideology_id

    db_session.commit()
    logger.info(f"Índices atualizados para {len(contexts)} veículos.")

    logger.info("=== ESPECTRO ATUAL ===")
    for v in spectrum:
        logger.info(
            f"  {v['source_name']:25s} | "
            f"BiasScore={v['bias_score']:.2f} | "
            f"Ideologia={v['ideology_score']:+.2f} ({v['position_label']})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  FLOW PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(window_days: int = 30) -> None:
    """
    Pipeline completo: coleta → classifica → persiste → agrega → contextualiza.
    Pode ser agendado via Prefect:
        prefect deployment apply pipeline/deployment.yaml
    """
    with get_session() as session:
        articles = task_collect(session)
        bias_results = task_classify(articles)
        task_persist(articles, bias_results, session)
        task_aggregate_contextualize(bias_results, session, window_days)


if __name__ == "__main__":
    run_pipeline()
