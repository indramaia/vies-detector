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

from datetime import datetime, timedelta, timezone

from loguru import logger
from collector import fetch_all_feeds, Deduplicator, ArticleData
from classifier import SentenceClassifier
from aggregation import compute_article_bias, aggregate_by_vehicle, ArticleBiasResult
from ideological import contextualize_all, get_spectrum_summary

from sqlalchemy import func

from scripts.setup_db import (
    get_session,
    ArticleRecord,
    SentenceRecord,
    VehicleIndexRecord,
    HomeSummaryRecord,
)


# ══════════════════════════════════════════════════════════════════════════════
#  TASKS — cada camada é uma task Prefect com retry automático
# ══════════════════════════════════════════════════════════════════════════════

def task_collect(existing_hashes: set) -> list[ArticleData]:
    """Camada 1: coleta RSS com deduplicação baseada no banco de dados.

    Recebe os hashes já como conjunto (a sessão DB deve estar fechada antes
    de chamar esta função — fetch_all_feeds leva vários minutos e uma conexão
    aberta durante esse tempo é morta pelo Neon por inatividade).
    """
    dedup    = Deduplicator(backend=existing_hashes)
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
        total_extracted = len(art.sentences)
        to_classify     = art.sentences[:_MAX_SENTENCES_CLASSIFY]
        sentence_results = clf.classify_batch(to_classify)
        bias_result = compute_article_bias(
            url_hash=art.url_hash,
            source_name=art.source_name,
            ideology_id=art.ideology_id,
            sentence_results=sentence_results,
        )
        results.append(bias_result)
        logger.debug(
            f"[{art.source_name}] '{art.title[:60]}' | "
            f"extraídas={total_extracted} classificadas={len(to_classify)} "
            f"(skipped={total_extracted - len(to_classify)}) | "
            f"BiasScore={bias_result.bias_score:.2f} ({bias_result.interpretation})"
        )

    logger.info(f"Artigos classificados: {len(results)}")
    return results


_MAX_SENTENCES_CLASSIFY = 100 # primeiras N sentenças por artigo — jornalismo concentra viés no lide

_RETENTION_DAYS = 45  # window_days (30) + margem — nada no app consulta artigos mais antigos que isso


def task_purge_old_data(db_session, retention_days: int = _RETENTION_DAYS) -> None:
    """Remove artigos e sentenças mais antigos que retention_days.

    Sem isso o banco cresce sem limite: sentences guarda o texto de cada
    sentença classificada para sempre, e é o que domina o tamanho do banco.
    O Neon free tier tem teto de 512 MB, e nenhuma consulta do app (agregação,
    API) olha além de window_days — então dados antigos não têm função.
    Roda antes da coleta, na mesma sessão que lê os hashes existentes.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    old_hashes = db_session.query(ArticleRecord.url_hash).filter(
        ArticleRecord.published_at < cutoff
    )
    n_sentences = (
        db_session.query(SentenceRecord)
        .filter(SentenceRecord.url_hash.in_(old_hashes))
        .delete(synchronize_session=False)
    )
    n_articles = (
        db_session.query(ArticleRecord)
        .filter(ArticleRecord.published_at < cutoff)
        .delete(synchronize_session=False)
    )
    db_session.commit()
    logger.info(
        f"Retenção ({retention_days}d) — removidos {n_articles} artigos e "
        f"{n_sentences} sentenças anteriores a {cutoff.date()}."
    )


def _clean(value: str | None) -> str | None:
    """Remove bytes nulos (\x00) que o PostgreSQL rejeita em string literals."""
    if value is None:
        return None
    return value.replace("\x00", "")


def task_persist(
    articles: list[ArticleData],
    bias_results: list[ArticleBiasResult],
    db_session,
) -> tuple[int, int]:
    """Persiste metadados, resultados de artigos e sentenças no banco.

    bulk_insert_mappings envia todos os registros em dois round-trips ao Neon
    (artigos + sentenças), eliminando os ~10.000 round-trips individuais do ORM.

    Retorna (artigos_inseridos, sentenças_inseridas) — usado por
    task_update_home_summary para manter o contador cumulativo, já que
    a retenção (task_purge_old_data) remove linhas da tabela ao longo do tempo.
    """
    bias_map = {r.url_hash: r for r in bias_results}

    article_rows: list[dict] = []
    sentence_rows: list[dict] = []

    for art in articles:
        bias = bias_map.get(art.url_hash)

        article_rows.append(dict(
            url_hash=art.url_hash,
            url=_clean(art.url),
            title=_clean(art.title),
            source_name=art.source_name,
            ideology_id=art.ideology_id,
            published_at=art.published_at,
            collected_at=art.collected_at,
            snippet=_clean(art.snippet),
            sentence_count=art.sentence_count,
            bias_score=bias.bias_score if bias else None,
            bias_interpretation=bias.interpretation if bias else None,
            n_factual=bias.n_factual if bias else None,
            n_biased=bias.n_biased if bias else None,
            n_strongly_biased=bias.n_strongly_biased if bias else None,
            image_url=_clean(art.image_url),
        ))

        if bias:
            for sr in bias.sentence_results:
                sentence_rows.append(dict(
                    url_hash=art.url_hash,
                    sentence=_clean(sr.sentence),
                    label=sr.label,
                    label_id=sr.label_id,
                    confidence=sr.confidence,
                    score_factual=sr.scores.get("factual", 0.0),
                    score_biased=sr.scores.get("enviesada", 0.0),
                    score_strongly_biased=sr.scores.get("fortemente_enviesada", 0.0),
                ))

    if article_rows:
        db_session.bulk_insert_mappings(ArticleRecord, article_rows)
    if sentence_rows:
        db_session.bulk_insert_mappings(SentenceRecord, sentence_rows)

    db_session.commit()
    logger.info(f"Persistidos {len(article_rows)} artigos e {len(sentence_rows)} sentenças no banco.")
    return len(article_rows), len(sentence_rows)


def task_aggregate_contextualize(
    bias_results: list[ArticleBiasResult],
    db_session,
    window_days: int = 30,
) -> None:
    """Camadas 3 + 4: agrega por veículo e salva índice com contexto ideológico.

    Consulta o banco para TODOS os artigos na janela de window_days dias —
    não apenas os da corrida atual — para que mean_bias reflita o histórico
    completo de 30 dias, e não só os artigos recém-coletados.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    window_records = (
        db_session.query(ArticleRecord)
        .filter(
            ArticleRecord.published_at >= cutoff,
            ArticleRecord.bias_score.isnot(None),
        )
        .all()
    )

    if not window_records:
        logger.info("Nenhum artigo com bias_score na janela de %d dias.", window_days)
        return

    logger.info(
        f"Agregando {len(window_records)} artigos dos últimos {window_days} dias "
        f"(corrida atual contribuiu com {len(bias_results)})."
    )

    window_results = [
        ArticleBiasResult(
            url_hash=rec.url_hash,
            source_name=rec.source_name,
            ideology_id=rec.ideology_id,
            bias_score=rec.bias_score,
            interpretation=rec.bias_interpretation or "",
            sentence_count=rec.sentence_count or 0,
            n_factual=rec.n_factual or 0,
            n_biased=rec.n_biased or 0,
            n_strongly_biased=rec.n_strongly_biased or 0,
            sentence_results=[],
        )
        for rec in window_records
    ]

    vehicle_indices = aggregate_by_vehicle(window_results, window_days=window_days)
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


# Fique ok > baixa o app, liga o 0800, whatsapp. 
# nutricionista, psicólogo, 

def task_update_home_summary(db_session, new_articles: int = 0, new_sentences: int = 0) -> None:
    """Atualiza os totais cumulativos da homepage (upsert id=1).

    total_articles/total_sentences são um CONTADOR CUMULATIVO — soma new_articles/
    new_sentences ao valor já gravado, em vez de recalcular via COUNT(*) na tabela.
    Isso é proposital: desde que task_purge_old_data passou a apagar dados com
    mais de _RETENTION_DAYS dias, COUNT(*) refletiria só a janela viva, e esse
    número aparece no frontend como "sentenças já classificadas" — uma métrica
    de vitrine que não pode regredir a cada purga.

    total_vehicles usa max() pelo mesmo motivo: não deve cair só porque um
    veículo ficou temporariamente fora da janela de retenção.
    """
    prior = db_session.get(HomeSummaryRecord, 1)
    prior_articles  = prior.total_articles if prior else 0
    prior_sentences = prior.total_sentences if prior else 0
    prior_vehicles  = prior.total_vehicles if prior else 0

    total_articles  = prior_articles + new_articles
    total_sentences = prior_sentences + new_sentences
    live_vehicles   = db_session.query(
        func.count(func.distinct(ArticleRecord.ideology_id))
    ).scalar() or 0
    total_vehicles = max(prior_vehicles, live_vehicles)
    last_updated = db_session.query(func.max(ArticleRecord.published_at)).scalar()

    db_session.merge(HomeSummaryRecord(
        id=1,
        total_articles=total_articles,
        total_sentences=total_sentences,
        total_vehicles=total_vehicles,
        last_updated=last_updated,
    ))
    db_session.commit()
    logger.info(
        f"home_summary atualizado (cumulativo): {total_articles} artigos | "
        f"{total_sentences} sentenças | {total_vehicles} veículos."
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
    # Sessão 1: purga dados antigos, busca hashes e fecha ANTES de iniciar a
    # coleta RSS. fetch_all_feeds leva vários minutos (scraping por artigo) —
    # manter a conexão aberta durante esse tempo causa SSL timeout no Neon.
    with get_session() as session:
        task_purge_old_data(session)
        existing_hashes = {row.url_hash for row in session.query(ArticleRecord.url_hash).all()}
    logger.info(f"Hashes já registrados no banco: {len(existing_hashes)}")

    articles = task_collect(existing_hashes)

    scraped_full  = sum(1 for a in articles if a.scraped)
    scrape_failed = len(articles) - scraped_full
    logger.info(
        f"Scraping — completo: {scraped_full} | "
        f"fallback (RSS/snippet): {scrape_failed} | "
        f"taxa: {scraped_full/len(articles)*100:.1f}%" if articles else "Scraping — 0 artigos."
    )

    # Resumo de scraping por veículo
    from collections import defaultdict
    vehicle_stats: dict[str, list] = defaultdict(list)
    for a in articles:
        vehicle_stats[a.source_name].append(a.scraped)
    logger.info("=== SCRAPING POR VEÍCULO ===")
    for source, flags in sorted(vehicle_stats.items()):
        ok = sum(flags)
        logger.info(f"  {source:30s} {ok:3d}/{len(flags):3d} ({ok/len(flags)*100:5.1f}%)")

    # Classificação sem nenhuma conexão aberta.
    bias_results = task_classify(articles)
    logger.info(
        f"Classificação — {len(bias_results)} artigos | "
        f"com corpo completo: {sum(1 for a in articles if a.scraped and a.url_hash in {r.url_hash for r in bias_results})} | "
        f"só snippet: {sum(1 for a in articles if not a.scraped and a.url_hash in {r.url_hash for r in bias_results})}"
    )

    # Sessão 2a: persiste artigos e sentenças (batch commits internos).
    with get_session() as session:
        new_articles, new_sentences = task_persist(articles, bias_results, session)

    # Sessão 2b: agrega índices por veículo (conexão fresca, rápido).
    with get_session() as session:
        task_aggregate_contextualize(bias_results, session, window_days)

    # Sessão 2c: soma os totais cumulativos da homepage (uma linha, leitura barata na API).
    with get_session() as session:
        task_update_home_summary(session, new_articles, new_sentences)


if __name__ == "__main__":
    run_pipeline()
