"""
api/app.py
──────────
REST API Flask que expõe os resultados do pipeline.

Endpoints:
    GET /api/vehicles                  → índices de todos os veículos
    GET /api/vehicles/<ideology_id>    → índice de um veículo específico
    GET /api/spectrum                  → resumo do espectro ideológico
    GET /api/articles?source=<id>      → artigos recentes de um veículo
    GET /api/stories                   → stories multi-veículo agrupadas por TF-IDF
    GET /api/stats                     → totais para landing page
    GET /api/health                    → status da API + diagnóstico de cache

Todos os endpoints retornam JSON com cabeçalho CORS.

Resiliência a cold start do Render free tier
────────────────────────────────────────────
Render adormece o servidor após 15 min de inatividade. O cold start leva
30-90 s, excedendo o timeout do frontend Lovable (12 s). Três camadas de
proteção implementadas:

  1. Cache TTL in-memory (por worker gunicorn)
       vehicles / stats / spectrum : TTL 15 min
       stories                     : TTL  5 min
       articles por fonte          : TTL  5 min
     → Após a primeira requisição bem-sucedida, as seguintes são servidas
       em memória sem tocar o banco Neon.

  2. Fallback estático de ideological_references.json
     → Se o DB estiver vazio (pipeline ainda não rodou), /api/vehicles e
       /api/spectrum retornam os veículos com posicionamento ideológico e
       article_count=0 em vez de lista vazia.

  3. Stale-on-error
     → Se o DB lançar exceção (SSL timeout Neon, cold start), qualquer
       entrada expirada do cache é servida em vez de retornar erro/vazio.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from threading import RLock

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from loguru import logger

from sqlalchemy import func
from scripts.setup_db import get_session, VehicleIndexRecord, ArticleRecord, SentenceRecord
from ideological import get_spectrum_summary, contextualize_all
from aggregation import VehicleIndex

app = Flask(__name__)

ALLOWED_ORIGINS = [
    "https://biasradar.lovable.app",
    "http://localhost:3000",
    "http://localhost:5173",
    r"https://.*\.lovable\.app",
]

CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})


# ── Cache TTL in-memory (por worker gunicorn) ─────────────────────────────────
# Cada worker mantém seu próprio dicionário. Gunicorn com 2 workers significa
# que o warm-up ocorre na primeira requisição de cada worker — aceitável.

_CACHE: dict[str, tuple[float, object]] = {}  # key → (expires_monotonic, payload)
_CACHE_LOCK = RLock()

_TTL_VEHICLES = 900   # 15 min  — pipeline roda a cada 12 h; dados mudam pouco
_TTL_STORIES  = 300   #  5 min  — artigos novos chegam com frequência
_TTL_STATS    = 900   # 15 min
_TTL_SPECTRUM = 900   # 15 min
_TTL_ARTICLES = 300   #  5 min


def _cache_get(key: str):
    """Retorna payload se ainda dentro do TTL, None caso contrário."""
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry and time.monotonic() < entry[0]:
            return entry[1]
    return None


def _cache_set(key: str, payload, ttl: int) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic() + ttl, payload)


def _cache_stale(key: str):
    """Retorna qualquer entrada existente (expirada ou não) — último recurso."""
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        return entry[1] if entry else None


def _cache_keys_status() -> dict[str, str]:
    now = time.monotonic()
    with _CACHE_LOCK:
        return {k: ("valid" if now < v[0] else "stale") for k, v in _CACHE.items()}


# ── Fallback estático ─────────────────────────────────────────────────────────

def _fallback_vehicles() -> list[dict]:
    """
    Quando o DB está vazio ou inacessível, constrói a lista de veículos
    a partir de ideological_references.json — sem dados de viés, mas com
    posicionamento ideológico — para que o frontend nunca fique zerado.
    """
    try:
        from ideological.reference_map import load_reference_map
        profiles = load_reference_map()
        return [
            {
                "ideology_id": pid,
                "source_name": p.name,
                "computed_at": None,
                "window_days": 30,
                "article_count": 0,
                "mean_bias": None,
                "ideology_score": p.ideology_score,
                "uncertainty": p.uncertainty,
                "position_label": p.position_label,
                "contextualization": p.description,
                "caveat": (
                    "⚠️  Dados de viés ainda não disponíveis — pipeline em execução."
                ),
            }
            for pid, p in profiles.items()
        ]
    except Exception:
        logger.exception("Erro ao carregar fallback de veículos")
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vehicle_index_to_dict(rec: VehicleIndexRecord) -> dict:
    return {
        "ideology_id": rec.ideology_id,
        "source_name": rec.source_name,
        "computed_at": (
            rec.computed_at.isoformat()
            if hasattr(rec.computed_at, "isoformat")
            else rec.computed_at
        ) if rec.computed_at else None,
        "window_days": rec.window_days,
        "article_count": rec.article_count,
        "mean_bias": rec.mean_bias,
        "ideology_score": rec.ideology_score,
        "uncertainty": rec.uncertainty,
        "position_label": rec.position_label,
        "contextualization": rec.contextualization,
        "caveat": (
            "⚠️  O BiasScore é uma estimativa probabilística. "
            "Consulte a documentação para limitações metodológicas."
        ),
    }


def _article_to_dict(rec: ArticleRecord) -> dict:
    return {
        "url_hash": rec.url_hash,
        "title": rec.title,
        "url": rec.url,
        "source_name": rec.source_name,
        "published_at": (
            rec.published_at.isoformat()
            if hasattr(rec.published_at, "isoformat")
            else rec.published_at
        ) if rec.published_at else None,
        "image_url": rec.image_url,
        "bias_score": rec.bias_score,
        "bias_interpretation": rec.bias_interpretation,
        "sentence_count": rec.sentence_count,
        "n_factual": rec.n_factual,
        "n_biased": rec.n_biased,
        "n_strongly_biased": rec.n_strongly_biased,
    }


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": str(e.description)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e.description)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Erro interno do servidor."}), 500

@app.errorhandler(503)
def service_unavailable(e):
    return jsonify({"error": str(e.description)}), 503


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Health check com diagnóstico de cache — não toca no banco."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache": _cache_keys_status(),
    })


@app.get("/api/stats")
def stats():
    """Retorna totais gerais para a landing page."""
    cached = _cache_get("stats")
    if cached is not None:
        return jsonify(cached)

    try:
        with get_session() as session:
            total_articles  = session.query(func.count(ArticleRecord.url_hash)).scalar() or 0
            total_sentences = session.query(func.count(SentenceRecord.id)).scalar() or 0
            total_vehicles  = session.query(
                func.count(func.distinct(ArticleRecord.ideology_id))
            ).scalar() or 0
            last_updated = session.query(func.max(ArticleRecord.published_at)).scalar()

        data = {
            "total_articles":  total_articles,
            "total_sentences": total_sentences,
            "total_vehicles":  total_vehicles,
            "last_updated": (
                last_updated.isoformat()
                if hasattr(last_updated, "isoformat")
                else last_updated
            ) if last_updated else None,
        }
    except Exception:
        logger.exception("DB error em /api/stats — tentando cache stale")
        stale = _cache_stale("stats")
        if stale:
            return jsonify(stale)
        return jsonify({
            "total_articles": 0, "total_sentences": 0,
            "total_vehicles": 0, "last_updated": None,
        })

    _cache_set("stats", data, _TTL_STATS)
    return jsonify(data)


@app.get("/api/vehicles")
def list_vehicles():
    """Retorna o índice editorial de todos os veículos monitorados."""
    cached = _cache_get("vehicles")
    if cached is not None:
        return jsonify(cached)

    try:
        with get_session() as session:
            records = session.query(VehicleIndexRecord).all()
            data = [_vehicle_index_to_dict(r) for r in records]
    except Exception:
        logger.exception("DB error em /api/vehicles — servindo fallback")
        return jsonify(_cache_stale("vehicles") or _fallback_vehicles())

    # DB vazio antes do primeiro pipeline: usa referências estáticas
    if not data:
        data = _fallback_vehicles()

    if data:
        _cache_set("vehicles", data, _TTL_VEHICLES)

    return jsonify(data)


@app.get("/api/vehicles/<ideology_id>")
def get_vehicle(ideology_id: str):
    """Retorna o índice editorial de um veículo específico."""
    # Aproveita cache agregado antes de ir ao DB
    cached_all = _cache_get("vehicles") or _cache_stale("vehicles")
    if cached_all:
        match = next((v for v in cached_all if v["ideology_id"] == ideology_id), None)
        if match:
            return jsonify(match)

    try:
        with get_session() as session:
            rec = session.get(VehicleIndexRecord, ideology_id)

        if rec is None:
            # Tenta fallback estático antes de 404
            from ideological.reference_map import load_reference_map
            profiles = load_reference_map()
            if ideology_id not in profiles:
                abort(404, description=f"Veículo '{ideology_id}' não encontrado.")
            p = profiles[ideology_id]
            return jsonify({
                "ideology_id":      ideology_id,
                "source_name":      p.name,
                "computed_at":      None,
                "window_days":      30,
                "article_count":    0,
                "mean_bias":        None,
                "ideology_score":   p.ideology_score,
                "uncertainty":      p.uncertainty,
                "position_label":   p.position_label,
                "contextualization": p.description,
                "caveat": "⚠️  Dados de viés ainda não disponíveis — pipeline em execução.",
            })

        return jsonify(_vehicle_index_to_dict(rec))

    except Exception:
        logger.exception(f"DB error em /api/vehicles/{ideology_id}")
        abort(503, description="Serviço temporariamente indisponível.")


@app.get("/api/spectrum")
def spectrum():
    """
    Retorna os veículos ordenados no espectro ideológico,
    do mais progressista ao mais conservador.
    """
    cached = _cache_get("spectrum")
    if cached is not None:
        return jsonify(cached)

    try:
        with get_session() as session:
            records = session.query(VehicleIndexRecord).all()
    except Exception:
        logger.exception("DB error em /api/spectrum")
        return jsonify(_cache_stale("spectrum") or [])

    from aggregation.window_aggregator import VehicleIndex
    now = datetime.now(timezone.utc)

    vehicle_indices = {}
    for r in records:
        vi = VehicleIndex(
            source_name=r.source_name,
            ideology_id=r.ideology_id,
            window_days=r.window_days,
            reference_date=r.computed_at or now,
            article_count=r.article_count,
            mean_bias=r.mean_bias,
            median_bias=r.mean_bias,
            std_bias=0.0,
            min_bias=0.0,
            max_bias=2.0,
            trend=None,
            window_start=now,
            window_end=now,
        )
        vehicle_indices[r.ideology_id] = vi

    contexts = contextualize_all(vehicle_indices)
    data = get_spectrum_summary(contexts)

    if data:
        _cache_set("spectrum", data, _TTL_SPECTRUM)

    return jsonify(data)


@app.get("/api/articles/<url_hash>/similar")
def similar_articles(url_hash: str):
    """
    Dado um artigo (url_hash), retorna artigos similares de outros veículos
    sobre o mesmo assunto, ordenados por similaridade TF-IDF decrescente.

    Usado pelo frontend para exibir "mesma notícia em outros veículos"
    ao clicar num artigo.

    Query params:
        limit     : max artigos retornados (padrão: 10, máx: 30)
        threshold : similaridade cosine mínima (padrão: 0.15)
        hours     : janela de busca em horas (padrão: 168 = 7 dias)
    """
    from aggregation.topic_clusterer import find_similar
    from datetime import timedelta

    limit     = min(int(request.args.get("limit",     10)),   30)
    threshold = float(request.args.get("threshold",   0.15))
    hours     = min(int(request.args.get("hours",    168)),  720)

    cache_key = f"similar:{url_hash}:{limit}:{threshold}:{hours}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with get_session() as session:
            anchor = session.get(ArticleRecord, url_hash)
            if anchor is None:
                abort(404, description=f"Artigo '{url_hash}' não encontrado.")

            candidates = (
                session.query(ArticleRecord)
                .filter(
                    ArticleRecord.published_at >= cutoff,
                    ArticleRecord.url_hash != url_hash,
                )
                .order_by(ArticleRecord.published_at.desc())
                .limit(300)
                .all()
            )
            result = find_similar(anchor, candidates, threshold=threshold, limit=limit)

        if result:
            _cache_set(cache_key, result, ttl=300)
        return jsonify(result)

    except Exception as exc:
        logger.exception(f"Erro em /api/articles/{url_hash}/similar")
        return jsonify(_cache_stale(cache_key) or [])


@app.get("/api/articles")
def list_articles():
    """
    Retorna artigos recentes de um veículo.

    Query params:
        source  : ideology_id do veículo (obrigatório)
        limit   : número máximo de artigos (padrão: 20, máx: 100)
    """
    ideology_id = request.args.get("source")
    if not ideology_id:
        abort(400, description="Parâmetro 'source' obrigatório.")

    limit = min(int(request.args.get("limit", 20)), 100)
    cache_key = f"articles:{ideology_id}:{limit}"

    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)

    try:
        with get_session() as session:
            records = (
                session.query(ArticleRecord)
                .filter(ArticleRecord.ideology_id == ideology_id)
                .order_by(ArticleRecord.published_at.desc())
                .limit(limit)
                .all()
            )
            data = [_article_to_dict(r) for r in records]
    except Exception:
        logger.exception(f"DB error em /api/articles?source={ideology_id}")
        return jsonify(_cache_stale(cache_key) or [])

    if data:
        _cache_set(cache_key, data, _TTL_ARTICLES)

    return jsonify(data)


@app.get("/api/stories")
def stories():
    """
    Agrupa artigos recentes de diferentes veículos sobre o mesmo assunto.

    Query params:
        hours      : janela temporal em horas (padrão: 48, máx: 168)
        limit      : número máximo de stories (padrão: 20, máx: 50)
        threshold  : similaridade TF-IDF mínima, float 0-1 (padrão: 0.25)
        min_sources: mínimo de veículos distintos por story (padrão: 2)

    Retorna SEMPRE JSON válido — nunca 500 — para que o frontend
    possa usar fallback sem tratar exceções de rede distintas.
    """
    from aggregation.topic_clusterer import cluster_articles
    from datetime import timedelta

    try:
        hours     = min(int(request.args.get("hours",    48)), 168)
        limit     = min(int(request.args.get("limit",    20)),  50)
        threshold = float(request.args.get("threshold",  0.25))
        min_src   = int(request.args.get("min_sources",  2))
    except (ValueError, TypeError) as exc:
        return jsonify({"stories": [], "ok": False, "error": f"Parâmetro inválido: {exc}"}), 400

    cache_key = f"stories:{hours}:{limit}:{threshold}:{min_src}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)

    try:
        # Limite de artigos carregados em RAM para o TF-IDF.
        # Render free tier tem 512 MB; acima de ~300 artigos com bigrams → OOM.
        _MAX_ARTICLES_TFIDF = 300

        with get_session() as session:
            # Tenta a janela solicitada; se vazia, expande progressivamente até
            # encontrar artigos no DB — garante que dados históricos nunca sejam
            # descartados só porque o pipeline não rodou recentemente.
            records = []
            effective_hours = hours
            for candidate_hours in [hours, 168, 720, None]:  # 48h → 7d → 30d → sem limite
                if candidate_hours is not None and candidate_hours < hours:
                    continue  # não encolhe a janela abaixo do pedido pelo cliente
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(hours=candidate_hours)
                    if candidate_hours is not None else None
                )
                q = (
                    session.query(ArticleRecord)
                    .order_by(ArticleRecord.published_at.desc())
                    .limit(_MAX_ARTICLES_TFIDF)
                )
                if cutoff is not None:
                    q = (
                        session.query(ArticleRecord)
                        .filter(ArticleRecord.published_at >= cutoff)
                        .order_by(ArticleRecord.published_at.desc())
                        .limit(_MAX_ARTICLES_TFIDF)
                    )
                records = q.all()
                effective_hours = candidate_hours
                if records:
                    break

            result = cluster_articles(
                records,
                similarity_threshold=threshold,
                min_sources=min_src,
                max_stories=limit,
            )

        payload = {
            "stories": result,
            "ok": True,
            "error": None,
            "effective_hours": effective_hours,  # informa o frontend qual janela foi usada
        }
        if result:
            _cache_set(cache_key, payload, _TTL_STORIES)
        return jsonify(payload)

    except Exception as exc:
        logger.exception("Erro em /api/stories")
        stale = _cache_stale(cache_key)
        if stale:
            return jsonify(stale)
        return jsonify({"stories": [], "ok": False, "error": str(exc)}), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host  = os.getenv("API_HOST",  "0.0.0.0")
    port  = int(os.getenv("API_PORT", 5000))
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    logger.info(f"API iniciando em http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
