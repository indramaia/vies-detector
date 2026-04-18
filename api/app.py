"""
api/app.py
──────────
REST API Flask que expõe os resultados do pipeline.

Endpoints:
    GET /api/vehicles                  → índices de todos os veículos
    GET /api/vehicles/<ideology_id>    → índice de um veículo específico
    GET /api/spectrum                  → resumo do espectro ideológico
    GET /api/articles?source=<id>      → artigos recentes de um veículo
    GET /api/health                    → status da API

Todos os endpoints retornam JSON com cabeçalho CORS.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from loguru import logger

from scripts.setup_db import get_session, VehicleIndexRecord, ArticleRecord
from ideological import get_spectrum_summary, contextualize_all
from aggregation import VehicleIndex

app = Flask(__name__)

_cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS(app, origins=[
    "https://biasradar.lovable.app",
    "https://id-preview--587bd150-dcce-40ab-aa7c-1c07eed8c13a.lovable.app",
    "http://localhost:*"
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vehicle_index_to_dict(rec: VehicleIndexRecord) -> dict:
    return {
        "ideology_id": rec.ideology_id,
        "source_name": rec.source_name,
        "computed_at": rec.computed_at.isoformat() if rec.computed_at else None,
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
        "published_at": rec.published_at.isoformat() if rec.published_at else None,
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})


@app.get("/api/vehicles")
def list_vehicles():
    """Retorna o índice editorial de todos os veículos monitorados."""
    with get_session() as session:
        records = session.query(VehicleIndexRecord).all()
        return jsonify([_vehicle_index_to_dict(r) for r in records])


@app.get("/api/vehicles/<ideology_id>")
def get_vehicle(ideology_id: str):
    """Retorna o índice editorial de um veículo específico."""
    with get_session() as session:
        rec = session.get(VehicleIndexRecord, ideology_id)
        if rec is None:
            abort(404, description=f"Veículo '{ideology_id}' não encontrado.")
        return jsonify(_vehicle_index_to_dict(rec))


@app.get("/api/spectrum")
def spectrum():
    """
    Retorna os veículos ordenados no espectro ideológico,
    do mais progressista ao mais conservador.
    """
    with get_session() as session:
        records = session.query(VehicleIndexRecord).all()

    # Reconstrói VehicleIndex para reutilizar get_spectrum_summary
    from aggregation.window_aggregator import VehicleIndex
    from datetime import datetime, timezone
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
            median_bias=r.mean_bias,   # Simplificação: mediana não persiste separadamente
            std_bias=0.0,
            min_bias=0.0,
            max_bias=2.0,
            trend=None,
            window_start=now,
            window_end=now,
        )
        vehicle_indices[r.ideology_id] = vi

    contexts = contextualize_all(vehicle_indices)
    summary = get_spectrum_summary(contexts)
    return jsonify(summary)


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

    with get_session() as session:
        records = (
            session.query(ArticleRecord)
            .filter(ArticleRecord.ideology_id == ideology_id)
            .order_by(ArticleRecord.published_at.desc())
            .limit(limit)
            .all()
        )
        return jsonify([_article_to_dict(r) for r in records])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 5000))
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    logger.info(f"API iniciando em http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
