"""
aggregation/topic_clusterer.py
──────────────────────────────
Agrupa artigos recentes de diferentes veículos sobre o mesmo evento/assunto.

Algoritmo:
    1. TF-IDF sobre título + snippet de cada artigo (janela: últimas N horas)
    2. Cosine similarity entre todos os pares
    3. Clustering greedy single-pass: cada artigo se junta ao cluster cujo
       centróide tem maior similaridade, se acima do threshold
    4. Filtra clusters com ≥ 2 fontes distintas (multi-perspectiva)
    5. Mapeia position_label (Esquerda/Centro/Direita) via VehicleIndexRecord

Referências:
    SALTON; MCGILL. Introduction to Modern Information Retrieval (1983).
    MANNING et al. Introduction to Information Retrieval (2008).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Stopwords PT-BR mínimas (sem dependência de NLTK) ────────────────────────
_PT_STOPWORDS = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
    "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como",
    "mas", "ao", "ele", "das", "à", "seu", "sua", "ou", "quando", "muito",
    "nos", "já", "eu", "também", "só", "pelo", "pela", "até", "isso",
    "ela", "entre", "depois", "sem", "mesmo", "aos", "ter", "seus", "suas",
    "foi", "há", "seja", "ser", "são", "está", "foram", "este", "essa",
    "sobre", "após", "durante", "segundo", "ainda", "vai", "pode", "não",
]

# ── Mapeamento de position_label para espectro simplificado (3 abas) ─────────
_SPECTRUM_MAP = {
    "Esquerda":        "Esquerda",
    "Centro-esquerda": "Esquerda",
    "Centro":          "Centro",
    "Centro-direita":  "Direita",
    "Direita":         "Direita",
}

# Carrega position_label de cada ideology_id a partir do JSON estático
_JSON_PATH = Path(__file__).parent.parent / "ideological" / "data" / "ideological_references.json"

def _load_position_map() -> dict[str, str]:
    data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    return {
        vid: _SPECTRUM_MAP.get(v.get("position_label", "Centro"), "Centro")
        for vid, v in data["vehicles"].items()
    }

_POSITION_MAP: dict[str, str] = _load_position_map()


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_articles(
    articles: list[Any],
    similarity_threshold: float = 0.25,
    min_sources: int = 2,
    max_stories: int = 30,
) -> list[dict]:
    """
    Recebe lista de ArticleRecord e retorna stories agrupadas.

    similarity_threshold: mínimo de similaridade TF-IDF para juntar artigos
                          0.20 = mais inclusivo | 0.35 = mais restritivo
    min_sources: número mínimo de veículos distintos para formar uma story
    max_stories: limite de stories retornadas (ordenadas por n_articles desc)
    """
    if not articles:
        return []

    # Texto para vetorização: título tem mais peso, repetido 2x
    texts = [
        f"{(a.title or '')} {(a.title or '')} {(a.snippet or '')}"
        for a in articles
    ]

    try:
        vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words=_PT_STOPWORDS,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    sim_matrix = cosine_similarity(tfidf)

    # Clustering greedy: cada artigo vai para o cluster mais similar
    n = len(articles)
    assigned = [-1] * n
    cluster_members: list[list[int]] = []

    for i in range(n):
        if assigned[i] != -1:
            continue
        # Inicia novo cluster com i
        cluster_idx = len(cluster_members)
        cluster_members.append([i])
        assigned[i] = cluster_idx

        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            # Similaridade média com todos os membros do cluster atual
            members = cluster_members[cluster_idx]
            avg_sim = float(np.mean([sim_matrix[i][k] for k in members]))
            if avg_sim >= similarity_threshold:
                cluster_members[cluster_idx].append(j)
                assigned[j] = cluster_idx

    # Constrói stories a partir dos clusters
    stories: list[dict] = []

    for members in cluster_members:
        cluster_arts = [articles[i] for i in members]

        # Filtra clusters com menos de min_sources veículos distintos
        sources = {a.ideology_id for a in cluster_arts}
        if len(sources) < min_sources:
            continue

        # Artigo mais recente como representante do título da story
        representative = max(
            cluster_arts,
            key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
        )

        # Contagem por espectro
        spectrum_counts: dict[str, int] = {"Esquerda": 0, "Centro": 0, "Direita": 0}
        articles_out = []

        for art in sorted(
            cluster_arts,
            key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        ):
            spectrum = _POSITION_MAP.get(art.ideology_id, "Centro")
            spectrum_counts[spectrum] += 1
            articles_out.append({
                "url_hash":    art.url_hash,
                "title":       art.title,
                "url":         art.url,
                "source_name": art.source_name,
                "ideology_id": art.ideology_id,
                "spectrum":    spectrum,
                "bias_score":  art.bias_score,
                "bias_interpretation": art.bias_interpretation,
                "published_at": (
                    art.published_at.isoformat()
                    if art.published_at else None
                ),
                "image_url":   art.image_url,
            })

        stories.append({
            "topic":         representative.title or "Sem título",
            "article_count": len(cluster_arts),
            "left_count":    spectrum_counts["Esquerda"],
            "center_count":  spectrum_counts["Centro"],
            "right_count":   spectrum_counts["Direita"],
            "latest_at":     (
                representative.published_at.isoformat()
                if representative.published_at else None
            ),
            "articles": articles_out,
        })

    # Ordena por número de artigos (mais coberto primeiro)
    stories.sort(key=lambda s: s["article_count"], reverse=True)
    return stories[:max_stories]
