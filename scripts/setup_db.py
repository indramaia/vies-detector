"""
scripts/setup_db.py
────────────────────
Definição dos modelos SQLAlchemy e inicialização do banco de dados.

Princípio LGPD aplicado:
    - ArticleRecord armazena apenas metadados + snippet (≤ 500 chars)
    - SentenceRecord armazena a sentença e os scores — nunca o artigo completo
    - Nenhuma informação pessoal é coletada ou armazenada

Uso:
    python scripts/setup_db.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text,
    ForeignKey, create_engine, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///vies_detector.db")
engine = create_engine(
    os.environ["DATABASE_URL"],
    use_insertmanyvalues=False,
    implicit_returning=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


# ── Modelos ───────────────────────────────────────────────────────────────────

class ArticleRecord(Base):
    """
    Metadados e resultado de análise de um artigo.

    NÃO armazena o texto completo (LGPD).
    snippet é limitado a 500 caracteres no pré-processamento.
    """
    __tablename__ = "articles"

    url_hash = Column(String(64), primary_key=True)   # SHA-256 hex
    url = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    source_name = Column(String(128), nullable=False)
    ideology_id = Column(String(64), nullable=False)
    published_at = Column(DateTime, nullable=True)
    collected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    snippet = Column(String(500), nullable=True)       # ≤ 500 chars (LGPD)
    sentence_count = Column(Integer, nullable=True)
    # Resultados da Camada 3
    bias_score = Column(Float, nullable=True)
    bias_interpretation = Column(String(64), nullable=True)
    n_factual = Column(Integer, nullable=True)
    n_biased = Column(Integer, nullable=True)
    n_strongly_biased = Column(Integer, nullable=True)
    image_url = Column(Text, nullable=True)



class SentenceRecord(Base):
    """
    Resultado de classificação de uma sentença individual.
    Vinculada ao artigo pelo url_hash.
    """
    __tablename__ = "sentences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url_hash = Column(String(64), ForeignKey("articles.url_hash"), nullable=False)
    sentence = Column(Text, nullable=False)
    label = Column(String(32), nullable=False)          # factual | enviesada | fortemente_enviesada
    label_id = Column(Integer, nullable=False)          # 0 | 1 | 2
    confidence = Column(Float, nullable=False)
    score_factual = Column(Float, nullable=False)
    score_biased = Column(Float, nullable=False)
    score_strongly_biased = Column(Float, nullable=False)


class VehicleIndexRecord(Base):
    """
    Índice editorial de um veículo em uma janela temporal.
    Substituído a cada execução do pipeline (upsert por ideology_id).
    """
    __tablename__ = "vehicle_indices"

    ideology_id = Column(String(64), primary_key=True)
    source_name = Column(String(128), nullable=False)
    computed_at = Column(DateTime, nullable=False)
    window_days = Column(Integer, nullable=False)
    article_count = Column(Integer, nullable=False)
    mean_bias = Column(Float, nullable=False)
    ideology_score = Column(Float, nullable=True)
    uncertainty = Column(Float, nullable=True)
    position_label = Column(String(64), nullable=True)
    contextualization = Column(Text, nullable=True)


# ── Utilitários ───────────────────────────────────────────────────────────────

def init_db() -> None:
    """Cria todas as tabelas se não existirem."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Banco de dados inicializado: {DATABASE_URL}")


@contextmanager
def get_session():
    """Context manager que garante commit/rollback e fechamento da sessão."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    logger.info("Setup do banco concluído.")
