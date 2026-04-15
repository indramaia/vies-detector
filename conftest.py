"""
conftest.py
────────────
Fixtures globais do pytest compartilhadas entre todos os módulos de teste.
"""

import sys
import os
from pathlib import Path

# Garante que a raiz do projeto esteja no PYTHONPATH durante os testes
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scripts.setup_db import Base


@pytest.fixture(scope="session")
def test_engine():
    """Engine SQLite em memória para testes — nunca toca o banco real."""
    engine = create_engine("sqlite:///:memory:", echo=False, future=True)
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def test_session(test_engine):
    """Sessão de banco isolada por teste — rollback automático ao final."""
    connection = test_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
