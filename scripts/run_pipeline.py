"""
scripts/run_pipeline.py
────────────────────────
Executa o pipeline manualmente (sem Prefect), útil para desenvolvimento
e depuração local.

Uso:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --window 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garante que a raiz do projeto está no PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from scripts.setup_db import init_db
from pipeline.main_flow import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Executa o pipeline Viés Detector")
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Janela temporal em dias para o índice por veículo (padrão: 30)",
    )
    args = parser.parse_args()

    logger.info("Inicializando banco de dados…")
    init_db()

    logger.info(f"Iniciando pipeline (janela={args.window} dias)…")
    run_pipeline(window_days=args.window)
    logger.info("Pipeline concluído com sucesso.")


if __name__ == "__main__":
    main()
