"""
classifier/model_loader.py
──────────────────────────
Carregamento do BERTimbau fine-tuned para classificação de viés.

O modelo é carregado uma única vez (singleton) e mantido em memória
durante o ciclo de vida do processo, evitando overhead de I/O repetido.

Referências:
    SOUZA et al. BERTimbau (2020).
    WOLF et al. HuggingFace Transformers (2020).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# ── Constantes ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/bertimbau-bias"  # modelo fine-tuned no FactNews
)
HF_MODEL_REPO = "IndraSeixas/bertimbau-bias"  # fallback público no HuggingFace Hub

# Mapeamento de rótulos conforme remapeamento do FactNews
# (VARGAS et al., 2023): -1 → 2, 0 → 0, 1 → 1
LABEL2ID: dict[str, int] = {
    "factual": 0,
    "enviesada": 1,
    "fortemente_enviesada": 2,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

NUM_LABELS = len(LABEL2ID)
MAX_LENGTH = 512


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("GPU disponível — usando CUDA.")
        return torch.device("cuda")
    logger.info("GPU não disponível — usando CPU.")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_model(model_path: str = DEFAULT_MODEL_PATH) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Carrega o tokenizador e o modelo de classificação.

    Usa @lru_cache para garantir carregamento único (singleton).
    O modelo é movido para GPU automaticamente se disponível.

    Args:
        model_path : caminho local (após fine-tuning) ou ID HuggingFace

    Returns:
        Tupla (model, tokenizer) prontos para inferência.
    """
    path = Path(model_path)
    if path.exists():
        logger.info(f"Carregando modelo local: {model_path}")
        source = model_path
    else:
        logger.warning(
            f"Modelo local não encontrado em '{model_path}'. "
            f"Baixando modelo fine-tuned do HuggingFace Hub: {HF_MODEL_REPO}"
        )
        source = HF_MODEL_REPO

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # Necessário ao carregar modelo base sem cabeça
    )

    device = _get_device()
    model = model.to(device)
    model.eval()

    logger.info(f"Modelo carregado com sucesso. Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer
