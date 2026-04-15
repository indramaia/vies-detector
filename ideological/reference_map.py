"""
ideological/reference_map.py
─────────────────────────────
Carregamento e acesso ao mapeamento ideológico baseado em referências acadêmicas.

O arquivo ideological_references.json contém posicionamentos derivados de:
    - Manchetômetro (FERES JÚNIOR et al., 2013-), IESP/UERJ
    - ORTELLADO; RIBEIRO (2018), GPOPAI/USP
    - INTERVOZES (2017), Donos da Mídia
    - REUTERS INSTITUTE (2024), Digital News Report

Escala [-1, +1]:
    -1.0 → Progressista / Esquerda
     0.0 → Centro
    +1.0 → Conservador / Direita
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

DATA_FILE = Path(__file__).parent / "data" / "ideological_references.json"


@dataclass(frozen=True)
class IdeologicalProfile:
    ideology_id: str
    name: str
    url: str
    ideology_score: float      # [-1.0, +1.0]
    uncertainty: float         # Estimativa de incerteza (±)
    position_label: str        # Rótulo textual (ex: "Centro-direita")
    description: str
    academic_refs: list[str]


@lru_cache(maxsize=1)
def load_reference_map() -> dict[str, IdeologicalProfile]:
    """
    Carrega o mapeamento ideológico do JSON e retorna um dicionário
    {ideology_id: IdeologicalProfile}.

    Usa @lru_cache para evitar leitura repetida de arquivo.
    """
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)

    profiles: dict[str, IdeologicalProfile] = {}
    for vid, v in data["vehicles"].items():
        profiles[vid] = IdeologicalProfile(
            ideology_id=vid,
            name=v["name"],
            url=v["url"],
            ideology_score=v["ideology_score"],
            uncertainty=v["uncertainty"],
            position_label=v["position_label"],
            description=v["description"],
            academic_refs=v["academic_refs"],
        )

    return profiles


def get_profile(ideology_id: str) -> IdeologicalProfile | None:
    return load_reference_map().get(ideology_id)
