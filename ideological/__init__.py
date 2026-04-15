"""
ideological
───────────
Camada 4 — Contextualização ideológica baseada em referências acadêmicas.
"""

from .spectrum import contextualize, contextualize_all, get_spectrum_summary, IdeologicalContext
from .reference_map import load_reference_map, get_profile, IdeologicalProfile

__all__ = [
    "contextualize",
    "contextualize_all",
    "get_spectrum_summary",
    "IdeologicalContext",
    "load_reference_map",
    "get_profile",
    "IdeologicalProfile",
]
