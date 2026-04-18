"""
pipeline
────────
Orquestração das quatro camadas via Prefect. --> removi Prefect para simplificar, mas a estrutura de camadas permanece.
"""
from .main_flow import run_pipeline

__all__ = ["run_pipeline"]
