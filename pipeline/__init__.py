"""
pipeline
────────
Orquestração das quatro camadas via Prefect.
"""
from .main_flow import run_pipeline

__all__ = ["run_pipeline"]
