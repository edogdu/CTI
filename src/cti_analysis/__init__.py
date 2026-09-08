"""
CTI Analysis package.

This package wires together data normalization, triple extraction,
graph alignment, and analysis utilities. The main orchestration entrypoint
is `cti_analysis.pipeline.run_pipeline`.
"""

from .pipeline import run_pipeline  # re-export for convenience

__all__ = ["run_pipeline"]

