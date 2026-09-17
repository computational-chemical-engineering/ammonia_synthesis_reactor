"""Paper pipeline: case tables, cached solves, KPIs and figures.

The publication notebook is a thin driver over this package — a settings
cell, one short cell per pipeline stage, and one cell per figure. Every
solved case is cached on disk, so re-running the notebook from a clean
kernel takes seconds and never depends on volatile kernel state.

Usage::

    from reactor.paper import cases, figures, runner, settings
"""
from __future__ import annotations

from reactor.paper import cache, cases, figures, kpis, provenance, runner, settings

__all__ = ["cache", "cases", "figures", "kpis", "provenance", "runner", "settings"]
