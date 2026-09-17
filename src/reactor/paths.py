"""Shared path helpers for scripts and analysis tools."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return repository root for editable installs, falling back sensibly."""
    package_file = Path(__file__).resolve()
    candidate = package_file.parents[2]
    if (candidate / "data" / "inputs").exists() and (candidate / "scripts").exists():
        return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "data" / "inputs").exists() and (cwd / "scripts").exists():
        return cwd

    return candidate


def data_inputs_dir() -> Path:
    """Return the data/inputs directory under the project root."""
    return project_root() / "data" / "inputs"


def results_dir() -> Path:
    """Return the results directory under the project root."""
    return project_root() / "results"


def default_input_csv(filename: str) -> Path:
    """Return the path to ``filename`` inside the data/inputs directory."""
    return data_inputs_dir() / filename


def default_results_path(*parts: str) -> Path:
    """Join path components onto the results directory."""
    return results_dir().joinpath(*parts)
