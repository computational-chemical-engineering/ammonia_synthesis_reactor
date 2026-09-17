"""On-disk cache of solved cases.

Layout::

    results/paper/<resolution>/<model>/<case_id>/
        config.json         the ReactorConfig the case was solved with
        fields.npz          2D/1D field arrays
        flows.npz           axial and membrane molar flows
        profiles_axial.csv  axial profiles
        kpis.json           KPIs in presentation units (+ *_si raw values)
        solve_status.json   solver diagnostics
        meta.json           completeness flag, wall time, provenance

A case is *complete* when ``meta.json`` says so, its schema version matches,
and every payload file is present. The runner skips complete cases, which is
what makes the notebook re-runnable and kernel-death-proof.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reactor.paper.case_setup import save_json

from reactor.paper import settings

_PAYLOAD_FILES = (
    "config.json", "fields.npz", "flows.npz",
    "profiles_axial.csv", "kpis.json", "solve_status.json",
)
# A failed case caches its diagnosis but has no fields to store.
_FAILED_PAYLOAD_FILES = ("config.json", "kpis.json", "solve_status.json")


def read_json(path: Path) -> Any:
    """Read and parse one JSON file."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def matches_config(case_dir: Path, expect: dict[str, Any]) -> bool:
    """True when the cached solve's ``config.json`` carries these values.

    Checked against the case's own ``config.json`` rather than a value copied
    into the metadata, so the answer comes from what the solver was actually
    given. Without this a settings change would silently reuse results
    computed under the old settings.
    """
    try:
        config = read_json(Path(case_dir) / "config.json")
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return False
    return all(config.get(key) == value for key, value in expect.items())


def matches_resolution(case_dir: Path, resolution: settings.Resolution) -> bool:
    """2D expectation: this resolution's grid and (weighted) tolerance."""
    return matches_config(case_dir, {
        "num_r": resolution.num_r,
        "num_z": resolution.num_z,
        "steady_state_atol": resolution.steady_state_atol,
    })


def is_complete(
    case_dir: Path,
    resolution: settings.Resolution | None = None,
    expect: dict[str, Any] | None = None,
) -> bool:
    """True when this directory holds a usable cached solve.

    Pass ``resolution`` to require the 2D grid/tolerance expectation, or
    ``expect`` (a config key -> value dict) for model-specific expectations
    (the 1D models share num_z with the tier but have their own tolerance
    semantics — see ``runner.cache_expectation``).
    """
    case_dir = Path(case_dir)
    meta_path = case_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = read_json(meta_path)
    except (json.JSONDecodeError, OSError):
        return False
    if not meta.get("complete"):
        return False
    if meta.get("schema_version") != settings.CACHE_SCHEMA_VERSION:
        return False
    required = _FAILED_PAYLOAD_FILES if meta.get("status") == "failed" else _PAYLOAD_FILES
    if not all((case_dir / name).exists() for name in required):
        return False
    if expect is not None:
        return matches_config(case_dir, expect)
    return resolution is None or matches_resolution(case_dir, resolution)


def clear(case_dir: Path) -> None:
    """Drop a cached case (used by FORCE_RERUN)."""
    case_dir = Path(case_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)


def write(
    case_dir: Path,
    *,
    config: Any,
    kpis: dict[str, Any],
    status: Any,
    meta: dict[str, Any],
    fields: dict[str, np.ndarray] | None = None,
    flows: dict[str, np.ndarray] | None = None,
    profiles: pd.DataFrame | None = None,
) -> None:
    """Write one case bundle. ``meta.json`` is written last, so an interrupted
    write leaves an incomplete — and therefore re-run — directory."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    save_json(case_dir / "config.json", config.to_dict() if hasattr(config, "to_dict") else config)
    save_json(case_dir / "kpis.json", kpis)
    save_json(case_dir / "solve_status.json", status)
    if fields is not None:
        np.savez_compressed(case_dir / "fields.npz", **fields)
    if flows is not None:
        np.savez_compressed(case_dir / "flows.npz", **flows)
    if profiles is not None:
        profiles.to_csv(case_dir / "profiles_axial.csv", index=False)
    save_json(case_dir / "meta.json", {**meta, "schema_version": settings.CACHE_SCHEMA_VERSION})


def load_kpis(case_dir: Path) -> dict[str, Any]:
    """Load the cached ``kpis.json`` of a case."""
    return read_json(Path(case_dir) / "kpis.json")


def load_meta(case_dir: Path) -> dict[str, Any]:
    """Load the cached ``meta.json`` of a case."""
    return read_json(Path(case_dir) / "meta.json")


def load_fields(case_dir: Path) -> dict[str, np.ndarray]:
    """Load the cached field arrays into a plain dict.

    Materialises the arrays so the npz handle does not stay open — figure
    cells load dozens of these.
    """
    with np.load(Path(case_dir) / "fields.npz") as data:
        return {key: data[key] for key in data.files}


def load_flows(case_dir: Path) -> dict[str, np.ndarray]:
    """Load the cached flow arrays (``flows.npz``) into a plain dict."""
    with np.load(Path(case_dir) / "flows.npz") as data:
        return {key: data[key] for key in data.files}


def load_profiles(case_dir: Path) -> pd.DataFrame:
    """Load the cached axial profiles (``profiles_axial.csv``) as a DataFrame."""
    return pd.read_csv(Path(case_dir) / "profiles_axial.csv")


def status_table(resolution_name: str, model: str, case_ids: list[str]) -> pd.DataFrame:
    """What the cache currently holds for these cases — the notebook's
    'is the sweep done yet' view."""
    res = settings.resolution(resolution_name)
    rows = []
    for case_id in case_ids:
        case_dir = settings.case_cache_dir(resolution_name, model, case_id)
        if not is_complete(case_dir, res):
            rows.append({"Case_ID": case_id, "cached": False, "status": "missing",
                         "runtime_s": np.nan, "steady_state_norm": np.nan})
            continue
        meta = load_meta(case_dir)
        rows.append({
            "Case_ID": case_id,
            "cached": True,
            "status": meta.get("status", "unknown"),
            "runtime_s": meta.get("runtime_s", np.nan),
            "steady_state_norm": meta.get("steady_state_norm", np.nan),
        })
    return pd.DataFrame(rows)
