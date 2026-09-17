"""Case-table loading and cross-checking against the archived configs.

Source of truth is ``data/inputs/cases_to_run.xlsx`` (G1-G8). The archived
configs under ``Dataset_paper/fields/<case>/config.json`` are how the sweep
was actually run; where the two disagree the archived config wins and the
disagreement is reported rather than silently patched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reactor import ReactorConfig
from reactor.paper import settings

# The xlsx sheet carries a banner row, a blank row and a title row before
# the real column header.
_SHEET = "Cases to Run"
_HEADER_ROW = 3

# xlsx column name -> the name scripts._shared.build_case_config expects.
_COLUMN_ALIASES = {
    "P_ret_bar": "p_ret_bar",
    "Sweep ratio": "Sweep_Ratio",
    "Is_Counter-Current": "Is_Counter_Current",
}

REQUIRED_COLUMNS = (
    "Case_ID", "Description", "N_mem", "L_m", "r_max_m", "Dcat",
    "GHSV_h", "Sweep_Ratio", "H2_N2_ratio", "p_ret_bar", "p_perm_bar",
    "T_ret_K", "T_perm_K", "Is_Counter_Current",
)


def _to_bool(value: Any) -> bool:
    """Coerce spreadsheet truthiness to a real bool.

    Excel writes "FALSE" as a string here, and ``bool("FALSE")`` is True —
    the kind of trap that silently flips every case to counter-current.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0", "", "nan"}:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def load_case_table(path: Path | None = None) -> pd.DataFrame:
    """Load the G1-G8 case table with columns the study runners expect."""
    path = Path(path) if path is not None else settings.cases_xlsx()
    df = pd.read_excel(path, sheet_name=_SHEET, header=_HEADER_ROW)
    df = df.rename(columns=_COLUMN_ALIASES)
    df = df[df["Case_ID"].notna()].copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df["Case_ID"] = df["Case_ID"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Is_Counter_Current"] = df["Is_Counter_Current"].map(_to_bool)
    df["N_mem"] = df["N_mem"].astype(int)
    for col in ("L_m", "r_max_m", "Dcat", "GHSV_h", "Sweep_Ratio",
                "H2_N2_ratio", "p_ret_bar", "p_perm_bar", "T_ret_K", "T_perm_K"):
        df[col] = df[col].astype(float)

    df["family"] = df["Case_ID"].str.extract(r"^([A-Z]\d+)")[0]
    return df.reset_index(drop=True)


def select_cases(df: pd.DataFrame, include_598k: str | None = None) -> pd.DataFrame:
    """Apply the 598 K reporting switch to a case table *for reporting only*.

    The runner always solves every case, including both 598 K runs — D3
    controls what enters figures and dataset, never what is computed.
    """
    mode = settings.INCLUDE_598K if include_598k is None else include_598k
    if mode not in settings.INCLUDE_598K_CHOICES:
        raise ValueError(
            f"INCLUDE_598K must be one of {settings.INCLUDE_598K_CHOICES}, got {mode!r}"
        )
    if mode == "exclude":
        return df[~df["Case_ID"].isin(settings.CASES_598K)].copy()
    return df.copy()


# ── Archived configs ─────────────────────────────────────────────────

def archived_case_dir(case_id: str) -> Path:
    """Directory of one archived case under ``Dataset_paper/fields``."""
    return settings.archive_root() / "fields" / case_id


def load_archived_config(case_id: str) -> ReactorConfig:
    """Load an archived ``config.json`` back into a ReactorConfig.

    ``species`` must stay a list of str; the numeric lists (including the
    nested ``y_ret_in``) become numpy arrays, which ``ReactorConfig``'s
    ``__post_init__`` already does for the composition fields.
    """
    path = archived_case_dir(case_id) / "config.json"
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    raw["species"] = [str(s) for s in raw["species"]]
    return ReactorConfig.from_dict(raw)


def archived_case_is_converged(case_id: str) -> bool:
    """True when the archived case carries a usable solution.

    Two archived cases are unconverged (the 598 K pair) and have no
    ``fields.npz`` — every archive read must be guarded.
    """
    return (archived_case_dir(case_id) / "fields.npz").exists()


# Physical parameters that must agree between the reconstructed config and
# the archived one. Grid and solver knobs are deliberately excluded: those
# are set per resolution by this pipeline.
_PHYSICAL_KEYS = (
    "L", "Lsealing", "r_min", "r_max", "r_min_perm", "r_max_perm",
    "Nm", "eps", "Dcat", "rho_c", "dp", "nu",
    "P0_NH3", "EA_NH3", "P0_H2", "EA_H2", "P0_N2", "EA_N2",
    "F_ret_in", "F_perm_in", "is_counter_current",
    "p_ret_out", "p_perm_out", "is_isothermal",
    "T_ret_in", "T_perm_in", "T_ret_init", "T_perm_init",
    "y_ret_in", "y_perm_in", "y_ret_init", "y_perm_init",
)


@dataclass
class ConfigDiff:
    """One disagreeing physical parameter between a reconstructed and an archived config."""
    case_id: str
    key: str
    reconstructed: Any
    archived: Any
    rel_error: float


def compare_with_archive(
    config: ReactorConfig,
    case_id: str,
    *,
    rtol: float = 1e-9,
) -> list[ConfigDiff]:
    """Compare a reconstructed config against the archived one.

    Returns one entry per disagreeing physical parameter (empty when they
    match). The archived config wins by policy — this reports, it does not
    patch.
    """
    archived = load_archived_config(case_id)
    diffs: list[ConfigDiff] = []
    for key in _PHYSICAL_KEYS:
        mine = getattr(config, key)
        theirs = getattr(archived, key)
        if isinstance(mine, np.ndarray) or isinstance(theirs, np.ndarray):
            a = np.asarray(mine, dtype=float).ravel()
            b = np.asarray(theirs, dtype=float).ravel()
            if a.shape != b.shape:
                diffs.append(ConfigDiff(case_id, key, a.tolist(), b.tolist(), np.inf))
                continue
            scale = np.maximum(np.abs(b), 1e-30)
            err = float(np.max(np.abs(a - b) / scale))
            if err > rtol:
                diffs.append(ConfigDiff(case_id, key, a.tolist(), b.tolist(), err))
        elif isinstance(mine, (bool, np.bool_)) or isinstance(theirs, (bool, np.bool_)):
            if bool(mine) != bool(theirs):
                diffs.append(ConfigDiff(case_id, key, bool(mine), bool(theirs), np.inf))
        elif isinstance(mine, (int, float, np.integer, np.floating)):
            err = abs(float(mine) - float(theirs)) / max(abs(float(theirs)), 1e-30)
            if err > rtol:
                diffs.append(ConfigDiff(case_id, key, float(mine), float(theirs), err))
        elif mine != theirs:
            diffs.append(ConfigDiff(case_id, key, mine, theirs, np.inf))
    return diffs


def crosscheck_report(
    case_ids: list[str],
    build_config,
    *,
    rtol: float = 1e-9,
) -> pd.DataFrame:
    """Cross-check reconstructed configs against the archive for a few cases.

    ``build_config`` takes a Case_ID and returns a ReactorConfig. Cases with
    no archived config (or an unconverged archive entry) are reported as
    ``skipped`` rather than raising.
    """
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        if not (archived_case_dir(case_id) / "config.json").exists():
            rows.append({"Case_ID": case_id, "status": "skipped",
                         "reason": "no archived config.json", "key": "", "rel_error": np.nan})
            continue
        diffs = compare_with_archive(build_config(case_id), case_id, rtol=rtol)
        if not diffs:
            rows.append({"Case_ID": case_id, "status": "match",
                         "reason": "", "key": "", "rel_error": 0.0})
            continue
        for d in diffs:
            rows.append({
                "Case_ID": case_id, "status": "MISMATCH",
                "reason": f"reconstructed={d.reconstructed!r} archived={d.archived!r}",
                "key": d.key, "rel_error": d.rel_error,
            })
    return pd.DataFrame(rows)
