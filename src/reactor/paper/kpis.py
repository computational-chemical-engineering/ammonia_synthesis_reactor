"""KPI extraction, presentation units, and the CP metric.

The KPI *definitions* are imported from ``scripts._shared`` rather than
reimplemented — comparing against the archived ``summary_kpis_2d.csv`` only
means anything if both sides use identical formulas.

Two unit conventions matter and are both carried explicitly:

* ``compute_kpis_from_flows`` returns SI/fractional values (X_H2 as a
  fraction, NH3 production in mol kg_cat^-1 s^-1).
* The archived summary CSVs — which the published figure scripts read and
  label ``[%]`` / ``[mmol g_cat^-1 h^-1]`` — hold those same quantities
  scaled by 100 and 3600. This module writes the scaled values under the
  archived column names and keeps the raw ones under ``*_si`` names.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from reactor.paper.case_setup import (  # noqa: F401  (re-exported deliberately)
    F_SMALL,
    check_element_balance,
    compute_kpis_from_flows,
    compute_score,
    family_code,
    solver_acceptance,
)

from reactor.paper import settings

# Species index of NH3 in the [H2, N2, NH3] ordering used everywhere.
INH3 = 2

# Archived-CSV presentation scaling, per KPI column.
#   fractions -> percent
#   mol kg_cat^-1 s^-1 -> mmol g_cat^-1 h^-1  (x1000 mmol/mol / 1000 g/kg x 3600)
PRESENTATION_SCALE = {
    "X_H2_out": 100.0,
    "NH3_rec_out": 100.0,
    "NH3_yield_out": 100.0,
    "NH3_purity_out": 100.0,
    "NH3_prod_out": 3600.0,
}

PRESENTATION_UNITS = {
    "X_H2_out": "%",
    "NH3_rec_out": "%",
    "NH3_yield_out": "%",
    "NH3_purity_out": "%",
    "NH3_prod_out": "mmol g_cat^-1 h^-1",
    "DeltaT_max": "K",
    "delta_p_ret_bar": "bar",
    "J_H2_avg": "mol m^-2 s^-1",
}

# The six KPIs the 1D-vs-2D sweep figures (6-8) plot, in panel order.
SWEEP_KPIS = (
    "X_H2_out", "NH3_prod_out", "NH3_rec_out",
    "NH3_yield_out", "NH3_purity_out", "DeltaT_max",
)


def to_presentation_units(kpis: dict[str, float]) -> dict[str, float]:
    """Scale SI/fractional KPIs to the archived CSV's presentation units.

    Raw values are preserved alongside under ``<name>_si``.
    """
    out = dict(kpis)
    for key, factor in PRESENTATION_SCALE.items():
        if key in kpis:
            out[f"{key}_si"] = float(kpis[key])
            out[key] = float(kpis[key]) * factor
    return out


def whsv(
    *,
    ghsv_h: float,
    vol_flow_std_m3_s: float,
    w_cat_kg: float,
    dcat: float,
    eps: float,
    convention: str | None = None,
) -> float:
    """Weight hourly space velocity, in the requested convention.

    See ``settings.WHSV_CONVENTION``. The published figures use the
    ``"archived"`` convention, which is GHSV per catalyst *volume* [1/h]
    despite carrying a mass-based axis label.
    """
    convention = settings.WHSV_CONVENTION if convention is None else convention
    if convention not in settings.WHSV_CONVENTION_CHOICES:
        raise ValueError(
            f"WHSV_CONVENTION must be one of {settings.WHSV_CONVENTION_CHOICES}, "
            f"got {convention!r}"
        )
    if convention == "archived":
        return float(ghsv_h / (dcat * (1.0 - eps)))
    # mLn g_cat^-1 h^-1: m^3/s -> mL/h is x1e6 x3600; kg -> g is x1e3.
    return float(vol_flow_std_m3_s / max(w_cat_kg, F_SMALL) * 3600.0 * 1e3)


# ── Concentration polarisation ───────────────────────────────────────

def area_weights(r_faces: np.ndarray) -> np.ndarray:
    """Normalised annular area weights from radial face positions."""
    areas = np.pi * (np.asarray(r_faces)[1:] ** 2 - np.asarray(r_faces)[:-1] ** 2)
    return areas / areas.sum()


def cp_profile(
    y_ret: np.ndarray,
    r_f_ret: np.ndarray,
    species: int = INH3,
) -> np.ndarray:
    """Axial concentration-polarisation profile CP(z).

    ``CP(z) = y_wall(z) / <y(z)>``, the membrane-wall mole fraction over the
    area-weighted cross-sectional mean. The 1D models assume CP = 1. Radial
    index 0 of the retentate domain is the membrane wall.
    """
    y = np.asarray(y_ret)[:, :, species]
    weights = area_weights(r_f_ret)
    y_mean = (y * weights[np.newaxis, :]).sum(axis=1)
    return y[:, 0] / np.maximum(y_mean, 1e-10)


def cp_summary(
    y_ret: np.ndarray,
    r_f_ret: np.ndarray,
    z_c: np.ndarray,
    *,
    z_min: float = 0.05,
    species: int = INH3,
) -> dict[str, float]:
    """Scalar CP metrics over the membrane-active length (z > sealing)."""
    cp = cp_profile(y_ret, r_f_ret, species=species)
    mask = np.asarray(z_c) > z_min
    active = cp[mask]
    if active.size == 0:
        return {"CP_min": float("nan"), "CP_mean": float("nan"), "CP_out": float("nan")}
    return {
        "CP_min": float(np.min(active)),
        "CP_mean": float(np.mean(active)),
        "CP_out": float(active[-1]),
    }


# ── Comparison against the archived summary ──────────────────────────

def compare_to_archive(
    summary: pd.DataFrame,
    archived_csv,
    *,
    columns: tuple[str, ...] = SWEEP_KPIS,
    exclude_cases: tuple[str, ...] = settings.CASES_598K,
) -> pd.DataFrame:
    """Relative differences between a fresh summary and the archived CSV.

    Both sides are in presentation units. The two unconverged 598 K rows
    that the archived CSV still carries (residual ~0.76 states, not
    solutions) are excluded by default.
    """
    archived = pd.read_csv(archived_csv)
    keep = ["Case_ID", *[c for c in columns if c in archived.columns]]
    archived = archived[keep]
    merged = summary.merge(archived, on="Case_ID", suffixes=("_new", "_archived"))
    merged = merged[~merged["Case_ID"].isin(exclude_cases)]

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        record: dict[str, Any] = {"Case_ID": row["Case_ID"]}
        for col in columns:
            new, old = row.get(f"{col}_new"), row.get(f"{col}_archived")
            if new is None or old is None or not np.isfinite(float(old)):
                record[f"{col}_rel"] = np.nan
                continue
            record[f"{col}_rel"] = abs(float(new) - float(old)) / max(abs(float(old)), F_SMALL)
        rows.append(record)
    return pd.DataFrame(rows)
