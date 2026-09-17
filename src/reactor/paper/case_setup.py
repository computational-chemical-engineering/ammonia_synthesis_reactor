"""Case setup and KPI utilities for the paper pipeline.

Everything needed to turn a row of the G1-G8 case table into a solvable
:class:`reactor.ReactorConfig` (2D and 1D variants), plus the KPI
definitions, elemental-balance checks, solver-acceptance rules and JSON
helpers shared by the cached runner. The KPI definitions here are the
single source of truth: the dataset's summary CSVs and every figure use
them unchanged.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.constants as const

from reactor import DEFAULTS, ReactorConfig

F_SMALL = 1e-14

# Sweep families of the paper's case table (G1-G8,
# and G1-G8 (the paper dataset). Keep the N entries first: plot_study_sweeps
# zips these keys against a 3x3 axes grid.
FAMILY_SPECS = {
    "N1": ("GHSV_h", r"GHSV [h$^{-1}$]"),
    "N2": ("L_m", r"Membrane length [m]"),
    "N3": ("r_max_m", r"Outer radius [m]"),
    "N4": ("p_ret_bar", r"Retentate pressure [bar]"),
    "N5": ("T_ret_K", r"Retentate inlet temperature [K]"),
    "N6": ("T_perm_K", r"Permeate inlet temperature [K]"),
    "N7": ("H2_N2_ratio", r"$H_2/N_2$ feed ratio [-]"),
    "N8": ("Sweep_Ratio", r"Sweep ratio [-]"),
    "N9": ("Dcat", r"Catalyst dilution [-]"),
    "G1": ("GHSV_h", "GHSV [h⁻¹]"),
    "G2": ("GHSV_h", "GHSV [h⁻¹]"),
    "G3": ("p_ret_bar", "Pressure [bar]"),
    "G4": ("p_ret_bar", "Pressure [bar]"),
    "G5": ("T_ret_K", "Temperature [K]"),
    "G6": ("T_ret_K", "Temperature [K]"),
    "G7": ("r_max_m", "Radius [m]"),
    "G8": ("r_max_m", "Radius [m]"),
}

# The parameter column that varies per family (for continuation)
FAMILY_PARAM_COL = {
    "N1": "GHSV_h",
    "N2": "L_m",
    "N3": "r_max_m",
    "N4": "p_ret_bar",
    "N5": "T_ret_K",
    "N6": "T_perm_K",
    "N7": "H2_N2_ratio",
    "N8": "Sweep_Ratio",
    "N9": "Dcat",
    "G1": "GHSV_h",
    "G2": "GHSV_h",
    "G3": "p_ret_bar",
    "G4": "p_ret_bar",
    "G5": "T_ret_K",
    "G6": "T_ret_K",
    "G7": "r_max_m",
    "G8": "r_max_m",
}

# ── JSON encoding ─────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that also handles numpy scalars/arrays and dataclasses."""
    def default(self, obj: Any) -> Any:
        """Convert numpy scalars/arrays and dataclasses to JSON-serializable builtins."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def to_builtin(value: Any) -> Any:
    """Recursively convert numpy types, dataclasses and Paths to plain builtins."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    """Write a payload as indented JSON, converting numpy/dataclass values."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2, cls=NumpyEncoder)


# ── Flow calculation ──────────────────────────────────────────────────

def calculate_flows(
    GHSV, eps, Dcat, rho_c, L_membrane, Lsealing,
    r_max, r_min, Nm, sweep_ratio, H2_N2_ratio,
    T_STP=273.15, P_STP=101325.0,
):
    """Convert operating conditions to inlet molar flows and design metrics."""
    L_bed = L_membrane + Lsealing
    A_reactor = np.pi * (r_max**2 - r_min**2 * Nm)
    A_membrane = 2 * np.pi * r_min * L_membrane * Nm
    vol_reactor = A_reactor * L_bed
    vol_cat = Dcat * (1 - eps) * vol_reactor
    W_cat = vol_cat * rho_c
    S_over_V = 2 * (r_min * Nm) / (r_max**2 - Nm * r_min**2)
    vol_flow_std = (GHSV * vol_cat) / 3600.0
    WHSV = vol_flow_std / W_cat *3600/1000
    F_ret_in = (P_STP * vol_flow_std) / (const.R * T_STP)
    y_N2_in = 1.0 / (H2_N2_ratio + 1.0)
    y_H2_in = 1.0 - y_N2_in
    F_perm_in = F_ret_in * sweep_ratio
    return (F_ret_in, F_perm_in, y_H2_in, y_N2_in,
            W_cat, WHSV, S_over_V, vol_reactor, A_membrane, A_reactor)


# ── Family helpers ────────────────────────────────────────────────────

def family_code(case_id: str) -> str:
    # Handles both "N1_GHSV_50_50bar" → "N1" and "G1 — GHSV sweep_50" → "G1"
    """Extract the sweep-family code ("N1", "G5", ...) from a Case_ID."""
    import re
    m = re.match(r'^([A-Z]\d+)', case_id.strip())
    return m.group(1) if m else case_id.split("_")[0]


def family_label(fam: str) -> str:
    """Human-readable sweep label of an N-family code (other codes pass through)."""
    mapping = {
        "N1": "GHSV sweep",
        "N2": "Length sweep",
        "N3": "Radius sweep",
        "N4": "Pressure sweep",
        "N5": "Retentate temperature sweep",
        "N6": "Permeate temperature sweep",
        "N7": "Feed ratio sweep",
        "N8": "Sweep-ratio sweep",
        "N9": "Catalyst dilution sweep",
    }
    return mapping.get(fam, fam)


# ── Elemental balance ────────────────────────────────────────────────

def element_flow_from_species(species_flows: np.ndarray) -> dict[str, float]:
    """Elemental H and N molar flows from [H2, N2, NH3] species flows."""
    return {
        "H": float(2.0 * species_flows[0] + 3.0 * species_flows[2]),
        "N": float(2.0 * species_flows[1] + species_flows[2]),
    }


def check_element_balance(
    flows_ret_ax: np.ndarray,
    flows_perm_ax: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Check elemental balance closure from inlet/outlet flows."""
    total_inlet = flows_ret_ax[0, :] + flows_perm_ax[0, :]
    total_outlet = flows_ret_ax[-1, :] + flows_perm_ax[-1, :]
    inlet_el = element_flow_from_species(total_inlet)
    outlet_el = element_flow_from_species(total_outlet)
    balance_species = total_inlet - total_outlet
    balance_H = float(inlet_el["H"] - outlet_el["H"])
    balance_N = float(inlet_el["N"] - outlet_el["N"])
    balance_abs_max = float(np.max(np.abs(balance_species)))
    balance_H_rel = abs(balance_H) / max(abs(inlet_el["H"]), F_SMALL)
    balance_N_rel = abs(balance_N) / max(abs(inlet_el["N"]), F_SMALL)
    element_ok = (
        abs(balance_H) <= atol or balance_H_rel <= rtol
    ) and (
        abs(balance_N) <= atol or balance_N_rel <= rtol
    )
    return {
        "balance_H": balance_H,
        "balance_N": balance_N,
        "balance_H_rel": balance_H_rel,
        "balance_N_rel": balance_N_rel,
        "balance_species": balance_species,
        "element_H_in_mol_s": inlet_el["H"],
        "element_N_in_mol_s": inlet_el["N"],
        "balance_abs_max": balance_abs_max,
        "element_balance_ok": bool(element_ok),
    }


# ── KPI computation ──────────────────────────────────────────────────

def compute_kpis_from_flows(
    flows_ret_ax: np.ndarray,
    flows_perm_ax: np.ndarray,
    W_cat: float,
    A_membrane: float,
) -> dict[str, float]:
    """Compute standard KPIs from axial flow arrays."""
    F_H2_ret = flows_ret_ax[:, 0]
    F_H2_perm = flows_perm_ax[:, 0]
    F_NH3_ret = flows_ret_ax[:, 2]
    F_NH3_perm = flows_perm_ax[:, 2]

    F_H2_in_ret = F_H2_ret[0]
    F_H2_in_perm = F_H2_perm[0]
    F_H2_in_tot = F_H2_in_ret + F_H2_in_perm
    F_H2_tm = F_H2_perm - F_H2_perm[0]
    F_H2_tm_star = np.where(F_H2_tm <= 0.0, F_H2_tm, 0.0)
    X_H2 = (F_H2_in_ret - F_H2_ret - F_H2_tm) / np.clip(F_H2_in_ret - F_H2_tm_star, F_SMALL, None)
    NH3_rec = F_NH3_perm / np.clip(F_NH3_perm + F_NH3_ret, F_SMALL, None)
    NH3_yield = 3.0 * (F_NH3_perm + F_NH3_ret) / np.clip(2.0 * F_H2_in_tot, F_SMALL, None)

    total_perm_out = np.sum(flows_perm_ax[-1, :])
    NH3_purity_out = float(flows_perm_ax[-1, 2] / np.clip(total_perm_out, F_SMALL, None))
    # Total NH3 produced = (NH3 out both sides) - (NH3 in both sides)
    F_NH3_in_tot  = F_NH3_ret[0]  + F_NH3_perm[0]   # NH3 entering (ret + perm inlets)
    F_NH3_out_tot = F_NH3_ret[-1] + F_NH3_perm[-1]  # NH3 leaving  (ret + perm outlets)
    NH3_prod_out  = float((F_NH3_out_tot - F_NH3_in_tot) / W_cat) if W_cat > 0 else 0.0
    J_H2_avg = float(flows_ret_ax[0, 0] - flows_ret_ax[-1, 0]) / max(A_membrane, F_SMALL) if A_membrane > 0 else 0.0

    return {
        "X_H2_out": float(X_H2[-1]),
        "NH3_prod_out": NH3_prod_out,
        "NH3_rec_out": float(NH3_rec[-1]),
        "NH3_yield_out": float(NH3_yield[-1]),
        "NH3_purity_out": NH3_purity_out,
        "J_H2_avg": J_H2_avg,
        "X_H2_profile": X_H2,
        "NH3_rec_profile": NH3_rec,
        "NH3_yield_profile": NH3_yield,
    }


# ── Solver acceptance ─────────────────────────────────────────────────

def solver_acceptance(
    status: Any,
    steady_state_atol: float,
    accept_factor: float = 1.1,
) -> dict[str, Any]:
    """Determine whether a solve is acceptable for the paper dataset."""
    final_norm = float(status.steady_state_norm)
    best_norm = float(status.best_steady_state_norm)
    norm_for_acceptance = min(final_norm, best_norm)
    target = steady_state_atol
    accept_limit = accept_factor * target
    close_to_target = np.isfinite(norm_for_acceptance) and norm_for_acceptance <= accept_limit

    if bool(status.converged):
        # A "floored" outcome (residual stagnant at its floor, KPIs stagnant)
        # is a converged iteration against an unreachable threshold; the
        # solver already accepts it (floored/oscillatory are accepted classes).
        outcome = getattr(status, "outcome", None)
        reason = ("solver_converged" if outcome in (None, "converged")
                  else f"accepted_{outcome}")
        return {
            "accepted": True,
            "accepted_close": False,
            "reason": reason,
            "target_norm": target,
            "accept_limit": accept_limit,
            "norm_for_acceptance": norm_for_acceptance,
        }
    if close_to_target:
        return {
            "accepted": True,
            "accepted_close": True,
            "reason": "accepted_near_steady_state_tolerance",
            "target_norm": target,
            "accept_limit": accept_limit,
            "norm_for_acceptance": norm_for_acceptance,
        }
    return {
        "accepted": False,
        "accepted_close": False,
        "reason": "steady_state_norm_above_accept_limit",
        "target_norm": target,
        "accept_limit": accept_limit,
        "norm_for_acceptance": norm_for_acceptance,
    }


# ── Score ─────────────────────────────────────────────────────────────

def compute_score(kpis: dict[str, float], delta_t_max: float, delta_p_bar: float) -> float:
    """Scalar merit score: weighted KPIs minus temperature-rise and pressure-drop penalties."""
    return float(
        kpis["NH3_prod_out"] * 1e3
        + 0.35 * kpis["X_H2_out"]
        + 0.25 * kpis["NH3_rec_out"]
        + 0.15 * kpis.get("NH3_purity_out", 0.0)
        - 0.002 * max(delta_t_max, 0.0)
        - 0.01 * max(delta_p_bar, 0.0)
    )


# ── Config building ──────────────────────────────────────────────────

def build_case_config(
    row: pd.Series,
    *,
    trace_nh3: float = 1e-3,
    perm_nh3_default: float = 1e-8,
    # Solver settings
    factor_react: float = 1.0,
    factor_p: float = 1.0,
    factor_T: float = 1e-1,
    num_r: int = 40,
    num_z: int = 100,
    num_timesteps: int = 30,
    max_newton_iterations: int = 5,
    is_isothermal: bool = False,
    dt_max: float = 1e6,
    rtol: float = 1e-2,
    atol: float = 1e-4,
    steady_state_atol: float = 1e-3,
    steady_state_rtol: float = 1e-2,
    extra_config: dict | None = None,
) -> tuple[ReactorConfig, dict[str, float], np.ndarray]:
    """Build a ReactorConfig from a case-study CSV row.

    ``extra_config`` merges additional ReactorConfig fields (highest
    priority) — used by the paper pipeline for the steady-state norm
    settings (norm_kind, wrms_* block tolerances, KPI-stop knobs).
    """

    cfg_base = ReactorConfig.from_defaults(
        r_max=float(row["r_max_m"]),
        Nm=int(row["N_mem"]),
        Dcat=float(row["Dcat"]),
        p_ret_out=float(row["p_ret_bar"]) * 1e5,
        p_perm_out=float(row["p_perm_bar"]) * 1e5,
        T_ret_in=float(row["T_ret_K"]),
        T_perm_in=float(row["T_perm_K"]),
        is_counter_current=bool(row["Is_Counter_Current"]),
       
    )

    r_min = cfg_base.r_min
    r_max = cfg_base.r_max
    Nm = cfg_base.Nm
    L_membrane = float(row["L_m"])
    Lsealing = cfg_base.Lsealing
    L_total=L_membrane + Lsealing
    Dcat = cfg_base.Dcat
    eps = cfg_base.eps
    rho_c = cfg_base.rho_c
    GHSV = float(row["GHSV_h"])
    sweep_ratio = float(row["Sweep_Ratio"])
    H2_N2_ratio = float(row["H2_N2_ratio"])
   
    (F_ret_in, F_perm_in, y_H2_in, y_N2_in,
     W_cat, WHSV, S_over_V, vol_reactor, A_membrane, A_reactor) = calculate_flows(
        GHSV=GHSV, eps=eps, rho_c=rho_c,
        sweep_ratio=sweep_ratio, H2_N2_ratio=H2_N2_ratio,
        r_min=r_min, r_max=r_max, Nm=Nm,
        Lsealing=Lsealing, L_membrane=L_membrane, Dcat=Dcat,
    )

    y_ret_in = np.array([(1.0 - trace_nh3) * y_H2_in, (1.0 - trace_nh3) * y_N2_in, trace_nh3])

    reactor_cfg = ReactorConfig.from_dict({
        **cfg_base.to_dict(),
        "L": L_total,
        "Lsealing": Lsealing,
        "F_ret_in": F_ret_in,
        "F_perm_in": F_perm_in,
        "y_ret_in": y_ret_in,
        "T_ret_init": cfg_base.T_ret_in,
        "T_perm_init": cfg_base.T_perm_in,
        "factor_react": factor_react,
        "factor_p": factor_p,
        "factor_T": factor_T,
        "num_r": num_r,
        "num_z": num_z,
        "num_timesteps": num_timesteps,
        "max_newton_iterations": max_newton_iterations,
        "is_isothermal": is_isothermal,
        "dt_max": dt_max,
        "rtol": rtol,
        "atol": atol,
        "steady_state_atol": steady_state_atol,
        "steady_state_rtol": steady_state_rtol,
        **(extra_config or {}),
    })

    design_meta = {
        "W_cat": W_cat,
        "WHSV": WHSV,
        "S_over_V": S_over_V,
        "vol_reactor_m3": vol_reactor,
        "A_membrane_m2": A_membrane,
        "A_reactor_m2": A_reactor,
        "L_membrane_m": L_membrane,
        "L_sealing_m": Lsealing,
        "L_total_m": L_total,
    }
    return reactor_cfg, design_meta, y_ret_in


def build_case_config_1d(
    row: pd.Series,
    *,
    trace_nh3: float = 1e-3,
    perm_nh3_default: float = 1e-8,
    num_z: int = 100,
    num_timesteps: int = 30,
    max_newton_iterations: int = 5,
    is_isothermal: bool = False,
    dt_max: float = 1e6,
    rtol: float = 1e-2,
    atol: float = 1e-4,
    steady_state_atol: float = 1e-3,
    steady_state_rtol: float = 1e-2,
    extra_config: dict | None = None,
) -> tuple[ReactorConfig, dict[str, float], np.ndarray]:
    """Build a ReactorConfig for the 1D model from a case-study CSV row.

    ``extra_config`` merges additional ReactorConfig fields (highest
    priority) — used by the paper pipeline for the fitted Sherwood-closure
    coefficients of the corrected 1D model.
    """
    r_min = DEFAULTS["r_min"]
    r_max = float(row["r_max_m"])
    Nm = int(row["N_mem"])
    L_membrane = float(row["L_m"])
    Lsealing = DEFAULTS["Lsealing"]
    Dcat = float(row["Dcat"])
    eps = DEFAULTS["eps"]
    rho_c = DEFAULTS["rho_c"]
    GHSV = float(row["GHSV_h"])
    sweep_ratio = float(row["Sweep_Ratio"])
    H2_N2_ratio = float(row["H2_N2_ratio"])

    (F_ret_in, F_perm_in, y_H2_in, y_N2_in,
     W_cat, WHSV, S_over_V, vol_reactor, A_membrane, A_reactor) = calculate_flows(
        GHSV=GHSV, eps=eps, rho_c=rho_c,
        sweep_ratio=sweep_ratio, H2_N2_ratio=H2_N2_ratio,
        r_min=r_min, r_max=r_max, Nm=Nm,
        Lsealing=Lsealing, L_membrane=L_membrane, Dcat=Dcat,
    )

    y_ret_in = np.array([(1.0 - trace_nh3) * y_H2_in, (1.0 - trace_nh3) * y_N2_in, trace_nh3])

    p_ret_out = float(row["p_ret_bar"]) * 1e5
    p_perm = float(row["p_perm_bar"]) * 1e5
    T_ret_in = float(row["T_ret_K"])
    T_perm_in = float(row["T_perm_K"])
    is_counter_current = bool(row["Is_Counter_Current"])
    L_total=L_membrane + Lsealing

    cfg = ReactorConfig.from_defaults(
        L=L_total,
        Lsealing=Lsealing,      
        r_min=r_min,
        r_max=r_max,
        p_ret_out=p_ret_out,
        p_perm_out=p_perm,
        T_ret_in=T_ret_in,
        T_perm_in=T_perm_in,
        T_ret_init=T_ret_in,
        T_perm_init=T_perm_in,
        F_ret_in=F_ret_in,
        F_perm_in=F_perm_in,
        y_ret_in=y_ret_in,
        Nm=Nm,
        Dcat=Dcat,
        is_counter_current=is_counter_current,
        factor_react=1.0,
        factor_p=1.0,
        factor_T=1e-1,
        num_z=num_z,
        num_timesteps=num_timesteps,
        max_newton_iterations=max_newton_iterations,
        is_isothermal=is_isothermal,
        dt_max=dt_max,
        rtol=rtol,
        atol=atol,
        steady_state_atol=steady_state_atol,
        steady_state_rtol=steady_state_rtol,
    )
    if extra_config:
        cfg = ReactorConfig.from_dict({**cfg.to_dict(), **extra_config})

    design_meta = {
        "W_cat": W_cat,
        "WHSV": WHSV,
        "S_over_V": S_over_V,
        "vol_reactor_m3": vol_reactor,
        "A_membrane_m2": A_membrane,
        "A_reactor_m2": A_reactor,
        "L_membrane_m": L_membrane,
        "L_sealing_m": Lsealing,
        "L_total_m": L_total,
    }
    return cfg, design_meta, y_ret_in


# ── Normalized trend helper ──────────────────────────────────────────

def normalized(values: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1] (all ones when constant)."""
    if values.empty:
        return values
    vmin, vmax = float(values.min()), float(values.max())
    if math.isclose(vmax, vmin):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - vmin) / (vmax - vmin)

