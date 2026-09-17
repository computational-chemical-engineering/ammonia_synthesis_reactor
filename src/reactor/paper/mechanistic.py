"""Mechanistic-closure analysis: per-case evaluation, exact tracing, field scans.

Everything Section 3.3.3 of the manuscript quotes about the reaction-screened
closure regenerates from here, out of the certified 2D cache:

* :func:`case_profile` — the screened closure evaluated on a case's 2D
  cross-sectionally averaged state (the basis of Figure 16).
* :func:`exact_trace_kpis` — the corrected 1D model re-solved with the
  *exact* flux-matched ``k_cp(z)`` override per case; what remains is the
  irreducible non-transport error of the 1D description.
* :func:`field_scan` — the two field measurements behind the mechanism
  identification: the NH3 deficit-layer half-thickness and the
  permeation-suction Peclet number on that thickness.

All functions read only the cache; the expensive artifacts are cached as
CSV/JSON next to the KPI summaries and ship with the dataset export.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reactor.cp_closure import annulus_screen_factor, conduction_shape_factor, screening_length
from reactor.paper import cache, cases, closures, kpis as kpi_mod, settings
from reactor.paper.case_setup import build_case_config_1d

EXACT_TRACE_NAME = "exact_trace_kpis.csv"
FIELD_SCAN_NAME = "mech_field_scan.json"


def exact_trace_path(resolution_name: str) -> Path:
    """Cached KPI table of the exact-k_cp corrected-1D runs."""
    return settings.cache_root(resolution_name) / EXACT_TRACE_NAME


def field_scan_path(resolution_name: str) -> Path:
    """Cached deficit-layer / suction field measurements."""
    return settings.cache_root(resolution_name) / FIELD_SCAN_NAME


def _corrected_reactor(row: pd.Series, resolution_name: str):
    """A corrected-1D reactor on the resolution's axial grid (Sh(kappa) fit)."""
    import importlib

    res = settings.resolution(resolution_name)
    fit = closures.load_sh_fit(resolution_name)
    config, design_meta, _ = build_case_config_1d(
        row, trace_nh3=settings.TRACE_NH3, num_z=res.num_z,
        extra_config={"sh_cp_coeff": fit["coeff"], "sh_cp_exp": fit["exp"]},
        **settings.SOLVER_1D,
    )
    mod = importlib.import_module(f"reactor.{settings.ONE_D_CORRECTED_VARIANT}")
    return mod.MembraneReactor1D(config=config), design_meta


def case_profile(resolution_name: str, case_id: str) -> dict[str, Any]:
    """Screened-closure quantities on the case's 2D mean state, per (z, species).

    Returns the exact flux-matched closure alongside the mechanistic
    evaluation: ``kcp_exact``, ``kcp_screen`` (calibrated), ``delta``,
    ``D``, the membrane flux ``J``, the axial grid and the geometry.
    """
    case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id)
    cl = closures.case_closure(case_dir)
    cfg2d = cl["config"]
    fields = cache.load_fields(case_dir)
    w = kpi_mod.area_weights(fields["r_f_ret"])
    c_mean = np.tensordot(fields["c_ret"], w, axes=([1], [0]))
    T_mean = np.tensordot(fields["T_ret"], w, axes=([1], [0]))
    p_mean = np.tensordot(fields["p_ret_bar"] * 1e5, w, axes=([1], [0]))

    table = cases.load_case_table()
    row = table[table["Case_ID"] == case_id].iloc[0]
    reactor, _ = _corrected_reactor(row, resolution_name)

    nz, nc = c_mean.shape
    c3 = c_mean.reshape(nz, 1, nc)
    T2 = T_mean.reshape(nz, 1)
    p2 = p_mean.reshape(nz, 1)
    y3 = c3 / np.maximum(np.sum(c3, axis=-1, keepdims=True), 1e-16)
    D = reactor.correlation.diffusion(y3, T2, p2)
    while D.ndim > 2:
        D = D.squeeze(axis=1)
    D = np.maximum(D, 1e-12)
    dRdc = reactor._kinetics_diag_sensitivity(c3, T2, p2)
    a, b = float(cfg2d["r_min"]), float(cfg2d["r_max"])
    scr = closures.load_screened_fit(resolution_name)
    delta = screening_length(D, dRdc)
    kcp_screen = scr["c_screen"] * np.where(
        np.isfinite(delta),
        D / np.maximum(delta, 1e-300) * annulus_screen_factor(a, b, delta), 0.0)
    return {
        "z_c": cl["z_c"], "J": cl["J"], "kcp_exact": cl["kcp"],
        "kcp_screen": kcp_screen, "delta": delta, "D": D,
        "r_mem": a, "r_max": b, "c_mean": c_mean, "T_mean": T_mean,
        "species": list(cfg2d["species"]),
        "i_nh3": cfg2d["species"].index("NH3"),
        "kcp_cond": conduction_shape_factor(a / b) * D / (b - a),
    }


# ── exact tracing ────────────────────────────────────────────────────

def exact_trace_kpis(resolution_name: str, *, force: bool = False,
                     verbose: int = 1) -> pd.DataFrame:
    """Corrected-1D KPIs with the exact per-case flux-matched k_cp(z) override.

    With the species closure made exact, the remaining deviation from the
    2D KPIs is the irreducible non-transport error (thermal observable
    definition + radial averaging of the nonlinear kinetics). Cached at
    :func:`exact_trace_path`; the solves take a few minutes for the full
    table.
    """
    path = exact_trace_path(resolution_name)
    if path.exists() and not force:
        return pd.read_csv(path)

    from reactor.paper.runner import _scalar_kpis_1d, _seed_reactor_from_2d

    table = cases.load_case_table()
    records = []
    for _, row in table.iterrows():
        case_id = str(row["Case_ID"])
        case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id)
        if not (case_dir / "fields.npz").exists():
            continue
        cl = closures.case_closure(case_dir)
        reactor, design_meta = _corrected_reactor(row, resolution_name)
        # The exact closure is undefined where the membrane carries no flux
        # (sealing region): fill with a large k_cp, i.e. no film resistance
        # where there is nothing to resist.
        kcp = np.asarray(cl["kcp"], float)
        kcp = np.where(np.isfinite(kcp) & (kcp > 0), kcp, 1.0)
        reactor.kcp_override = kcp
        # Seed from the 2D cross-sectional averages: the diagnostic asks
        # whether the 1D model reproduces the 2D solution when transport is
        # exact, so it must sit on the SAME operating branch (the low-GHSV
        # cases live in the multiplicity window). Deterministic, never a
        # warm start between cases.
        _seed_reactor_from_2d(reactor, case_dir)
        start = time.perf_counter()
        status = reactor.solve(dt_init=settings.DT_INIT_1D, return_status=True,
                               verbose=0)
        norm = status.steady_state_norm
        rec = {
            "Case_ID": case_id,
            "family": kpi_mod.family_code(case_id),
            "solve_converged": bool(status.converged),
            "steady_state_norm": float(norm) if norm is not None else np.nan,
            "runtime_s": time.perf_counter() - start,
        }
        rec.update(kpi_mod.to_presentation_units(_scalar_kpis_1d(reactor, design_meta)))
        records.append(rec)
        if verbose:
            print(f"  exact-trace {case_id}: converged={rec['solve_converged']}")
    df = pd.DataFrame.from_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def exact_trace_summary(resolution_name: str) -> dict[str, float]:
    """Median |relative deviation| of the exact-traced 1D KPIs from 2D (reported cases)."""
    trace = exact_trace_kpis(resolution_name)
    s2 = pd.read_csv(settings.summary_csv(resolution_name, settings.MODEL_2D))
    drop = set(settings.CASES_598K)
    out: dict[str, float] = {}
    for kpi in ("X_H2_out", "NH3_rec_out", "NH3_prod_out"):
        m = s2[~s2.Case_ID.isin(drop)][["Case_ID", kpi]].merge(
            trace[["Case_ID", kpi]], on="Case_ID", suffixes=("_2d", "_1d"))
        dev = 100.0 * np.abs(m[f"{kpi}_1d"] - m[f"{kpi}_2d"]) / np.abs(m[f"{kpi}_2d"])
        out[f"{kpi}_median_dev_pct"] = float(dev.median())
        out[f"{kpi}_max_dev_pct"] = float(dev.max())
    out["n_cases"] = int(len(m))
    return out


# ── field measurements ───────────────────────────────────────────────

def _case_layer_and_suction(resolution_name: str, case_id: str) -> dict[str, float] | None:
    """Deficit-layer half-thickness and suction Peclet for one 2D case.

    Half-thickness: per axial cell, the radial distance from the membrane
    at which the NH3 deficit (bulk mean minus local) has decayed to half
    its wall value; flux-weighted over the membrane-active length.
    Suction Peclet: the permeation-induced radial velocity at the wall
    times the half-thickness over the NH3 diffusivity.
    """
    case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id)
    if not (case_dir / "fields.npz").exists():
        return None
    prof = case_profile(resolution_name, case_id)
    fields = cache.load_fields(case_dir)
    i = prof["i_nh3"]
    c = fields["c_ret"][:, :, i]                      # (nz, nr)
    r_f = fields["r_f_ret"]
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    w = kpi_mod.area_weights(r_f)
    c_mean = c @ w
    c_wall = c[:, 0]                                   # first cell at the membrane
    deficit = c_mean[:, None] - c                      # (nz, nr), >=0 near the wall
    wall_def = c_mean - c_wall

    J = prof["J"][:, i]
    wgt = np.abs(J)
    active = (wgt > 0) & (wall_def > 1e-12 * np.maximum(c_mean, 1e-30))
    if not np.any(active):
        return None

    halves = []
    for iz in np.nonzero(active)[0]:
        target = 0.5 * wall_def[iz]
        d = deficit[iz]
        below = np.nonzero(d <= target)[0]
        if len(below) == 0:
            halves.append(r_c[-1] - prof["r_mem"])
            continue
        j = below[0]
        if j == 0:
            halves.append(r_c[0] - prof["r_mem"])
            continue
        # linear interpolation between the bracketing cells
        f = (d[j - 1] - target) / max(d[j - 1] - d[j], 1e-300)
        r_half = r_c[j - 1] + f * (r_c[j] - r_c[j - 1])
        halves.append(r_half - prof["r_mem"])
    halves = np.asarray(halves)
    wsel = wgt[active]
    ell = float(np.sum(halves * wsel) / np.sum(wsel))

    # suction velocity from the total molar membrane flux and the local
    # bulk molar density; Peclet on the measured half-thickness
    c_tot = fields["c_ret"][:, :, :].sum(axis=2) @ w
    v_w = np.abs(prof["J"]).sum(axis=1) / np.maximum(c_tot, 1e-30)
    D_nh3 = prof["D"][:, i]
    pe = float(np.sum((v_w[active] * halves / D_nh3[active]) * wsel) / np.sum(wsel))
    return {"half_thickness_m": ell, "suction_peclet": pe,
            "d_h_m": prof["r_max"] - prof["r_mem"]}


def field_scan(resolution_name: str, *, force: bool = False) -> dict[str, Any]:
    """Deficit-layer half-thickness and suction Peclet across all cached cases."""
    path = field_scan_path(resolution_name)
    if path.exists() and not force:
        return json.loads(path.read_text())

    table = cases.load_case_table()
    per_case: dict[str, dict[str, float]] = {}
    for _, row in table.iterrows():
        rec = _case_layer_and_suction(resolution_name, str(row["Case_ID"]))
        if rec is not None:
            per_case[str(row["Case_ID"])] = rec
    ells = np.array([r["half_thickness_m"] for r in per_case.values()])
    pes = np.array([r["suction_peclet"] for r in per_case.values()])
    dhs = np.array([r["d_h_m"] for r in per_case.values()])
    out = {
        "resolution": resolution_name,
        "n_cases": len(per_case),
        "half_thickness_mm": {"min": float(ells.min() * 1e3),
                              "median": float(np.median(ells) * 1e3),
                              "max": float(ells.max() * 1e3)},
        "suction_peclet": {"min": float(pes.min()),
                           "median": float(np.median(pes)),
                           "max": float(pes.max())},
        "d_h_span_factor": float(dhs.max() / dhs.min()),
        "per_case": per_case,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out
