"""Cached case runner for the paper sweeps.

Every case is solved from a cold start with the escalation ladder on (the
default), then cached. Re-running the sweep hits the cache and returns in
seconds; only missing or forced cases are recomputed.

Long sweeps belong in a background script (``scripts/run_paper_sweep.py``),
not in a live notebook cell — the notebook cell just calls
:func:`run_sweep`, finds everything cached, and builds the summary.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable

import importlib
from functools import partial

import numpy as np
import pandas as pd

from reactor import MembraneReactor, ReactorConfig
from reactor import stability
from reactor.paper.case_setup import (
    build_case_config,
    build_case_config_1d,
    compute_kpis_from_flows,
)

from reactor.paper import cache, closures, kpis as kpi_mod, provenance, settings


# ── Configuration ─────────────────────────────────────────────────────

def build_config_2d(row: pd.Series, resolution_name: str) -> tuple[ReactorConfig, dict[str, float]]:
    """Build the 2D config for one case at the given resolution.

    Solver settings come from :data:`settings.SOLVER`, which mirrors the
    defaults of ``scripts/run_study_2d.py`` — the script that produced the
    archived dataset. Only the grid and the steady-state tolerance vary with
    resolution.
    """
    res = settings.resolution(resolution_name)
    # No below-the-floor guard anymore: with the weighted norm and
    # KPI-stagnation stopping, a tolerance below a case's floor ends in a
    # cheap, accepted "floored" outcome instead of a burned step budget.
    config, design_meta, _ = build_case_config(
        row,
        trace_nh3=settings.TRACE_NH3,
        num_r=res.num_r,
        num_z=res.num_z,
        steady_state_atol=res.steady_state_atol,
        extra_config=settings.NORM_CONFIG,
        **settings.SOLVER,
    )
    if config.pressure_equation != settings.PRESSURE_EQUATION:
        config = ReactorConfig.from_dict(
            {**config.to_dict(), "pressure_equation": settings.PRESSURE_EQUATION}
        )
    return config, design_meta


# ── Field / flow extraction ──────────────────────────────────────────

def _centerline_average(face_values: np.ndarray) -> np.ndarray:
    """Average adjacent face values onto cell centers."""
    return 0.5 * (face_values[1:] + face_values[:-1])


def _total_flux_fields(reactor: MembraneReactor):
    """Total (diffusive + convective) axial and radial flux fields, both sides."""
    ret_ax, ret_rad, perm_ax, perm_rad = reactor._compute_fluxes_diff()
    c_ret_ax, c_ret_rad, c_perm_ax, c_perm_rad = reactor._compute_fluxes_conv()
    return (ret_ax + c_ret_ax, ret_rad + c_ret_rad,
            perm_ax + c_perm_ax, perm_rad + c_perm_rad)


def _reaction_source(reactor: MembraneReactor) -> np.ndarray:
    """Reaction source term on the retentate side from the current reactor state."""
    c = reactor.cpT[..., :-2]
    p = reactor.cpT[..., -2]
    T = reactor.cpT[..., -1]
    _, c_ret = reactor._split_perm_and_ret(c)
    _, p_ret = reactor._split_perm_and_ret(p)
    _, T_ret = reactor._split_perm_and_ret(T)
    # kinetics carries internal state — always pass T explicitly.
    return reactor.kinetics(reactor._reaction_partial_pressures(c_ret, T_ret, p_ret), T_ret)


def extract_fields(reactor: MembraneReactor) -> dict[str, np.ndarray]:
    """The field arrays cached per case (and shipped in the dataset)."""
    c = reactor.cpT[..., :-2]
    p = reactor.cpT[..., -2]
    T = reactor.cpT[..., -1]
    y = c / np.clip(np.sum(c, axis=-1, keepdims=True), kpi_mod.F_SMALL, None)

    c_perm, c_ret = reactor._split_perm_and_ret(c)
    p_perm, p_ret = reactor._split_perm_and_ret(p)
    T_perm, T_ret = reactor._split_perm_and_ret(T)
    y_perm, y_ret = reactor._split_perm_and_ret(y)
    flux_ret_ax, flux_ret_rad, flux_perm_ax, flux_perm_rad = _total_flux_fields(reactor)

    return {
        "z_c": np.asarray(reactor.z_c),
        "z_f": np.asarray(reactor.z_f),
        "r_c_perm": np.asarray(reactor.r_c_perm),
        "r_c_ret": np.asarray(reactor.r_c_ret),
        "r_f_perm": np.asarray(reactor.r_f_perm),
        "r_f_ret": np.asarray(reactor.r_f_ret),
        "T_perm": T_perm, "T_ret": T_ret,
        "p_perm_bar": p_perm / 1e5, "p_ret_bar": p_ret / 1e5,
        "y_perm": y_perm, "y_ret": y_ret,
        "c_perm": c_perm, "c_ret": c_ret,
        "u_perm_ax": np.asarray(reactor.u_perm_ax),
        "u_perm_rad": np.asarray(reactor.u_perm_rad),
        "u_ret_ax": np.asarray(reactor.u_ret_ax),
        "u_ret_rad": np.asarray(reactor.u_ret_rad),
        "flux_ret_ax": flux_ret_ax, "flux_ret_rad": flux_ret_rad,
        "flux_perm_ax": flux_perm_ax, "flux_perm_rad": flux_perm_rad,
        "reaction_source_ret": _reaction_source(reactor),
    }


def extract_profiles(
    reactor: MembraneReactor,
    fields: dict[str, np.ndarray],
    flows: dict[str, np.ndarray],
    design_meta: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Axial profiles and the outlet KPIs, both from ``compute_flows``."""
    flows_ret_ax = flows["flows_ret_ax"]
    flows_perm_ax = flows["flows_perm_ax"]
    raw = kpi_mod.compute_kpis_from_flows(
        flows_ret_ax, flows_perm_ax, design_meta["W_cat"], design_meta["A_membrane_m2"]
    )
    X_H2 = _centerline_average(raw.pop("X_H2_profile"))
    NH3_rec = _centerline_average(raw.pop("NH3_rec_profile"))
    NH3_yield = _centerline_average(raw.pop("NH3_yield_profile"))

    z = fields["z_c"]
    w_ret = kpi_mod.area_weights(fields["r_f_ret"])
    w_perm = kpi_mod.area_weights(fields["r_f_perm"])
    mean_ret = lambda f: np.tensordot(f, w_ret, axes=([1], [0]))  # noqa: E731
    mean_perm = lambda f: np.tensordot(f, w_perm, axes=([1], [0]))  # noqa: E731

    p_ret_mean = mean_ret(fields["p_ret_bar"])
    profiles = pd.DataFrame({
        "z_m": z,
        "F_H2_ret_mol_s": _centerline_average(flows_ret_ax[:, 0]),
        "F_N2_ret_mol_s": _centerline_average(flows_ret_ax[:, 1]),
        "F_NH3_ret_mol_s": _centerline_average(flows_ret_ax[:, 2]),
        "F_H2_perm_mol_s": _centerline_average(flows_perm_ax[:, 0]),
        "F_N2_perm_mol_s": _centerline_average(flows_perm_ax[:, 1]),
        "F_NH3_perm_mol_s": _centerline_average(flows_perm_ax[:, 2]),
        "X_H2": X_H2,
        "NH3_recovery": NH3_rec,
        "NH3_yield": NH3_yield,
        "T_ret_mean_K": mean_ret(fields["T_ret"]),
        "T_perm_mean_K": mean_perm(fields["T_perm"]),
        "p_ret_mean_bar": p_ret_mean,
        "p_perm_mean_bar": mean_perm(fields["p_perm_bar"]),
        "y_NH3_ret_mean": mean_ret(fields["y_ret"][:, :, kpi_mod.INH3]),
        "y_NH3_perm_mean": mean_perm(fields["y_perm"][:, :, kpi_mod.INH3]),
        "CP_NH3": kpi_mod.cp_profile(fields["y_ret"], fields["r_f_ret"]),
        "J_NH3_membrane_mol_s_m": 2.0 * np.pi * fields["r_f_ret"][0] * fields["flux_ret_rad"][:, 0, 2],
        "J_H2_membrane_mol_s_m": 2.0 * np.pi * fields["r_f_ret"][0] * fields["flux_ret_rad"][:, 0, 0],
    })

    # DeltaT_max is the pointwise 2D hot-spot value; DeltaT_max_avg is the
    # maximum of the radially averaged profile — the like-for-like quantity
    # for any 1D comparison (a 1D model has no radial hot spot by
    # construction; comparing it against the pointwise value builds in a
    # median 8.5% observable mismatch no closure can bridge).
    T_ret_mean = mean_ret(fields["T_ret"])
    dT_point = float(np.max(fields["T_ret"]) - reactor.config.T_ret_in)
    dT_avg = float(np.max(T_ret_mean) - reactor.config.T_ret_in)
    extras = {
        "DeltaT_max": dT_point,
        "DeltaT_max_avg": dT_avg,
        "T_hotspot_excess_K": dT_point - dT_avg,
        "delta_p_ret_bar": float(p_ret_mean[0] - p_ret_mean[-1]),
    }
    return profiles, {**raw, **extras}


# ── Convergence and stability certificates ───────────────────────────

def _scalar_kpis(reactor: MembraneReactor, design_meta: dict[str, float]) -> dict[str, float]:
    """The published scalar KPIs recomputed from the current reactor state."""
    fr_ax, _, fp_ax, _ = reactor.compute_flows()
    raw = compute_kpis_from_flows(
        fr_ax, fp_ax, design_meta["W_cat"], design_meta["A_membrane_m2"]
    )
    out = {k: float(v) for k, v in raw.items() if np.ndim(v) == 0}
    _, T_ret = reactor._split_perm_and_ret(reactor.cpT[..., -1])
    out["DeltaT_max"] = float(np.max(T_ret) - reactor.config.T_ret_in)
    return out


def certify_convergence(
    reactor: MembraneReactor,
    status: Any,
    design_meta: dict[str, float],
    *,
    n_steps: int | None = None,
) -> dict[str, Any]:
    """Convergence certificate: march ``n_steps``
    extra pseudo-transient steps and record the resulting KPI drift.

    Converts "we chose this tolerance" into a per-case, machine-checkable
    claim that the published numbers had stopped moving. Mutates the reactor
    state (call it AFTER extracting fields/flows); restores the pre-march
    state if the certification march itself fails.
    """
    from reactor.numerical_safety import RecoverableNumericalError

    n_steps = settings.N_CERTIFY if n_steps is None else n_steps
    cert: dict[str, Any] = {
        "achieved_residual": float(status.steady_state_norm),
        "residual_kind": status.norm_kind,
        "class": status.outcome,
        "certify_steps": 0,
        "kpi_drift_rel": None,
    }
    if status.outcome not in ("converged", "floored"):
        cert["skipped"] = f"outcome {status.outcome!r} is not an accepted steady state"
        return cert
    before = _scalar_kpis(reactor, design_meta)
    backup = reactor.cpT.copy()
    dt = reactor.dt_max
    for done in range(1, n_steps + 1):
        # A near-marginal slow mode can make a single dt_max step overflow
        # (seen on G7_0.05, max Re lambda ~ -5e-4): retry the step with line
        # search at a smaller dt before declaring the march failed.
        step_ok = False
        step_backup = reactor.cpT.copy()
        step_dt = dt
        for _ in range(3):
            c_old = reactor.cpT[..., :-2].copy()
            T_old = reactor.cpT[..., -1].copy()
            reactor._p_old = reactor.cpT[..., -2].copy()
            try:
                res = reactor._solve_cpT(c_old, T_old, step_dt,
                                         use_line_search=True)
            except RecoverableNumericalError:
                res = None
            if res is not None and np.isfinite(res.g_norm):
                step_ok = True
                break
            reactor._restore_state(step_backup)
            step_dt *= 0.1
        if not step_ok:
            reactor._restore_state(backup)
            cert["skipped"] = "certification march failed; state restored"
            return cert
    after = _scalar_kpis(reactor, design_meta)
    cert["certify_steps"] = n_steps
    drift = {
        k: abs(after[k] - before[k]) / max(abs(before[k]), 1e-12) for k in before
    }
    cert["kpi_drift_rel"] = drift
    # A certified state must not move: 20 dt_max steps on a genuinely
    # floored/converged state drift ~1e-9 (measured). Anything near the
    # publication accuracy target means the solve stopped too early.
    cert["kpi_drift_ok"] = bool(max(drift.values()) <= settings.CERTIFICATE_DRIFT_MAX)
    cert["achieved_residual"] = float(reactor._compute_steady_state_norm())
    return cert


def stability_certificate(reactor: MembraneReactor, k: int = 10) -> dict[str, Any]:
    """Eigenvalue certificate for a (suspected-unstable) steady state."""
    try:
        return stability.leading_eigenvalues(reactor, k=k)
    except Exception as exc:  # Arpack non-convergence, singular factorization
        return {"error": f"{type(exc).__name__}: {exc}"}


# ── Single case ───────────────────────────────────────────────────────

def run_case_2d(
    row: pd.Series,
    resolution_name: str,
    *,
    force: bool = False,
    verbose: int = 1,
) -> dict[str, Any]:
    """Solve (or load) one 2D case and return its cached KPI record."""
    case_id = str(row["Case_ID"])
    case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id)

    if force:
        cache.clear(case_dir)
    if cache.is_complete(case_dir, settings.resolution(resolution_name)):
        return cache.load_kpis(case_dir)

    config, design_meta = build_config_2d(row, resolution_name)
    reactor = MembraneReactor(config=config)
    start = time.perf_counter()
    status = reactor.solve(verbose=verbose, dt_init=settings.DT_INIT, return_status=True)
    runtime_s = time.perf_counter() - start

    acceptance = kpi_mod.solver_acceptance(
        status, config.steady_state_atol, settings.STEADY_STATE_ACCEPT_FACTOR
    )
    base = {
        "Case_ID": case_id,
        "Description": str(row["Description"]),
        "family": kpi_mod.family_code(case_id),
        "model": settings.MODEL_2D,
        "resolution": resolution_name,
        "solve_converged": bool(status.converged),
        "solve_outcome": str(status.outcome),
        "norm_kind": str(status.norm_kind),
        "used_temperature_continuation": bool(status.used_temperature_continuation),
        "solver_accepted": bool(acceptance["accepted"]),
        "solver_accepted_close": bool(acceptance["accepted_close"]),
        "solver_acceptance_reason": acceptance["reason"],
        "runtime_s": runtime_s,
        "solver_steps_attempted": int(status.num_steps_attempted),
        "solver_steps_accepted": int(status.num_steps_accepted),
        "steady_state_norm": float(status.steady_state_norm),
        "best_steady_state_norm": float(status.best_steady_state_norm),
        "last_failure_message": status.last_failure_message,
        "N_mem": int(row["N_mem"]),
        "L_m": float(row["L_m"]),
        "r_max_m": float(row["r_max_m"]),
        "Dcat": float(row["Dcat"]),
        "GHSV_h": float(row["GHSV_h"]),
        "Sweep_Ratio": float(row["Sweep_Ratio"]),
        "H2_N2_ratio": float(row["H2_N2_ratio"]),
        "p_ret_bar": float(row["p_ret_bar"]),
        "p_perm_bar": float(row["p_perm_bar"]),
        "T_ret_K": float(row["T_ret_K"]),
        "T_perm_K": float(row["T_perm_K"]),
        **design_meta,
        "WHSV": kpi_mod.whsv(
            ghsv_h=float(row["GHSV_h"]),
            vol_flow_std_m3_s=float(row["GHSV_h"]) * design_meta["W_cat"] / config.rho_c / 3600.0,
            w_cat_kg=design_meta["W_cat"],
            dcat=config.Dcat,
            eps=config.eps,
        ),
        "WHSV_convention": settings.WHSV_CONVENTION,
    }

    if not acceptance["accepted"]:
        record = {**base, "status": "failed",
                  "message": status.last_failure_message or "steady-state norm above accept limit",
                  "oscillation_pseudo_transient": status.oscillation}
        cache.write(case_dir, config=config, kpis=record, status=status,
                    meta={"complete": True, "status": "failed", "runtime_s": runtime_s,
                          "steady_state_norm": float(status.steady_state_norm),
                          "solve_outcome": str(status.outcome),
                          "provenance": provenance.stamp()})
        if verbose:
            print(f"  FAILED ({status.outcome}): {record['message']}", flush=True)
        return record

    # Item 4 of the convergence plan: every accepted steady state gets the
    # eigenvalue certificate (a few seconds against the case's solve time).
    # Heuristics are not enough — G6_598 was landed by an early steady-Newton
    # jump with no ladder and no observed orbit, and Newton is blind to
    # dynamic stability on any path. A positive-real pair means the physical
    # attractor is a limit cycle, so the case is reported "oscillatory" with
    # its unstable steady state attached. Run BEFORE certification: an
    # unstable state must never be marched.
    eig_cert = stability_certificate(reactor)
    unstable = bool(eig_cert and eig_cert.get("max_real_part", 0.0) > 0.0)

    if unstable:
        # Do not march an unstable steady state — the drift would measure the
        # instability, not the convergence. The eigenvalues are the evidence.
        convergence_cert: dict[str, Any] = {
            "achieved_residual": float(status.steady_state_norm),
            "residual_kind": str(status.norm_kind),
            "class": "oscillatory",
            "certify_steps": 0,
            "kpi_drift_rel": None,
            "skipped": ("dynamically unstable steady state: a certification "
                        "march would walk away; see eigenvalue_certificate"),
        }
    else:
        # Certify-and-resume: the certificate is enforced, not just
        # recorded. A failing drift means the published numbers were still
        # moving (jump-landed states that Newton left short of the marched
        # floor, or slow-creep floored states) — resume the same
        # deterministic march and re-certify. This is not a warm start
        # between cases; it is the same case continuing its own solve.
        resume_rounds = 0
        while True:
            convergence_cert = certify_convergence(reactor, status, design_meta)
            if (convergence_cert.get("kpi_drift_ok") is not False
                    or resume_rounds >= settings.CERTIFY_MAX_RESUMES):
                break
            resume_rounds += 1
            if verbose:
                drift = max(convergence_cert["kpi_drift_rel"].values())
                print(f"  certificate drift {drift:.1e} > "
                      f"{settings.CERTIFICATE_DRIFT_MAX:g} — resuming march "
                      f"(round {resume_rounds})", flush=True)
            reactor.solve(num_timesteps=settings.CERTIFY_RESUME_STEPS,
                          dt_init=status.final_dt, verbose=0,
                          return_status=True)
        convergence_cert["resume_rounds"] = resume_rounds
        if resume_rounds:
            # The state moved meaningfully: refresh the spectrum.
            eig_cert = stability_certificate(reactor)

    # Extract AFTER certification: the published fields, flows and KPIs are
    # those of the certified (settled) state.
    fields = extract_fields(reactor)
    fr_ax, fr_mem, fp_ax, fp_mem = reactor.compute_flows()
    flows = {"flows_ret_ax": fr_ax, "flows_ret_mem": fr_mem,
             "flows_perm_ax": fp_ax, "flows_perm_mem": fp_mem}
    profiles, raw_kpis = extract_profiles(reactor, fields, flows, design_meta)

    balance = kpi_mod.check_element_balance(
        fr_ax, fp_ax, settings.ELEMENT_BALANCE_RTOL, settings.ELEMENT_BALANCE_ATOL
    )
    if unstable:
        case_status = "oscillatory"
    elif not balance["element_balance_ok"]:
        case_status = "flagged_balance"
    else:
        case_status = str(status.outcome)  # "converged" or "floored"
    record = {
        **base,
        "status": case_status,
        "dynamically_unstable": unstable,
        **kpi_mod.to_presentation_units(raw_kpis),
        **kpi_mod.cp_summary(fields["y_ret"], fields["r_f_ret"], fields["z_c"]),
        **{k: v for k, v in balance.items() if k != "balance_species"},
        "score": kpi_mod.compute_score(raw_kpis, raw_kpis["DeltaT_max"], raw_kpis["delta_p_ret_bar"]),
        "convergence_certificate": convergence_cert,
        "eigenvalue_certificate": eig_cert,
        # Classification evidence only: pseudo-transient time is not physical
        # time, so this amplitude is not the physical cycle's amplitude.
        "oscillation_pseudo_transient": status.oscillation,
    }
    cache.write(
        case_dir, config=config, kpis=record, status=status,
        fields=fields, flows=flows, profiles=profiles,
        meta={"complete": True, "status": record["status"], "runtime_s": runtime_s,
              "steady_state_norm": float(status.steady_state_norm),
              "solve_outcome": str(status.outcome),
              "provenance": provenance.stamp()},
    )
    if verbose:
        drift_note = ""
        if convergence_cert.get("kpi_drift_rel") is not None and not convergence_cert.get("kpi_drift_ok", True):
            drift_note = (f" | CERT DRIFT "
                          f"{max(convergence_cert['kpi_drift_rel'].values()):.1e}")
        print(
            f"  {record['status']} | prod={record['NH3_prod_out']:.3f} mmol/g/h | "
            f"X_H2={record['X_H2_out']:.2f}% | CP_min={record['CP_min']:.3f} | "
            f"{runtime_s:.1f}s{drift_note}",
            flush=True,
        )
    return record


# ── 1D cases ──────────────────────────────────────────────────────────

def cache_expectation(resolution_name: str, model: str) -> dict[str, Any]:
    """Config values a cached solve must carry to be reused for this model.

    2D: the tier's grid and weighted tolerance. 1D: the tier's num_z with
    the 1D solver's own (absolute-norm) tolerance; the corrected model
    additionally pins the fitted Sherwood coefficients, so a refit
    invalidates its cache.
    """
    res = settings.resolution(resolution_name)
    if model == settings.MODEL_2D:
        return {"num_r": res.num_r, "num_z": res.num_z,
                "steady_state_atol": res.steady_state_atol}
    expect: dict[str, Any] = {
        "num_z": res.num_z,
        "steady_state_atol": settings.SOLVER_1D["steady_state_atol"],
    }
    if model == settings.MODEL_1D_CORRECTED:
        try:
            fit = closures.load_sh_fit(resolution_name)
            expect["sh_cp_coeff"] = fit["coeff"]
            expect["sh_cp_exp"] = fit["exp"]
        except FileNotFoundError:
            pass
    elif model == settings.MODEL_1D_SCREENED:
        expect["cp_closure"] = "screened"
        try:
            fit = closures.load_screened_fit(resolution_name)
            expect["screen_c1"] = fit["c_screen"]
            expect["screen_c2"] = fit["c_cond_floor"]
        except FileNotFoundError:
            pass
    return expect


def _one_d_class(model: str):
    """The ``MembraneReactor1D`` class of the variant wired to this model name."""
    module_name = {
        settings.MODEL_1D: settings.ONE_D_VARIANT,
        settings.MODEL_1D_CORRECTED: settings.ONE_D_CORRECTED_VARIANT,
        settings.MODEL_1D_SCREENED: settings.ONE_D_CORRECTED_VARIANT,
    }[model]
    return importlib.import_module(f"reactor.{module_name}").MembraneReactor1D


def build_config_1d(row: pd.Series, resolution_name: str, model: str):
    """1D config for one case: same axial grid as the 2D tier.

    The corrected model reads its Sherwood coefficients from the fit that
    ``closures.fit_sh_kappa`` wrote next to the 2D cache — never from
    hand-copied constants — so a refit invalidates the corrected-1D cache
    through the config comparison.
    """
    res = settings.resolution(resolution_name)
    extra = None
    if model == settings.MODEL_1D_CORRECTED:
        fit = closures.load_sh_fit(resolution_name)
        extra = {"sh_cp_coeff": fit["coeff"], "sh_cp_exp": fit["exp"]}
    elif model == settings.MODEL_1D_SCREENED:
        fit = closures.load_screened_fit(resolution_name)
        extra = {"cp_closure": "screened",
                 "screen_c1": fit["c_screen"],
                 "screen_c2": fit["c_cond_floor"]}
    config, design_meta, _ = build_case_config_1d(
        row,
        trace_nh3=settings.TRACE_NH3,
        num_z=res.num_z,
        extra_config=extra,
        **settings.SOLVER_1D,
    )
    return config, design_meta


def _scalar_kpis_1d(reactor, design_meta: dict[str, float]) -> dict[str, float]:
    """Scalar KPIs (plus DeltaT_max) recomputed from the current 1D reactor state."""
    fr_ax, _, fp_ax, _ = reactor.compute_flows()
    raw = compute_kpis_from_flows(
        fr_ax, fp_ax, design_meta["W_cat"], design_meta["A_membrane_m2"]
    )
    out = {k: float(v) for k, v in raw.items() if np.ndim(v) == 0}
    out["DeltaT_max"] = float(np.max(reactor.cpT[:, 1, -1]) - reactor.T_ret_in)
    return out


def certify_convergence_1d(
    reactor,
    status: Any,
    design_meta: dict[str, float],
    *,
    n_steps: int | None = None,
) -> dict[str, Any]:
    """1D twin of :func:`certify_convergence` (same discipline, own API)."""
    from reactor.numerical_safety import RecoverableNumericalError

    n_steps = settings.N_CERTIFY if n_steps is None else n_steps
    cert: dict[str, Any] = {
        "achieved_residual": float(status.steady_state_norm),
        "residual_kind": "absolute",
        "class": "converged" if status.converged else "failed",
        "certify_steps": 0,
        "kpi_drift_rel": None,
    }
    before = _scalar_kpis_1d(reactor, design_meta)
    backup = reactor.cpT.copy()
    for _ in range(n_steps):
        step_backup = reactor.cpT.copy()
        step_dt = reactor.dt_max
        step_ok = False
        for _retry in range(3):
            c_old = reactor.cpT[..., :-2].copy()
            T_old = reactor.cpT[..., -1].copy()
            try:
                res = reactor._solve_cpT(c_old, T_old, step_dt,
                                         use_line_search=True)
            except RecoverableNumericalError:
                res = None
            if res is not None and np.isfinite(res.g_norm):
                step_ok = True
                break
            reactor._restore_state(step_backup)
            step_dt *= 0.1
        if not step_ok:
            reactor._restore_state(backup)
            cert["skipped"] = "certification march failed; state restored"
            return cert
    after = _scalar_kpis_1d(reactor, design_meta)
    cert["certify_steps"] = n_steps
    drift = {k: abs(after[k] - before[k]) / max(abs(before[k]), 1e-12)
             for k in before}
    cert["kpi_drift_rel"] = drift
    cert["kpi_drift_ok"] = bool(max(drift.values()) <= settings.CERTIFICATE_DRIFT_MAX)
    cert["achieved_residual"] = float(reactor._compute_steady_state_norm())
    return cert


def extract_fields_1d(reactor) -> dict[str, np.ndarray]:
    """The 1D field arrays cached per case (permeate row 0, retentate row 1)."""
    c = reactor.cpT[..., :-2]
    return {
        "z_c": np.asarray(reactor.z_c),
        "z_f": np.asarray(reactor.z_f),
        "T_perm": reactor.cpT[:, 0, -1].copy(),
        "T_ret": reactor.cpT[:, 1, -1].copy(),
        "p_perm_bar": reactor.cpT[:, 0, -2] / 1e5,
        "p_ret_bar": reactor.cpT[:, 1, -2] / 1e5,
        "c_perm": c[:, 0, :].copy(),
        "c_ret": c[:, 1, :].copy(),
        "u_perm_ax": np.asarray(reactor.u_perm_ax),
        "u_ret_ax": np.asarray(reactor.u_ret_ax),
    }


def _seed_reactor_from_2d(reactor, case_dir_2d) -> None:
    """Deterministic initial state from the certified 2D solution's
    cross-sectional averages.

    Used only as a rescue when a cold 1D solve fails: the low-GHSV cases sit
    in the same multiplicity window as their 2D counterparts, and the 1D
    solver has no escalation ladder, so cold marching can trap in the wrong
    basin (G2_50: dt pinned at 1e-5, residual at 37, for thousands of
    steps). Seeding from the 2D average is fully scripted and reproducible
    — never a warm start between cases — and lands the 1D solution on the
    SAME operating branch as the 2D one, which is the branch a 1D-vs-2D
    comparison must be made on.
    """
    fields = cache.load_fields(case_dir_2d)
    w_ret = kpi_mod.area_weights(fields["r_f_ret"])
    w_perm = kpi_mod.area_weights(fields["r_f_perm"])
    seeded = reactor.cpT.copy()
    seeded[:, 1, :-2] = np.tensordot(fields["c_ret"], w_ret, axes=([1], [0]))
    seeded[:, 0, :-2] = np.tensordot(fields["c_perm"], w_perm, axes=([1], [0]))
    seeded[:, 1, -2] = np.tensordot(fields["p_ret_bar"] * 1e5, w_ret, axes=([1], [0]))
    seeded[:, 0, -2] = np.tensordot(fields["p_perm_bar"] * 1e5, w_perm, axes=([1], [0]))
    seeded[:, 1, -1] = np.tensordot(fields["T_ret"], w_ret, axes=([1], [0]))
    seeded[:, 0, -1] = np.tensordot(fields["T_perm"], w_perm, axes=([1], [0]))
    reactor._restore_state(seeded)


def run_case_1d(
    row: pd.Series,
    resolution_name: str,
    *,
    model: str = settings.MODEL_1D,
    force: bool = False,
    verbose: int = 1,
) -> dict[str, Any]:
    """Solve (or load) one 1D case and return its cached KPI record."""
    case_id = str(row["Case_ID"])
    case_dir = settings.case_cache_dir(resolution_name, model, case_id)

    if force:
        cache.clear(case_dir)
    if cache.is_complete(case_dir, expect=cache_expectation(resolution_name, model)):
        return cache.load_kpis(case_dir)

    config, design_meta = build_config_1d(row, resolution_name, model)
    reactor = _one_d_class(model)(config=config)
    start = time.perf_counter()
    status = reactor.solve(dt_init=settings.DT_INIT_1D, return_status=True,
                           verbose=0 if verbose < 2 else verbose)
    runtime_s = time.perf_counter() - start

    acceptance = kpi_mod.solver_acceptance(
        status, config.steady_state_atol, settings.STEADY_STATE_ACCEPT_FACTOR
    )
    base = {
        "Case_ID": case_id,
        "Description": str(row["Description"]),
        "family": kpi_mod.family_code(case_id),
        "model": model,
        "resolution": resolution_name,
        "solve_converged": bool(status.converged),
        "solver_accepted": bool(acceptance["accepted"]),
        "solver_acceptance_reason": acceptance["reason"],
        "runtime_s": runtime_s,
        "solver_steps_attempted": int(status.num_steps_attempted),
        "steady_state_norm": float(status.steady_state_norm),
        "N_mem": int(row["N_mem"]),
        "L_m": float(row["L_m"]),
        "r_max_m": float(row["r_max_m"]),
        "Dcat": float(row["Dcat"]),
        "GHSV_h": float(row["GHSV_h"]),
        "Sweep_Ratio": float(row["Sweep_Ratio"]),
        "H2_N2_ratio": float(row["H2_N2_ratio"]),
        "p_ret_bar": float(row["p_ret_bar"]),
        "p_perm_bar": float(row["p_perm_bar"]),
        "T_ret_K": float(row["T_ret_K"]),
        "T_perm_K": float(row["T_perm_K"]),
        **design_meta,
        "sh_cp_coeff": float(config.sh_cp_coeff),
        "sh_cp_exp": float(config.sh_cp_exp),
        "cp_closure": str(config.cp_closure),
        "screen_c1": float(config.screen_c1),
        "screen_c2": float(config.screen_c2),
    }

    # Certify-and-resume, same discipline as 2D. The 1D solver has no
    # floored classifier, so the certificate doubles as the floored test: a
    # state whose KPIs do not move over the certification march is a
    # converged iteration even if the residual threshold was unreachable.
    def _certify_loop(current_status):
        """Certify the KPI drift, resuming the march while it fails (bounded rounds)."""
        rounds = 0
        while True:
            cert = certify_convergence_1d(reactor, current_status, design_meta)
            if (cert.get("kpi_drift_ok") is not False
                    or rounds >= settings.CERTIFY_MAX_RESUMES):
                break
            rounds += 1
            if verbose:
                drift = max(cert["kpi_drift_rel"].values())
                print(f"  certificate drift {drift:.1e} — resuming march "
                      f"(round {rounds})", flush=True)
            reactor.solve(num_timesteps=settings.CERTIFY_RESUME_STEPS,
                          dt_init=current_status.final_dt, verbose=0,
                          return_status=True)
        cert["resume_rounds"] = rounds
        return cert

    def _is_certified(cert) -> bool:
        # A passing KPI-drift certificate is only meaningful near a steady
        # state: at a lost state (residual >> tolerance) dt_max Newton steps
        # barely move, so the KPIs look stagnant and the drift test passes
        # falsely (caught on G1_50: accepted at residual 27 with nonsense
        # KPIs). The residual sanity bound closes that hole.
        """True when the certificate passes both the KPI-drift and residual sanity checks."""
        return (cert.get("kpi_drift_ok") is True
                and float(cert.get("achieved_residual", np.inf))
                <= 10.0 * config.steady_state_atol)

    convergence_cert = _certify_loop(status)
    init_mode = "cold"
    certified = _is_certified(convergence_cert)
    if not (acceptance["accepted"] or certified):
        # Rescue: restart from the certified 2D solution's cross-sectional
        # average (see _seed_reactor_from_2d). Only fires when the cold
        # solve is genuinely lost.
        case_dir_2d = settings.case_cache_dir(
            resolution_name, settings.MODEL_2D, case_id
        )
        if cache.is_complete(case_dir_2d, settings.resolution(resolution_name)):
            if verbose:
                print(f"  cold 1D solve lost (ss={status.steady_state_norm:.2e}) "
                      f"— reseeding from the 2D average", flush=True)
            _seed_reactor_from_2d(reactor, case_dir_2d)
            status = reactor.solve(
                num_timesteps=settings.SOLVER_1D["num_timesteps"],
                dt_init=1e-4, return_status=True,
                verbose=0 if verbose < 2 else verbose,
            )
            runtime_s = time.perf_counter() - start
            acceptance = kpi_mod.solver_acceptance(
                status, config.steady_state_atol,
                settings.STEADY_STATE_ACCEPT_FACTOR,
            )
            convergence_cert = _certify_loop(status)
            init_mode = "2d_seeded"
            certified = _is_certified(convergence_cert)
    base["init"] = init_mode
    base["solve_converged"] = bool(status.converged)
    # The 1D solver restores its best-visited state at the end, so the
    # published state's residual is the certificate's achieved_residual;
    # the solver status carries the (possibly much larger) pre-restore
    # last-step norm. Record the honest one as the headline.
    base["solver_final_norm"] = float(status.steady_state_norm)
    base["steady_state_norm"] = float(
        convergence_cert.get("achieved_residual", status.steady_state_norm)
    )
    # solver_accepted is what downstream filters (figures) key on: a
    # certificate-accepted (floored) case IS an accepted result.
    base["solver_accepted"] = bool(acceptance["accepted"] or certified)
    if not acceptance["accepted"] and certified:
        base["solver_acceptance_reason"] = "accepted_certified_floored"

    if not (acceptance["accepted"] or certified):
        record = {**base, "status": "failed",
                  "message": status.last_failure_message or "steady-state norm above accept limit",
                  "convergence_certificate": convergence_cert}
        cache.write(case_dir, config=config, kpis=record, status=status,
                    meta={"complete": True, "status": "failed", "runtime_s": runtime_s,
                          "steady_state_norm": float(status.steady_state_norm),
                          "provenance": provenance.stamp()})
        if verbose:
            print(f"  FAILED: {record['message']}", flush=True)
        return record
    outcome_1d = "converged" if status.converged else "floored"
    convergence_cert["class"] = outcome_1d

    fields = extract_fields_1d(reactor)
    fr_ax, fr_mem, fp_ax, fp_mem = reactor.compute_flows()
    flows = {"flows_ret_ax": fr_ax, "flows_ret_mem": fr_mem,
             "flows_perm_ax": fp_ax, "flows_perm_mem": fp_mem}
    raw = kpi_mod.compute_kpis_from_flows(
        fr_ax, fp_ax, design_meta["W_cat"], design_meta["A_membrane_m2"]
    )
    X_H2 = _centerline_average(raw.pop("X_H2_profile"))
    NH3_rec = _centerline_average(raw.pop("NH3_rec_profile"))
    NH3_yield = _centerline_average(raw.pop("NH3_yield_profile"))
    profiles = pd.DataFrame({
        "z_m": fields["z_c"],
        "F_H2_ret_mol_s": _centerline_average(fr_ax[:, 0]),
        "F_N2_ret_mol_s": _centerline_average(fr_ax[:, 1]),
        "F_NH3_ret_mol_s": _centerline_average(fr_ax[:, 2]),
        "F_H2_perm_mol_s": _centerline_average(fp_ax[:, 0]),
        "F_N2_perm_mol_s": _centerline_average(fp_ax[:, 1]),
        "F_NH3_perm_mol_s": _centerline_average(fp_ax[:, 2]),
        "X_H2": X_H2,
        "NH3_recovery": NH3_rec,
        "NH3_yield": NH3_yield,
        "T_ret_K": fields["T_ret"],
        "T_perm_K": fields["T_perm"],
        "p_ret_bar": fields["p_ret_bar"],
        "p_perm_bar": fields["p_perm_bar"],
    })
    raw["DeltaT_max"] = float(np.max(fields["T_ret"]) - config.T_ret_in)
    p_ret = fields["p_ret_bar"]
    raw["delta_p_ret_bar"] = float(p_ret[0] - p_ret[-1])

    balance = kpi_mod.check_element_balance(
        fr_ax, fp_ax, settings.ELEMENT_BALANCE_RTOL, settings.ELEMENT_BALANCE_ATOL
    )
    record = {
        **base,
        "status": outcome_1d if balance["element_balance_ok"] else "flagged_balance",
        **kpi_mod.to_presentation_units(raw),
        **{k: v for k, v in balance.items() if k != "balance_species"},
        "convergence_certificate": convergence_cert,
    }
    cache.write(
        case_dir, config=config, kpis=record, status=status,
        fields=fields, flows=flows, profiles=profiles,
        meta={"complete": True, "status": record["status"], "runtime_s": runtime_s,
              "steady_state_norm": float(status.steady_state_norm),
              "provenance": provenance.stamp()},
    )
    if verbose:
        drift_note = ""
        if convergence_cert.get("kpi_drift_rel") is not None and not convergence_cert.get("kpi_drift_ok", True):
            drift_note = (f" | CERT DRIFT "
                          f"{max(convergence_cert['kpi_drift_rel'].values()):.1e}")
        print(
            f"  {record['status']} | prod={record['NH3_prod_out']:.3f} mmol/g/h | "
            f"X_H2={record['X_H2_out']:.2f}% | {runtime_s:.1f}s{drift_note}",
            flush=True,
        )
    return record


def trace_case_1d(
    row: pd.Series,
    resolution_name: str = "publication",
    *,
    verbose: int = 0,
) -> dict[str, Any]:
    """Exact-tracing validation: corrected 1D with the 2D-extracted k_cp.

    Runs the corrected 1D model with ``kcp_override`` set to the closure
    extracted from this case's cached 2D solution, and returns the 1D-traced
    KPIs next to the certified 2D ones. The remaining gap measures the
    closures the trace does NOT carry (heat, axial dispersion, averaging
    nonlinearity).
    """
    case_id = str(row["Case_ID"])
    case_dir_2d = settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id)
    if not cache.is_complete(case_dir_2d, settings.resolution(resolution_name)):
        raise FileNotFoundError(f"no complete 2D cache for {case_id}")
    closure = closures.case_closure(case_dir_2d)

    config, design_meta, _ = build_case_config_1d(
        row, trace_nh3=settings.TRACE_NH3,
        num_z=settings.resolution(resolution_name).num_z,
        **settings.SOLVER_1D,
    )
    reactor = _one_d_class(settings.MODEL_1D_CORRECTED)(config=config)
    if not np.allclose(reactor.z_c, closure["z_c"], rtol=1e-10, atol=1e-12):
        raise RuntimeError("1D and 2D axial grids differ; cannot trace directly")
    kcp = closure["kcp"]
    # Where the 2D solution shows no measurable polarization the closure is
    # undefined; a large k_cp (no CP resistance) is the faithful limit.
    reactor.kcp_override = np.where(np.isfinite(kcp), kcp, 1e3)
    status = reactor.solve(dt_init=settings.DT_INIT_1D, return_status=True,
                           verbose=verbose)
    traced = _scalar_kpis_1d(reactor, design_meta)
    kpis_2d = cache.load_kpis(case_dir_2d)
    out = {"Case_ID": case_id, "converged": bool(status.converged),
           "steady_state_norm": float(status.steady_state_norm)}
    for key in ("X_H2_out", "NH3_prod_out", "NH3_rec_out", "NH3_purity_out",
                "DeltaT_max"):
        t = traced[key]
        d2 = kpis_2d.get(f"{key}_si", kpis_2d.get(key))
        out[f"{key}_traced"] = t
        out[f"{key}_2d"] = d2
        out[f"{key}_rel_err"] = abs(t - d2) / max(abs(d2), 1e-12)
    return out


# ── Sweep ─────────────────────────────────────────────────────────────

def run_sweep(
    case_table: pd.DataFrame,
    resolution_name: str,
    *,
    model: str = settings.MODEL_2D,
    force: bool = False,
    verbose: int = 1,
    runner: Callable[..., dict[str, Any]] | None = None,
    only: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Solve every case (skipping complete cache entries) and return the summary.

    Also writes ``summary_kpis_<model>.csv`` next to the cache.
    """
    if runner is None:
        runner = (run_case_2d if model == settings.MODEL_2D
                  else partial(run_case_1d, model=model))
    expect = cache_expectation(resolution_name, model)
    wanted = set(only) if only is not None else None

    total = len(case_table)
    for position, (_, row) in enumerate(case_table.iterrows(), start=1):
        case_id = str(row["Case_ID"])
        if wanted is not None and case_id not in wanted:
            continue
        case_dir = settings.case_cache_dir(resolution_name, model, case_id)
        cached = cache.is_complete(case_dir, expect=expect) and not force
        if verbose:
            marker = "cached" if cached else "solving"
            print(f"[{position}/{total}] {case_id} ({marker})", flush=True)
        runner(row, resolution_name, force=force, verbose=0 if cached else verbose)

    # Build the summary from the cache rather than from this run's records, so
    # a partial run (--case-id, or a resumed sweep) still writes a complete
    # table of everything solved so far.
    summary = build_summary(case_table, resolution_name, model=model)
    out = settings.summary_csv(resolution_name, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    if verbose:
        n_ok = int((summary["status"] != "failed").sum()) if not summary.empty else 0
        print(f"\n{n_ok}/{len(summary)} accepted, "
              f"{total - len(summary)} not yet cached -> {out}", flush=True)
    return summary


def build_summary(
    case_table: pd.DataFrame,
    resolution_name: str,
    *,
    model: str = settings.MODEL_2D,
) -> pd.DataFrame:
    """Collect the KPI records of every cached case, in case-table order."""
    expect = cache_expectation(resolution_name, model)
    records: list[dict[str, Any]] = []
    for _, row in case_table.iterrows():
        case_dir = settings.case_cache_dir(resolution_name, model, str(row["Case_ID"]))
        if cache.is_complete(case_dir, expect=expect):
            records.append(cache.load_kpis(case_dir))
    return pd.DataFrame(records)


def cp_extremes(
    case_table: pd.DataFrame,
    resolution_name: str,
    *,
    z_min: float = 0.05,
) -> pd.DataFrame:
    """Per-case CP_NH3 range over the membrane-active length.

    The cached KPI record carries ``CP_min``; testing the paper's claim that
    CP < 1 *everywhere* needs the maximum, so this reads it back from the
    cached fields. Failed cases have no fields and are skipped.
    """
    res = settings.resolution(resolution_name)
    rows: list[dict[str, Any]] = []
    for case_id in case_table["Case_ID"]:
        case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, str(case_id))
        if not cache.is_complete(case_dir, res):
            continue
        if cache.load_meta(case_dir).get("status") == "failed":
            continue
        fields = cache.load_fields(case_dir)
        cp = kpi_mod.cp_profile(fields["y_ret"], fields["r_f_ret"])
        active = cp[fields["z_c"] > z_min]
        rows.append({"Case_ID": case_id,
                     "CP_min": float(active.min()),
                     "CP_max": float(active.max()),
                     "CP_below_1_everywhere": bool(active.max() < 1.0)})
    return pd.DataFrame(rows)


def load_summary(resolution_name: str, model: str = settings.MODEL_2D) -> pd.DataFrame:
    """Read a summary CSV written by :func:`run_sweep`."""
    path = settings.summary_csv(resolution_name, model)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run the sweep first "
            f"(scripts/run_paper_sweep.py --resolution {resolution_name})"
        )
    return pd.read_csv(path)


def case_dir_for(resolution_name: str, case_id: str, model: str = settings.MODEL_2D) -> Path:
    """Cache directory of one case (thin wrapper over ``settings.case_cache_dir``)."""
    return settings.case_cache_dir(resolution_name, model, case_id)
