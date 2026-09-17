"""Exact 1D closures extracted from converged 2D solutions, and the Sh refit.

The cross-sectionally averaged 2D equations are a 1D model plus closure
terms. This module extracts the species closure *exactly* from cached 2D
cases:

    k_cp,i(z) = J_i(z) / (c_mean,i(z) - c_wall,i(z))

with ``J_i`` the membrane-face flux toward the permeate, ``c_mean`` the
area-weighted cross-sectional mean and ``c_wall`` the first-cell (membrane
wall) concentration — the same wall convention as the paper's CP figures.

Three uses:

* **Exact tracing** (`trace_case_1d`): run the corrected 1D model with the
  extracted ``k_cp`` as an override. By construction the 1D model then
  carries the 2D closure, so the remaining 1D-vs-2D gap measures everything
  *else* (heat closure, axial dispersion, averaging nonlinearity) — the
  cleanest decomposition of the dimensionality effect.
* **Refit** (`fit_sh_kappa`): fit ``Sh = coeff * kappa**exp`` (kappa =
  r_mem/r_max < 1) to the flux-weighted exact Sherwood numbers of the
  certified dataset, replacing the manuscript's hard-coded fit that was made
  against the under-converged archive with an ambiguous kappa definition.
  The fit is written to ``sh_cp_fit.json`` next to the 2D cache; the
  corrected-1D runner reads it from there — no hand-copied constants.
* **Diagnostics**: per-case Sh(z) profiles for Figures 12-14.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reactor.gas_mixture_correlations import GasMixtureCorrelations

from reactor.paper import cache, kpis as kpi_mod, settings

#: Species used for the scalar Sherwood fit: NH3 is the permeating product
#: whose polarization the paper quantifies (CP figures are NH3).
FIT_SPECIES = "NH3"

#: Relative concentration-difference floor below which k_cp = J/(dc) is
#: numerically undefined (no measurable polarization).
DC_REL_FLOOR = 1e-4


def sh_fit_path(resolution_name: str, kind: str = "flux_matched") -> Path:
    """Path of the Sh(kappa) fit JSON ("flux_matched" or "wall") for a resolution."""
    name = {"flux_matched": "sh_cp_fit.json", "wall": "sh_wall_fit.json"}[kind]
    return settings.cache_root(resolution_name) / name


# ── Exact closure extraction ─────────────────────────────────────────

def exact_kcp(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact CP mass-transfer coefficients from one converged 2D case.

    Returns a dict with per-(z, species) arrays:
    ``kcp`` [m/s] (NaN where polarization is too small to define it),
    ``J`` (membrane-face flux toward permeate, mol m^-2 s^-1),
    ``c_mean``, ``c_wall`` [mol m^-3].
    """
    c_ret = fields["c_ret"]                      # (nz, nr, nc)
    weights = kpi_mod.area_weights(fields["r_f_ret"])
    c_mean = np.tensordot(c_ret, weights, axes=([1], [0]))   # (nz, nc)
    c_wall = c_ret[:, 0, :]                                  # (nz, nc)
    # Membrane face is radial index 0; +r points away from the membrane, so
    # the flux toward the permeate is the negative radial flux.
    J = -fields["flux_ret_rad"][:, 0, :]                     # (nz, nc)

    dc = c_mean - c_wall
    scale = np.maximum(np.abs(c_mean), 1e-30)
    defined = np.abs(dc) > DC_REL_FLOOR * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        kcp = np.where(defined, J / np.where(defined, dc, 1.0), np.nan)
    # A negative k_cp means flux against the concentration difference —
    # possible transiently for a non-permeating species; not a closure.
    kcp = np.where(kcp > 0.0, kcp, np.nan)
    return {"kcp": kcp, "J": J, "c_mean": c_mean, "c_wall": c_wall}


def flux_matched_kcp(
    fields: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """The k_cp that makes the 1D membrane law reproduce the 2D flux.

    The corrected 1D law is ``J = beta * J_ideal`` with ``beta =
    k_cp/(k_cp + Perm*Rg*T_ret)`` and ``J_ideal`` the uncorrected flux
    evaluated at the bulk (cross-sectional mean) states. Solving
    ``beta = J_2D / J_ideal`` for k_cp folds *everything* the law lumps —
    retentate-side polarization, permeate-side polarization, and the
    temperature weighting — into the one coefficient the law actually has.
    This is the self-consistent closure for both the exact trace and the
    Sherwood fit (the wall-based :func:`exact_kcp` remains the physical
    film coefficient for the CP-narrative figures).

    Where ``beta`` falls outside (0, 1) — flux at or above the ideal, or
    against the ideal driving force — no positive k_cp exists; k_cp is NaN
    there and callers substitute the no-resistance limit.
    """
    from reactor.config import ReactorConfig, get_membrane_permeances

    w_ret = kpi_mod.area_weights(fields["r_f_ret"])
    w_perm = kpi_mod.area_weights(fields["r_f_perm"])
    c_ret = np.tensordot(fields["c_ret"], w_ret, axes=([1], [0]))     # (nz, nc)
    c_perm = np.tensordot(fields["c_perm"], w_perm, axes=([1], [0]))  # (nz, nc)
    T_ret = np.tensordot(fields["T_ret"], w_ret, axes=([1], [0]))     # (nz,)
    T_perm = np.tensordot(fields["T_perm"], w_perm, axes=([1], [0]))  # (nz,)
    J = -fields["flux_ret_rad"][:, 0, :]                              # (nz, nc)

    cfg_obj = ReactorConfig.from_dict(config)
    Rg = cfg_obj.Rg
    P0, EA = get_membrane_permeances(
        cfg_obj.species, cfg_obj, fields["z_c"], cfg_obj.Lsealing
    )
    perm = P0 * np.exp(-EA / (Rg * T_ret[:, None]))                   # (nz, nc)

    J_ideal = perm * Rg * (T_ret[:, None] * c_ret - T_perm[:, None] * c_perm)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(np.abs(J_ideal) > 0.0, J / J_ideal, np.nan)
        valid = np.isfinite(beta) & (beta > 0.0) & (beta < 1.0)
        kcp = np.where(
            valid,
            perm * Rg * T_ret[:, None] * beta / np.maximum(1.0 - beta, 1e-300),
            np.nan,
        )
    return {"kcp": kcp, "beta": beta, "J": J, "J_ideal": J_ideal,
            "c_mean": c_ret, "c_perm_mean": c_perm}


def exact_sh(
    fields: dict[str, np.ndarray],
    config: dict[str, Any],
    kcp: np.ndarray,
) -> np.ndarray:
    """Sherwood profile Sh_i(z) = k_cp * d_h / D_i on the mean state."""
    corr = GasMixtureCorrelations(config["species"], config["database"])
    weights = kpi_mod.area_weights(fields["r_f_ret"])
    y_mean = np.tensordot(fields["y_ret"], weights, axes=([1], [0]))  # (nz, nc)
    T_mean = np.tensordot(fields["T_ret"], weights, axes=([1], [0]))  # (nz,)
    p_mean = np.tensordot(fields["p_ret_bar"] * 1e5, weights, axes=([1], [0]))
    nz, nc = y_mean.shape
    D = corr.diffusion(
        y_mean.reshape(nz, 1, nc), T_mean.reshape(nz, 1), p_mean.reshape(nz, 1)
    )
    D = np.asarray(D)
    while D.ndim > 2:
        D = D.squeeze(axis=1)
    d_h = float(config["r_max"]) - float(config["r_min"])
    return kcp * d_h / np.maximum(D, 1e-30)


def case_closure(case_dir: Path, *, kind: str = "flux_matched") -> dict[str, Any]:
    """Exact closure bundle for one cached, accepted 2D case.

    ``kind="flux_matched"`` (default) is the self-consistent closure for the
    1D law (used for tracing and the Sh fit); ``kind="wall"`` is the
    physical film coefficient matching the paper's CP definition (used for
    the CP-narrative diagnostics).
    """
    fields = cache.load_fields(case_dir)
    config = cache.read_json(Path(case_dir) / "config.json")
    if kind == "flux_matched":
        ex = flux_matched_kcp(fields, config)
    elif kind == "wall":
        ex = exact_kcp(fields)
    else:
        raise ValueError(f"unknown closure kind {kind!r}")
    sh = exact_sh(fields, config, ex["kcp"])
    return {**ex, "sh": sh, "z_c": fields["z_c"], "config": config,
            "kind": kind}


def representative_sh(
    closure: dict[str, Any],
    species: str = FIT_SPECIES,
) -> float:
    """Flux-weighted axial average of the exact Sh for one species.

    Weighted by the membrane flux magnitude so the sealing region (zero
    flux) and low-transfer cells (noisy k_cp) contribute nothing.
    """
    i = closure["config"]["species"].index(species)
    sh = closure["sh"][:, i]
    w = np.abs(closure["J"][:, i])
    ok = np.isfinite(sh) & (w > 0.0)
    if not np.any(ok):
        return float("nan")
    return float(np.sum(sh[ok] * w[ok]) / np.sum(w[ok]))


# ── The Sh(kappa) refit ──────────────────────────────────────────────

def fit_sh_kappa(
    case_table: pd.DataFrame,
    resolution_name: str = "publication",
    *,
    species: str = FIT_SPECIES,
    kind: str = "flux_matched",
    write: bool = True,
) -> dict[str, Any]:
    """Fit ``Sh = coeff * kappa**exp`` on the certified 2D cache.

    ``kind="flux_matched"`` (default) fits the self-consistent 1D-law
    closure the corrected model consumes; ``kind="wall"`` fits the physical
    film coefficient (the paper's CP definition) for the CP-narrative
    figures. The two are written to separate files.

    kappa = r_mem/r_max (< 1), the canonical definition also used by
    ``membrane_reactor_1d_corrected.compute_kcp_sh``. Cases are pooled per kappa
    (log-mean over the sweep conditions sharing a geometry), then a
    least-squares line in log-log space gives (coeff, exp). Only accepted
    cases with cached fields enter; per-case values and the pooled scatter
    are recorded so the fit is auditable.
    """
    res = settings.resolution(resolution_name)
    per_case: list[dict[str, Any]] = []
    for case_id in case_table["Case_ID"]:
        case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, str(case_id))
        if not cache.is_complete(case_dir, res):
            continue
        if cache.load_meta(case_dir).get("status") == "failed":
            continue
        if not (Path(case_dir) / "fields.npz").exists():
            continue
        closure = case_closure(case_dir, kind=kind)
        cfg = closure["config"]
        kappa = float(cfg["r_min"]) / float(cfg["r_max"])
        sh = representative_sh(closure, species=species)
        if np.isfinite(sh) and sh > 0.0:
            per_case.append({"Case_ID": str(case_id), "kappa": kappa, "Sh": sh})

    if len(per_case) < 3:
        raise RuntimeError(
            f"only {len(per_case)} usable cases in the {resolution_name} 2D "
            f"cache — run the 2D sweep first"
        )

    df = pd.DataFrame(per_case)
    pooled = (
        df.assign(lnSh=np.log(df["Sh"]))
          .groupby("kappa")["lnSh"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    x = np.log(pooled["kappa"].to_numpy())
    y = pooled["mean"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    fit = {
        "coeff": float(np.exp(intercept)),
        "exp": float(slope),
        "kappa_definition": "r_mem / r_max",
        "closure_kind": kind,
        "species": species,
        "r_squared_pooled": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_cases": len(per_case),
        "n_kappa_groups": int(len(pooled)),
        "pooled": pooled.assign(Sh_group=np.exp(pooled["mean"]))
                        .drop(columns=["mean"]).to_dict("records"),
        "per_case": per_case,
        "resolution": resolution_name,
    }
    if write:
        out = sh_fit_path(resolution_name, kind)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fit, indent=2))
    return fit


def load_sh_fit(resolution_name: str, kind: str = "flux_matched") -> dict[str, Any]:
    """Load the Sh(kappa) fit written by :func:`fit_sh_kappa`; raise if absent."""
    path = sh_fit_path(resolution_name, kind)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run reactor.paper.closures.fit_sh_kappa "
            f"on the {resolution_name} 2D cache first"
        )
    return json.loads(path.read_text())


# ── Mechanistic (screened) closure calibration ───────────────────────

def screened_fit_path(resolution_name: str) -> Path:
    """Path of the calibrated screened-closure fit JSON for a resolution."""
    return settings.cache_root(resolution_name) / "screened_cp_fit.json"


def calibrate_screened_closure(
    case_table: pd.DataFrame,
    resolution_name: str = "publication",
    *,
    species: str = FIT_SPECIES,
    write: bool = True,
) -> dict[str, Any]:
    """Calibrate C_screen of ``reactor.cp_closure`` on the certified cache.

    The screening channel ``k = C_screen * (D/delta) * f_ann`` needs one O(1)
    profile-shape constant; with a single multiplicative parameter the
    flux-weighted log-least-squares optimum is closed-form:
    ``ln C = <ln(k_target / k_screen)>_w``. The conduction fallback constant
    is fixed at 1.0 (a physical bound, not a fit — it engages only for
    species the kinetics cannot heal). The fit is auditable: per-kappa
    geometric-mean bias of the calibrated model is recorded and should sit
    within ~10% of unity if the screening mechanism is dominant.
    """
    import importlib

    from reactor.cp_closure import annulus_screen_factor, screening_length
    from reactor.paper.case_setup import build_case_config_1d

    mod = importlib.import_module(f"reactor.{settings.ONE_D_CORRECTED_VARIANT}")
    res = settings.resolution(resolution_name)
    lnr, wts, kap_all = [], [], []
    for case_id in case_table["Case_ID"]:
        case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, str(case_id))
        if not cache.is_complete(case_dir, res):
            continue
        if not (Path(case_dir) / "fields.npz").exists():
            continue
        cl = case_closure(case_dir)
        cfg2d = cl["config"]
        fields = cache.load_fields(case_dir)
        w_ret = kpi_mod.area_weights(fields["r_f_ret"])
        c_mean = np.tensordot(fields["c_ret"], w_ret, axes=([1], [0]))
        T_mean = np.tensordot(fields["T_ret"], w_ret, axes=([1], [0]))
        p_mean = np.tensordot(fields["p_ret_bar"] * 1e5, w_ret, axes=([1], [0]))

        row = case_table[case_table["Case_ID"] == case_id].iloc[0]
        cfg1d, _, _ = build_case_config_1d(
            row, trace_nh3=settings.TRACE_NH3, num_z=res.num_z, **settings.SOLVER_1D
        )
        reactor = mod.MembraneReactor1D(config=cfg1d)
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
        delta = screening_length(D, dRdc)
        k_screen = np.where(
            np.isfinite(delta),
            D / np.maximum(delta, 1e-300) * annulus_screen_factor(a, b, delta),
            0.0,
        )
        i = cfg2d["species"].index(species)
        k_t = cl["kcp"][:, i]
        w = np.abs(cl["J"][:, i])
        ok = np.isfinite(k_t) & (k_t > 0) & (k_screen[:, i] > 0) & (w > 0)
        lnr.extend(np.log(k_t[ok] / k_screen[ok, i]))
        wts.extend(w[ok])
        kap_all.extend([a / b] * int(ok.sum()))

    lnr = np.asarray(lnr)
    wts = np.asarray(wts)
    kap_all = np.asarray(kap_all)
    if lnr.size < 100:
        raise RuntimeError("not enough calibration points — run the 2D sweep first")
    c_screen = float(np.exp(np.average(lnr, weights=wts)))
    bias = {}
    kap_rounded = np.round(kap_all, 6)
    for kv in sorted(set(kap_rounded)):
        m = kap_rounded == kv
        bias[f"{kv:.4f}"] = float(np.exp(np.average(lnr[m], weights=wts[m])) / c_screen)
    fit = {
        "c_screen": c_screen,
        "c_cond_floor": 1.0,
        "species": species,
        "n_points": int(lnr.size),
        "log_scatter_sd": float(np.sqrt(np.average((lnr - np.log(c_screen)) ** 2, weights=wts))),
        "per_kappa_bias": bias,
        "resolution": resolution_name,
    }
    if write:
        out = screened_fit_path(resolution_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fit, indent=2))
    return fit


def load_screened_fit(resolution_name: str) -> dict[str, Any]:
    """Load the calibration written by :func:`calibrate_screened_closure`; raise if absent."""
    path = screened_fit_path(resolution_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run "
            f"reactor.paper.closures.calibrate_screened_closure first"
        )
    return json.loads(path.read_text())
