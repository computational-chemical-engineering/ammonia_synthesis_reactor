"""Model validation for the paper: Rossetti kinetics and Weisz-Prater.

Two self-contained pieces feeding Figures 2-4:

* **Rossetti 1D kinetics validation** (Figures 3-4): every experimental
  point of Rossetti et al. (`data/inputs/ammonia_synthesis_data_rossetti_
  et_al.csv`) re-solved with the isothermal, membrane-free 1D model. The
  runner logic is ported from ``scripts/run_rossetti_1d.py`` (geometry and
  flow conversion kept identical); results are cached as a CSV next to the
  resolution's sweep caches so the figure cells re-read in milliseconds.
* **Weisz-Prater criterion** (Figure 2): no archived script existed. The
  criterion is evaluated against the *certified 2D fields*: the scan finds,
  per species, the worst local state (largest ``|R_i| / (D_eff,i c_i)``)
  over every accepted cached case, and the figure draws the resulting
  ``Phi_WP(r_p)`` curves. That anchors the "internal gradients negligible"
  claim to the actual operating envelope, hot spots included, rather than
  to a nominal inlet state.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reactor import MembraneReactor1D, ReactorConfig
from reactor.ammonia_synthesis_kinetics import AmmoniaSynthesisKinetics
from reactor.gas_mixture_correlations import GasMixtureCorrelations

from reactor.paper import cache, settings

# ── Rossetti 1D kinetics validation ──────────────────────────────────

#: Fixed-bed micro-reactor of the Rossetti experiments (ported verbatim
#: from scripts/run_rossetti_1d.py — the geometry the validation was
#: originally run with).
ROSSETTI_BED = {
    "W_cat_kg": 0.1e-3,     # catalyst mass
    "rho_c": 590.0,         # catalyst density [kg/m3]
    "eps": 0.45,            # bed porosity
    "dilution": 1.0 / 32.0, # catalyst : (catalyst+quartz) volume ratio
    "D_tube_m": 0.009,      # tube diameter
}

ROSSETTI_SOLVER = {
    "num_timesteps": 50,
    "max_newton_iterations": 20,
    "steady_state_atol": 1e-3,
    "steady_state_rtol": None,
    "dt_max": 1e3,
}
ROSSETTI_DT_INIT = 1e-3

# Acceptance follows the certificate-over-threshold discipline of the paper
# sweeps: the absolute-norm floor of these stiff micro-bed cases sits at
# ~1e-2 (measured: test 1 / GHSV 50000 floors at 9e-3 while the outlet NH3
# fraction moves <0.3% between 50 and 200 steps), so the residual label
# alone would mark every point failed. A point is accepted when the
# monitored KPI (outlet NH3 fraction) drifts less than KPI_DRIFT_MAX over a
# CERTIFY_STEPS continuation; otherwise the march resumes, up to
# MAX_RESUMES rounds of RESUME_STEPS.
ROSSETTI_CERTIFY = {
    "certify_steps": 10,
    "kpi_drift_max": 1e-3,
    "resume_steps": 50,
    "max_resumes": 3,
}


#: Radial cells for the 2D validation runs. The tube is isothermal and
#: membrane-free, so radial gradients are minimal — a 3-point spot check
#: at 24 cells agrees with the 1D model to <=0.24% on the outlet NH3
#: fraction across the GHSV range.
ROSSETTI_NUM_R_2D = 24

ROSSETTI_MODELS = ("1d", "2d")


def rossetti_results_path(model: str = "2d") -> Path:
    """Cached Rossetti validation table for one model ("1d" or "2d")."""
    if model not in ROSSETTI_MODELS:
        raise ValueError(f"model must be one of {ROSSETTI_MODELS}, got {model!r}")
    return settings.validation_dir() / f"rossetti_{model}.csv"


def load_rossetti_table() -> pd.DataFrame:
    """The experimental dataset: one row per test, one GHSV_* column per point."""
    return pd.read_csv(settings.rossetti_csv(), comment="#")


def build_rossetti_config(row: pd.Series, ghsv_h: float) -> tuple[ReactorConfig, dict[str, float]]:
    """Isothermal, membrane-free 1D config for one experimental point."""
    cfg = ReactorConfig.from_defaults()

    T_K = float(row["Temperature (°C)"]) + 273.15
    p_Pa = float(row["Pressure (bar)"]) * 1e5
    cfg.is_isothermal = True
    cfg.T_ret_in = cfg.T_perm_in = cfg.T_ret_init = cfg.T_perm_init = T_K
    cfg.p_ret_out = cfg.p_perm_out = p_Pa
    cfg.Perm_NH3 = 0.0
    cfg.Nm = 0

    # Feed composition from the H2/N2 ratio, with a small NH3 trace.
    ratio = float(row["H2/N2 Ratio"])
    y_H2 = ratio / (1.0 + ratio)
    y_N2 = 1.0 / (1.0 + ratio)
    y_ret = np.array([[[y_H2, y_N2, 1e-7]]], dtype=float)
    y_perm = np.array([[[0.0, 1.0, 0.0]]], dtype=float)
    cfg.y_ret_in = y_ret
    cfg.y_ret_init = y_ret.copy()
    cfg.y_perm_in = y_perm
    cfg.y_perm_init = y_perm.copy()

    # Bed geometry: catalyst diluted 1:32 in quartz, packed at porosity eps.
    bed = ROSSETTI_BED
    V_cat = bed["W_cat_kg"] / bed["rho_c"]
    V_bed = (V_cat / bed["dilution"]) / (1.0 - bed["eps"])
    A_bed = np.pi * bed["D_tube_m"] ** 2 / 4.0
    cfg.L = V_bed / A_bed
    cfg.r_min = 1e-6
    cfg.r_max = bed["D_tube_m"] / 2.0
    cfg.r_min_perm = 0.0
    cfg.r_max_perm = 0.0035
    cfg.eps = bed["eps"]
    cfg.rho_c = bed["rho_c"]
    cfg.Dcat = bed["dilution"]

    # GHSV (per catalyst volume, NTP) -> total inlet molar flow.
    T_NTP, p_atm, R_atm = 273.15, 1.0, 0.08206e-3  # K, atm, m3*atm/(mol*K)
    vol_flow_s = ghsv_h * V_cat / 3600.0
    cfg.F_ret_in = vol_flow_s * p_atm / (R_atm * T_NTP)
    cfg.F_perm_in = cfg.F_ret_in * 1e-6
    cfg.is_counter_current = False
    cfg.num_z = 100

    for key, value in ROSSETTI_SOLVER.items():
        setattr(cfg, key, value)
    return cfg, {"GHSV_h": float(ghsv_h), "V_cat_m3": V_cat}


def _solve_rossetti_point(row: pd.Series, ghsv_h: float, exp_volpct: float,
                          model: str) -> dict[str, Any]:
    """Solve one experimental point, certify by KPI drift, return the record."""
    from reactor import MembraneReactor

    cfg, meta = build_rossetti_config(row, ghsv_h)
    if model == "2d":
        cfg.num_r = ROSSETTI_NUM_R_2D
        reactor = MembraneReactor(config=cfg)
    else:
        reactor = MembraneReactor1D(config=cfg)
    status = reactor.solve(return_status=True, dt_init=ROSSETTI_DT_INIT, verbose=0)

    def _y_nh3() -> float:
        """Outlet NH3 mole fraction of the retentate at the current reactor state."""
        F_out = reactor.compute_flows()[0][-1, :]
        return float(F_out[2] / max(F_out.sum(), 1e-30))

    # Certify: KPI drift over a short continuation, resume if moving.
    cert = ROSSETTI_CERTIFY
    y_NH3_model = _y_nh3()
    drift = float("inf")
    for round_ in range(cert["max_resumes"] + 1):
        status = reactor.solve(
            num_timesteps=cert["certify_steps"], return_status=True,
            dt_init=status.final_dt, verbose=0)
        y_new = _y_nh3()
        drift = abs(y_new - y_NH3_model) / max(abs(y_new), 1e-12)
        y_NH3_model = y_new
        if drift <= cert["kpi_drift_max"] or round_ == cert["max_resumes"]:
            break
        status = reactor.solve(
            num_timesteps=cert["resume_steps"], return_status=True,
            dt_init=status.final_dt, verbose=0)
        y_NH3_model = _y_nh3()
    return {
        "Test": int(row["Test"]),
        "Pressure_bar": float(row["Pressure (bar)"]),
        "T_C": float(row["Temperature (°C)"]),
        "H2_N2_ratio": float(row["H2/N2 Ratio"]),
        "GHSV_h": ghsv_h,
        "NH3_exp_volpct": exp_volpct,
        "NH3_model_volpct": 100.0 * y_NH3_model,
        "solve_converged": bool(drift <= cert["kpi_drift_max"]),
        "kpi_drift_rel": drift,
        "steady_state_norm": float(status.steady_state_norm),
        "model": model,
        "num_z": int(cfg.num_z),
        "num_r": int(ROSSETTI_NUM_R_2D) if model == "2d" else 1,
        **meta,
    }


def run_rossetti(
    *,
    model: str = "1d",
    tests: "list[int] | None" = None,
    out_path: Path | None = None,
    force: bool = False,
    verbose: int = 0,
) -> pd.DataFrame:
    """Solve every Rossetti point; cache the table per model.

    The runs are resolution-independent (fixed 100-cell grid, isothermal,
    membrane-free); the cache lives under ``settings.validation_dir()``
    and is shared by both resolution tiers. The 1D table is ~120 solves at
    15-30 s each; the 2D table (``model="2d"``, 24 radial cells) is
    ~200 s per point — run it as a background script, optionally sharded
    over ``tests`` with per-shard ``out_path``s merged afterwards.
    """
    if model not in ROSSETTI_MODELS:
        raise ValueError(f"model must be one of {ROSSETTI_MODELS}, got {model!r}")
    out_path = rossetti_results_path(model) if out_path is None else Path(out_path)
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    table = load_rossetti_table()
    if tests is not None:
        table = table[table["Test"].isin(list(tests))]
    ghsv_cols = [c for c in table.columns if c.startswith("GHSV_")]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for _, row in table.iterrows():
        for col in ghsv_cols:
            raw = str(row[col]).strip()
            if pd.isna(row[col]) or raw in ("", ","):
                continue
            ghsv_h = float(col.split("_")[1])
            rows.append(_solve_rossetti_point(row, ghsv_h, float(raw), model))
            if verbose:
                rec = rows[-1]
                print(f"[{model}] Test {rec['Test']:>2} GHSV {ghsv_h:>8.0f}: "
                      f"exp {rec['NH3_exp_volpct']:5.2f}%  "
                      f"model {rec['NH3_model_volpct']:5.2f}%  "
                      f"drift {rec['kpi_drift_rel']:.1e} "
                      f"{'ok' if rec['solve_converged'] else 'STILL MOVING'}",
                      flush=True)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"Rossetti validation: {len(df)} points in "
              f"{time.perf_counter() - t0:.1f} s -> {out_path}", flush=True)
    return df


# ── Weisz-Prater criterion ───────────────────────────────────────────

#: Mole-fraction floor: below this a species' local Phi is not evaluated.
#: It excludes the inlet trace-NH3 region, where c_i -> 0 makes the
#: modulus diverge without physical meaning (NH3 is produced, not
#: consumed, there).
WP_Y_FLOOR = 1e-3

#: Effective-diffusivity factor eps_p/tau_p for the porous catalyst.
#: A typical value, NOT a measured one — the figure also draws the
#: molecular-diffusivity bound so the conclusion's sensitivity to this
#: assumption is visible.
WP_EFF_DIFF_FACTOR = 0.1


def weisz_prater_path(resolution_name: str) -> Path:
    """Cached Weisz-Prater scan JSON for one resolution."""
    return settings.cache_root(resolution_name) / "weisz_prater_scan.json"


def _weisz_prater_nominal(*, T_K: float = 623.0, p_bar: float = 80.0,
                          h2_n2_ratio: float = 2.0) -> dict[str, Any]:
    """Phi/(f_eff r_p^2) per reactant at the nominal worst operating state."""
    cfg = ReactorConfig.from_defaults()
    species = list(cfg.species)
    y = np.array([[h2_n2_ratio / (1 + h2_n2_ratio),
                   1.0 / (1 + h2_n2_ratio), 1e-9]])
    y = y / y.sum()
    T = np.full((1,), T_K)
    p = np.full((1,), p_bar * 1e5)
    c = y * (p / (cfg.Rg * T))[:, None]

    kinetics = AmmoniaSynthesisKinetics(species, T=T, p=p,
                                        rho_b=cfg.rho_b, rho_c=cfg.rho_c)
    R_bed = kinetics(c * cfg.Rg * T[:, None], T=T)
    R_part = np.abs(R_bed) / (cfg.Dcat * (1.0 - cfg.eps))

    corr = GasMixtureCorrelations(species, cfg.database)
    D = np.asarray(corr.diffusion(y.reshape(1, 1, 3), T.reshape(1, 1),
                                  p.reshape(1, 1)))
    while D.ndim > 2:
        D = D.squeeze(axis=1)
    D = np.maximum(D, 1e-12)
    q = (R_part / (D * np.maximum(c, 1e-30)))[0]
    return {
        "T_K": T_K, "p_bar": p_bar, "h2_n2_ratio": h2_n2_ratio,
        "phi_over_feff_rp2": {s: float(q[j]) for j, s in enumerate(species)
                              if s != "NH3"},
    }


def weisz_prater_scan(
    case_table: pd.DataFrame,
    resolution_name: str = "publication",
    *,
    force: bool = False,
    y_floor: float = WP_Y_FLOOR,
) -> dict[str, Any]:
    """Worst-case ``|R_i| / (D_i c_i)`` per species over the certified cache.

    For every accepted cached 2D case, the local volumetric reaction rate
    (per *catalyst-particle* volume: the bed-volume rate divided by
    ``Dcat*(1-eps)`` — conservative, treating dilution as inert particles)
    and the local mixture diffusivity are evaluated on the full (z, r)
    grid. The species-wise maximum of ``|R_i|/(D_i c_i)`` is
    ``Phi_WP / r_p^2`` before the effective-diffusivity factor; the figure
    multiplies in ``r_p^2`` and ``1/WP_EFF_DIFF_FACTOR``.
    """
    out_path = weisz_prater_path(resolution_name)
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    res = settings.resolution(resolution_name)
    worst: dict[str, dict[str, Any]] = {}
    n_cases = 0
    for case_id in case_table["Case_ID"]:
        case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, str(case_id))
        if not cache.is_complete(case_dir, res):
            continue
        if cache.load_meta(case_dir).get("status") == "failed":
            continue
        if not (Path(case_dir) / "fields.npz").exists():
            continue
        fields = cache.load_fields(case_dir)
        config = cache.read_json(Path(case_dir) / "config.json")
        species = list(config["species"])
        cfg = ReactorConfig.from_dict(config)

        c = fields["c_ret"]                    # (nz, nr, nc) [mol/m3]
        T = fields["T_ret"]                    # (nz, nr) [K]
        p = fields["p_ret_bar"] * 1e5          # (nz, nr) [Pa]
        c_tot = np.maximum(c.sum(axis=-1), 1e-30)
        y = c / c_tot[..., None]

        kinetics = AmmoniaSynthesisKinetics(
            species, T=T, p=p, rho_b=cfg.rho_b, rho_c=cfg.rho_c
        )
        p_partial = c * cfg.Rg * T[..., None]
        R_bed = kinetics(p_partial, T=T)       # (nz, nr, nc) [mol/m3_bed/s]
        R_part = np.abs(R_bed) / (cfg.Dcat * (1.0 - cfg.eps))

        corr = GasMixtureCorrelations(species, config["database"])
        D = np.asarray(corr.diffusion(y, T, p))
        while D.ndim > 3:
            D = D.squeeze(axis=-2)
        D = np.maximum(D, 1e-12)

        q = R_part / (D * np.maximum(c, 1e-30))   # Phi / (f_eff r_p^2)
        q = np.where(y > y_floor, q, 0.0)

        for i, sp in enumerate(species):
            flat = int(np.argmax(q[..., i]))
            iz, ir = np.unravel_index(flat, q[..., i].shape)
            val = float(q[iz, ir, i])
            if val > worst.get(sp, {}).get("phi_over_feff_rp2", -1.0):
                worst[sp] = {
                    "phi_over_feff_rp2": val,        # [1/m2]
                    "case_id": str(case_id),
                    "z_m": float(fields["z_c"][iz]),
                    "T_K": float(T[iz, ir]),
                    "p_bar": float(p[iz, ir] / 1e5),
                    "y": float(y[iz, ir, i]),
                    "c_mol_m3": float(c[iz, ir, i]),
                    "D_m2_s": float(D[iz, ir, i]),
                    "R_particle_mol_m3_s": float(R_part[iz, ir, i]),
                }
        # Snapshot at the point of maximum NH3 content: the state where
        # the (product) NH3 modulus is physically meaningful — the
        # membrane-stripped corners that dominate the raw per-species
        # maxima are exhausted states, not kinetically active ones.
        flat = int(np.argmax(y[..., 2]))
        iz, ir = np.unravel_index(flat, y[..., 2].shape)
        if float(y[iz, ir, 2]) > worst.get("_state_at_max_yNH3", {}).get("y_NH3", -1.0):
            worst["_state_at_max_yNH3"] = {
                "case_id": str(case_id),
                "z_m": float(fields["z_c"][iz]),
                "T_K": float(T[iz, ir]),
                "p_bar": float(p[iz, ir] / 1e5),
                "y_NH3": float(y[iz, ir, 2]),
                "phi_over_feff_rp2": {
                    s: float(q[iz, ir, j]) for j, s in enumerate(species)
                },
            }
        n_cases += 1

    if not worst:
        raise RuntimeError(
            f"no usable cases in the {resolution_name} 2D cache — run the 2D sweep first"
        )

    # Nominal-inlet evaluation for the REACTANTS — the manuscript's
    # convention ("maximum pressure, optimal temperature, inlet
    # composition"). NH3 is a trace at the inlet, so its modulus is
    # undefined here; the figure takes it from the max-y_NH3 state above.
    nominal = _weisz_prater_nominal()

    scan = {
        "nominal": nominal,
        "worst": worst,
        "n_cases": n_cases,
        "y_floor": y_floor,
        "dp_m": float(ReactorConfig.from_defaults().dp),
        "eff_diff_factor": WP_EFF_DIFF_FACTOR,
        "resolution": resolution_name,
        "definition": ("Phi_WP = R_particle * r_p^2 / (f_eff * D_i * c_i), "
                       "f_eff = eps_p/tau_p; worst local state over all "
                       "accepted cached 2D cases with y_i > y_floor"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scan, indent=2))
    return scan


# ── Membrane permeance validation (Figure 5) ─────────────────────────

#: The single-gas permeation test matrix behind Figure 5.
def permeation_cases_csv() -> Path:
    """The single-gas permeation test matrix CSV behind Figure 5."""
    return settings.PERMEATION_CASES_CSV


def permeation_exp_csv() -> Path:
    """The measured (digitized) single-gas permeance CSV behind Figure 5."""
    return settings.PERMEATION_EXP_CSV


def permeation_fit_path() -> Path:
    """Cached per-species Arrhenius permeance fit JSON."""
    return settings.validation_dir() / "permeation_arrhenius_fit.json"


def permeation_results_path() -> Path:
    """Cached apparent-permeance table of the single-gas module runs."""
    return settings.validation_dir() / "permeation_1d.csv"


def load_permeation_experiment() -> pd.DataFrame:
    """The measured single-gas permeances (currently digitized from the
    manuscript's Figure 5 — see the CSV header for provenance)."""
    return pd.read_csv(permeation_exp_csv(), comment="#")


def fit_permeance_arrhenius(*, write: bool = True) -> dict[str, Any]:
    """Per-species Arrhenius fit Perm = P0*exp(-EA/(Rg*T)) to the data.

    This is the membrane-transport sub-model of the manuscript's Figure 5
    ("2D Model (Arrhenius fit)"). Note these are the REAL CMS membrane's
    parameters; the reactor sweeps use the optimized target permeances of
    the config, which differ by orders of magnitude for H2 and N2.
    """
    import json

    exp = load_permeation_experiment()
    Rg = ReactorConfig.from_defaults().Rg
    fit: dict[str, Any] = {"law": "Perm = P0*exp(-EA/(Rg*T))",
                           "source": str(permeation_exp_csv().name)}
    for sp, sub in exp.groupby("species"):
        x = 1.0 / sub["T_K"].to_numpy()
        y = np.log(sub["permeance_mol_m-2_s-1_Pa-1"].to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        fit[sp] = {"P0": float(np.exp(intercept)), "EA": float(-slope * Rg),
                   "r_squared": float(r**2), "n_points": int(len(sub))}
    if write:
        path = permeation_fit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fit, indent=2))
    return fit


def build_permeation_config(row: pd.Series, fit: dict[str, Any]) -> ReactorConfig:
    """Single-gas module config, ported from scripts/run_permeation_single_gas_1d.py.

    Three deliberate deviations from the archived runner:

    * the membrane-fit Arrhenius parameters replace the reactor's
      optimized targets for ALL species;
    * ``Lsealing = 0`` — the config default (0.05 m) would silently
      deactivate half of the 0.097 m module (sealing applies to
      z <= Lsealing);
    * ``r_min = r_max_perm`` (the membrane interface, the paper-case
      convention). The archived runner set ``r_min = 1e-6``, which put
      the membrane transfer area at the axis — 5000x too small; as
      written it could never reproduce a measured permeance.
    """
    cfg = ReactorConfig.from_defaults()
    cfg.is_isothermal = True
    cfg.T_ret_in = cfg.T_ret_init = float(row["T_ret_K"])
    cfg.T_perm_in = cfg.T_perm_init = float(row["T_perm_K"])
    cfg.p_ret_out = float(row["p_ret_bar"]) * 1e5
    cfg.p_perm_out = float(row["p_perm_bar"]) * 1e5
    cfg.factor_react = 0.0
    cfg.L = float(row["L_m"])
    cfg.Lsealing = 0.0
    cfg.r_min = float(row["r_max_perm_m"])
    cfg.r_max = float(row["r_max_m"])
    cfg.r_min_perm = 0.0
    cfg.r_max_perm = float(row["r_max_perm_m"])
    cfg.Nm = int(row["N_mem"])
    cfg.Dcat = float(row["Dcat"])

    for sp in ("H2", "N2", "NH3"):
        setattr(cfg, f"P0_{sp}", fit[sp]["P0"])
        setattr(cfg, f"EA_{sp}", fit[sp]["EA"])

    F_feed = float(row["F_v_feed_m3_s"]) * 1e5 / (cfg.Rg * 273.15)
    cfg.F_ret_in = F_feed
    cfg.F_perm_in = F_feed * float(row["Sweep_Ratio"])

    comp = str(row["Component"]).strip().upper()
    idx = {"H2": 0, "N2": 1, "NH3": 2}[comp]
    y_ret = np.zeros((1, 1, 3))
    y_ret[..., idx] = 1.0
    cfg.y_ret_in = y_ret
    cfg.y_ret_init = y_ret.copy()
    y_perm = np.zeros((1, 1, 3))
    y_perm[..., 1] = 1.0                      # pure N2 sweep
    cfg.y_perm_in = y_perm
    cfg.y_perm_init = y_perm.copy()

    cfg.is_counter_current = bool(row["Is_Counter_Current"])
    cfg.num_z = 100
    cfg.num_timesteps = 200
    cfg.max_newton_iterations = 30
    cfg.steady_state_atol = 1e-4
    cfg.steady_state_rtol = 1e-3
    cfg.dt_max = 1e2
    cfg.rtol = 1e-6
    cfg.atol = 1e-8
    return cfg


def run_permeation(*, force: bool = False, verbose: int = 0) -> pd.DataFrame:
    """Solve the 15 single-gas cases (1D module); cache apparent permeances.

    The apparent permeance is the permeated flow of the test component
    over membrane area times the log-mean partial-pressure difference —
    the quantity a permeation experiment reports. It should reproduce the
    Arrhenius input law up to driving-force averaging.
    """
    out_path = permeation_results_path()
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    fit = fit_permeance_arrhenius()
    table = pd.read_csv(permeation_cases_csv(), encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        cfg = build_permeation_config(row, fit)
        comp = str(row["Component"]).strip().upper()
        idx = {"H2": 0, "N2": 1, "NH3": 2}[comp]
        reactor = MembraneReactor1D(config=cfg)
        status = reactor.solve(return_status=True, dt_init=1e-4, verbose=0)

        flows_ret_ax, _, flows_perm_ax, _ = reactor.compute_flows()
        F_perm = flows_perm_ax[:, idx]
        permeated = float(F_perm[-1] - F_perm[0])
        A_mem = 2.0 * np.pi * cfg.r_max_perm * cfg.L * cfg.Nm

        # log-mean partial-pressure difference between module ends
        def partial_p(flows_ax, p_pa):
            """Partial-pressure profile of the test component along the module."""
            F_tot = np.maximum(flows_ax.sum(axis=1), 1e-30)
            return flows_ax[:, idx] / F_tot * p_pa

        p_ret = partial_p(flows_ret_ax, cfg.p_ret_out)
        p_perm = partial_p(flows_perm_ax, cfg.p_perm_out)
        d1 = max(p_ret[0] - p_perm[0], 1e-6)
        d2 = max(p_ret[-1] - p_perm[-1], 1e-6)
        lmdf = (d1 - d2) / np.log(d1 / d2) if abs(d1 - d2) > 1e-9 else d1

        T = float(row["T_ret_K"])
        perm_law = fit[comp]["P0"] * np.exp(-fit[comp]["EA"] / (cfg.Rg * T))
        rows.append({
            "Case_ID": int(row["Case_ID"]),
            "Component": comp,
            "T_K": T,
            "p_ret_bar": float(row["p_ret_bar"]),
            "permeance_apparent": permeated / (A_mem * lmdf),
            "permeance_law": float(perm_law),
            "solve_converged": bool(status.converged),
            "steady_state_norm": float(status.steady_state_norm),
        })
        if verbose:
            r = rows[-1]
            print(f"{comp} T={T:.0f} K: apparent {r['permeance_apparent']:.3e} "
                  f"vs law {perm_law:.3e} "
                  f"({'ok' if status.converged else 'NOT CONV'})", flush=True)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
