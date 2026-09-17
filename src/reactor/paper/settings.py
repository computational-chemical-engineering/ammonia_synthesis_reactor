"""Settings for the paper pipeline: resolutions, paths and deferred decisions.

Everything the notebook is allowed to change lives here. The three switches
below are reporting-level switches. Each is a
one-line change plus a re-run — nothing downstream hard-codes past them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reactor.paths import project_root

# ── Cache/schema version ─────────────────────────────────────────────
# Bump when the cached artifact layout or the KPI definitions change, so
# stale caches are recomputed rather than silently reused.
# v2: steady-state norm switched from unscaled-absolute to error-weighted
#     RMS (CVODE convention). The tolerance
#     values look similar but MEAN something different; the resolution-aware
#     cache check cannot see that, so the schema bump invalidates everything.
CACHE_SCHEMA_VERSION = 2


# ── Resolutions ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Resolution:
    """Grid and tolerance settings for one resolution tier."""

    name: str
    num_r: int
    num_z: int
    steady_state_atol: float

    @property
    def slug(self) -> str:
        """Directory-name slug of this resolution (its name)."""
        return self.name


# Accuracy context (measured, floor-converged states; see
# measured during development): relative to 40x100, the draft grid
# shifts NH3 production by <=0.2% for most families but by ~8% for the
# low-GHSV G1 cases, and NH3 recovery by ~3% everywhere.
#
# Since the convergence rework (2026-08-17) the tolerance is the ERROR-WEIGHTED RMS norm
# (norm_kind="weighted", CVODE convention): wrms <= 1.0 is the canonical
# converged test, and "converged" additionally requires the outlet KPIs to
# be stagnant. Cases whose residual floor sits above the tolerance stop at
# their floor with stagnant KPIs and are accepted as "floored" — they are
# converged iterations against an unreachable threshold, not failures.
#
# Measured per-case floors in this norm (floor probes, cold starts,
# unreachable tolerance, stagnation-stopped):
#
#   40x100:  G2_50 0.82 (X_H2 51.548%, -0.65% vs the 51.884% two-route
#            reference), G8_0.06 0.87, G8_0.02 1.06, G7_0.02 1.67
#   24x60:   G1_50 0.26, G2_50 0.62
#
# versus absolute floors 7.3e-4 ... 1.9e-3 that shared one un-comparable
# scale. One tolerance of 1.0 now means the same thing for every case and
# both grids; the old per-tier values (draft 3e-2, publication 2e-3,
# absolute norm) are retired.
#
# The KPI-stagnation gate is what protects the slow low-GHSV cases: their
# X_H2 still climbs by 6-13 points AFTER the residual first crosses 1.0, so
# a pure threshold stop would sit 12-25% off; the gate keeps them marching
# to their floor (measured: G2_50 40x100 crosses wrms=1.0 at X_H2=45.7% and
# floors at 51.548%).
WRMS_TOLERANCE = 1.0

RESOLUTIONS: dict[str, Resolution] = {
    "draft": Resolution("draft", num_r=24, num_z=60, steady_state_atol=WRMS_TOLERANCE),
    "publication": Resolution("publication", num_r=40, num_z=100, steady_state_atol=WRMS_TOLERANCE),
}


def resolution(name: str) -> Resolution:
    """Look up a resolution tier by name; raise ValueError for unknown names."""
    try:
        return RESOLUTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown RESOLUTION {name!r}; expected one of {sorted(RESOLUTIONS)}"
        ) from None


# ── Solver settings shared by every paper case ───────────────────────
# These mirror the defaults of scripts/run_study_2d.py, which is how the
# archived dataset was produced. Do not change them per case: the solver
# work is finished and validated, this pipeline only consumes it.
SOLVER = {
    "factor_react": 1.0,
    "factor_p": 1.0,
    "factor_T": 1e-1,
    "num_timesteps": 3000,
    "max_newton_iterations": 10,
    "dt_max": 1e6,
    "rtol": 5e-2,
    "atol": 1e-3,
    "steady_state_rtol": 5e-2,
}

# Steady-state norm and stopping (weighted-RMS + KPI-stagnation gate).
# The wrms_* block tolerances are the calibrated ReactorConfig defaults —
# repeated here explicitly so the cached config.json records them and a
# recalibration invalidates the cache via the config comparison.
NORM_CONFIG = {
    "norm_kind": "weighted",
    "wrms_rtol_c": 1e-4,
    "wrms_atol_c_rel": 1e-7,
    "wrms_rtol_p": 1e-4,
    "wrms_atol_p_rel": 1e-7,
    "wrms_rtol_T": 1e-5,
    "wrms_atol_T": 1e-3,
    "kpi_stop": True,
    "kpi_check_every": 10,
    "kpi_rtol": 1e-3,
    "kpi_n_stagnant": 3,
    "steady_soft_step_cap": 400,
}

# Convergence certificate: extra steps marched at the end of an accepted
# solve to record the KPI drift (plan Item 5; ~10% overhead). A certified
# state drifts ~1e-9 relative (measured); a drift above the cap below flags
# a solve that stopped early (kpi_drift_ok=False in the certificate).
#
# The certificate is ENFORCED, not just recorded: when it fails, the runner
# resumes the pseudo-transient march (same deterministic solve, same case —
# not a warm start between cases) and re-certifies, up to
# CERTIFY_MAX_RESUMES rounds. Measured need: 7/52 publication cases —
# jump-landed states whose march kept improving (up to 3.5% on ΔT_max) and
# floored states still creeping ~0.3%/20 steps.
N_CERTIFY = 20
CERTIFICATE_DRIFT_MAX = 1e-3
CERTIFY_MAX_RESUMES = 3
CERTIFY_RESUME_STEPS = 400

DT_INIT = 1e-6
STEADY_STATE_ACCEPT_FACTOR = 1.5

# ── 1D solver settings ───────────────────────────────────────────────
# The 1D models keep their original absolute residual norm (the weighted
# machinery is 2D-only). They are cheap enough to run to a tight absolute
# tolerance, and the runner certifies the KPI drift exactly like the 2D
# path, so an unreachable tolerance ends certified, not failed.
SOLVER_1D = {
    # Soft budget: the 1D solver has no floored classifier, so it would
    # march its whole budget against an unreachable tolerance. Cases reach
    # their floor within ~200 steps; the certify-and-resume loop extends
    # the march (up to 3 x CERTIFY_RESUME_STEPS) only when the KPIs are
    # still moving.
    "num_timesteps": 400,
    "max_newton_iterations": 10,
    "dt_max": 1e6,
    "rtol": 5e-2,
    "atol": 1e-3,
    # Reachable stopping threshold (the 1D floors sit at ~1e-4..1e-3 in the
    # absolute norm). Accuracy does NOT rest on this number: the certificate
    # is the authority — a case whose KPIs still move at the crossing fails
    # its drift check and the runner resumes the march. Measured: 1e-3 is
    # ~4x faster than an unreachable 1e-5 (5-13 s/case vs ~27 s/case) with
    # KPIs identical to <0.1%.
    "steady_state_atol": 1e-3,
    "steady_state_rtol": None,
}
DT_INIT_1D = 1e-6

# Trace NH3 in the retentate feed. The archived configs carry 1e-9 in
# y_ret_in[2]; scripts/run_study_2d.py defaults to the same. (Note that
# scripts._shared.build_case_config's own default is 1e-3 — always pass
# this explicitly.)
TRACE_NH3 = 1e-9

# Paper data uses the EOS pressure row. "continuity" is experimental and has
# one unresolved failure in development testing.
PRESSURE_EQUATION = "eos"

ELEMENT_BALANCE_RTOL = 1e-3
ELEMENT_BALANCE_ATOL = 1e-8


# ── Deferred decisions ────────────────────────────────────────────────

# Published 1D variants.
# which 1D variant is the published one. Defaults match the current wiring
# of reactor/__init__.py (plain 1D) and scripts/run_1d_corrected.py
# (Sherwood-corrected 1D). Flipping either is a one-line change here plus a
# 1D cache re-run (minutes — the 1D models are cheap).
ONE_D_VARIANT = "membrane_reactor_1d"
ONE_D_CORRECTED_VARIANT = "membrane_reactor_1d_corrected"

# Figure 5 inputs: the single-gas permeation test matrix and the measured
# permeances.
PERMEATION_CASES_CSV = project_root() / "data" / "inputs" / "permeation_cases_1d.csv"

# The MEASURED laboratory permeances behind Figure 5's symbols (received
# 2026-08-25; provenance in the CSV header). The digitized placeholder
# permeation_exp_digitized_fig5.csv is kept alongside as the record of the
# pre-recovery state — the two agree to <= 0.5% on every point. The
# Arrhenius fit and the figure re-render from whatever this points to.
PERMEATION_EXP_CSV = project_root() / "data" / "inputs" / "permeation_exp_measured_fig5.csv"

# SI Figure S.1 input: the Chapman et al. diffusivity points digitized by
# the co-author (received 2026-08-25; provenance in the CSV header).
# Columns: panel ("P"|"T"), series ("Chapman" experimental points |
# "Correlation" digitized correlation curve, used only as a cross-check),
# p_bar, T_K, D_m2_s. Only the "Chapman" series is plotted.
S1_DIFFUSION_DATA_CSV: str | None = str(
    project_root() / "data" / "inputs" / "s1_diffusivity_chapman.csv")

# Reporting of the two 598 K cases in figures and dataset.
#   "exclude" - matches the paper's current 50 cases (default)
#   "include" - shown in the sweeps, labelled as dynamically unstable
#   "section" - promoted to a result section (multiplicity subsection)
# The pipeline ALWAYS solves and caches both 598 K cases and always runs the
# multiplicity analysis, so all three outcomes are a re-render, not a
# re-compute. The 598 K steady states are eigenvalue-certified per case.
INCLUDE_598K = "exclude"
INCLUDE_598K_CHOICES = ("exclude", "include", "section")

# Case IDs of the two 598 K runs. Note the second is misnamed in the case
# table and in the archive (no underscore before 598) — keep it verbatim.
CASES_598K = (
    "G5 — Temperature sweep_598",
    "G6 — Temperature sweep598",
)


# ── Space-velocity axis convention ──────────────────────────────────
# DECIDED 2026-08-19 (agreed with the co-author): figures and text label
# the space-velocity axis GHSV [1/h], per catalyst volume, using the
# case-table values directly. The originally published axes carried
# WHSV = GHSV/(Dcat*(1-eps)) = GHSV*5 mislabelled as a mass-based value
# (off by rho_c/(1000*Dcat*(1-eps)) ~ 2.95 from the true mass basis);
# "archived" reproduces those axes if ever needed, "physical" is the
# true mass-based value in mLn/(g_cat h). The stored WHSV column in the
# per-case KPI records keeps the archived convention for traceability.
WHSV_CONVENTION = "archived"
WHSV_CONVENTION_CHOICES = ("archived", "physical")


# ── Paths ─────────────────────────────────────────────────────────────

def cases_xlsx() -> Path:
    """Source of truth for the G1-G8 case table."""
    return project_root() / "data" / "inputs" / "cases_to_run.xlsx"


def rossetti_csv() -> Path:
    """The Rossetti et al. experimental kinetics dataset."""
    return project_root() / "data" / "inputs" / "ammonia_synthesis_data_rossetti_et_al.csv"


def archive_root() -> Path:
    """Archived paper dataset from the earlier study (gitignored)."""
    return project_root() / "Dataset_paper"


def cache_root(resolution_name: str) -> Path:
    """Root of the cached paper results for one resolution tier."""
    return project_root() / "results" / "paper" / resolution_name


def case_cache_dir(resolution_name: str, model: str, case_id: str) -> Path:
    """Cache directory for one solved case.

    Case IDs contain em-dashes ("G1 — GHSV sweep_50"); always build these
    paths with pathlib and never interpolate them into a shell command.
    """
    return cache_root(resolution_name) / model / case_id


def summary_csv(resolution_name: str, model: str) -> Path:
    """Per-model KPI summary CSV next to the cache."""
    return cache_root(resolution_name) / f"summary_kpis_{model}.csv"


def figures_dir(resolution_name: str) -> Path:
    """Figure output directory for one resolution tier."""
    return cache_root(resolution_name) / "figures"


def validation_dir() -> Path:
    """Resolution-independent validation artifacts (the Rossetti runs use a
    fixed 1D grid regardless of the 2D resolution tier)."""
    return project_root() / "results" / "paper" / "validation"


def dataset_dir() -> Path:
    """Export target for the archived dataset. Never committed to git."""
    return project_root() / "dataset"


MODEL_2D = "2d"
MODEL_1D = "1d"
MODEL_1D_CORRECTED = "1d_corrected"
# Same model class as 1d_corrected but with the mechanistic reaction-
# screening CP closure (reactor.cp_closure) instead of the fitted Sh(kappa)
# power law — a D1-adjacent candidate: local, chemistry-portable.
MODEL_1D_SCREENED = "1d_screened"
