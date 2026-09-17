"""Guards for the paper pipeline — one test per trap that cost real time.

These are cheap: nothing here solves a reactor. They protect the case-table
loading, the KPI unit conventions, and the cache-completeness contract that
make the notebook re-runnable.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from reactor.convergence import (
    ConvergenceMonitor,
    steady_state_weights,
    weighted_rms,
)
from reactor.paper import cache, cases, kpis, runner, settings


# ── Case table ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def case_table() -> pd.DataFrame:
    """The paper case table, loaded once per module."""
    return cases.load_case_table()


def test_case_table_has_all_52_cases(case_table):
    """The case table holds all 52 cases spanning families G1 through G8."""
    assert len(case_table) == 52
    assert sorted(case_table["family"].unique()) == [f"G{i}" for i in range(1, 9)]


def test_counter_current_flag_is_a_real_bool(case_table):
    """The xlsx stores "FALSE" as a string, and bool("FALSE") is True."""
    assert case_table["Is_Counter_Current"].dtype == bool
    assert not case_table["Is_Counter_Current"].any()


@pytest.mark.parametrize("text,expected", [
    ("FALSE", False), ("false", False), ("TRUE", True), ("yes", True),
    (0, False), (1, True), (np.False_, False),
])
def test_to_bool(text, expected):
    """cases._to_bool maps spreadsheet truthy/falsy spellings to real Python bools."""
    assert cases._to_bool(text) is expected


def test_598k_case_ids_are_in_the_table(case_table):
    """Including the misnamed one — no underscore before 598."""
    present = set(case_table["Case_ID"])
    for case_id in settings.CASES_598K:
        assert case_id in present


def test_598k_switch_controls_reporting_only(case_table):
    """select_cases drops the two 598 K cases only for 'exclude' and rejects unknown modes."""
    assert len(cases.select_cases(case_table, "exclude")) == 50
    assert len(cases.select_cases(case_table, "include")) == 52
    assert len(cases.select_cases(case_table, "section")) == 52
    with pytest.raises(ValueError):
        cases.select_cases(case_table, "maybe")


# ── Configuration ─────────────────────────────────────────────────────

def test_trace_nh3_matches_the_archived_runs(case_table):
    """The archived configs carry y_ret_in[2] = 1e-9, not build_case_config's
    own 1e-3 default."""
    row = case_table.iloc[0]
    config, _ = runner.build_config_2d(row, "draft")
    assert config.y_ret_in.ravel()[2] == pytest.approx(settings.TRACE_NH3)


def test_paper_cases_use_the_eos_pressure_row(case_table):
    """Paper 2D configs use the 'eos' pressure equation."""
    config, _ = runner.build_config_2d(case_table.iloc[0], "draft")
    assert config.pressure_equation == "eos"


@pytest.mark.parametrize("name,num_r,num_z", [
    ("draft", 24, 60),
    ("publication", 40, 100),
])
def test_resolution_settings(case_table, name, num_r, num_z):
    """Each resolution tier sets its own grid but shares the weighted tolerance, norm kind, and KPI stopping."""
    config, _ = runner.build_config_2d(case_table.iloc[0], name)
    assert (config.num_r, config.num_z) == (num_r, num_z)
    # One weighted tolerance means the same thing for every case and grid.
    assert config.steady_state_atol == settings.WRMS_TOLERANCE
    assert config.norm_kind == "weighted"
    assert config.kpi_stop is True


def test_one_tolerance_for_both_tiers():
    """The whole point of the weighted norm: the tolerance is no longer a
    per-grid tuning knob (draft 3e-2 / publication 2e-3 in the absolute
    norm) but one number with one meaning — wrms <= 1."""
    assert settings.RESOLUTIONS["draft"].steady_state_atol == settings.WRMS_TOLERANCE
    assert settings.RESOLUTIONS["publication"].steady_state_atol == settings.WRMS_TOLERANCE
    assert settings.WRMS_TOLERANCE == 1.0


def test_norm_change_bumped_the_cache_schema():
    """Changing the MEANING of steady_state_atol is invisible to the
    resolution-aware cache check; the schema version is what invalidates
    caches solved under the absolute norm."""
    assert settings.CACHE_SCHEMA_VERSION >= 2
    assert settings.NORM_CONFIG["norm_kind"] == "weighted"


@pytest.mark.skipif(not settings.archive_root().exists(), reason="Dataset_paper not available")
def test_reconstructed_configs_match_the_archive(case_table):
    """Configs rebuilt from the case table match the archived Dataset_paper configs for one case per family."""
    by_id = case_table.set_index("Case_ID")
    probe = [case_table[case_table["family"] == f]["Case_ID"].iloc[0]
             for f in sorted(case_table["family"].unique())]
    report = cases.crosscheck_report(
        probe, lambda cid: runner.build_config_2d(by_id.loc[cid].rename(cid), "publication")[0]
    )
    assert (report["status"] == "match").all(), report[report["status"] == "MISMATCH"].to_string()


@pytest.mark.skipif(not settings.archive_root().exists(), reason="Dataset_paper not available")
def test_unconverged_archive_entries_are_guarded():
    """Both 598 K archive directories have no fields.npz."""
    for case_id in settings.CASES_598K:
        assert not cases.archived_case_is_converged(case_id)


# ── KPI units ─────────────────────────────────────────────────────────

def test_presentation_units_keep_the_raw_values():
    """to_presentation_units scales values for display while keeping the SI originals under *_si keys."""
    raw = {"X_H2_out": 0.25, "NH3_prod_out": 1e-3, "DeltaT_max": 100.0}
    out = kpis.to_presentation_units(raw)
    assert out["X_H2_out"] == pytest.approx(25.0)
    assert out["X_H2_out_si"] == pytest.approx(0.25)
    assert out["NH3_prod_out"] == pytest.approx(3.6)     # mmol g^-1 h^-1
    assert out["NH3_prod_out_si"] == pytest.approx(1e-3)  # mol kg^-1 s^-1
    assert out["DeltaT_max"] == 100.0                     # unscaled


def test_whsv_conventions_differ_by_the_documented_factor():
    """The 'archived' and 'physical' WHSV conventions give their documented values, and unknown conventions raise."""
    kwargs = dict(ghsv_h=10000.0, vol_flow_std_m3_s=1.0e-3, w_cat_kg=0.34,
                  dcat=1.0 / 3.0, eps=0.4)
    archived = kpis.whsv(**kwargs, convention="archived")
    assert archived == pytest.approx(50000.0)  # GHSV x 5, the published axis
    physical = kpis.whsv(**kwargs, convention="physical")
    assert physical == pytest.approx(1.0e-3 / 0.34 * 3600.0 * 1e3)
    with pytest.raises(ValueError):
        kpis.whsv(**kwargs, convention="nonsense")


def test_cp_is_wall_over_area_mean():
    """cp_profile is the wall value divided by the area-weighted mean concentration."""
    # Two equal-area annuli: wall value 1, bulk value 3 -> CP = 1 / 2.
    r_f = np.array([0.0, 1.0, np.sqrt(2.0)])
    y = np.zeros((1, 2, 3))
    y[0, 0, kpis.INH3] = 1.0
    y[0, 1, kpis.INH3] = 3.0
    assert kpis.cp_profile(y, r_f)[0] == pytest.approx(0.5)


# ── Cache contract ────────────────────────────────────────────────────

def test_incomplete_cache_is_not_reused(tmp_path):
    """A cache dir missing its payload files is never complete, even when meta.json claims it is."""
    case_dir = tmp_path / "G1 — GHSV sweep_50"   # em-dash, as on disk
    case_dir.mkdir()
    assert not cache.is_complete(case_dir)

    (case_dir / "meta.json").write_text(json.dumps(
        {"complete": True, "status": "converged",
         "schema_version": settings.CACHE_SCHEMA_VERSION}))
    # meta says complete but the payload is missing
    assert not cache.is_complete(case_dir)


def test_stale_schema_version_invalidates_the_cache(tmp_path):
    """A meta.json schema_version different from the current one invalidates an otherwise complete cache."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    for name in ("config.json", "kpis.json", "solve_status.json"):
        (case_dir / name).write_text("{}")
    meta = {"complete": True, "status": "failed",
            "schema_version": settings.CACHE_SCHEMA_VERSION}
    (case_dir / "meta.json").write_text(json.dumps(meta))
    assert cache.is_complete(case_dir)

    (case_dir / "meta.json").write_text(
        json.dumps({**meta, "schema_version": settings.CACHE_SCHEMA_VERSION + 1}))
    assert not cache.is_complete(case_dir)


def test_cache_from_a_different_resolution_is_not_reused(tmp_path):
    """Changing a resolution's grid or tolerance must invalidate its cache,
    or a settings change silently reuses results from the old settings."""
    draft = settings.RESOLUTIONS["draft"]
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    for name in ("kpis.json", "solve_status.json"):
        (case_dir / name).write_text("{}")
    (case_dir / "config.json").write_text(json.dumps({
        "num_r": draft.num_r, "num_z": draft.num_z,
        "steady_state_atol": draft.steady_state_atol}))
    (case_dir / "meta.json").write_text(json.dumps(
        {"complete": True, "status": "failed",
         "schema_version": settings.CACHE_SCHEMA_VERSION}))

    assert cache.is_complete(case_dir, draft)
    assert not cache.is_complete(case_dir, settings.RESOLUTIONS["publication"])

    # A tolerance change alone is enough to invalidate.
    (case_dir / "config.json").write_text(json.dumps({
        "num_r": draft.num_r, "num_z": draft.num_z,
        "steady_state_atol": draft.steady_state_atol * 10}))
    assert not cache.is_complete(case_dir, draft)
    assert cache.is_complete(case_dir)  # still structurally complete


def test_cache_paths_survive_em_dash_case_ids():
    """case_cache_dir preserves em-dash case IDs verbatim in the directory name."""
    case_id = "G1 — GHSV sweep_50"
    path = settings.case_cache_dir("draft", settings.MODEL_2D, case_id)
    assert path.name == case_id
    assert path.parent.name == settings.MODEL_2D


# ── Weighted residual norm (convergence plan, Item 1) ────────────────

def test_weighted_rms_is_the_cvode_norm():
    """wrms = sqrt(mean((g_i/w_i)^2)); unit weighted components give 1."""
    g = np.array([1.0, -2.0, 3.0])
    w = np.array([1.0, 2.0, 3.0])
    assert weighted_rms(g, w) == pytest.approx(1.0)
    assert weighted_rms(0.5 * g, w) == pytest.approx(0.5)


def _weights_kwargs():
    """Shared keyword arguments for steady_state_weights in these tests."""
    return dict(rtol_c=1e-2, atol_c_rel=1e-5, rtol_p=1e-2, atol_p_rel=1e-5,
                rtol_T=1e-4, atol_T=1e-2, Rg=8.314, c_ref=30e5 / (8.314 * 650.0))


def test_weights_are_per_block_and_from_the_current_state():
    """Weights come from the current state per block: rtol*|x| plus each block's own absolute floor."""
    cpT = np.zeros((1, 1, 5))  # 3 species + p + T
    cpT[0, 0, :3] = [100.0, 50.0, 1e-6]  # major, major, trace
    cpT[0, 0, 3] = 30e5
    cpT[0, 0, 4] = 650.0
    kw = _weights_kwargs()
    c_ref = kw["c_ref"]
    w = steady_state_weights(cpT, **kw)
    # concentration rows: rtol*|c| + relative absolute floor
    assert w[0, 0, 0] == pytest.approx(1e-2 * 100.0 + 1e-5 * c_ref)
    # a trace species is floored by the absolute term, not drowned by rtol
    assert w[0, 0, 2] == pytest.approx(1e-2 * 1e-6 + 1e-5 * c_ref)
    # the algebraic pressure row scales with |p|/(Rg*T) — its own scale,
    # never a concentration-field scale
    assert w[0, 0, 3] == pytest.approx(1e-2 * 30e5 / (8.314 * 650.0) + 1e-5 * c_ref)
    # temperature rows scale with |T|, floored in Kelvin
    assert w[0, 0, 4] == pytest.approx(1e-4 * 650.0 + 1e-2)


def test_one_weighted_tolerance_transfers_across_pressure_scales():
    """The central defect of the absolute norm: the same *relative* residual
    at a 10x pressure scale gives a 10x absolute norm but an identical
    weighted norm."""
    kw = _weights_kwargs()
    kw["atol_c_rel"] = 0.0
    kw["atol_p_rel"] = 0.0
    for scale in (1.0, 10.0):
        cpT = np.zeros((1, 1, 5))
        cpT[0, 0, :3] = scale * np.array([100.0, 50.0, 10.0])
        cpT[0, 0, 3] = scale * 30e5
        cpT[0, 0, 4] = 650.0
        g = np.zeros_like(cpT)
        g[0, 0, :3] = 1e-3 * cpT[0, 0, :3]  # 0.1% relative everywhere
        g[0, 0, 3] = 1e-3 * scale * 30e5 / (8.314 * 650.0)
        g[0, 0, 4] = 1e-3 * 650.0 * 1e-4 / 1e-4  # matched to rtol_T scale
        g[0, 0, 4] = 1e-3 * 650.0
        w = steady_state_weights(cpT, **kw)
        wrms = weighted_rms(g[..., :4], w[..., :4])  # c and p blocks
        assert wrms == pytest.approx(0.1, rel=1e-9)  # 1e-3 / rtol 1e-2


# ── Outcome classifier (convergence plan, Items 2+3) ─────────────────

def _monitor(**overrides):
    """Build a ConvergenceMonitor with small test-sized thresholds."""
    kw = dict(check_every=10, kpi_rtol=1e-3, n_stagnant=3, window=8,
              dt_max=1e6, soft_step_cap=400,
              kpi_floor=np.array([1e-9, 1e-9, 1.0]))
    kw.update(overrides)
    return ConvergenceMonitor(**kw)


def _feed(monitor, rows):
    """rows: (step, g_ss_norm, dt, kpi_vector). Returns last classification."""
    out = None
    for step, norm, dt, kpi in rows:
        out = monitor.observe(step=step, g_ss_norm=norm, dt=dt,
                              kpi=np.asarray(kpi, float))
    return out


def test_classifier_progressing_while_residual_drops():
    """A steadily dropping residual keeps the classification at progressing."""
    m = _monitor()
    rows = [(10 * (i + 1), 1e-1 * 0.5 ** i, 1e3, [1.0, 0.5, 700.0 - i])
            for i in range(6)]
    assert _feed(m, rows).outcome == "progressing"


def test_classifier_floored_when_kpis_stagnate_at_dt_max():
    """Stagnant KPIs at dt_max with a flat residual classify as floored."""
    m = _monitor()
    kpi = [1.0, 0.5, 700.0]
    rows = [(10 * (i + 1), 9.5e-4 * (1.0 + 1e-4 * i), 1e6, kpi) for i in range(6)]
    verdict = _feed(m, rows)
    assert verdict.outcome == "floored"
    assert verdict.evidence["dt_at_dt_max"] is True
    assert verdict.evidence["stagnant_checks"] >= 3


def test_classifier_not_floored_while_kpis_still_move():
    """Stalled residual alone is not enough — the monitor watches what is
    published."""
    m = _monitor()
    rows = [(10 * (i + 1), 9.5e-4, 1e6, [1.0 * (1.0 + 0.01 * i), 0.5, 700.0])
            for i in range(6)]
    assert _feed(m, rows).outcome == "progressing"


def test_classifier_floored_past_soft_cap_without_dt_max():
    """Past the soft step cap, stagnant KPIs classify as floored even without dt at dt_max."""
    m = _monitor()
    kpi = [1.0, 0.5, 700.0]
    rows = [(390 + 10 * (i + 1), 9.5e-4, 1.0, kpi) for i in range(6)]
    verdict = _feed(m, rows)
    assert verdict.outcome == "floored"
    assert verdict.evidence["dt_at_dt_max"] is False


def test_classifier_oscillatory_on_sign_alternating_kpis():
    """Limit-cycle orbit: KPI reverses direction at constant amplitude while
    the residual sits far above its floor."""
    m = _monitor()
    rows = [(10 * (i + 1), 0.8, 1.0, [1.0, 0.5, 700.0 + 20.0 * (-1) ** i])
            for i in range(8)]
    verdict = _feed(m, rows)
    assert verdict.outcome == "oscillatory"
    assert verdict.evidence["sign_changes"] >= 4
    assert verdict.evidence["amplitude_rel"] > 0.0


def test_classifier_decaying_spiral_is_not_oscillatory():
    """A decaying KPI oscillation classifies as progressing, not oscillatory."""
    rows = [(10 * (i + 1), 0.8 * 0.9 ** i, 1.0,
             [1.0, 0.5, 700.0 + 20.0 * 0.5 ** i * (-1) ** i])
            for i in range(8)]
    m = _monitor()
    assert _feed(m, rows).outcome == "progressing"


def test_classifier_diverging_on_growth_and_nonfinite():
    """Exponential residual growth or a non-finite residual classifies as diverging."""
    m = _monitor()
    rows = [(10 * (i + 1), 1e-2 * 2.0 ** i, 1e-3, [1.0, 0.5, 700.0])
            for i in range(8)]
    assert _feed(m, rows).outcome == "diverging"
    m = _monitor()
    assert _feed(m, [(10, np.inf, 1e-3, [1.0, 0.5, 700.0])]).outcome == "diverging"
