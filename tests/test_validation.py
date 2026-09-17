"""Tests for reactor.paper.validation (Rossetti runs, Weisz-Prater)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reactor.paper import settings, validation


@pytest.fixture(scope="module")
def rossetti_table():
    """The digitized Rossetti table, loaded once per module."""
    return validation.load_rossetti_table()


def test_rossetti_table_shape(rossetti_table):
    """The Rossetti table has 19 tests by 7 GHSV columns with more than 100 numeric points."""
    # 19 tests, each with up to 7 GHSV columns; every used column is numeric.
    assert len(rossetti_table) == 19
    ghsv_cols = [c for c in rossetti_table.columns if c.startswith("GHSV_")]
    assert len(ghsv_cols) == 7
    n_points = rossetti_table[ghsv_cols].apply(
        pd.to_numeric, errors="coerce").notna().sum().sum()
    assert n_points > 100


def test_rossetti_config_is_isothermal_and_membrane_free(rossetti_table):
    """Rossetti configs are isothermal and membrane-free, taking T and p straight from the table row."""
    row = rossetti_table.iloc[0]
    cfg, meta = validation.build_rossetti_config(row, 50_000.0)
    assert cfg.is_isothermal
    assert cfg.Perm_NH3 == 0.0
    assert cfg.Nm == 0
    assert cfg.T_ret_in == pytest.approx(row["Temperature (°C)"] + 273.15)
    assert cfg.p_ret_out == pytest.approx(row["Pressure (bar)"] * 1e5)


def test_rossetti_feed_composition_from_ratio(rossetti_table):
    """The feed mole fractions honor the H2/N2 ratio from the row and sum to one."""
    row = rossetti_table.iloc[0].copy()
    row["H2/N2 Ratio"] = 3.0
    cfg, _ = validation.build_rossetti_config(row, 50_000.0)
    y = np.asarray(cfg.y_ret_in).ravel()
    assert y[0] / y[1] == pytest.approx(3.0)
    assert y.sum() == pytest.approx(1.0, abs=1e-6)


def test_rossetti_flow_scales_linearly_with_ghsv(rossetti_table):
    """Inlet molar flow scales linearly with GHSV while the geometry stays fixed."""
    row = rossetti_table.iloc[0]
    cfg1, _ = validation.build_rossetti_config(row, 50_000.0)
    cfg2, _ = validation.build_rossetti_config(row, 100_000.0)
    assert cfg2.F_ret_in == pytest.approx(2.0 * cfg1.F_ret_in)
    # Geometry does not depend on GHSV.
    assert cfg2.L == pytest.approx(cfg1.L)


def test_rossetti_cache_is_resolution_independent():
    """Rossetti results cache in the shared validation dir, not under any resolution's cache root."""
    assert validation.rossetti_results_path().parent == settings.validation_dir()


def test_weisz_prater_scan_requires_a_cache(tmp_path, monkeypatch):
    """With no usable cached cases, weisz_prater_scan raises RuntimeError instead of silently returning."""
    # With an empty cache the scan must fail loudly, never silently return.
    monkeypatch.setattr(settings, "cache_root", lambda name: tmp_path / name)
    table = pd.DataFrame({"Case_ID": ["nope"]})
    with pytest.raises(RuntimeError, match="no usable cases"):
        validation.weisz_prater_scan(table, "draft")


def test_permeation_arrhenius_fit_signs():
    """The Arrhenius fit reproduces the sign fingerprint: negative apparent EA for NH3, positive for H2 and N2."""
    # NH3 permeance falls with T (negative apparent EA, surface diffusion);
    # H2 and N2 rise (positive EA). The digitized dataset must reproduce
    # that qualitative fingerprint or the fit inputs are wrong.
    fit = validation.fit_permeance_arrhenius(write=False)
    assert fit["NH3"]["EA"] < 0
    assert fit["H2"]["EA"] > 0
    assert fit["N2"]["EA"] > 0
    for sp in ("NH3", "N2", "H2"):
        assert fit[sp]["n_points"] == 5
        assert fit[sp]["r_squared"] > 0.8


def test_permeation_config_geometry():
    """Permeation configs put the retentate at the membrane, disable sealing and reaction, and use the fitted permeances."""
    table = pd.read_csv(validation.permeation_cases_csv(), encoding="utf-8-sig")
    fit = validation.fit_permeance_arrhenius(write=False)
    cfg = validation.build_permeation_config(table.iloc[0], fit)
    # Membrane interface convention: retentate starts AT the membrane.
    assert cfg.r_min == cfg.r_max_perm
    # The tiny module must not be half-deactivated by the default sealing.
    assert cfg.Lsealing == 0.0
    assert cfg.factor_react == 0.0
    # Membrane-fit parameters, not the reactor's optimized targets.
    assert cfg.P0_N2 == fit["N2"]["P0"]
