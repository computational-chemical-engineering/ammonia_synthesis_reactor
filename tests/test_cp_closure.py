"""Tests for the mechanistic CP closure mathematics (reactor.cp_closure)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from reactor.cp_closure import (annulus_screen_factor, conduction_shape_factor,
                                screened_kcp, screening_length)
from reactor.paper import settings

A, B = 0.005, 0.06  # the paper's kappa = 0.083 annulus [m]


def test_screening_length_infinite_without_kinetic_response():
    """s_i <= 0 (no product inhibition) gives an infinite screening length."""
    D = np.array([1e-5, 1e-5])
    delta = screening_length(D, np.array([+1.0, 0.0]))
    assert np.isinf(delta).all()
    delta = screening_length(D, np.array([-1.0e2, -1.0e4]))
    assert np.allclose(delta, np.sqrt(D / np.array([1e2, 1e4])))


def test_annulus_screen_factor_limits():
    """Thin layer -> planar film-theory limit (f -> 1); delta -> inf -> 0."""
    thin = annulus_screen_factor(A, B, np.array([1e-5 * (B - A)]))
    assert thin[0] == pytest.approx(1.0, abs=1e-3)
    # at the physically realized x_a ~ 1.7 the curvature enhancement is ~1.3
    xa = 1.7
    mid = annulus_screen_factor(A, B, np.array([A / xa]))
    assert 1.2 < mid[0] < 1.4
    wide = annulus_screen_factor(A, B, np.array([1e6]))
    assert wide[0] == pytest.approx(0.0, abs=1e-3)


def test_conduction_shape_factor_values():
    """The closed-form G(kappa) at the paper's extreme geometries."""
    assert conduction_shape_factor(0.083) == pytest.approx(6.24, abs=0.01)
    assert conduction_shape_factor(0.25) == pytest.approx(3.78, abs=0.01)
    # weak dependence: effective exponent ~ -0.46 over the fitted range
    expo = (np.log(conduction_shape_factor(0.25) / conduction_shape_factor(0.083))
            / np.log(0.25 / 0.083))
    assert expo == pytest.approx(-0.46, abs=0.03)


def test_screened_kcp_takes_the_better_conductor():
    """max() selection: conduction floor without kinetics, screening above it."""
    D = np.full((1, 1), 1e-5)
    no_kinetics = screened_kcp(D, np.zeros((1, 1)), r_mem=A, r_max=B)
    g = conduction_shape_factor(A / B)
    assert no_kinetics[0, 0] == pytest.approx(g * 1e-5 / (B - A), rel=1e-12)
    strong = screened_kcp(D, np.full((1, 1), -1e4), r_mem=A, r_max=B)
    assert strong[0, 0] > no_kinetics[0, 0]


@pytest.mark.skipif(
    not (settings.cache_root("publication") / "screened_cp_fit.json").exists(),
    reason="publication cache not present")
def test_calibration_constant_pinned():
    """The shipped calibration: C = 0.464, per-kappa bias flat to +-5%."""
    fit = json.loads(
        (settings.cache_root("publication") / "screened_cp_fit.json").read_text())
    assert fit["c_screen"] == pytest.approx(0.464, abs=0.005)
    for bias in fit["per_kappa_bias"].values():
        assert 0.95 < bias < 1.05
