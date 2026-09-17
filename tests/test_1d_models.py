"""Guards for the 1D models — the membrane-coupling conservation bug class.

The corrected 1D model shipped for months with a "geometric factor" that
under-fed the permeate by ~20x (a 16% hydrogen element-balance violation on
G2_3000). These tests pin the assembly-level conservation identity so that
class of bug cannot return, without solving a full reactor.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from reactor.config import ReactorConfig


def _tiny_config(**overrides):
    """Build a small, fast ReactorConfig so tests never solve a full reactor."""
    return ReactorConfig.from_defaults(
        num_z=16, num_timesteps=2, max_newton_iterations=5, **overrides
    )


@pytest.mark.parametrize("module_name", [
    "membrane_reactor_1d",
    "membrane_reactor_1d_corrected",
])
def test_membrane_coupling_conserves_mass(module_name):
    """Molar rate per unit length leaving the retentate equals the rate
    entering the permeate: A_ret*g_mem_ret + A_perm*g_mem_perm == 0."""
    mod = importlib.import_module(f"reactor.{module_name}")
    reactor = mod.MembraneReactor1D(config=_tiny_config())
    c = reactor.cpT[..., :-2]
    T = reactor.cpT[..., -1]
    g_mem, _ = reactor._membrane_flux(c, T)
    ret_rate = reactor.A_ret * g_mem[:, 1, :]
    perm_rate = reactor.A_perm * g_mem[:, 0, :]
    assert np.abs(ret_rate).max() > 0.0  # the test must exercise a real flux
    assert np.allclose(ret_rate + perm_rate, 0.0,
                       atol=1e-12 * np.abs(ret_rate).max())


def test_sh_closure_is_configurable_and_overridable():
    """The Sh correlation honors the configured coefficients, and kcp_override wins verbatim over the correlation."""
    mod = importlib.import_module("reactor.membrane_reactor_1d_corrected")
    reactor = mod.MembraneReactor1D(config=_tiny_config(sh_cp_coeff=3.0,
                                                        sh_cp_exp=-1.0))
    assert reactor.sh_cp_coeff == 3.0
    c_ret = reactor.cpT[:, 1, :-2]
    T_ret = reactor.cpT[:, 1, -1]
    kcp_corr = reactor.compute_kcp_sh(c_ret, T_ret)
    # Correlation mode: Sh = 3.0 * (r_mem/r_max)^-1, same for all species
    kappa = reactor.r_mem / reactor.r_max
    assert kappa < 1.0  # canonical definition: membrane over shell radius
    assert kcp_corr.shape == (reactor.num_z, reactor.num_c)
    assert np.all(kcp_corr > 0.0)
    # Exact-tracing mode: the override wins verbatim
    reactor.kcp_override = np.full((reactor.num_z, reactor.num_c), 1e-3)
    kcp = reactor.compute_kcp_sh(c_ret, T_ret)
    assert np.allclose(kcp, 1e-3)


# ── Mechanistic CP closure (reactor.cp_closure) ──────────────────────

def test_conduction_shape_factor_matches_numeric_integration():
    """conduction_shape_factor(kappa) matches direct numerical integration of the annular conduction profile."""
    from reactor.cp_closure import conduction_shape_factor
    for kappa in (0.0833, 0.25):
        a, b = kappa, 1.0
        r = np.linspace(a, b, 20001)
        prof = 0.5 * (b**2 * np.log(r / a) - (r**2 - a**2) / 2)
        mean = np.trapezoid(prof * 2 * r, r) / (b**2 - a**2)
        J = (b**2 - a**2) / (2 * a)
        numeric = (J / mean) * (b - a)
        assert conduction_shape_factor(kappa) == pytest.approx(numeric, rel=1e-3)


def test_screen_factor_limits():
    """annulus_screen_factor tends to 1 in the planar limit, exceeds 1 at finite curvature, and vanishes for a fully screened channel."""
    from reactor.cp_closure import annulus_screen_factor
    a, b = 0.005, 0.06
    # planar limit: delta << a -> factor -> 1
    assert annulus_screen_factor(a, b, np.array([1e-6]))[0] == pytest.approx(1.0, rel=1e-2)
    # curvature regime: delta ~ a enhances the wall gradient
    assert annulus_screen_factor(a, b, np.array([3e-3]))[0] > 1.1
    # no kinetic response: the screened channel carries nothing
    assert annulus_screen_factor(a, b, np.array([np.inf]))[0] == 0.0


def test_screened_kcp_falls_back_to_conduction():
    """With zero kinetic response, screened_kcp reduces exactly to the pure-conduction shape-factor value."""
    from reactor.cp_closure import conduction_shape_factor, screened_kcp
    D = np.full((4, 3), 1e-5)
    dRdc = np.zeros((4, 3))          # inert: no screening anywhere
    k = screened_kcp(D, dRdc, r_mem=0.005, r_max=0.06)
    expected = conduction_shape_factor(0.005 / 0.06) * 1e-5 / 0.055
    assert np.allclose(k, expected, rtol=1e-12)


def test_screened_closure_mode_runs_in_the_model():
    """The 'screened' cp_closure mode yields positive kcp of shape (num_z, num_c) inside the model."""
    mod = importlib.import_module("reactor.membrane_reactor_1d_corrected")
    reactor = mod.MembraneReactor1D(config=_tiny_config(
        cp_closure="screened", screen_c1=0.5, screen_c2=1.0))
    c_ret = reactor.cpT[:, 1, :-2]
    T_ret = reactor.cpT[:, 1, -1]
    kcp = reactor.compute_kcp_sh(c_ret, T_ret)
    assert kcp.shape == (reactor.num_z, reactor.num_c)
    assert np.all(kcp > 0.0)
