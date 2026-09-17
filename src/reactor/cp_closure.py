"""Mechanistic concentration-polarization closure for the 1D membrane model.

Replaces the fitted power law ``Sh = coeff * kappa**exp`` with a model of the
mechanisms measured on the certified 2D dataset (2026-08-17 analysis):

* The polarization deficit at the membrane is a **reaction-screened
  diffusion layer**, not a flow boundary layer: Sh moves < 2x across a
  1200x GHSV span, and the deficit's absolute thickness (~2-5 mm) is nearly
  geometry-independent. Removing product at the wall locally lifts the
  (product-inhibited) reaction rate, which heals the deficit within a
  screening length

      delta_i = sqrt(D_i / s_i),     s_i = max(-dR_i/dc_i, 0),

  set by *local* kinetics sensitivity and diffusivity only.
* Geometry enters mechanistically: cylindrical curvature and the finite
  annular gap via the Helmholtz solution around the membrane, plus the
  uniform-source conduction background that carries species with no
  kinetic response (s_i -> 0).

The closure, evaluated from local (z-dependent) bulk properties — both
channels feed the same deficit, so the better conductor controls:

    k_cp,i(z) = max( C_screen * (D_i/delta_i) * f_ann(r_mem/delta_i, r_max/delta_i),
                     C_cond   * G_cond(kappa) * D_i / d_h )

with two O(1) dimensionless constants calibrated once against the
flux-matched exact closure of the 2D dataset (structure selection showed
the screening term alone carries the geometry dependence; any additive
conduction admixture re-biases it). Because the mechanisms — not
the numbers — carry the geometry and chemistry dependence, the model
generalizes to other kinetics, temperatures and geometries as long as
screening and conduction remain the dominant transport structure. (It does
NOT cover flow-dominated polarization, e.g. an inert high-Peclet channel:
there a Graetz/Leveque closure would be needed.)

All functions are pure and array-friendly.
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0e, i1e, k0e, k1e

#: Calibrated shape constants (see reactor.paper closures calibration; O(1)
#: by construction — they absorb the linearization and the effective-D
#: convention, not the physics).
DEFAULT_C_SCREEN = 1.0
DEFAULT_C_COND = 1.0


def screening_length(D: np.ndarray, dRdc: np.ndarray) -> np.ndarray:
    """delta_i = sqrt(D_i / max(-dR_i/dc_i, 0)); inf where the kinetics do
    not respond (no screening)."""
    s = np.maximum(-np.asarray(dRdc, float), 0.0)
    with np.errstate(divide="ignore"):
        return np.where(s > 0.0, np.sqrt(np.asarray(D, float) / np.maximum(s, 1e-300)),
                        np.inf)


def annulus_screen_factor(a: float, b: float, delta: np.ndarray) -> np.ndarray:
    """Dimensionless wall-gradient factor of the screened annulus.

    Solves D*(1/r)(r c')' = c'/delta^2 on [a, b] with no flux at r = b; the
    factor is -delta*c'(a)/c'(a-value), so k_screen = (D/delta)*f. Uses
    exponentially scaled Bessel functions for stability. Limits: thin layer
    (delta << b-a): f -> K1/K0(a/delta) -> 1 for delta << a (planar);
    delta -> inf: f -> 0 (a pure deficit cannot sustain steady flux without
    the reaction response — the conduction term takes over there).
    """
    delta = np.asarray(delta, float)
    out = np.zeros_like(delta)
    finite = np.isfinite(delta) & (delta > 0.0)
    if not np.any(finite):
        return out
    xa = a / delta[finite]
    xb = b / delta[finite]
    damp = np.exp(-2.0 * np.clip(xb - xa, 0.0, 700.0))
    num = k1e(xa) * i1e(xb) - i1e(xa) * k1e(xb) * damp
    den = k0e(xa) * i1e(xb) + i0e(xa) * k1e(xb) * damp
    out[finite] = np.where(den > 0.0, num / np.maximum(den, 1e-300), 0.0)
    return np.maximum(out, 0.0)


def conduction_shape_factor(kappa: float) -> float:
    """G_cond = k_cond * d_h / D for the uniform-source annulus.

    Steady diffusion on [a, b] (kappa = a/b) with a uniform volumetric
    source, all of it extracted at the inner wall, no flux at the outer
    wall; k_cond = J / (c_mean - c_wall). This is the flow-independent
    background-supply limit (the mass-transfer analog of a fully developed
    Nusselt number) and the fallback for species without kinetic screening.
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    a, b = kappa, 1.0
    L = np.log(b / a)
    bracket = (
        b**2 * (b**2 / 2.0 * L - (b**2 - a**2) / 4.0)
        - (b**4 - a**4) / 8.0
        + a**2 * (b**2 - a**2) / 4.0
    )
    k_over_D = (b**2 - a**2) ** 2 / (2.0 * a * bracket)
    return float(k_over_D * (b - a))


def screened_kcp(
    D: np.ndarray,
    dRdc: np.ndarray,
    *,
    r_mem: float,
    r_max: float,
    c_screen: float = DEFAULT_C_SCREEN,
    c_cond: float = DEFAULT_C_COND,
) -> np.ndarray:
    """The mechanistic CP mass-transfer coefficient, per (z, species).

    Parameters are local bulk properties: ``D`` the effective diffusivities
    [m^2/s] and ``dRdc`` the diagonal kinetics sensitivities
    dR_i/dc_i [1/s], both shaped (nz, nc).
    """
    D = np.asarray(D, float)
    delta = screening_length(D, dRdc)
    f = annulus_screen_factor(r_mem, r_max, delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_screen = np.where(np.isfinite(delta), D / np.maximum(delta, 1e-300) * f, 0.0)
    d_h = r_max - r_mem
    g_cond = conduction_shape_factor(r_mem / r_max)
    k_cond = g_cond * D / d_h
    # The channels feed the same deficit: the better conductor controls.
    # For screened species (this chemistry: all three, k_screen ~ 4x k_cond)
    # the screening term carries the closure — calibration shows it alone
    # reproduces the measured kappa-dependence to +-5% where any additive
    # mix re-biases it. Conduction is the physical lower bound that keeps
    # unscreened species (s_i -> 0) finite.
    return np.maximum(c_screen * k_screen, c_cond * k_cond)
