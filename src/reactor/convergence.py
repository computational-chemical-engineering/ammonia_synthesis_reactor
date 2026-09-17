"""Steady-state convergence: weighted residual norm and stopping classifier.

Implements the steady-state stopping machinery:

* an error-weighted RMS residual norm (CVODE/IDA convention) with per-block
  tolerances for the concentration, pressure and temperature rows, so that
  one tolerance means the same thing for every case;
* a monitor that watches the published quantities (outlet flows) instead of
  only the residual, stops on KPI stagnation, and classifies the outcome as
  ``converged`` / ``floored`` / ``oscillatory`` / ``diverging`` instead of
  a binary pass/fail.

Everything here is pure (no reactor state), so it is unit-testable on
synthetic residuals and histories.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

import numpy as np

NORM_KINDS = ("absolute", "weighted")

#: Classifier outcomes. ``progressing`` is the transient "keep marching"
#: verdict; the other five can end a solve.
OUTCOMES = (
    "progressing", "converged", "floored", "oscillatory", "diverging", "failed",
)


# ── Item 1: error-weighted RMS norm ───────────────────────────────────

def weighted_rms(g: np.ndarray, weights: np.ndarray) -> float:
    """``sqrt(mean((g_i / w_i)**2))`` — the CVODE/IDA WRMS norm.

    ``wrms <= tol`` (canonically 1.0) is the convergence test.
    """
    g = np.asarray(g, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    return float(np.sqrt(np.mean((g / w) ** 2)))


def steady_state_weights(
    cpT: np.ndarray,
    *,
    rtol_c: float,
    atol_c_rel: float,
    rtol_p: float,
    atol_p_rel: float,
    rtol_T: float,
    atol_T: float,
    Rg: float,
    c_ref: float,
) -> np.ndarray:
    """Per-row error weights ``w_i = rtol_block*scale_i + atol_block``.

    Weights come from the *current* state (recomputed every step, never
    frozen at the initial condition). Blocks:

    * concentration rows scale with the local ``|c_i|`` [mol m^-3]; the
      absolute floor is ``atol_c_rel * c_ref`` so trace species do not
      dominate the norm;
    * the pressure row is algebraic under ``pressure_equation="eos"`` (its
      residual is the EOS density mismatch ``c_tot - sum(c)``, in
      concentration units): it scales with the local total molar density
      ``|p|/(Rg*T)`` implied by the pressure — its own scale, never a
      concentration-field scale;
    * temperature rows scale with the local ``|T|`` [K], floored at
      ``atol_T`` Kelvin.

    ``c_ref`` is the case's reference total molar density (e.g.
    ``p_ret_out/(Rg*T_ret_in)``), used only for the absolute floors so a
    single relative setting transfers across the pressure sweep.
    """
    cpT = np.asarray(cpT, dtype=float)
    w = np.empty_like(cpT)
    c = cpT[..., :-2]
    p = cpT[..., -2]
    T = cpT[..., -1]
    w[..., :-2] = rtol_c * np.abs(c) + atol_c_rel * c_ref
    c_tot_scale = np.abs(p) / (Rg * np.maximum(np.abs(T), 1.0))
    w[..., -2] = rtol_p * c_tot_scale + atol_p_rel * c_ref
    w[..., -1] = rtol_T * np.abs(T) + atol_T
    return w


# ── Items 2 + 3: KPI stagnation stopping and outcome classification ──

@dataclass
class CheckRecord:
    """One monitor sample, taken every ``check_every`` accepted steps."""

    step: int
    g_ss_norm: float
    dt: float
    kpi: np.ndarray
    rel_change: Optional[float] = None  # vs the previous check


@dataclass
class Classification:
    """Classifier verdict plus the evidence it rests on."""

    outcome: str
    evidence: dict[str, Any] = field(default_factory=dict)


class ConvergenceMonitor:
    """Trailing-history classifier for the pseudo-transient march.

    Fed one sample every ``check_every`` accepted steps (a KPI vector — for
    the membrane reactor: outlet molar flows per species on both sides plus
    the peak temperature — the steady-state residual norm, and the current
    dt), it classifies the trajectory:

    ``converged``    is decided by the residual test in ``solve()`` itself,
                     never here.
    ``floored``      KPIs stagnant for ``n_stagnant`` consecutive checks and
                     the residual no longer decreasing, with dt at dt_max
                     (or the soft step cap exceeded): the iteration has
                     converged even though the residual threshold is
                     unreachable. Accept.
    ``oscillatory``  a KPI component keeps changing direction at roughly
                     constant peak-to-peak amplitude while the residual is
                     not improving: a limit-cycle orbit. Stop marching, hand
                     to the escalation ladder.
    ``diverging``    residual non-finite, or growing monotonically across
                     the window.
    ``progressing``  anything else: keep marching.
    """

    def __init__(
        self,
        *,
        check_every: int = 10,
        kpi_rtol: float = 1e-3,
        n_stagnant: int = 3,
        window: int = 8,
        dt_max: float = np.inf,
        soft_step_cap: int = 400,
        kpi_floor: Optional[np.ndarray] = None,
        residual_stall_rel: float = 0.02,
        osc_min_sign_changes: int = 4,
        osc_amplitude_factor: float = 10.0,
    ) -> None:
        """Validate thresholds and window sizes and allocate the trailing history buffer."""
        if check_every < 1:
            raise ValueError("check_every must be >= 1")
        if window < 4:
            raise ValueError("window must be >= 4 to classify oscillation")
        self.check_every = check_every
        self.kpi_rtol = kpi_rtol
        self.n_stagnant = n_stagnant
        self.window = window
        self.dt_max = dt_max
        self.soft_step_cap = soft_step_cap
        self.kpi_floor = None if kpi_floor is None else np.asarray(kpi_floor, float)
        self.residual_stall_rel = residual_stall_rel
        self.osc_min_sign_changes = osc_min_sign_changes
        self.osc_amplitude_factor = osc_amplitude_factor
        self.history: Deque[CheckRecord] = deque(maxlen=window)
        self.stagnant_checks = 0
        self.last: Optional[Classification] = None

    def due(self, accepted_steps: int) -> bool:
        """Return True when a classification check is due, i.e. every ``check_every`` accepted steps."""
        return accepted_steps > 0 and accepted_steps % self.check_every == 0

    # -- internals ----------------------------------------------------

    def _rel_change(self, new: np.ndarray, old: np.ndarray) -> float:
        """Return the max floored relative change between two KPI vectors."""
        floor = self.kpi_floor if self.kpi_floor is not None else 0.0
        scale = np.maximum(np.abs(new), floor)
        scale = np.maximum(scale, 1e-300)
        return float(np.max(np.abs(new - old) / scale))

    def _residual_stalled(self) -> bool:
        """Best residual in the newer half no longer beats the older half."""
        if len(self.history) < 4:
            return False
        norms = [rec.g_ss_norm for rec in self.history]
        half = len(norms) // 2
        best_old = min(norms[:half])
        best_new = min(norms[half:])
        return best_new >= best_old * (1.0 - self.residual_stall_rel)

    def _residual_growing(self) -> bool:
        """Return True if the residual grew monotonically across a full window to more than 3x its start."""
        if len(self.history) < self.window:
            return False
        norms = [rec.g_ss_norm for rec in self.history]
        increasing = all(b >= a for a, b in zip(norms, norms[1:]))
        return increasing and norms[-1] > 3.0 * norms[0]

    def _oscillation(self) -> Optional[dict[str, Any]]:
        """Sign-alternating KPI increments at roughly constant amplitude."""
        if len(self.history) < max(5, self.window - 2):
            return None
        kpis = np.array([rec.kpi for rec in self.history])  # (n_checks, n_kpi)
        floor = self.kpi_floor if self.kpi_floor is not None else 0.0
        scale = np.maximum(np.max(np.abs(kpis), axis=0), floor)
        scale = np.maximum(scale, 1e-300)
        deltas = np.diff(kpis, axis=0) / scale  # relative increments
        noise = 0.1 * self.kpi_rtol
        best: Optional[dict[str, Any]] = None
        for k in range(kpis.shape[1]):
            d = deltas[:, k]
            signs = np.sign(np.where(np.abs(d) > noise, d, 0.0))
            signs = signs[signs != 0.0]
            if signs.size < 2:
                continue
            sign_changes = int(np.sum(signs[1:] != signs[:-1]))
            if sign_changes < self.osc_min_sign_changes:
                continue
            series = kpis[:, k] / scale[k]
            half = len(series) // 2
            amp_old = float(np.ptp(series[:half]))
            amp_new = float(np.ptp(series[half:]))
            amp = float(np.ptp(series))
            if amp < self.osc_amplitude_factor * self.kpi_rtol:
                continue  # jitter around stagnation, not a cycle
            if min(amp_old, amp_new) <= 0.0 or max(amp_old, amp_new) > 3.0 * min(amp_old, amp_new):
                continue  # amplitude not roughly constant (decaying spiral etc.)
            candidate = {
                "component": k,
                "sign_changes": sign_changes,
                "amplitude_rel": amp,
                "cycles_observed": sign_changes / 2.0,
                "checks_in_window": len(self.history),
            }
            if best is None or candidate["amplitude_rel"] > best["amplitude_rel"]:
                best = candidate
        return best

    # -- the one entry point ------------------------------------------

    def observe(
        self,
        *,
        step: int,
        g_ss_norm: float,
        dt: float,
        kpi: np.ndarray,
    ) -> Classification:
        """Record one check and classify the trailing history."""
        kpi = np.asarray(kpi, dtype=float)
        rel = None
        if self.history:
            rel = self._rel_change(kpi, self.history[-1].kpi)
            if rel < self.kpi_rtol:
                self.stagnant_checks += 1
            else:
                self.stagnant_checks = 0
        self.history.append(
            CheckRecord(step=step, g_ss_norm=float(g_ss_norm), dt=float(dt),
                        kpi=kpi, rel_change=rel)
        )

        if not np.isfinite(g_ss_norm) or not np.all(np.isfinite(kpi)):
            result = Classification("diverging", {"reason": "non-finite residual or KPI"})
        elif self._residual_growing():
            result = Classification(
                "diverging",
                {"reason": "residual growing across the window",
                 "norms": [rec.g_ss_norm for rec in self.history]},
            )
        else:
            osc = self._oscillation()
            if osc is not None and self._residual_stalled():
                result = Classification("oscillatory", osc)
            elif (
                self.stagnant_checks >= self.n_stagnant
                and self._residual_stalled()
                and (dt >= 0.99 * self.dt_max or step > self.soft_step_cap)
            ):
                result = Classification(
                    "floored",
                    {"stagnant_checks": self.stagnant_checks,
                     "kpi_rel_change": rel,
                     "achieved_residual": float(g_ss_norm),
                     "dt_at_dt_max": bool(dt >= 0.99 * self.dt_max)},
                )
            else:
                result = Classification("progressing", {"kpi_rel_change": rel})
        self.last = result
        return result
