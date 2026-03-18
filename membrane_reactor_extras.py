import logging

import numpy as np

from numerical_safety import RecoverableNumericalError


logger = logging.getLogger(__name__)

EPS_CHEM_TIMESTEP = 1e-8


class LegacyAdaptiveSolveMixin:
    """Legacy helper methods retained outside the core reactor class body.

    Host classes are expected to provide the reactor state attributes used here
    (for example cpT, num_r_perm, kinetics, and last_solver_failure_message)
    together with the core helper methods invoked by these legacy routines
    (_split_perm_and_ret, _reaction_partial_pressures, _solve_cpT, and
    _restore_state).
    """

    def _compute_dt_chem_min(self):
        """Return minimum explicit chemical time step avoiding negative c."""
        c_ret = self.cpT[:, self.num_r_perm :, :-2]
        _, T_ret = self._split_perm_and_ret(self.cpT[..., -1])
        _, p_ret = self._split_perm_and_ret(self.cpT[..., -2])
        rates = self.kinetics(
            self._reaction_partial_pressures(c_ret, T_ret, p_ret),
            T_ret,
        )
        dt_chem_local = np.where(
            rates < 0,
            np.maximum(c_ret, EPS_CHEM_TIMESTEP) / (-rates + EPS_CHEM_TIMESTEP),
            np.inf,
        )
        dt_chem_min = np.min(dt_chem_local)
        return dt_chem_min

    def _solve_step(self, c_old, T_old, dt):
        """Compatibility wrapper for the monolithic correction solve."""
        return self._solve_cpT(c_old, T_old, dt)

    def _solve_adaptive_dt(
        self,
        dt,
        c_old,
        T_old,
        dt_init=None,
        dt_min=None,
        dt_max=None,
        dt_factor_increase=1.2,
        dt_factor_decrease=0.5,
        verbose=0,
    ):
        """Solve using adaptive pseudo-time stepping.

        Integrates from t=0 to t=dt using adaptive step sizes. Step size
        increases after successful convergence and decreases after failure.

        Args:
            dt: Target pseudo-time to integrate to (final time).
            c_old: Previous concentration field for transient term.
            T_old: Previous temperature field for transient term.
            dt_init: Initial step size. Defaults to min of chemical timescale and dt.
            dt_min: Minimum step size before giving up. Defaults to 0.2*dt_chem.
            dt_max: Maximum step size. Defaults to dt.
            dt_factor_increase: Step size multiplier after success (default 1.2).
            dt_factor_decrease: Step size multiplier after failure (default 0.5).
            verbose: Verbosity level (0=quiet, 1=warnings, 2=progress).

        Returns:
            True if converged, False otherwise.
        """
        dt_chem = None
        if dt_init is None:
            dt_chem = self._compute_dt_chem_min()
            dt_init = min(dt_chem, dt)
        if dt_min is None:
            if dt_chem is None:
                dt_chem = self._compute_dt_chem_min()
            dt_min = min(0.2 * dt_chem, dt_init, dt)
        if dt_max is None:
            dt_max = max(dt_init, dt)
        t_final = dt
        dt = dt_init

        cpT_prev = self.cpT.copy()
        t = 0.0
        is_converged = False
        while t < t_final or not is_converged:
            try:
                result = self._solve_cpT(c_old, T_old, dt)
                is_converged = result.converged
                is_converging = is_converged
            except RecoverableNumericalError as exc:
                self.last_solver_failure_message = (
                    f"adaptive dt step failed at dt={dt:.2e}: {exc}"
                )
                is_converging = False
            except Exception:
                is_converging = False
            if is_converging:
                cpT_prev = self.cpT.copy()
                t += dt
                dt = min(t + dt * dt_factor_increase, t + dt_max, t_final) - t
                if verbose > 1:
                    logger.info("Increasing timestep to dt = %.4e, t = %.4e", dt, t)
            else:
                if dt <= dt_min:
                    break
                elif t >= t_final:
                    break
                else:
                    self._restore_state(cpT_prev)
                    dt = max(dt * dt_factor_decrease, dt_min)
                    self.kinetics.set_T_and_p(
                        T=self.cpT[..., -1][:, self.num_r_perm :],
                        p=self.cpT[:, self.num_r_perm :, -2],
                    )
                    if verbose > 1:
                        if self.last_solver_failure_message is not None:
                            logger.info(
                                "Reduced dt to %.4e after %s",
                                dt,
                                self.last_solver_failure_message,
                            )
                        else:
                            logger.info("Reduced dt to %.4e", dt)
        if verbose > 1 and not is_converged:
            logger.warning("Failed: could not converge at t = %.4e", t)
        return is_converged
