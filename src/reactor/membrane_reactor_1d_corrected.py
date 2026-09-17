"""
Concentration-polarization-corrected 1D membrane reactor model.

The 1D counterpart of :mod:`reactor.membrane_reactor` with a configurable
closure for the membrane-side concentration polarization that the plain 1D
model (:mod:`reactor.membrane_reactor_1d`) neglects: either the fitted
Sherwood power law ``Sh = sh_cp_coeff * kappa**sh_cp_exp`` (kappa =
r_mem/r_max, coefficients supplied by the config and normally taken from
the fit on the certified 2D dataset) or the mechanistic reaction-screening
closure of :mod:`reactor.cp_closure` (``cp_closure="screened"``).

The module mirrors the architecture of the 2D model in membrane_reactor.py.  The key
idea: num_r=2, with one cell for permeate (index 0) and one for retentate
(index 1).  cpT has shape (num_z, 2, num_c+2) and uses the identical packing
convention as the 2D model.

Pressure–velocity coupling, Ergun (retentate), Poiseuille (permeate),
membrane permeation, and heat transfer follow the same formulation as the 2D
code.  The radial operators are replaced by algebraic membrane‐coupling
source terms whose Jacobians are analytical and assembled into the monolithic
Newton system (fully implicit).
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy.sparse.linalg as sla
from scipy.sparse import csc_array

from pymrm import (
    NumJac,
    construct_coefficient_matrix,
    construct_convflux_upwind,
    construct_div,
    construct_grad,
    interp_cntr_to_stagg,
    interp_cntr_to_stagg_tvd,
    interp_stagg_to_cntr,
    non_uniform_grid,
    update_csc_array_indices,
    upwind,
)

from .ammonia_synthesis_kinetics import AmmoniaSynthesisKinetics
from .config import ReactorConfig, get_membrane_permeances
from .gas_mixture_correlations import GasMixtureCorrelations
from .physics import (
    BC_DIRICHLET,
    BC_DIRICHLET_HOM,
    BC_NEUMANN,
    BC_NEUMANN_HOM,
    BC_NONE,
    compute_packed_bed_permeability,
    make_dirichlet_bc,
    get_axial_bcs_for_flow,
)
from .solvers import NewtonConfig, armijo_line_search
from .numerical_safety import RecoverableNumericalError

logger = logging.getLogger(__name__)

# Mesh refinement constants (same as mesh.py)
GRID_REFINEMENT_OFFSET_AXIAL = 8
GRID_REFINEMENT_MIN_FACTOR = 0.8
GRID_STRETCH_RATIO = 1.2

# Solver constants (mirror 2D)
CFL_INIT = 0.1
STEADY_STATE_DT = 1e10
STEADY_STATE_PLATEAU_REQUIRED = 20
STEADY_STATE_PLATEAU_TOL = 0.02
DEFAULT_NEWTON_CONFIG = NewtonConfig()
ARMIJO_COEFF = DEFAULT_NEWTON_CONFIG.armijo_coeff
MIN_LINE_SEARCH_ALPHA = DEFAULT_NEWTON_CONFIG.min_line_search_alpha


# =========================================================================
# Result dataclasses (identical to 2D)
# =========================================================================

@dataclass
class SolveResult:
    """Result of one Newton solve of the coupled cpT system."""
    converged: bool
    num_iterations: int
    g_norm: float
    g_norm_init: float

    @property
    def convergence_factor(self) -> float:
        """Ratio of final to initial Newton residual norm."""
        return self.g_norm / self.g_norm_init if self.g_norm_init > 0 else 0.0

    def as_tuple(self):
        """Return (num_iterations, g_norm, converged, convergence_factor) for legacy tuple unpacking."""
        return (self.num_iterations, self.g_norm, self.converged, self.convergence_factor)

    def __iter__(self):
        return iter(self.as_tuple())


@dataclass
class SteadyStateSolveStatus:
    """Status record from the adaptive pseudo-transient steady-state solve."""
    converged: bool
    num_steps_attempted: int
    num_steps_accepted: int
    final_dt: float
    steady_state_norm: Optional[float]
    initial_steady_state_norm: Optional[float]
    best_steady_state_norm: Optional[float]
    baseline_steady_state_norm: Optional[float]
    last_failure_message: Optional[str]


# =========================================================================
# 1D membrane reactor class
# =========================================================================

class MembraneReactor1D:
    """1D axisymmetric membrane reactor model.

    Architecture: cpT has shape (num_z, 2, num_c+2).
      - Index [:, 0, :] = permeate
      - Index [:, 1, :] = retentate
      - cpT[..., :-2] = species concentrations
      - cpT[..., -2]  = pressure
      - cpT[..., -1]  = temperature
    """

    # Expose BC templates for external code
    BC_NONE = BC_NONE
    BC_DIRICHLET_HOM = BC_DIRICHLET_HOM
    BC_NEUMANN_HOM = BC_NEUMANN_HOM
    BC_DIRICHLET = BC_DIRICHLET
    BC_NEUMANN = BC_NEUMANN

    def __init__(
        self,
        config_file: Optional[str] = None,
        config: Optional[ReactorConfig] = None,
        c: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        # --- config ---
        """Build the 1D reactor: load the config, expose scalar parameters,
        set up mesh/fields/Jacobians, and run a short non-reactive
        pressure-velocity initialization solve.
        """
        if config is not None:
            if not isinstance(config, ReactorConfig):
                raise TypeError("config must be ReactorConfig")
            self._config = config
        else:
            self._config = ReactorConfig.from_defaults(config_file, **kwargs)
        self.param_dict = self._config.to_dict()

        # Expose scalar config attributes for convenience
        cfg = self._config
        self.species = cfg.species
        self.num_c = cfg.num_c
        self.num_z = cfg.num_z
        self.Rg = cfg.Rg
        self.L = cfg.L
        self.Lsealing = cfg.Lsealing
        self.nu = cfg.nu
        self.r_min = cfg.r_min
        self.r_max = cfg.r_max
        self.r_min_perm = cfg.r_min_perm
        self.r_max_perm = cfg.r_max_perm
        self.Nu_ret = cfg.Nu_ret
        self.Nu_perm = cfg.Nu_perm
        self.lambda_mem = cfg.lambda_mem
        self.Nm = cfg.Nm
        self.eps = cfg.eps
        self.Dcat = cfg.Dcat
        self.rho_c = cfg.rho_c
        self.rho_b = cfg.rho_b
        self.dp = cfg.dp
        self.F_ret_in = cfg.F_ret_in
        self.F_perm_in = cfg.F_perm_in
        self.is_counter_current = cfg.is_counter_current
        self.p_ret_out = cfg.p_ret_out
        self.p_perm_out = cfg.p_perm_out
        self.is_isothermal = cfg.is_isothermal
        self.T_ret_in = cfg.T_ret_in
        self.T_perm_in = cfg.T_perm_in
        self.T_ret_init = cfg.T_ret_init
        self.T_perm_init = cfg.T_perm_init
        self.y_ret_init = cfg.y_ret_init
        self.y_perm_init = cfg.y_perm_init
        self.y_ret_in = cfg.y_ret_in
        self.y_perm_in = cfg.y_perm_in
        self.factor_react = cfg.factor_react
        self.factor_p = cfg.factor_p
        self.factor_T = cfg.factor_T
        self.max_newton_iterations = cfg.max_newton_iterations
        self.rtol = cfg.rtol
        self.atol = cfg.atol
        self.ord_norm = cfg.ord_norm
        self.num_timesteps = cfg.num_timesteps
        self.dt_init = cfg.dt_init
        self.dt_min = cfg.dt_min
        self.dt_max = cfg.dt_max
        self.dt_increase_factor = cfg.dt_increase_factor
        self.dt_decrease_factor = cfg.dt_decrease_factor
        self.adaptive_dt_threshold = cfg.adaptive_dt_threshold
        self.steady_state_atol = cfg.steady_state_atol
        self.steady_state_rtol = cfg.steady_state_rtol
        self.newton_conv_rate_min = cfg.newton_conv_rate_min
        self.kinetics_uses_auxiliary_pressure = cfg.kinetics_uses_auxiliary_pressure
        self.database = cfg.database
        # Sherwood-CP closure (see compute_kcp_sh); getattr keeps old
        # serialized configs loadable.
        self.sh_cp_coeff = getattr(cfg, "sh_cp_coeff", 6.13)
        self.sh_cp_exp = getattr(cfg, "sh_cp_exp", -0.859)
        self.cp_closure = getattr(cfg, "cp_closure", "sh_kappa")
        self.screen_c1 = getattr(cfg, "screen_c1", 1.0)
        self.screen_c2 = getattr(cfg, "screen_c2", 1.0)
        # Exact-tracing hook: set to a (num_z, num_c) array of k_cp values
        # extracted from a converged 2D solution to bypass the correlation.
        self.kcp_override = None

        # Convenience: "num_r" for the 1D model is always 2
        self.num_r = 2
        self.num_r_perm = 1
        self.num_r_ret = 1

        self._init_derived()
        self._init_fields(c=c, p=p, T=T)
        self._init_jac()

        # Initial non-reactive pressure-velocity solve
        self.factor_norm = (self.cpT.size) ** (-1.0 / self.ord_norm)
        self.last_solve_status = None
        self.conv_factor_min = math.exp(-self.newton_conv_rate_min)

        dt_cfl = self._compute_dt_cfl(cfl=CFL_INIT)
        factor_react_save = self.factor_react
        self.factor_react = 0.0
        self.solve(dt=dt_cfl, num_timesteps=2)
        self.factor_react = factor_react_save

    # =====================================================================
    # Initialisation helpers
    # =====================================================================

    def _init_derived(self):
        """Set up mesh, correlations, permeabilities, and inlet fluxes."""
        self.correlation = GasMixtureCorrelations(self.species, self.database)

        # ----- axial mesh (same as 2D mesh.py) -----
        num_z_sealing = int(np.round(self.Lsealing / self.L * self.num_z))
        z_f_uniform = np.linspace(0, self.Lsealing, num_z_sealing + 1)
        dz_nonuniform = (self.L - self.Lsealing) / max(
            self.num_z - num_z_sealing - GRID_REFINEMENT_OFFSET_AXIAL,
            GRID_REFINEMENT_MIN_FACTOR * (self.num_z - num_z_sealing),
        )
        z_f_non_uniform = non_uniform_grid(
            self.Lsealing, self.L,
            self.num_z + 1 - num_z_sealing,
            dz_nonuniform, GRID_STRETCH_RATIO,
        )
        self.z_f = np.concatenate((z_f_uniform, z_f_non_uniform[1:]), axis=0)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        self.dz = self.z_f[1:] - self.z_f[:-1]

        # ----- membrane permeances -----
        self.P0, self.EA = get_membrane_permeances(
            species=self.species,
            config=self._config,
            z_c=self.z_c,
            Lsealing=self.Lsealing,
        )

        # ----- cross-sectional areas -----
        self.A_ret = np.pi * (self.r_max**2 - self.r_min**2)  # retentate annulus
        self.A_perm = np.pi * self.r_max_perm**2  # permeate tube

        # ----- membrane geometric factors -----
        # S_mem = membrane_perimeter * dz / cell_volume  per axial cell
        # For species source: J_i enters per unit membrane area;
        # we need it per unit cell volume.
        # Retentate cell volume per unit length: A_ret
        # Permeate cell volume per unit length: A_perm
        # Membrane perimeter (inner cylinder): 2 pi r_min

        self.r_mem = self.r_min

        self.S_mem_ret = 2.0 * np.pi * self.r_mem / self.A_ret
        self.S_mem_perm = 2.0 * np.pi * self.r_mem / self.A_perm

        # Species transfer geometric factor. The residuals are per unit of
        # each side's OWN cross-section volume, and S_mem_ret/S_mem_perm are
        # already the per-side specific membrane areas (2*pi*r_mem / A_side),
        # so conservation of the molar rate per unit length reads
        #   A_ret * (S_mem_ret * J) == A_perm * (S_mem_perm * J)  == 2*pi*r_mem*J
        # which holds with factor_geom_mem = 1.0 exactly.
        #
        # BUG FIX 2026-08-17: this used to be A_perm/A_ret, mis-derived from
        # requiring S_mem_perm*factor == S_mem_ret (a same-volume-basis
        # identity that does not apply here). With the paper geometry that
        # under-fed the permeate by ~20x: measured on G2_3000, a 16%
        # hydrogen element-balance violation and NH3 recovery of 0.47%
        # instead of ~58%. The plain 1D model always had 1.0 and conserves
        # to ~5e-6.
        self.factor_geom_mem = 1.0

        # Geometry diagnostics (debug level only)
        logger.debug("r_min = %s", self.r_min)
        logger.debug("r_max_perm = %s", self.r_max_perm)
        logger.debug("A_ret = %s", self.A_ret)
        logger.debug("A_perm = %s", self.A_perm)
        logger.debug("S_mem_ret = %s", self.S_mem_ret)
        logger.debug("S_mem_perm = %s", self.S_mem_perm)
        logger.debug("factor_geom_mem = %s", self.factor_geom_mem)
        logger.debug("S_mem_perm_eff = %s", self.S_mem_perm * self.factor_geom_mem)

        # ----- inlet molar fluxes (mol/(m^2 s)) -----
        # Retentate: uniform over annulus
        self.flux_ret_in = (
            self.F_ret_in / self.A_ret
            * self.y_ret_in.reshape((1, 1, self.num_c))
        )  # (1, 1, nc)
        # Permeate: average over tube (cross-section avg of Poiseuille = F/A)
        self.flux_perm_in = (
            self.F_perm_in / self.A_perm
            * self.y_perm_in.reshape((1, 1, self.num_c))
        )  # (1, 1, nc)

        _nh3_trace = 1e-4
        self.inflow_conc_ret_backflow = (
            self.p_ret_out / (self.Rg * self.T_ret_in)
            * np.array([[[0.0, 1.0 - _nh3_trace, _nh3_trace]]])
        )

    def _init_fields(self, c=None, p=None, T=None):
        """Initialise cpT, velocity arrays, counters.

        If c, p, T are provided they are used as initial state instead of
        the default equilibrium/uniform initialisation.
        """
        nz, nc = self.num_z, self.num_c
        self.cpT = np.empty((nz, 2, nc + 2))

        if c is not None:
            self.cpT[..., :nc] = c
        else:
            # Retentate concentrations
            c_ret_0 = (
                self.correlation.molar_density(self.y_ret_init, self.T_ret_init, self.p_ret_out)
                * self.y_ret_init
            ).ravel()
            self.cpT[:, 1, :nc] = c_ret_0

            # Permeate concentrations
            c_perm_0 = (
                self.correlation.molar_density(self.y_perm_init, self.T_perm_init, self.p_perm_out)
                * self.y_perm_init
            ).ravel()
            self.cpT[:, 0, :nc] = c_perm_0

        if p is not None:
            self.cpT[..., -2] = p
        else:
            self.cpT[:, 1, -2] = self.p_ret_out
            self.cpT[:, 0, -2] = self.p_perm_out

        if T is not None:
            self.cpT[..., -1] = T
        else:
            self.cpT[:, 1, -1] = self.T_ret_init
            self.cpT[:, 0, -1] = self.T_perm_init

        # Scalar velocities at axial faces: shape (num_z+1,)
        c_ret_0 = self.cpT[0, 1, :nc]
        c_perm_0 = self.cpT[0, 0, :nc]
        c_tot_ret = float(np.sum(c_ret_0))
        c_tot_perm = float(np.sum(c_perm_0))
        u_ret_uniform = float(np.sum(self.flux_ret_in)) / max(c_tot_ret, 1e-20)
        u_perm_uniform = float(np.sum(self.flux_perm_in)) / max(c_tot_perm, 1e-20)
        self.u_ret_ax = np.full(nz + 1, u_ret_uniform if not self.is_counter_current else -u_ret_uniform)
        self.u_perm_ax = np.full(nz + 1, u_perm_uniform)
        self.div_u_ret = np.zeros(nz)
        self.div_u_perm = np.zeros(nz)

        self.cnt_num_solves_cpT = 0

    # =====================================================================
    # Jacobian / operator initialisation
    # =====================================================================

    def _init_jac(self):
        """Build FVM operators for the 1D model."""
        nz, nc = self.num_z, self.num_c

        # Shapes for pymrm operators (treat each side as (num_z, 1, nc))
        shape_c_side = (nz, 1, nc)
        shape_p_side = (nz, 1)
        shape_cpT_full = (nz, 2, nc + 2)

        # ---- Accumulation ----
        self.jac_c_accum_perm = construct_coefficient_matrix(1.0, shape_c_side)
        self.jac_c_accum_ret = construct_coefficient_matrix(self.eps, shape_c_side)
        self.jac_T_accum_perm = construct_coefficient_matrix(1.0, shape_p_side)
        self.jac_T_accum_ret = construct_coefficient_matrix(1.0, shape_p_side)

        # ---- Divergence operators (axial only, nu=0 for z-axis) ----
        self.div_c_perm_ax = construct_div(shape_c_side, self.z_f, nu=0, axis=0)
        self.div_c_ret_ax = construct_div(shape_c_side, self.z_f, nu=0, axis=0)
        self.div_p_perm_ax = construct_div(shape_p_side, self.z_f, nu=0, axis=0)
        self.div_p_ret_ax = construct_div(shape_p_side, self.z_f, nu=0, axis=0)

        # ---- Determine BC tuples based on flow direction ----
        if self.is_counter_current:
            bc_c_ret_ax = (BC_NEUMANN_HOM, BC_NONE)
            bc_p_ret_ax = (BC_DIRICHLET, BC_NONE)
            bc_T_ret_ax = (BC_NEUMANN_HOM, BC_DIRICHLET)
        else:
            bc_c_ret_ax = (BC_NONE, BC_NEUMANN_HOM)
            bc_p_ret_ax = (BC_NONE, BC_DIRICHLET)
            bc_T_ret_ax = (BC_DIRICHLET, BC_NEUMANN_HOM)

        # ---- Gradient operators ----
        # Permeate concentration
        self.grad_c_perm_ax, self.grad_bc_c_perm_ax = construct_grad(
            shape_c_side, self.z_f, self.z_c,
            bc=(BC_NONE, BC_NEUMANN_HOM), axis=0,
        )
        # Retentate concentration
        self.grad_c_ret_ax, self.grad_bc_c_ret_ax = construct_grad(
            shape_c_side, self.z_f, self.z_c, bc=bc_c_ret_ax, axis=0,
        )
        # Permeate pressure
        self.grad_p_perm_ax, self.grad_bc_p_perm_ax = construct_grad(
            shape_p_side, self.z_f, self.z_c,
            bc=(BC_NONE, BC_DIRICHLET), axis=0,
        )
        self.grad_bc_p_perm_ax *= self.p_perm_out
        # Retentate pressure
        self.grad_p_ret_ax, self.grad_bc_p_ret_ax = construct_grad(
            shape_p_side, self.z_f, self.z_c, bc=bc_p_ret_ax, axis=0,
        )
        self.grad_bc_p_ret_ax *= self.p_ret_out

        # Temperature gradients
        self.grad_T_perm_ax, self.grad_bc_T_perm_ax = construct_grad(
            shape_p_side, self.z_f, self.z_c,
            bc=(BC_DIRICHLET, BC_NEUMANN_HOM), axis=0,
        )
        self.grad_bc_T_perm_ax *= self.T_perm_in
        self.grad_T_ret_ax, self.grad_bc_T_ret_ax = construct_grad(
            shape_p_side, self.z_f, self.z_c, bc=bc_T_ret_ax, axis=0,
        )
        self.grad_bc_T_ret_ax *= self.T_ret_in

        # ---- Shift to monolithic indexing ----
        # Full shapes
        shape_c_full = (nz, 2, nc)
        shape_p_full = (nz, 2)
        offset_ret_c = (0, 1, 0)
        offset_ret_p = (0, 1)

        # Concentration accumulation → monolithic
        jcp = update_csc_array_indices(self.jac_c_accum_perm, shape_c_side, shape_c_full)
        jcr = update_csc_array_indices(self.jac_c_accum_ret, shape_c_side, shape_c_full, offset=offset_ret_c)
        self.jac_c_accum = jcp + jcr

        # Temperature accumulation → monolithic
        jTp = update_csc_array_indices(self.jac_T_accum_perm, shape_p_side, shape_p_full)
        jTr = update_csc_array_indices(self.jac_T_accum_ret, shape_p_side, shape_p_full, offset=offset_ret_p)
        self.jac_T_accum = jTp + jTr

        # Divergence operators → monolithic rows
        self.div_c_perm_ax_full = update_csc_array_indices(
            self.div_c_perm_ax, (shape_c_side, None), (shape_c_full, None))
        self.div_c_ret_ax_full = update_csc_array_indices(
            self.div_c_ret_ax, (shape_c_side, None), (shape_c_full, None),
            offset=(offset_ret_c, None))
        self.div_p_perm_ax_full = update_csc_array_indices(
            self.div_p_perm_ax, (shape_p_side, None), (shape_p_full, None))
        self.div_p_ret_ax_full = update_csc_array_indices(
            self.div_p_ret_ax, (shape_p_side, None), (shape_p_full, None),
            offset=(offset_ret_p, None))
        
        # ---- NumJac helpers ----
        self.numjac = NumJac(shape_c_side)
        self.numjac_p = NumJac(shape_p_full + (1,))
        shape_p_ret_nj = shape_p_side + (1,)
        self.numjac_cT = NumJac(shape_in=shape_p_ret_nj, shape_out=shape_c_side)
        self.numjac_Tc = NumJac(shape_in=shape_c_side, shape_out=shape_p_ret_nj)
        self.numjac_TT_react = NumJac(shape_p_ret_nj)

        # ---- Summation matrix (species → total density) ----
        self.sum_c = construct_coefficient_matrix(
            np.array([[[1.0]]]), shape=(shape_p_full + (1,), shape_c_full)
        )

        # ---- Inlet flux contribution to residual ----
        self.g_c_in = np.zeros(shape_c_full)
        flux_perm = self.flux_perm_in  # (1, 1, nc)
        flux_ret = self.flux_ret_in    # (1, 1, nc)
        # Permeate: always inlet at z=0
        g_c_in_vec = (self.div_c_perm_ax_full[:, :flux_perm.size] @ flux_perm.ravel())
        self.g_c_in.ravel()[:] += g_c_in_vec.ravel()
        # Retentate: depends on flow direction
        if self.is_counter_current:
            g_c_in_vec = -(self.div_c_ret_ax_full[:, -flux_ret.size:] @ flux_ret.ravel())
        else:
            g_c_in_vec = (self.div_c_ret_ax_full[:, :flux_ret.size] @ flux_ret.ravel())
        self.g_c_in.ravel()[:] += g_c_in_vec.ravel()

        # Kinetics
        T_ret = self.cpT[:, 1, -1]
        p_ret = self.cpT[:, 1, -2]
        self.kinetics = AmmoniaSynthesisKinetics(
            self.species,
            T=T_ret.reshape((-1, 1)),
            p=p_ret.reshape((-1, 1)),
            rho_b=self.rho_b,
            rho_c=self.rho_c,
        )

        self._jac = None
        self.last_solver_failure_message = None

    # =====================================================================
    # Pressure – velocity coupling
    # =====================================================================

    def _construct_darcy_matrices(self):
        """Build permeability-weighted Darcy matrices for both sides."""
        c = self.cpT[..., :-2]
        T = self.cpT[..., -1]
        p = self.cpT[..., -2]

        c_ret = c[:, 1:, :]      # (nz, 1, nc)
        T_ret = T[:, 1:]          # (nz, 1)
        p_ret = p[:, 1:]          # (nz, 1)
        c_perm = c[:, :1, :]
        T_perm = T[:, :1]

        shape_p_side = (self.num_z, 1)

        # ---- retentate: Ergun ----
        visc_ret = self.correlation.viscosity(c_ret, T_ret)
        rho_ret = self.correlation.density(c_ret, T_ret, p_ret)
        u_abs = np.abs(interp_stagg_to_cntr(
            self.u_ret_ax.reshape(-1, 1), self.z_f, self.z_c, axis=0
        ))  # (nz, 1)
        k_ret = compute_packed_bed_permeability(visc_ret, rho_ret, u_abs, self.eps, self.dp)
        k_ret_ax = interp_cntr_to_stagg(k_ret, self.z_f, self.z_c, axis=0)
        self.k_matrix_ret_ax = construct_coefficient_matrix(k_ret_ax, shape_p_side, axis=0)

        # ---- permeate: Poiseuille (cross-section average) ----
        visc_perm = self.correlation.viscosity(c_perm, T_perm)
        R = self.r_max_perm
        k_perm = R**2 / (8.0 * visc_perm)  # (nz, 1)
        k_perm_ax = interp_cntr_to_stagg(k_perm, self.z_f, self.z_c, axis=0)
        self.k_matrix_perm_ax = construct_coefficient_matrix(k_perm_ax, shape_p_side, axis=0)

    def _update_velocity_fields(self):
        """Compute axial velocities from pressure gradients (Darcy)."""
        p = self.cpT[..., -2]
        p_ret = p[:, 1:].reshape((-1, 1))  # (nz, 1)
        p_perm = p[:, :1].reshape((-1, 1))

        # u = -K grad(p) + bc
        vel_ret = (-self.k_matrix_ret_ax) @ self.grad_p_ret_ax
        vel_bc_ret = (-self.k_matrix_ret_ax) @ self.grad_bc_p_ret_ax
        self.u_ret_ax.reshape((-1, 1))[...] = vel_ret @ p_ret + vel_bc_ret

        vel_perm = (-self.k_matrix_perm_ax) @ self.grad_p_perm_ax
        vel_bc_perm = (-self.k_matrix_perm_ax) @ self.grad_bc_p_perm_ax
        self.u_perm_ax.reshape((-1, 1))[...] = vel_perm @ p_perm + vel_bc_perm

        # Extrapolate inlet face velocity (same trick as 2D)
        self.u_perm_ax[0] = self.u_perm_ax[1] - (
            (self.z_f[1] - self.z_f[0]) / (self.z_f[2] - self.z_f[1])
            * (self.u_perm_ax[2] - self.u_perm_ax[1])
        )
        if self.is_counter_current:
            self.u_ret_ax[-1] = self.u_ret_ax[-2] - (
                (self.z_f[-2] - self.z_f[-1]) / (self.z_f[-3] - self.z_f[-2])
                * (self.u_ret_ax[-3] - self.u_ret_ax[-2])
            )
        else:
            self.u_ret_ax[0] = self.u_ret_ax[1] - (
                (self.z_f[1] - self.z_f[0]) / (self.z_f[2] - self.z_f[1])
                * (self.u_ret_ax[2] - self.u_ret_ax[1])
            )

        # Divergence (scalar per cell)
        self.div_u_perm = (self.div_p_perm_ax @ self.u_perm_ax.reshape((-1, 1))).ravel()
        self.div_u_ret = (self.div_p_ret_ax @ self.u_ret_ax.reshape((-1, 1))).ravel()

    # =====================================================================
    # Membrane coupling (implicit)
    # =====================================================================
    def compute_kcp_sh(self, c_ret_cell, T_ret):
        """CP mass-transfer coefficient k_cp [m/s] per axial cell and species.

        Two modes:

        * ``self.kcp_override`` set (shape (num_z, num_c)): use it verbatim.
          This is the exact-tracing mode — the override is extracted from a
          converged 2D solution (k_cp,i(z) = J_i / (c_mean,i - c_wall,i)),
          so the 1D model reproduces the 2D closure by construction.
        * otherwise the fitted correlation ``Sh = sh_cp_coeff * kappa**sh_cp_exp``
          with the geometry ratio defined as ``kappa = r_mem / r_max`` (< 1,
          membrane radius over shell radius; the definition the fit in
          ``reactor.paper.closures`` uses) and ``k_cp = Sh * D_eff / d_h``
          with ``d_h = r_max - r_min``. The coefficients live in the config
          (``sh_cp_coeff``, ``sh_cp_exp``) so a refit is a config change,
          not a code change.
        """
        nz, nc = c_ret_cell.shape

        if self.kcp_override is not None:
            kcp = np.broadcast_to(
                np.asarray(self.kcp_override, dtype=float), (nz, nc)
            )
            return np.maximum(kcp, 1e-12)

        # Bulk mole fractions in the retentate cell
        ctot = np.maximum(np.sum(c_ret_cell, axis=1, keepdims=True), 1e-16)
        yret = c_ret_cell / ctot

        # correlation functions expect (nz, 1, nc) not (nz, nc) — add radial dim
        yret_3d = yret.reshape(nz, 1, nc)
        T_ret_2d = T_ret.reshape(nz, 1)
        p_ret_2d = self.cpT[:, 1, -2].reshape(nz, 1)

        # Effective diffusion coefficient D_eff (nz, nc)
        D_eff = self.correlation.diffusion(yret_3d, T_ret_2d, p_ret_2d)
        if D_eff.ndim >= 3:
            D_eff = D_eff.squeeze(axis=1)   # (nz, nc)
        D_eff = np.maximum(D_eff, 1e-12)   # avoid zero

        if self.cp_closure == "screened":
            # Mechanistic reaction-screening + conduction closure
            # (reactor.cp_closure), evaluated from the LOCAL bulk state —
            # the kinetics sensitivity makes it z-dependent and portable to
            # other chemistries.
            from .cp_closure import screened_kcp

            c_ret_3d = c_ret_cell.reshape(nz, 1, nc)
            dRdc = self._kinetics_diag_sensitivity(c_ret_3d, T_ret_2d, p_ret_2d)
            kcp = screened_kcp(
                D_eff, dRdc, r_mem=self.r_mem, r_max=self.r_max,
                c_screen=self.screen_c1, c_cond=self.screen_c2,
            )
            return np.maximum(kcp, 1e-12)

        # kappa = r_mem / r_max, in (0, 1). NOTE: the pre-fix code computed
        # r_max/r_mem while its comment claimed r_mem/r_max — an ~8x Sh
        # ambiguity. The definition here is canonical; sh_cp_coeff/sh_cp_exp
        # must come from a fit using the same definition.
        kappa = self.r_mem / self.r_max
        self.dh = self.r_max - self.r_min
        Sh_opt = self.sh_cp_coeff * (kappa ** self.sh_cp_exp)

        # Same Sh for all cells/species; species enter through D_eff.
        Sh = np.full_like(D_eff, Sh_opt)
        kcp = Sh * D_eff / self.dh
        return np.maximum(kcp, 1e-12)


    def _kinetics_diag_sensitivity(self, c_ret_3d, T_ret_2d, p_ret_2d):
        """Diagonal kinetics sensitivities dR_i/dc_i [1/s] at the bulk state.

        Forward differences per species; the kinetics is always called with
        an explicit T (it carries internal state). Used by the "screened"
        CP closure: s_i = max(-dR_i/dc_i, 0) is the local healing rate that
        sets the screening length.
        """
        pp0 = self._reaction_partial_pressures(c_ret_3d, T_ret_2d, p_ret_2d)
        R0 = self.factor_react * self.kinetics(pp0, T_ret_2d)
        nz, _, nc = c_ret_3d.shape
        out = np.empty((nz, nc))
        c_scale = np.maximum(np.sum(c_ret_3d, axis=-1), 1e-12)   # (nz, 1)
        for j in range(nc):
            h = 1e-5 * np.maximum(np.abs(c_ret_3d[..., j]), 1e-6 * c_scale)
            ch = c_ret_3d.copy()
            ch[..., j] += h
            Rj = self.factor_react * self.kinetics(
                self._reaction_partial_pressures(ch, T_ret_2d, p_ret_2d),
                T_ret_2d,
            )
            out[:, j] = ((Rj[..., j] - R0[..., j]) / h)[:, 0]
        return out

    def _membrane_flux(self, c, T):
        """
        Compute membrane species flux J_i with Sherwood-based concentration
        polarization correction and its analytical Jacobian contributions.

        Governing equations:
            J_i = perm_i * (Rg*T_ret*c_sm_i - P_perm_i)

            kcp_i * (c_ret_i - c_sm_i) = perm_i * (Rg*T_ret*c_sm_i - P_perm_i)

        Solving for c_sm_i:
            c_sm_i = (kcp_i*c_ret_i + perm_i*P_perm_i) / (kcp_i + perm_i*Rg*T_ret)

        Substituting back gives an effective linear flux in c_ret and c_perm:
            J_i = perm_i * kcp_i * Rg*T_ret / denom_i * c_ret_i
                - perm_i^2 * Rg*T_perm / denom_i * Rg*T_ret * ... (see below)

        The Jacobian is assembled analytically (4-block structure, as in the old model),
        which is faster and more accurate than numerical differentiation.

        Returns
        -------
        g_mem : ndarray, shape (nz, 2, nc)
            Species residual contribution.
        jac_mem_c : sparse matrix
            Analytical Jacobian of g_mem wrt species concentrations c,
            in species-space ((nz, 2, nc) -> (nz, 2, nc)).
        """
        nz, nc = self.num_z, self.num_c

        c_ret = c[:, 1, :]     # (nz, nc)
        c_perm = c[:, 0, :]    # (nz, nc)
        T_ret = T[:, 1]        # (nz,)
        T_perm = T[:, 0]       # (nz,)
        Rg = self.Rg

        perm = self.P0 * np.exp(-self.EA / (Rg * T_ret[:, None]))   # (nz, nc)

        # Permeate-side partial pressure
        P_perm_i = Rg * T_perm[:, None] * c_perm                    # (nz, nc)

        # Sherwood-based CP mass-transfer coefficient
        kcp = self.compute_kcp_sh(c_ret, T_ret)                     # (nz, nc)

        # Membrane-surface retentate concentration
        denom = kcp + perm * Rg * T_ret[:, None]                    # (nz, nc)
        denom_safe = np.maximum(denom, 1e-16)
        c_sm = (kcp * c_ret + perm * P_perm_i) / denom_safe         # (nz, nc)

        # CP-corrected membrane flux
        J = perm * (Rg * T_ret[:, None] * c_sm - P_perm_i)         # (nz, nc)

        # ---- Residual ----
        g_mem = np.zeros((nz, 2, nc))
        # Retentate loses species: +S_mem_ret * J  (positive = loss, added to residual)
        g_mem[:, 1, :] = self.S_mem_ret * J
        # Permeate gains species: -S_mem_perm * factor_geom_mem * J
        g_mem[:, 0, :] = -self.S_mem_perm * self.factor_geom_mem * J

        # ---- Diagnostics ----
        self.kcp_last = kcp
        self.c_sm_last = c_sm
        self.J_last = J
        self.y_sm_last = c_sm / np.maximum(np.sum(c_sm, axis=1, keepdims=True), 1e-16)

        if "NH3" in self.species:
            idx_nh3 = self.species.index("NH3")
            self.y_nh3_sm_last = c_sm[:, idx_nh3] / np.maximum(np.sum(c_sm, axis=1), 1e-16)
            self.y_nh3_bulk_last = c_ret[:, idx_nh3] / np.maximum(np.sum(c_ret, axis=1), 1e-16)

        # ---- Analytical Jacobian (4-block structure, same as old model) ----
        # J = perm * (Rg*T_ret * c_sm - P_perm_i)
        # c_sm = (kcp * c_ret + perm * Rg*T_perm * c_perm) / denom
        # => dJ/dc_ret = perm * Rg*T_ret * kcp / denom      (effective coeff for c_ret)
        # => dJ/dc_perm = perm * Rg*T_perm * (perm*Rg*T_ret/denom - 1)
        #               = -perm * Rg*T_perm * kcp / denom   (effective coeff for c_perm)
        #
        # dg_ret/dc_ret  = +S_mem_ret * dJ/dc_ret
        # dg_ret/dc_perm = +S_mem_ret * dJ/dc_perm
        # dg_perm/dc_ret  = -S_mem_perm * factor_geom_mem * dJ/dc_ret
        # dg_perm/dc_perm = -S_mem_perm * factor_geom_mem * dJ/dc_perm
        eff_ret  = perm * Rg * T_ret[:, None] * kcp / denom_safe    # (nz, nc)
        eff_perm = -perm * Rg * T_perm[:, None] * kcp / denom_safe  # (nz, nc)

        coeff_ret_ret   =  self.S_mem_ret * eff_ret                          # (nz, nc)
        coeff_ret_perm  =  self.S_mem_ret * eff_perm                         # (nz, nc)
        coeff_perm_ret  = -self.S_mem_perm * self.factor_geom_mem * eff_ret  # (nz, nc)
        coeff_perm_perm = -self.S_mem_perm * self.factor_geom_mem * eff_perm # (nz, nc)

        shape_sub  = (nz, 1, nc)
        shape_full = (nz, 2, nc)
        offset_ret  = (0, 1, 0)
        offset_perm = (0, 0, 0)

        # 1) ret ← ret
        jac_rr = construct_coefficient_matrix(coeff_ret_ret.reshape(nz, 1, nc))
        jac_rr = update_csc_array_indices(
            jac_rr, (shape_sub, shape_sub), (shape_full, shape_full),
            offset=(offset_ret, offset_ret),
        )
        # 2) ret ← perm
        jac_rp = construct_coefficient_matrix(coeff_ret_perm.reshape(nz, 1, nc))
        jac_rp = update_csc_array_indices(
            jac_rp, (shape_sub, shape_sub), (shape_full, shape_full),
            offset=(offset_ret, offset_perm),
        )
        # 3) perm ← ret
        jac_pr = construct_coefficient_matrix(coeff_perm_ret.reshape(nz, 1, nc))
        jac_pr = update_csc_array_indices(
            jac_pr, (shape_sub, shape_sub), (shape_full, shape_full),
            offset=(offset_perm, offset_ret),
        )
        # 4) perm ← perm
        jac_pp = construct_coefficient_matrix(coeff_perm_perm.reshape(nz, 1, nc))
        jac_pp = update_csc_array_indices(
            jac_pp, (shape_sub, shape_sub), (shape_full, shape_full),
            offset=(offset_perm, offset_perm),
        )

        jac_mem_c = jac_rr + jac_rp + jac_pr + jac_pp
        return g_mem, jac_mem_c

    def _membrane_heat_flux(self, T, c):
        """Compute membrane heat transfer contribution to temperature residual.

        Returns (g_Tmem, jac_Tmem) where shapes match temperature field (nz, 2).
        jac_Tmem is in monolithic temperature-only indexing.
        """
        nz = self.num_z
        T_ret = T[:, 1]    # (nz,)
        T_perm = T[:, 0]   # (nz,)
        c_ret = c[:, 1, :]  # (nz, nc)
        c_perm = c[:, 0, :]

        # Compute Nu, h, U (same logic as 2D _construct_g_T_cond)
        # ---- Retentate side ----
        c_tot_ret = np.maximum(np.abs(np.sum(c_ret, axis=-1, keepdims=True)), 1e-10)
        y_ret = c_ret / c_tot_ret
        visc_ret = self.correlation.viscosity(y_ret, T_ret)
        rho_ret = self.correlation.molecular_weight(c_ret)
        u_ret_c = np.abs(interp_stagg_to_cntr(
            self.u_ret_ax.reshape(-1, 1), self.z_f, self.z_c, axis=0
        ).ravel())
        lmbda_ret = self.correlation.thermal_conductivity(y_ret, T_ret).ravel()
        Re_ret = np.abs(rho_ret.ravel() * self.dp * u_ret_c / visc_ret.ravel())
        cp_ret = self.correlation.specific_heat(c_ret, T_ret).ravel()
        Pr_ret = np.abs(visc_ret.ravel() / rho_ret.ravel() * cp_ret / lmbda_ret)
        Nu_ret_val = self.Nu_ret(Re_ret, Pr_ret)
        h_min = 1.0
        h_ret = np.maximum(Nu_ret_val * lmbda_ret / self.dp, h_min)

        # ---- Permeate side ----
        c_tot_perm = np.maximum(np.abs(np.sum(c_perm, axis=-1, keepdims=True)), 1e-10)
        y_perm = c_perm / c_tot_perm
        d_tube = 2.0 * self.r_max_perm
        visc_perm = self.correlation.viscosity(y_perm, T_perm)
        rho_perm = self.correlation.molecular_weight(c_perm)
        u_perm_c = np.abs(interp_stagg_to_cntr(
            self.u_perm_ax.reshape(-1, 1), self.z_f, self.z_c, axis=0
        ).ravel())
        lmbda_perm = self.correlation.thermal_conductivity(y_perm, T_perm).ravel()
        Re_perm = np.abs(rho_perm.ravel() * d_tube * u_perm_c / visc_perm.ravel())
        cp_perm = self.correlation.specific_heat(c_perm, T_perm).ravel()
        Pr_perm = np.abs(visc_perm.ravel() / rho_perm.ravel() * cp_perm / lmbda_perm)
        Nu_perm_val = self.Nu_perm(Re_perm, Pr_perm)
        h_perm = np.maximum(Nu_perm_val * lmbda_perm / d_tube, h_min)

        T_mean = 0.5 * (T_ret + T_perm)
        lambda_mem = self.lambda_mem(T_mean)

        if self.nu == 1:
            resist_mem = (
             self.r_max_perm * np.log(self.r_min / self.r_max_perm)
           ) / lambda_mem
            factor_geom_h = self.r_min / self.r_max_perm
            U = 1.0 / (1.0 / h_perm + resist_mem + 1.0 / (factor_geom_h * h_ret))
        else:
            resist_mem = (self.r_max_perm - self.r_min) / lambda_mem
            factor_geom_h = 1.0
            U = 1.0 / (1.0 / h_ret + resist_mem + 1.0 / h_perm)

        # Q = S_mem * U * (T_ret - T_perm)  per unit volume per second
        dT = T_ret - T_perm  # (nz,)

        # g_T_mem(ret) = +S_mem_ret * U * dT / cp_ret  (heat lost by retentate)
        # g_T_mem(perm) = -S_mem_perm * factor_geom * U * dT / cp_perm (heat gained)
        g_Tmem = np.zeros((nz, 2))
        cp_ret_safe = np.maximum(cp_ret, 1.0)
        cp_perm_safe = np.maximum(cp_perm, 1.0)
        g_Tmem[:, 1] = self.S_mem_ret * U * dT / cp_ret_safe
        g_Tmem[:, 0] = -self.S_mem_perm * self.factor_geom_mem * U * dT / cp_perm_safe

        # ---- Jacobian (analytical, linear in T_ret, T_perm) ----
        shape_T = (nz, 2)
        # dg_T_ret/dT_ret = +S_mem_ret * U / cp_ret
        # dg_T_ret/dT_perm = -S_mem_ret * U / cp_ret
        # dg_T_perm/dT_ret = -S_mem_perm * fgeo * U / cp_perm
        # dg_T_perm/dT_perm = +S_mem_perm * fgeo * U / cp_perm
        diag_rr = self.S_mem_ret * U / cp_ret_safe         # (nz,)
        diag_rp = -diag_rr
        diag_pr = -self.S_mem_perm * self.factor_geom_mem * U / cp_perm_safe
        diag_pp = -diag_pr

        # Build sparse: rows/cols indexed in (nz, 2) flattened
        n = nz * 2
        row_ret = np.arange(nz) * 2 + 1   # retentate rows
        row_perm = np.arange(nz) * 2       # permeate rows
        col_ret = row_ret
        col_perm = row_perm

        data = np.concatenate([diag_rr, diag_rp, diag_pr, diag_pp])
        rows = np.concatenate([row_ret, row_ret, row_perm, row_perm])
        cols = np.concatenate([col_ret, col_perm, col_ret, col_perm])
        jac_Tmem = csc_array((data, (rows, cols)), shape=(n, n))

        return g_Tmem, jac_Tmem, U, cp_ret_safe, cp_perm_safe

    # =====================================================================
    # Reaction helpers
    # =====================================================================

    def _reaction_partial_pressures(self, c_ret, T_ret, p_ret):
        """Return retentate species partial pressures (Pa), from the auxiliary
        pressure field or the ideal-gas law depending on
        ``kinetics_uses_auxiliary_pressure``.
        """
        if self.kinetics_uses_auxiliary_pressure:
            c_tot = np.sum(c_ret, axis=-1, keepdims=True)
            c_tot_safe = np.maximum(np.abs(c_tot), 1e-10)
            return c_ret * (p_ret[..., np.newaxis] / c_tot_safe)
        return c_ret * (self.Rg * T_ret)[..., np.newaxis]

    # =====================================================================
    # Full residual + Jacobian assembly
    # =====================================================================

    def _construct_g_cpT(self, c_old, T_old, dt, compute_jac=False):
        
        """Monolithic residual and Jacobian for concentrations + pressure + temperature.

        Returns (g, jac) with g shape (num_z, 2, num_c+2).
        """
        cpT = self.cpT
        nz, nc = self.num_z, self.num_c

        c = cpT[..., :-2]   # (nz, 2, nc)
        p = cpT[..., -2]    # (nz, 2)
        T = cpT[..., -1]    # (nz, 2)

        factor_p = self.factor_p
        factor_T = self.factor_T
        g = np.empty(cpT.shape)

        c_ret = c[:, 1:, :]   # (nz, 1, nc)
        c_perm = c[:, :1, :]  # (nz, 1, nc)
        T_ret = T[:, 1:]      # (nz, 1)
        T_perm = T[:, :1]     # (nz, 1)
        p_ret = p[:, 1:]      # (nz, 1)
        p_perm = p[:, :1]     # (nz, 1)

        c_sum = np.sum(c, axis=-1)
        c_sum_safe = np.maximum(np.abs(c_sum), 1e-10)
        y = c / c_sum_safe[..., np.newaxis]

        # ---- Transport state ----
        self._construct_darcy_matrices()
        self._update_velocity_fields()

        # ---- Convection ----
        g_conv, jac_conv, jac_darcy = self._construct_g_conv(c, compute_jac=compute_jac)

        # ---- Axial diffusion ----
        g_diff, jac_diff = self._construct_g_diff(c, T, p, compute_jac=compute_jac)

        # ---- Membrane coupling (species, implicit) ----
        g_mem, jac_mem_c = self._membrane_flux(c, T)

        # ---- Reaction ----
        if compute_jac:
            g_react, jac_react_local = self.numjac(
                lambda cv: self.factor_react * self.kinetics(
                    self._reaction_partial_pressures(cv, T_ret, p_ret), T_ret
                ),
                c=c_ret,
            )
        else:
            g_react = self.factor_react * self.kinetics(
                self._reaction_partial_pressures(c_ret, T_ret, p_ret), T_ret
            )
            jac_react_local = None

        # ---- Temperature ----
        if self.is_isothermal:
            g_T_conv = np.zeros_like(T)
            g_T_cond = np.zeros_like(T)
            g_Tmem = np.zeros_like(T)
            jac_T_conv = None
            jac_T_darcy = None
            jac_T_cond = None
            jac_Tmem = None
            g_T_react = np.zeros_like(T_ret)
        else:
            g_T_conv, jac_T_conv, jac_T_darcy = self._construct_g_T_conv(
                T, compute_jac=compute_jac
            )
            g_T_cond, jac_T_cond, cp_inv_perm, cp_inv_ret = self._construct_g_T_cond(T, c)
            g_Tmem, jac_Tmem, U_val, cp_ret_safe, cp_perm_safe = self._membrane_heat_flux(T, c)

            enthalpies = self.correlation.species_enthalpies(T_ret)
            cp_inv_ret_flat = cp_inv_ret.ravel()

            if compute_jac:
                shape_T_ret_loc = (nz, 1, 1)
                shape_c_ret_loc = (nz, 1, nc)

                _, jac_cT_react = self.numjac_cT(
                    lambda Tv: self.factor_react * self.kinetics(
                        self._reaction_partial_pressures(c_ret, Tv[..., 0], p_ret),
                        Tv[..., 0],
                    ),
                    c=T_ret[..., np.newaxis],
                )

                _, jac_Tc_react = self.numjac_Tc(
                    lambda cv: (
                        np.sum(
                            self.factor_react * self.kinetics(
                                self._reaction_partial_pressures(cv, T_ret, p_ret),
                                T_ret,
                            ) * enthalpies,
                            axis=-1,
                        ) * cp_inv_ret_flat.reshape(T_ret.shape)
                    )[..., np.newaxis],
                    c=c_ret,
                )

                g_T_react_val, jac_TT_react_ret = self.numjac_TT_react(
                    lambda Tv: (
                        np.sum(
                            self.factor_react * self.kinetics(
                                self._reaction_partial_pressures(c_ret, Tv[..., 0], p_ret),
                                Tv[..., 0],
                            ) * self.correlation.species_enthalpies(Tv[..., 0]),
                            axis=-1,
                        ) * cp_inv_ret_flat.reshape(T_ret.shape)
                    )[..., np.newaxis],
                    c=T_ret[..., np.newaxis],
                )
                g_T_react = g_T_react_val.reshape(T_ret.shape)

            else:
                g_T_react = (
                    np.sum(g_react * enthalpies, axis=-1)
                    * cp_inv_ret_flat.reshape(T_ret.shape)
                )

        # ===== Assemble Jacobian =====
        if compute_jac:
            shape_c_full = (nz, 2, nc)
            shape_c_side = (nz, 1, nc)
            shape_p_full = (nz, 2)
            shape_p_side = (nz, 1)
            shape_cpT = cpT.shape

            offset_c_ret = (0, 1, 0)
            offset_p = (0,) * (cpT.ndim - 1) + (nc,)
            offset_T = (0,) * (cpT.ndim - 1) + (nc + 1,)

            # ---- Species block ----
            jac_cc = jac_conv + jac_diff
            if c_old is not None:
                jac_cc += (1.0 / dt) * self.jac_c_accum

            jac_mem_c_full = update_csc_array_indices(jac_mem_c, shape_c_full, shape_c_full)
            jac_cc += jac_mem_c_full

            jac_react_mono = update_csc_array_indices(
                jac_react_local, shape_c_side, shape_c_full, offset=offset_c_ret
            )
            jac_cc -= jac_react_mono

            # ---- Pressure block ----
            c_tot, dc_tot_dp = self.numjac_p(
                lambda pv: self.correlation.molar_density(y, T, pv),
                c=p,
            )
            _, dc_tot_dT = self.numjac_p(
                lambda Tv: self.correlation.molar_density(y, Tv, p),
                c=T,
                f_value=c_tot,
            )

            jac_pp = dc_tot_dp
            jac_pc = -self.sum_c
            jac_pT = dc_tot_dT
            jac_cp = jac_darcy

            # ---- Temperature block ----
            if self.is_isothermal:
                jac_TT = self.jac_T_accum
                jac_TP = None
            else:
                jac_TT = (1.0 / dt) * self.jac_T_accum + jac_T_conv + jac_T_cond
                jac_TP = jac_T_darcy

                jac_TT += update_csc_array_indices(
                    jac_TT_react_ret,
                    shape_T_ret_loc,
                    (nz, 2, 1),
                    offset=(0, 1, 0),
                )

            # ---- Map blocks to cpT space ----
            shape_p_1 = shape_p_full + (1,)

            jac_cc = update_csc_array_indices(jac_cc, shape_c_full, shape_cpT)
            jac_pp = update_csc_array_indices(jac_pp, shape_p_1, shape_cpT, offset=offset_p)
            jac_cp = update_csc_array_indices(
                jac_cp, (shape_c_full, shape_p_1), shape_cpT, offset=(None, offset_p)
            )
            jac_pc = update_csc_array_indices(
                jac_pc, (shape_p_1, shape_c_full), shape_cpT, offset=(offset_p, None)
            )
            jac_pT = update_csc_array_indices(
                jac_pT, (shape_p_1, shape_p_1), shape_cpT, offset=(offset_p, offset_T)
            )
            jac_TT = update_csc_array_indices(jac_TT, shape_p_1, shape_cpT, offset=offset_T)

            base_jac = (
                jac_cc
                + factor_p * jac_pp
                + jac_cp
                + factor_p * jac_pc
                + factor_p * jac_pT
                + factor_T * jac_TT
            )

            if not self.is_isothermal:
                if jac_TP is not None:
                    jac_TP = update_csc_array_indices(
                        jac_TP, shape_p_1, shape_cpT, offset=(offset_T, offset_p)
                    )
                    base_jac = base_jac + factor_T * jac_TP

                if jac_Tmem is not None:
                    jac_Tmem_cpT = update_csc_array_indices(
                        jac_Tmem, shape_p_1, shape_cpT, offset=offset_T
                    )
                    base_jac = base_jac + factor_T * jac_Tmem_cpT

                offset_c_ret_cpT = (0, 1, 0)
                offset_T_ret_cpT = (0, 1, nc + 1)

                jac_cT_mapped = update_csc_array_indices(
                    -jac_cT_react,
                    (shape_c_ret_loc, shape_T_ret_loc),
                    shape_cpT,
                    offset=(offset_c_ret_cpT, offset_T_ret_cpT),
                )
                jac_Tc_mapped = update_csc_array_indices(
                    jac_Tc_react,
                    (shape_T_ret_loc, shape_c_ret_loc),
                    shape_cpT,
                    offset=(offset_T_ret_cpT, offset_c_ret_cpT),
                )

                base_jac = base_jac + jac_cT_mapped + factor_T * jac_Tc_mapped

            self._jac = base_jac

        else:
            c_tot = self.correlation.molar_density(y, T, p)

        # ===== Assemble residual =====
        g_c = self.g_c_in + g_conv + g_diff + g_mem
        if c_old is not None:
            g_c += (
                self.jac_c_accum @ ((c - c_old).reshape((-1, 1)) / dt)
            ).reshape(c.shape)
        g_c[:, 1:, :] -= g_react

        g_p = c_tot - c_sum

        if self.is_isothermal:
            T_init = np.empty_like(T)
            T_init[:, 0] = self.T_perm_in
            T_init[:, 1] = self.T_ret_in
            g_T = T - T_init
        else:
            g_T = g_T_conv + g_T_cond + g_Tmem
            if T_old is not None:
                g_T += (
                    self.jac_T_accum @ ((T - T_old).reshape((-1, 1)) / dt)
                ).reshape(T.shape)
            g_T[:, 1:] += g_T_react

        g[..., :-2] = g_c
        g[..., -2] = factor_p * g_p
        g[..., -1] = factor_T * g_T

        return g, self._jac

    # =====================================================================
    # Convection residual
    # =====================================================================

    def _construct_g_conv(self, c, compute_jac=False):
        """Upwind convection residual for both sides."""
        nz, nc = self.num_z, self.num_c
        shape_c_side = (nz, 1, nc)
        shape_c_full = (nz, 2, nc)
        shape_p_side = (nz, 1)
        shape_p_full = (nz, 2)

        g = np.zeros(shape_c_full)
        g_vect = g.ravel()

        c_perm = c[:, :1, :]
        c_ret = c[:, 1:, :]

        u_perm = self.u_perm_ax.reshape(-1, 1, 1)
        u_ret_local = self.u_ret_ax.copy()

        # Retentate BCs
        bc_ret_ax = get_axial_bcs_for_flow(
            is_counter_current=self.is_counter_current,
            u_ax=u_ret_local.reshape(-1, 1),
            bc_inlet=BC_NONE,
            inflow_value=self.inflow_conc_ret_backflow,
        )
        u_ret = u_ret_local.reshape(-1, 1, 1)

        # Permeate convection
        bc_perm_ax = (BC_NONE, BC_NEUMANN_HOM)
        c_perm_ax, _ = interp_cntr_to_stagg_tvd(
            c_perm, self.z_f, self.z_c, bc=bc_perm_ax, v=u_perm, tvd_limiter=upwind, axis=0
        )
        flux_perm_ax = u_perm * c_perm_ax
        g_perm = (self.div_c_perm_ax @ flux_perm_ax.ravel()).reshape(shape_c_side)

        # Retentate convection
        c_ret_ax, _ = interp_cntr_to_stagg_tvd(
            c_ret, self.z_f, self.z_c, bc=bc_ret_ax, v=u_ret, tvd_limiter=upwind, axis=0
        )
        flux_ret_ax = u_ret * c_ret_ax
        g_ret = (self.div_c_ret_ax @ flux_ret_ax.ravel()).reshape(shape_c_side)

        g[:, :1, :] = g_perm
        g[:, 1:, :] = g_ret

        if compute_jac:
            # Permeate
            conv_mat_perm, _ = construct_convflux_upwind(
                shape_c_side, self.z_f, self.z_c, bc=bc_perm_ax, v=u_perm, axis=0
            )
            jac_perm = self.div_c_perm_ax @ conv_mat_perm

            # Darcy coupling: d(c*u)/dp  where u = -K dp/dz
            ck_perm = construct_coefficient_matrix(
                c_perm_ax, shape=(c_perm_ax.shape, c_perm_ax.shape[:-1] + (1,))
            ) @ self.k_matrix_perm_ax
            jac_darcy = self.div_c_perm_ax @ ((-ck_perm) @ self.grad_p_perm_ax)

            # Retentate
            conv_mat_ret, _ = construct_convflux_upwind(
                shape_c_side, self.z_f, self.z_c, bc=bc_ret_ax, v=u_ret, axis=0
            )
            jac_ret = self.div_c_ret_ax @ conv_mat_ret

            ck_ret = construct_coefficient_matrix(
                c_ret_ax, shape=(c_ret_ax.shape, c_ret_ax.shape[:-1] + (1,))
            ) @ self.k_matrix_ret_ax
            jac_darcy_ret = self.div_c_ret_ax @ ((-ck_ret) @ self.grad_p_ret_ax)

            # Map to monolithic (rows + cols from sub-block to full)
            jac_perm_full = update_csc_array_indices(jac_perm, shape_c_side, shape_c_full)
            jac_ret_full = update_csc_array_indices(
                jac_ret, shape_c_side, shape_c_full, offset=(0, 1, 0)
            )
            jac_conv = jac_perm_full + jac_ret_full

            # Darcy → map to (c_full, p_full)
            jac_darcy_perm_full = update_csc_array_indices(
                jac_darcy, (shape_c_side, shape_p_side), (shape_c_full, shape_p_full)
            )
            jac_darcy_ret_full = update_csc_array_indices(
                jac_darcy_ret, (shape_c_side, shape_p_side), (shape_c_full, shape_p_full),
                offset=((0, 1, 0), (0, 1))
            )
            jac_darcy_full = jac_darcy_perm_full + jac_darcy_ret_full

            return g, jac_conv, jac_darcy_full
        return g, None, None

    # =====================================================================
    # Axial diffusion
    # =====================================================================

    def _construct_g_diff(self, c, T, p, compute_jac=False):
        """Axial diffusion residual for both sides."""
        nz, nc = self.num_z, self.num_c
        shape_c_side = (nz, 1, nc)
        shape_c_full = (nz, 2, nc)

        c_perm = c[:, :1, :]
        c_ret = c[:, 1:, :]
        T_perm = T[:, :1]
        T_ret = T[:, 1:]
        p_perm = p[:, :1]
        p_ret = p[:, 1:]

        # Mole fractions
        c_tot_perm = np.maximum(np.abs(np.sum(c_perm, axis=-1, keepdims=True)), 1e-10)
        y_perm = c_perm / c_tot_perm
        c_tot_ret = np.maximum(np.abs(np.sum(c_ret, axis=-1, keepdims=True)), 1e-10)
        y_ret = c_ret / c_tot_ret

        # Diffusivities
        D_perm = self.correlation.diffusion(y_perm, T_perm, p_perm)
        D_ret = self.correlation.diffusion(y_ret, T_ret, p_ret)
         # diffusion() returns full matrix (nz, 1, nc, nc) — extract diagonal species diffusivities
        if D_perm.ndim == 4:
            D_perm = np.diagonal(D_perm, axis1=-2, axis2=-1)  # (nz, 1, nc)
        if D_ret.ndim == 4:
            D_ret = np.diagonal(D_ret, axis1=-2, axis2=-1)    # (nz, 1, nc)
        
        # Perm diffusion
        D_perm_ax = interp_cntr_to_stagg(D_perm, self.z_f, self.z_c, axis=0)
        D_perm_mat = construct_coefficient_matrix(D_perm_ax, shape_c_side, axis=0)
        jac_perm = self.div_c_perm_ax @ (-D_perm_mat) @ self.grad_c_perm_ax
        g_bc_perm = self.div_c_perm_ax @ (-D_perm_mat @ self.grad_bc_c_perm_ax)

        # Ret diffusion
        D_ret_ax = interp_cntr_to_stagg(D_ret, self.z_f, self.z_c, axis=0)
        D_ret_mat = construct_coefficient_matrix(D_ret_ax, shape_c_side, axis=0)
        jac_ret = self.div_c_ret_ax @ (-D_ret_mat) @ self.grad_c_ret_ax
        g_bc_ret = self.div_c_ret_ax @ (-D_ret_mat @ self.grad_bc_c_ret_ax)

        g = np.zeros(shape_c_full)
        g[:, :1, :] = (g_bc_perm + jac_perm @ c_perm.reshape((-1, 1))).reshape(shape_c_side)
        g[:, 1:, :] = (g_bc_ret + jac_ret @ c_ret.reshape((-1, 1))).reshape(shape_c_side)

        if compute_jac:
            jac_perm_full = update_csc_array_indices(jac_perm, shape_c_side, shape_c_full)
            jac_ret_full = update_csc_array_indices(
                jac_ret, shape_c_side, shape_c_full, offset=(0, 1, 0)
            )
            return g, jac_perm_full + jac_ret_full
        return g, None

    # =====================================================================
    # Temperature convection
    # =====================================================================

    def _construct_g_T_conv(self, T, compute_jac=False):
        """Temperature convection residual."""
        nz = self.num_z
        shape_p_side = (nz, 1)
        shape_p_full = (nz, 2)

        T_perm = T[:, :1]
        T_ret = T[:, 1:]

        bc_perm_dirichlet = make_dirichlet_bc(self.T_perm_in)
        bc_ret_dirichlet = make_dirichlet_bc(self.T_ret_in)

        u_perm = self.u_perm_ax.reshape(-1, 1)
        u_ret_local = self.u_ret_ax.copy().reshape(-1, 1)

        bc_ret_ax = get_axial_bcs_for_flow(
            is_counter_current=self.is_counter_current,
            u_ax=u_ret_local,
            bc_inlet=bc_ret_dirichlet,
            inflow_value=self.T_ret_in,
        )

        g = np.zeros(shape_p_full)

        # Permeate
        T_perm_ax, _ = interp_cntr_to_stagg_tvd(
            T_perm, self.z_f, self.z_c,
            bc=(bc_perm_dirichlet, BC_NEUMANN_HOM), v=u_perm, tvd_limiter=upwind, axis=0,
        )
        g[:, :1] = (self.div_p_perm_ax @ (u_perm * T_perm_ax).ravel()).reshape(shape_p_side)
        g[:, :1] -= (T_perm * self.div_u_perm.reshape(-1, 1))

        # Retentate
        T_ret_ax, _ = interp_cntr_to_stagg_tvd(
            T_ret, self.z_f, self.z_c,
            bc=bc_ret_ax, v=u_ret_local, tvd_limiter=upwind, axis=0,
        )
        g[:, 1:] = (self.div_p_ret_ax @ (u_ret_local * T_ret_ax).ravel()).reshape(shape_p_side)
        g[:, 1:] -= (T_ret * self.div_u_ret.reshape(-1, 1))

        if compute_jac:
            conv_perm, _ = construct_convflux_upwind(
                shape_p_side, self.z_f, self.z_c,
                bc=(bc_perm_dirichlet, BC_NEUMANN_HOM), v=u_perm, axis=0,
            )
            jac_perm = self.div_p_perm_ax @ conv_perm
            jac_perm -= construct_coefficient_matrix(self.div_u_perm.reshape(shape_p_side))

            # Darcy coupling for T convection
            Tk_perm = construct_coefficient_matrix(T_perm_ax) @ self.k_matrix_perm_ax
            jac_darcy = self.div_p_perm_ax @ ((-Tk_perm) @ self.grad_p_perm_ax)
            jac_darcy_div = self.div_p_perm_ax @ (self.k_matrix_perm_ax @ self.grad_p_perm_ax)

            conv_ret, _ = construct_convflux_upwind(
                shape_p_side, self.z_f, self.z_c,
                bc=bc_ret_ax, v=u_ret_local, axis=0,
            )
            jac_ret = self.div_p_ret_ax @ conv_ret
            jac_ret -= construct_coefficient_matrix(self.div_u_ret.reshape(shape_p_side))

            Tk_ret = construct_coefficient_matrix(T_ret_ax) @ self.k_matrix_ret_ax
            jac_darcy_ret = self.div_p_ret_ax @ ((-Tk_ret) @ self.grad_p_ret_ax)
            jac_darcy_div_ret = self.div_p_ret_ax @ (self.k_matrix_ret_ax @ self.grad_p_ret_ax)

            # Map to full (rows + cols from sub-block to full)
            jac_perm_full = update_csc_array_indices(jac_perm, shape_p_side, shape_p_full)
            jac_ret_full = update_csc_array_indices(
                jac_ret, shape_p_side, shape_p_full, offset=(0, 1)
            )
            jac_T = jac_perm_full + jac_ret_full

            # Darcy → T-p coupling
            jac_darcy_all = update_csc_array_indices(jac_darcy, (shape_p_side, shape_p_side), (shape_p_full, shape_p_full))
            jac_darcy_all += update_csc_array_indices(
                jac_darcy_ret, (shape_p_side, shape_p_side), (shape_p_full, shape_p_full),
                offset=((0, 1), (0, 1))
            )

            jac_darcy_div_full = update_csc_array_indices(jac_darcy_div, (shape_p_side, shape_p_side), (shape_p_full, shape_p_full))
            jac_darcy_div_full += update_csc_array_indices(
                jac_darcy_div_ret, (shape_p_side, shape_p_side), (shape_p_full, shape_p_full),
                offset=((0, 1), (0, 1))
            )
            jac_darcy_all += construct_coefficient_matrix(T) @ jac_darcy_div_full

            return g, jac_T, jac_darcy_all
        return g, None, None

    # =====================================================================
    # Temperature conduction
    # =====================================================================

    def _construct_g_T_cond(self, T, c, jac_only=True):
        """Axial conduction residual (no membrane — that's in _membrane_heat_flux)."""
        nz = self.num_z
        shape_p_side = (nz, 1)
        shape_p_full = (nz, 2)

        T_perm = T[:, :1]
        T_ret = T[:, 1:]

        c_perm = c[:, :1, :]
        c_ret = c[:, 1:, :]

        # Thermal conductivity
        c_tot_p = np.maximum(np.abs(np.sum(c_perm, axis=-1, keepdims=True)), 1e-10)
        y_perm = c_perm / c_tot_p
        lmbda_perm = self.correlation.thermal_conductivity(y_perm, T_perm)
        lmbda_perm_ax = interp_cntr_to_stagg(lmbda_perm, self.z_f, self.z_c, axis=0)
        lmbda_perm_mat = construct_coefficient_matrix(lmbda_perm_ax)

        c_tot_r = np.maximum(np.abs(np.sum(c_ret, axis=-1, keepdims=True)), 1e-10)
        y_ret = c_ret / c_tot_r
        lmbda_ret = self.correlation.thermal_conductivity(y_ret, T_ret)
        lmbda_ret_ax = interp_cntr_to_stagg(lmbda_ret, self.z_f, self.z_c, axis=0)
        lmbda_ret_mat = construct_coefficient_matrix(lmbda_ret_ax)

        # Conduction Jacobians (side-local)
        jac_perm = self.div_p_perm_ax @ (-lmbda_perm_mat @ self.grad_T_perm_ax)
        g_bc_perm = self.div_p_perm_ax @ (-lmbda_perm_mat @ self.grad_bc_T_perm_ax)

        jac_ret = self.div_p_ret_ax @ (-lmbda_ret_mat @ self.grad_T_ret_ax)
        g_bc_ret = self.div_p_ret_ax @ (-lmbda_ret_mat @ self.grad_bc_T_ret_ax)

        # cp⁻¹ scaling
        cp_perm = self.correlation.specific_heat(c_perm, T_perm)
        cp_ret = self.correlation.specific_heat(c_ret, T_ret)
        cp_inv_perm = 1.0 / np.maximum(cp_perm, 1.0)
        cp_inv_ret = 1.0 / np.maximum(cp_ret, 1.0)

        cp_inv_perm_mat = construct_coefficient_matrix(cp_inv_perm)
        cp_inv_ret_mat = construct_coefficient_matrix(cp_inv_ret)

        g_bc_perm_dense = np.asarray(g_bc_perm.todense()).ravel()
        g_bc_ret_dense = np.asarray(g_bc_ret.todense()).ravel()
        g_perm = (cp_inv_perm_mat @ (jac_perm @ T_perm.ravel() + g_bc_perm_dense)).reshape(shape_p_side)
        g_ret = (cp_inv_ret_mat @ (jac_ret @ T_ret.ravel() + g_bc_ret_dense)).reshape(shape_p_side)

        g = np.zeros(shape_p_full)
        g[:, :1] = g_perm
        g[:, 1:] = g_ret

        jac_perm_scaled = cp_inv_perm_mat @ jac_perm
        jac_ret_scaled = cp_inv_ret_mat @ jac_ret

        # Map to full (rows + cols from sub-block to full)
        jac_full = update_csc_array_indices(jac_perm_scaled, shape_p_side, shape_p_full)
        jac_full += update_csc_array_indices(
            jac_ret_scaled, shape_p_side, shape_p_full, offset=(0, 1)
        )

        if jac_only:
            return g, jac_full, cp_inv_perm, cp_inv_ret
        else:
            return g, jac_full, None, cp_inv_perm, cp_inv_ret

    # =====================================================================
    # Newton solver
    # =====================================================================

    def _solve_cpT(self, c_old, T_old, dt, verbose=0, use_line_search=False):
        """Newton-solve the coupled cpT system for one pseudo-time step of size dt; return a SolveResult."""
        cpT = self.cpT
        cpT_vec = cpT.ravel()
        factor_norm = self.factor_norm
        ord = self.ord_norm
        self.last_solver_failure_message = None

        def compute_norm(g):
            """Return the scaled ord-norm of a residual vector."""
            return np.linalg.norm(g.ravel(), ord=ord) * factor_norm

        def eval_residual(x_vec):
            """Write the trial state into cpT, refresh velocity fields, and return the residual."""
            cpT_vec[:] = x_vec
            self._update_velocity_fields()
            g, _ = self._construct_g_cpT(c_old, T_old, dt, compute_jac=False)
            return g

        g_norm_init = None
        g_norm = np.inf
        success = True

        for j in range(self.max_newton_iterations):
            try:
                g, jac = self._construct_g_cpT(c_old, T_old, dt, compute_jac=True)
            except RecoverableNumericalError as exc:
                self.last_solver_failure_message = f"residual assembly failed: {exc}"
                logger.debug(self.last_solver_failure_message)
                return SolveResult(False, j + 1, np.inf, g_norm_init or np.inf)

            g_norm = compute_norm(g)
            logger.debug("Newton iter %d: g_norm=%.4e", j, g_norm)

            if j == 0:
                g_norm_init = g_norm
            if not np.isfinite(g_norm) or (g_norm_init and g_norm > 10 * g_norm_init):
                return SolveResult(False, j + 1, g_norm, g_norm_init or np.inf)
            if g_norm <= max(self.rtol * g_norm_init, self.atol):
                return SolveResult(success, j + 1, g_norm, g_norm_init)

            try:
                dcpT = -sla.spsolve(jac, g.ravel())
            except RuntimeError:
                return SolveResult(False, j + 1, np.inf, g_norm_init or np.inf)
            if not np.all(np.isfinite(dcpT)):
                return SolveResult(False, j + 1, np.inf, g_norm_init or np.inf)
            self.cnt_num_solves_cpT += 1

            if use_line_search:
                try:
                    x_new, g_new_norm, alpha, ls_ok = armijo_line_search(
                        x=cpT_vec.copy(), dx=dcpT, g_norm=g_norm,
                        residual_fn=lambda x: eval_residual(x).ravel(),
                        norm_fn=compute_norm,
                        armijo_coeff=ARMIJO_COEFF,
                        min_alpha=MIN_LINE_SEARCH_ALPHA,
                    )
                except RecoverableNumericalError:
                    return SolveResult(False, j + 1, np.inf, g_norm_init or np.inf)
                cpT_vec[:] = x_new
                if not ls_ok:
                    success = False
            else:
                cpT_vec[:] += dcpT
            self._update_velocity_fields()

        try:
            g, _ = self._construct_g_cpT(c_old, T_old, dt, compute_jac=False)
        except RecoverableNumericalError:
            return SolveResult(False, self.max_newton_iterations, np.inf, g_norm_init or np.inf)

        g_norm = compute_norm(g)
        return SolveResult(
            success and g_norm <= max(self.rtol * (g_norm_init or np.inf), self.atol),
            self.max_newton_iterations, g_norm, g_norm_init or np.inf,
        )

    # =====================================================================
    # Adaptive pseudo-transient solver
    # =====================================================================

    def _compute_dt_cfl(self, cfl=CFL_INIT):
        """Return a CFL-limited time step (s) from the smallest axial cell and the peak axial velocity."""
        dz_min = np.min(self.z_f[1:] - self.z_f[:-1])
        u_max = max(np.max(np.abs(self.u_perm_ax)), np.max(np.abs(self.u_ret_ax)))
        return cfl * dz_min / max(u_max, 1e-30)

    def _compute_steady_state_norm(self):
        """Return the scaled norm of the steady-state residual (transient term suppressed via a huge dt)."""
        c_curr = self.cpT[..., :-2].copy()
        T_curr = self.cpT[..., -1].copy()
        g_ss, _ = self._construct_g_cpT(c_curr, T_curr, STEADY_STATE_DT, compute_jac=False)
        return np.linalg.norm(g_ss.ravel(), ord=self.ord_norm) * self.factor_norm

    def _restore_state(self, cpT_saved):
        """Restore cpT from a backup and rebuild the Darcy matrices and velocity fields."""
        self.cpT = cpT_saved.copy()
        self._construct_darcy_matrices()
        self._update_velocity_fields()

    def solve(
        self,
        num_timesteps=None, dt=None, dt_init=None,
        dt_min=None, dt_max=None,
        dt_increase_factor=None, dt_decrease_factor=None,
        adaptive_dt_threshold=None,
        use_adaptive_dt=True,
        steady_state_atol=None, steady_state_rtol=None,
        verbose=0, return_status=False,
        **kwargs,
    ):
        """Adaptive pseudo-transient steady-state solver (mirrors 2D)."""
        if kwargs:
            raise TypeError(f"solve() got unexpected keyword(s): {', '.join(sorted(kwargs))}")

        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        if dt is not None and dt_init is not None:
            raise ValueError("Provide either dt or dt_init, not both.")
        if dt is not None:
            dt_init = dt
            use_adaptive_dt = False
        elif dt_init is None:
            dt_init = self.dt_init

        self.cnt_num_solves_cpT = 0
        dt = dt_init
        dt_min = self.dt_min if dt_min is None else dt_min
        dt_max = self.dt_max if dt_max is None else dt_max
        dt_increase = self.dt_increase_factor if dt_increase_factor is None else dt_increase_factor
        dt_decrease = self.dt_decrease_factor if dt_decrease_factor is None else dt_decrease_factor
        threshold = self.adaptive_dt_threshold if adaptive_dt_threshold is None else adaptive_dt_threshold
        ss_atol = self.steady_state_atol if steady_state_atol is None else steady_state_atol
        ss_rtol = self.steady_state_rtol if steady_state_rtol is None else steady_state_rtol
        self.last_solver_failure_message = None

        g_ss_prev = None
        g_ss_0 = None
        g_ss = None
        g_ss_best = None
        cpT_best = None
        is_converged = False
        accepted = 0
        n_increasing = 0
        n_plateau = 0

        def finalize(conv, steps):
            """Build the SteadyStateSolveStatus, store it on self, and return it (or its converged flag)."""
            status = SteadyStateSolveStatus(
                converged=conv, num_steps_attempted=steps, num_steps_accepted=accepted,
                final_dt=dt, steady_state_norm=g_ss,
                initial_steady_state_norm=g_ss_0,
                best_steady_state_norm=g_ss_best,
                baseline_steady_state_norm=None,
                last_failure_message=self.last_solver_failure_message,
            )
            self.last_solve_status = status
            return status if return_status else status.converged

        if verbose >= 1:
            print(f"Starting 1D solve: dt_init={dt_init:.2e}, dt_range=[{dt_min:.2e}, {dt_max:.2e}]")

        def check_ss(norm):
            """Return True when norm satisfies the configured absolute and relative steady-state tolerances."""
            abs_ok = ss_atol > 0 and norm <= ss_atol
            rel_ok = ss_rtol is not None and g_ss_0 is not None and norm / g_ss_0 <= ss_rtol
            criteria = []
            if ss_atol > 0:
                criteria.append(abs_ok)
            if ss_rtol is not None:
                criteria.append(rel_ok)
            return bool(criteria) and all(criteria)

        for i in range(num_timesteps):
            T_old = self.cpT[..., -1].copy()
            c_old = self.cpT[..., :-2].copy()
            cpT_backup = self.cpT.copy()

            result = self._solve_cpT(c_old, T_old, dt)
            if not np.isfinite(result.g_norm):
                self._restore_state(cpT_backup)
                dt = max(dt * dt_decrease, dt_min)
                n_plateau = 0
                if verbose >= 2:
                    print(f"  Step {i}: REJECTED (NaN), dt->{dt:.2e}")
                continue

            try:
                g_ss = self._compute_steady_state_norm()
            except RecoverableNumericalError:
                self._restore_state(cpT_backup)
                dt = max(dt * dt_decrease, dt_min)
                n_plateau = 0
                continue

            if g_ss_prev is not None and g_ss > 10 * g_ss_prev:
                self._restore_state(cpT_backup)
                dt = max(dt * dt_decrease, dt_min)
                n_increasing = 0
                n_plateau = 0
                if verbose >= 2:
                    print(f"  Step {i}: REJECTED (g_ss grew), dt->{dt:.2e}")
                continue

            accepted += 1
            if g_ss_best is None or g_ss < g_ss_best:
                g_ss_best = g_ss
                cpT_best = self.cpT.copy()
            if g_ss_0 is None:
                g_ss_0 = max(g_ss, 1e-30)

            if use_adaptive_dt and g_ss_prev is not None:
                red = g_ss / g_ss_prev
                if red > 1.05:
                    dt = max(dt * dt_decrease, dt_min)
                    n_increasing += 1
                elif red > 1.0:
                    n_increasing += 1
                elif red < threshold:
                    dt = min(dt * dt_increase**2, dt_max)
                    n_increasing = 0
                else:
                    dt = min(dt * dt_increase, dt_max)
                    n_increasing = 0

                if abs(red - 1.0) < STEADY_STATE_PLATEAU_TOL:
                    n_plateau += 1
                else:
                    n_plateau = 0

                if n_increasing >= 10 and cpT_best is not None:
                    self._restore_state(cpT_best)
                    g_ss = g_ss_best
                    g_ss_prev = g_ss_best
                    n_increasing = 0
                    n_plateau = 0

                if verbose >= 2:
                    print(f"  Step {i}: red={red:.2f}, dt={dt:.2e}, ||g_ss||={g_ss:.2e}")
            elif verbose >= 2:
                print(f"  Step {i}: dt={dt:.2e}, ||g_ss||={g_ss:.2e}")

            if check_ss(g_ss):
                is_converged = True
                if verbose >= 1:
                    print(f"  Step {i}: CONVERGED, ||g_ss||={g_ss:.2e}")
                break

            g_ss_prev = g_ss

        if cpT_best is not None and (g_ss is None or g_ss_best < g_ss):
            self._restore_state(cpT_best)
            g_ss = g_ss_best
            if verbose >= 1:
                print(f"  Restored best: ||g_ss||={g_ss_best:.2e}")

        if verbose >= 1 and not is_converged:
            msg = f"  Not converged after {num_timesteps} steps"
            if g_ss is not None:
                msg += f", ||g_ss||={g_ss:.2e}"
            print(msg)

        return finalize(is_converged, i + 1 if num_timesteps > 0 else 0)

    # =====================================================================
    # Post-processing
    # =====================================================================

    def compute_flows(self):
        """Compute axial molar flows and membrane transfer (same interface as 2D).

        Returns:
            (flows_ret_ax, flows_ret_mem, flows_perm_ax, flows_perm_mem)
            - flows_ret_ax: (num_z+1, num_c) molar flows at retentate axial faces
            - flows_ret_mem: (num_c,) total membrane species transfer (retentate side)
            - flows_perm_ax: (num_z+1, num_c) molar flows at permeate axial faces
            - flows_perm_mem: (num_c,) total membrane species transfer (permeate side)

        The membrane flux uses the same Sherwood CP-corrected formulation as the
        solver (_membrane_flux), so flows_ret_mem is consistent with what was solved.
        """
        c = self.cpT[..., :-2]
        T = self.cpT[..., -1]
        nc = self.num_c

        c_ret = c[:, 1:, :]   # (nz, 1, nc)
        c_perm = c[:, :1, :]  # (nz, 1, nc)

        # Axial face concentrations via upwind
        u_ret = self.u_ret_ax.reshape(-1, 1, 1)
        u_perm = self.u_perm_ax.reshape(-1, 1, 1)

        bc_ret_ax = get_axial_bcs_for_flow(
            is_counter_current=self.is_counter_current,
            u_ax=self.u_ret_ax.reshape(-1, 1),
            bc_inlet=BC_NONE,
            inflow_value=self.inflow_conc_ret_backflow,
        )
        bc_perm_ax = (BC_NONE, BC_NEUMANN_HOM)

        c_ret_ax, _ = interp_cntr_to_stagg_tvd(
            c_ret, self.z_f, self.z_c, bc=bc_ret_ax, v=u_ret, tvd_limiter=upwind, axis=0
        )
        c_perm_ax, _ = interp_cntr_to_stagg_tvd(
            c_perm, self.z_f, self.z_c, bc=bc_perm_ax, v=u_perm, tvd_limiter=upwind, axis=0
        )

        # Molar flow = c_face * u_face * A
        flows_ret_ax = c_ret_ax.reshape(-1, nc) * self.u_ret_ax.reshape(-1, 1) * self.A_ret
        flows_perm_ax = c_perm_ax.reshape(-1, nc) * self.u_perm_ax.reshape(-1, 1) * self.A_perm

        # Overwrite inlet face with known inlet fluxes
        flows_perm_ax[0, :] = self.flux_perm_in.ravel() * self.A_perm
        if self.is_counter_current:
            flows_ret_ax[-1, :] = -self.flux_ret_in.ravel() * self.A_ret
        else:
            flows_ret_ax[0, :] = self.flux_ret_in.ravel() * self.A_ret

        # ---- Membrane transfer with Sherwood CP correction ----
        # Uses the same c_sm formula as _membrane_flux so post-processing is
        # consistent with what the solver actually converged to.
        T_ret = T[:, 1]               # (nz,)
        T_perm = T[:, 0]              # (nz,)
        c_ret_cell = c[:, 1, :]       # (nz, nc) bulk retentate concentration
        c_perm_cell = c[:, 0, :]      # (nz, nc) bulk permeate concentration

        Rg = self.Rg
        perm = self.P0 * np.exp(-self.EA / (Rg * T_ret[:, None]))   # (nz, nc)
        P_perm_i = Rg * T_perm[:, None] * c_perm_cell               # (nz, nc)
        kcp = self.compute_kcp_sh(c_ret_cell, T_ret)                 # (nz, nc)
        denom = kcp + perm * Rg * T_ret[:, None]
        c_sm = (kcp * c_ret_cell + perm * P_perm_i) / np.maximum(denom, 1e-16)
        J = perm * (Rg * T_ret[:, None] * c_sm - P_perm_i)          # (nz, nc)

        A_mem_per_cell = 2.0 * np.pi * self.r_min * self.dz          # (nz,)
        flows_ret_mem = np.sum(J * A_mem_per_cell[:, None], axis=0)
        flows_perm_mem = self.factor_geom_mem * flows_ret_mem

        # Store diagnostics (consistent with _membrane_flux)
        self.kcp_last = kcp
        self.c_sm_last = c_sm
        self.J_last = J
        self.y_sm_last = c_sm / np.maximum(np.sum(c_sm, axis=1, keepdims=True), 1e-16)
        if "NH3" in self.species:
            idx_nh3 = self.species.index("NH3")
            self.y_nh3_sm_last = c_sm[:, idx_nh3] / np.maximum(np.sum(c_sm, axis=1), 1e-16)
            self.y_nh3_bulk_last = c_ret_cell[:, idx_nh3] / np.maximum(np.sum(c_ret_cell, axis=1), 1e-16)

        return flows_ret_ax, flows_ret_mem, flows_perm_ax, flows_perm_mem

    def info(self):
        """Print reactor info (mirrors 2D)."""
        vol_ret = self.A_ret * (self.z_f[-1] - self.z_f[0])
        vol_perm = self.A_perm * (self.z_f[-1] - self.z_f[0])
        flow_vol_ret = self.F_ret_in * self.Rg * self.T_ret_in / self.p_ret_out
        flow_vol_perm = self.F_perm_in * self.Rg * self.T_perm_in / self.p_perm_out
        logger.info("Residence time retentate: %.4f s", vol_ret / flow_vol_ret)
        logger.info("Residence time permeate: %.4f s", vol_perm / flow_vol_perm)



