import importlib
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# test: diffusion coefficient is 0
# test: simplify ergun

import numpy as np
from numpy.typing import NDArray
import scipy.sparse.linalg as sla
from pymrm import (
    NumJac,
    compute_boundary_values,
    construct_boundary_value_matrices,
    construct_coefficient_matrix,
    construct_convflux_upwind,
    construct_div,
    construct_grad,
    construct_interface_matrices,
    interp_cntr_to_stagg,
    interp_cntr_to_stagg_tvd,
    interp_stagg_to_cntr,
    update_csc_array_indices,
    upwind,
)

import defaults  # Import the defaults module (still needed for reload)
from ammonia_synthesis_kinetics import AmmoniaSynthesisKinetics
from config import ReactorConfig
from gas_mixture_correlations import GasMixtureCorrelations
from mesh import ReactorMesh
from physics import (
    BC_DIRICHLET,
    BC_DIRICHLET_HOM,
    BC_NEUMANN,
    BC_NEUMANN_HOM,
    BC_NONE,
    compute_inlet_flux_permeate,
    compute_inlet_flux_retentate,
    compute_membrane_permeabilities,
    compute_packed_bed_permeability,
    compute_permeate_permeability,
    get_axial_bcs_for_flow,
    make_dirichlet_bc,
)
from solvers import ContinuationConfig, NewtonConfig, armijo_line_search

# Module-level logger
logger = logging.getLogger(__name__)

# =============================================================================
# Solver result types
# =============================================================================


@dataclass
class SegregatedSolveResult:
    """Result from segregated concentration-pressure Newton solve."""

    converged: bool
    num_iterations: int
    g_norm: float  # Concentration residual norm
    g_p_norm: float  # Pressure residual norm
    g_norm_init: float  # Initial concentration residual
    g_p_norm_init: float  # Initial pressure residual

    @property
    def convergence_factor(self) -> float:
        """Concentration convergence factor."""
        return self.g_norm / self.g_norm_init if self.g_norm_init > 0 else 0.0

    @property
    def convergence_factor_p(self) -> float:
        """Pressure convergence factor."""
        return self.g_p_norm / self.g_p_norm_init if self.g_p_norm_init > 0 else 0.0

    def as_tuple(self) -> Tuple[int, float, float, bool, float, float]:
        """Return legacy tuple format for backwards compatibility."""
        return (
            self.num_iterations,
            self.g_norm,
            self.g_p_norm,
            self.converged,
            self.convergence_factor,
            self.convergence_factor_p,
        )


# =============================================================================
# Numerical solver constants
# =============================================================================

# Default solver configurations (can be overridden per-instance)
DEFAULT_NEWTON_CONFIG = NewtonConfig()
DEFAULT_CONTINUATION_CONFIG = ContinuationConfig()

# Physical timestep constraints
CFL_INIT = 0.1  # Initial CFL number for timestep selection
EPS_CHEM_TIMESTEP = 1e-8  # Small epsilon for chemical timestep calculation
MAX_DT_PER_STEP = 500.0  # Maximum temperature change per solve step [K]

# Line search parameters (from NewtonConfig defaults)
ARMIJO_COEFF = DEFAULT_NEWTON_CONFIG.armijo_coeff
MIN_LINE_SEARCH_ALPHA = DEFAULT_NEWTON_CONFIG.min_line_search_alpha


class MembraneReactor:
    """2D axisymmetric membrane reactor model.

    Simulates coupled mass, momentum, and (optionally) energy transport with
    catalytic reaction and selective permeation between retentate and permeate
    regions on non-uniform grids. Supports continuation and pseudo‑transient
    strategies for robust steady-state convergence.
    """

    # =========================================================================
    # Standard boundary condition templates (imported from physics.py)
    # =========================================================================
    # Kept as class attributes for backward compatibility
    BC_NONE = BC_NONE
    BC_DIRICHLET_HOM = BC_DIRICHLET_HOM
    BC_NEUMANN_HOM = BC_NEUMANN_HOM
    BC_DIRICHLET = BC_DIRICHLET
    BC_NEUMANN = BC_NEUMANN

    def __init__(
        self,
        config_file: Optional[str] = None,
        c: Optional[NDArray[np.float64]] = None,
        p: Optional[NDArray[np.float64]] = None,
        T: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> None:
        """Construct reactor, load defaults, apply overrides, allocate fields.

        Args:
            config_file: Optional JSON file overriding defaults.
            c: Initial concentration field (num_z, num_r, num_c).
            p: Initial pressure field (num_z, num_r).
            T: Initial temperature field (num_z, num_r).
            **kwargs: Explicit overrides of default parameters.

        Side Effects:
            Creates spatial discretization, initializes kinetics & Jacobian
            matrices, performs a non-reactive pressure/velocity initialization.
        """
        self._init_config(config_file, kwargs)
        self._init_derived_parameters()
        self._init_fields(c, p, T)
        _, T_ret = self._split_perm_and_ret(self.cpT[..., -1])
        _, p_ret = self._split_perm_and_ret(self.cpT[..., -2])

        self.kinetics = AmmoniaSynthesisKinetics(
            self.species, T=T_ret, p=p_ret, rho_b=self.rho_b, rho_c=self.rho_c
        )
        self._init_jac()
        self.factor_norm_c = (self.cpT[..., :-2].size) ** (-1.0 / self.ord_norm)
        self.factor_norm_p = (self.cpT[..., -2].size) ** (-1.0 / self.ord_norm)

        dt_cfl = self._compute_dt_cfl(cfl=CFL_INIT)
        self.solve(dt=dt_cfl, use_adaptive_react=False, num_timesteps=2)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to config and mesh for backwards compatibility.

        Args:
            name: Attribute name to look up.

        Returns:
            Attribute value from _config or _mesh.

        Raises:
            AttributeError: If attribute not found in config or mesh.
        """
        # Avoid infinite recursion during initialization
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )

        # Try config first, then mesh
        try:
            config = object.__getattribute__(self, "_config")
            return getattr(config, name)
        except AttributeError:
            pass

        try:
            mesh = object.__getattribute__(self, "_mesh")
            return getattr(mesh, name)
        except AttributeError:
            pass
        
        return object.__getattribute__(self, name)


    def _init_config(self, config_file, kwargs):
        """Load and merge configuration from defaults, file, and kwargs.

        Configuration is applied in order of increasing priority:
        1. Default values from defaults.DEFAULTS
        2. Values from JSON config file (if provided)
        3. Explicit kwargs (highest priority)

        Args:
            config_file (str|None): Path to JSON configuration file.
            kwargs (dict): Explicit parameter overrides.

        Side Effects:
            Sets all configuration parameters as instance attributes.
            Stores merged dict as self.param_dict for serialization.
        """
        importlib.reload(defaults)

        # Create ReactorConfig (handles merging and validation)
        self._config = ReactorConfig.from_defaults(config_file, **kwargs)

        # Store param_dict for serialization (legacy)
        self.param_dict = self._config.to_dict()

    def _init_derived_parameters(self):
        """Compute geometry-dependent counts, densities, permeabilities & inlets.

        Populates:
            Mesh object, membrane permeability array (self.perm),
            inlet flux distributions, Reynolds/Schmidt groups.
        """
        self.correlation = GasMixtureCorrelations(self.species, self.database)

        # Create mesh (attributes accessed via __getattr__)
        self._mesh = ReactorMesh(self._config)

        # membrane permeabilities
        self.perm = compute_membrane_permeabilities(
            species=self.species,
            Perm_NH3=self.Perm_NH3,
            Sel_am_hy=self.Sel_am_hy,
            Sel_am_ni=self.Sel_am_ni,
            z_c=self.z_c,
            Lsealing=self.Lsealing,
        )

        # Reactor Flow Conditions - inlet fluxes
        self.flux_perm_in = compute_inlet_flux_permeate(
            F_in=self.F_perm_in,
            r_max_perm=self.r_max_perm,
            r_f_perm=self.r_f_perm,
            y_in=self.y_perm_in,
        )
        self.flux_ret_in = compute_inlet_flux_retentate(
            F_in=self.F_ret_in,
            r_min=self.r_min,
            r_max=self.r_max,
            y_in=self.y_ret_in,
            num_r_ret=self.num_r_ret,
            num_species=self.num_c,
        )

        rho_g = self.correlation.density(
            self.y_ret_init, self.T_ret_init, self.p_ret_out
        )  # Gas density [kg/m³]
        viscosity = self.correlation.viscosity(
            self.y_ret_init, self.T_ret_init
        )  # Gas viscosity [Pa s]
        D = self.correlation.diffusion(
            self.y_ret_init, self.T_ret_init, self.p_ret_out
        )  # Gas viscosity [Pa s]
        # Reynolds and Schmidt Numbers

        mass_flux = self.correlation.molecular_weight(
            self.flux_perm_in
        )  # Molecular weight [kg/mol]
        self.Re = mass_flux * 2.0 * self.r_max / viscosity
        self.Sc = viscosity / rho_g / np.mean(D)
        self.ReSc = self.Re * self.Sc
        self.rL = self.r_max / self.L
        self.Lr = self.L / self.r_max

        self.conv_factor_min = math.exp(-self.newton_conv_rate_min)

    def _init_fields(self, c=None, p=None, T=None):
        """Initialize concentration, pressure, temperature, and velocity fields.

        Args:
            c: Initial concentration field (num_z, num_r, num_c). If None, uses
                equilibrium values based on inlet compositions.
            p: Initial pressure field (num_z, num_r). If None, uses outlet pressures.
            T: Initial temperature field (num_z, num_r). If None, uses inlet temps.

        Returns:
            cpT initialized field array.
        """
        shape_cpT = (self.num_z, self.num_r, self.num_c + 2)
        shape_c = (self.num_z, self.num_r, self.num_c)
        shape_p = (self.num_z, self.num_r)
        shape_T = (self.num_z, self.num_r)

        self.cpT = np.empty(shape_cpT)
        if c is None:
            c = self.cpT[..., :-2]
            c_ret = c[:, self.num_r_perm :, :]
            c_ret_0 = (
                self.correlation.molar_density(
                    self.y_ret_init, self.T_ret_init, self.p_ret_out
                )
                * self.y_ret_init
            )
            c_ret[...] = np.broadcast_to(c_ret_0, c_ret.shape)
            c_perm = c[:, : self.num_r_perm, :]
            c_perm_0 = (
                self.correlation.molar_density(
                    self.y_perm_init, self.T_perm_init, self.p_perm_out
                )
                * self.y_perm_init
            )
            c_perm[...] = np.broadcast_to(c_perm_0.reshape((1, 1, -1)), c_perm.shape)
        else:
            self.cpT[..., :-2] = np.broadcast_to(np.array(c), shape_c).copy()
            c_ret = self.cpT[:, self.num_r_perm :, :-2]
            c_perm = self.cpT[:, : self.num_r_perm, :-2]

        self.c_ret_ax = interp_cntr_to_stagg(c_ret, x_f=self.z_f, x_c=self.z_c, axis=0)
        self.c_ret_rad = interp_cntr_to_stagg(
            c_ret, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1
        )
        self.c_perm_ax = interp_cntr_to_stagg(
            c_perm, x_f=self.z_f, x_c=self.z_c, axis=0
        )
        self.c_perm_rad = interp_cntr_to_stagg(
            c_perm, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1
        )

        if p is None:
            p = self.cpT[:, :, -2]
            p_ret = p[:, self.num_r_perm :]
            p_ret[:, :] = np.broadcast_to(
                np.array(self.p_ret_out).reshape((1, 1)), p_ret.shape
            )
            p_perm = p[:, : self.num_r_perm]
            p_perm[:, :] = np.broadcast_to(
                np.array(self.p_perm_out).reshape((1, 1)), p_perm.shape
            )
        else:
            self.cpT[..., -2] = np.broadcast_to(np.array(p), shape_p).copy()

        if T is None:
            T = self.cpT[..., -1]
            T_ret = T[:, self.num_r_perm :]
            T_ret[:, :] = np.broadcast_to(
                np.array(self.T_ret_init).reshape((1, 1)), T_ret.shape
            )
            T_perm = T[:, : self.num_r_perm]
            T_perm[:, :] = np.broadcast_to(
                np.array(self.T_perm_init).reshape((1, 1)), T_perm.shape
            )
        else:
            self.cpT[..., -1] = np.broadcast_to(np.array(T), shape_T).copy()

        c = self.cpT[..., :-2]
        c_perm = np.sum(c[:, 0 : self.num_r_perm, :], axis=-1)
        c_perm_ax = interp_cntr_to_stagg(c_perm, self.z_f, self.z_c, axis=0)
        c_ret = np.sum(c[:, self.num_r_perm :, :], axis=-1)
        c_ret_ax = interp_cntr_to_stagg(c_ret, self.z_f, self.z_c, axis=0)
        self.u_perm_ax = np.sum(self.flux_perm_in[0, :, :], axis=-1) / c_perm_ax
        self.u_ret_ax = np.sum(self.flux_ret_in[0, :, :], axis=-1) / c_ret_ax
        if self.is_counter_current:
            self.u_ret_ax = -self.u_ret_ax
        self.u_perm_rad = np.zeros((self.num_z, self.num_r_perm + 1))
        self.u_ret_rad = np.zeros((self.num_z, self.num_r_ret + 1))
        self.div_u = np.zeros((self.num_z, self.num_r))

        self.cnt_num_solves_cpT = 0
        self.cnt_num_solves_T = 0

        return self.cpT, self.cpT[..., -1]

    def _init_jac(self):
        """Initialize Jacobian matrices for the coupled system.

        Sets up divergence, gradient, and boundary operators for concentration,
        pressure, and temperature fields in both permeate and retentate regions,
        then shifts indices to form a monolithic discretization.
        """
        # Define field shapes
        self._shapes = {
            "c": (self.num_z, self.num_r, self.num_c),
            "c_ret": (self.num_z, self.num_r_ret, self.num_c),
            "c_perm": (self.num_z, self.num_r_perm, self.num_c),
            "p": (self.num_z, self.num_r),
            "p_ret": (self.num_z, self.num_r_ret),
            "p_perm": (self.num_z, self.num_r_perm),
        }

        # Get flow-direction-dependent boundary conditions
        bc_ret_ax, bc_p_ret_ax, bc_T_ret_ax = self._get_axial_bcs_for_flow_direction()

        # Initialize operators for each field type
        jac_c_accum_perm, jac_c_accum_ret = self._init_concentration_operators(
            bc_ret_ax
        )
        self._init_pressure_operators(bc_p_ret_ax)
        self._init_temperature_operators(bc_T_ret_ax)

        # Shift to monolithic indexing and combine accumulation matrices
        self._shift_to_monolithic_indices(jac_c_accum_perm, jac_c_accum_ret)

        # Initialize auxiliary matrices
        self._init_auxiliary_matrices()

    def _init_concentration_operators(self, bc_ret_ax):
        """Initialize divergence and gradient operators for concentration fields.

        Args:
            bc_ret_ax: Axial boundary conditions for retentate concentration.

        Returns:
            Tuple of (jac_c_accum_perm, jac_c_accum_ret) accumulation matrices.
        """
        shape_c_perm = self._shapes["c_perm"]
        shape_c_ret = self._shapes["c_ret"]

        # Accumulation matrices
        jac_c_accum_perm = construct_coefficient_matrix(1.0, shape_c_perm)
        jac_c_accum_ret = construct_coefficient_matrix(self.eps, shape_c_ret)

        # Divergence operators
        self.div_c_perm_ax = construct_div(shape_c_perm, self.z_f, nu=0, axis=0)
        self.div_c_perm_rad = construct_div(
            shape_c_perm, self.r_f_perm, nu=self.nu, axis=1
        )
        self.div_c_ret_ax = construct_div(shape_c_ret, self.z_f, nu=0, axis=0)
        self.div_c_ret_rad = construct_div(
            shape_c_ret, self.r_f_ret, nu=self.nu, axis=1
        )

        # Gradient operators - permeate
        self.grad_c_perm_ax, self.grad_bc_c_perm_ax = construct_grad(
            shape_c_perm,
            self.z_f,
            self.z_c,
            bc=(self.BC_NONE, self.BC_NEUMANN_HOM),
            axis=0,
        )
        self.grad_c_perm_rad, _ = construct_grad(
            shape_c_perm,
            self.r_f_perm,
            self.r_c_perm,
            bc=(self.BC_NEUMANN_HOM, self.BC_NONE),
            axis=1,
        )

        # Gradient operators - retentate
        self.grad_c_ret_ax, self.grad_bc_c_ret_ax = construct_grad(
            shape_c_ret, self.z_f, self.z_c, bc_ret_ax, axis=0
        )
        self.grad_c_ret_rad, _ = construct_grad(
            shape_c_ret,
            self.r_f_ret,
            self.r_c_ret,
            bc=(self.BC_NONE, self.BC_NEUMANN_HOM),
            axis=1,
        )

        # Membrane boundary value matrices
        self.c_matrix_perm_mem, _ = construct_boundary_value_matrices(
            shape_c_perm, self.r_f_perm, self.r_c_perm, bc=None, bound_id=1, axis=1
        )
        self.c_matrix_ret_mem, _ = construct_boundary_value_matrices(
            shape_c_ret, self.r_f_ret, self.r_c_ret, bc=None, bound_id=0, axis=1
        )

        return jac_c_accum_perm, jac_c_accum_ret

    def _init_pressure_operators(self, bc_p_ret_ax):
        """Initialize divergence and gradient operators for pressure fields.

        Args:
            bc_p_ret_ax: Axial boundary conditions for retentate pressure.
        """
        shape_p_perm = self._shapes["p_perm"]
        shape_p_ret = self._shapes["p_ret"]

        # Divergence operators
        self.div_p_perm_ax = construct_div(shape_p_perm, self.z_f, nu=0, axis=0)
        self.div_p_perm_rad = construct_div(
            shape_p_perm, self.r_f_perm, nu=self.nu, axis=1
        )
        self.div_p_ret_ax = construct_div(shape_p_ret, self.z_f, nu=0, axis=0)
        self.div_p_ret_rad = construct_div(
            shape_p_ret, self.r_f_ret, nu=self.nu, axis=1
        )

        # Gradient operators - permeate
        self.grad_p_perm_ax, self.grad_bc_p_perm_ax = construct_grad(
            shape_p_perm,
            self.z_f,
            self.z_c,
            bc=(self.BC_NONE, self.BC_DIRICHLET),
            axis=0,
        )
        self.grad_bc_p_perm_ax *= self.p_perm_out
        self.grad_p_perm_rad, _ = construct_grad(
            shape_p_perm,
            self.r_f_perm,
            self.r_c_perm,
            bc=(self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM),
            axis=1,
        )

        # Gradient operators - retentate
        self.grad_p_ret_ax, self.grad_bc_p_ret_ax = construct_grad(
            shape_p_ret, self.z_f, self.z_c, bc=bc_p_ret_ax, axis=0
        )
        self.grad_bc_p_ret_ax *= self.p_ret_out
        self.grad_p_ret_rad, _ = construct_grad(
            shape_p_ret,
            self.r_f_ret,
            self.r_c_ret,
            bc=(self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM),
            axis=1,
        )

    def _init_temperature_operators(self, bc_T_ret_ax):
        """Initialize gradient operators for temperature fields.

        Args:
            bc_T_ret_ax: Axial boundary conditions for retentate temperature.
        """
        shape_p = self._shapes["p"]
        shape_p_perm = self._shapes["p_perm"]
        shape_p_ret = self._shapes["p_ret"]

        # Accumulation matrix
        self.jac_T_accum = construct_coefficient_matrix(1.0, shape_p)

        # Gradient operators - permeate
        self.grad_T_perm_ax, self.grad_bc_T_perm_ax = construct_grad(
            shape_p_perm,
            self.z_f,
            self.z_c,
            bc=(self.BC_DIRICHLET, self.BC_NEUMANN_HOM),
            axis=0,
        )
        self.grad_bc_T_perm_ax *= self.T_perm_in
        self.grad_T_perm_rad, _, self.grad_bc_T_perm_rad = construct_grad(
            shape_p_perm,
            self.r_f_perm,
            self.r_c_perm,
            bc=(self.BC_NEUMANN_HOM, self.BC_DIRICHLET),
            axis=1,
            shapes_d=(None, (self.num_z, 1)),
        )

        # Gradient operators - retentate
        self.grad_T_ret_ax, self.grad_bc_T_ret_ax = construct_grad(
            shape_p_ret, self.z_f, self.z_c, bc_T_ret_ax, axis=0
        )
        self.grad_bc_T_ret_ax *= self.T_ret_in
        self.grad_T_ret_rad, self.grad_bc_T_ret_rad, _ = construct_grad(
            shape_p_ret,
            self.r_f_ret,
            self.r_c_ret,
            bc=(self.BC_DIRICHLET, self.BC_NEUMANN_HOM),
            axis=1,
            shapes_d=((self.num_z, 1), None),
        )

    def _shift_to_monolithic_indices(self, jac_c_accum_perm, jac_c_accum_ret):
        """Shift region-specific operators to monolithic (combined) indexing.

        Combines permeate and retentate operators into a single system by
        offsetting retentate indices appropriately.

        Args:
            jac_c_accum_perm: Permeate accumulation matrix.
            jac_c_accum_ret: Retentate accumulation matrix.
        """
        shape_c = self._shapes["c"]
        shape_c_ret = self._shapes["c_ret"]
        shape_c_perm = self._shapes["c_perm"]
        shape_p = self._shapes["p"]
        shape_p_ret = self._shapes["p_ret"]
        shape_p_perm = self._shapes["p_perm"]

        # Index offsets for retentate region
        offset_c = (0, self.num_r_perm, 0)
        offset_p = (0, self.num_r_perm)

        # Concentration accumulation
        jac_c_accum_perm = update_csc_array_indices(
            jac_c_accum_perm, shape_c_perm, shape_c
        )
        jac_c_accum_ret = update_csc_array_indices(
            jac_c_accum_ret, shape_c_ret, shape_c, offset=offset_c
        )
        self.jac_c_accum = jac_c_accum_perm + jac_c_accum_ret

        # Concentration divergence operators
        self.div_c_ret_ax = update_csc_array_indices(
            self.div_c_ret_ax,
            (shape_c_ret, None),
            (shape_c, None),
            offset=(offset_c, None),
        )
        self.div_c_ret_rad = update_csc_array_indices(
            self.div_c_ret_rad,
            (shape_c_ret, None),
            (shape_c, None),
            offset=(offset_c, None),
        )
        self.div_c_perm_ax = update_csc_array_indices(
            self.div_c_perm_ax, (shape_c_perm, None), (shape_c, None)
        )
        self.div_c_perm_rad = update_csc_array_indices(
            self.div_c_perm_rad, (shape_c_perm, None), (shape_c, None)
        )

        # Concentration gradient operators
        self.grad_c_ret_ax = update_csc_array_indices(
            self.grad_c_ret_ax,
            (None, shape_c_ret),
            (None, shape_c),
            offset=(None, offset_c),
        )
        self.grad_c_ret_rad = update_csc_array_indices(
            self.grad_c_ret_rad,
            (None, shape_c_ret),
            (None, shape_c),
            offset=(None, offset_c),
        )
        self.grad_c_perm_ax = update_csc_array_indices(
            self.grad_c_perm_ax, (None, shape_c_perm), (None, shape_c)
        )
        self.grad_c_perm_rad = update_csc_array_indices(
            self.grad_c_perm_rad, (None, shape_c_perm), (None, shape_c)
        )

        # Concentration membrane matrices
        self.c_matrix_ret_mem = update_csc_array_indices(
            self.c_matrix_ret_mem,
            (None, shape_c_ret),
            (None, shape_c),
            offset=(None, offset_c),
        )
        self.c_matrix_perm_mem = update_csc_array_indices(
            self.c_matrix_perm_mem, (None, shape_c_perm), (None, shape_c)
        )

        # Pressure divergence operators
        self.div_p_ret_ax = update_csc_array_indices(
            self.div_p_ret_ax,
            (shape_p_ret, None),
            (shape_p, None),
            offset=(offset_p, None),
        )
        self.div_p_ret_rad = update_csc_array_indices(
            self.div_p_ret_rad,
            (shape_p_ret, None),
            (shape_p, None),
            offset=(offset_p, None),
        )
        self.div_p_perm_ax = update_csc_array_indices(
            self.div_p_perm_ax, (shape_p_perm, None), (shape_p, None)
        )
        self.div_p_perm_rad = update_csc_array_indices(
            self.div_p_perm_rad, (shape_p_perm, None), (shape_p, None)
        )

        # Pressure gradient operators
        self.grad_p_ret_ax = update_csc_array_indices(
            self.grad_p_ret_ax,
            (None, shape_p_ret),
            (None, shape_p),
            offset=(None, offset_p),
        )
        self.grad_p_ret_rad = update_csc_array_indices(
            self.grad_p_ret_rad,
            (None, shape_p_ret),
            (None, shape_p),
            offset=(None, offset_p),
        )
        self.grad_p_perm_ax = update_csc_array_indices(
            self.grad_p_perm_ax, (None, shape_p_perm), (None, shape_p)
        )
        self.grad_p_perm_rad = update_csc_array_indices(
            self.grad_p_perm_rad, (None, shape_p_perm), (None, shape_p)
        )

        # Temperature gradient operators
        self.grad_T_ret_ax = update_csc_array_indices(
            self.grad_T_ret_ax,
            (None, shape_p_ret),
            (None, shape_p),
            offset=(None, offset_p),
        )
        self.grad_T_ret_rad = update_csc_array_indices(
            self.grad_T_ret_rad,
            (None, shape_p_ret),
            (None, shape_p),
            offset=(None, offset_p),
        )
        self.grad_T_perm_ax = update_csc_array_indices(
            self.grad_T_perm_ax, (None, shape_p_perm), (None, shape_p)
        )
        self.grad_T_perm_rad = update_csc_array_indices(
            self.grad_T_perm_rad, (None, shape_p_perm), (None, shape_p)
        )

    def _init_auxiliary_matrices(self):
        """Initialize Darcy matrices, numerical Jacobian helpers, and inlet flux."""
        shape_c = self._shapes["c"]
        shape_c_ret = self._shapes["c_ret"]
        shape_p = self._shapes["p"]

        self._construct_darcy_matrices()

        # Numerical Jacobian helpers for reaction and pressure coupling
        self.numjac = NumJac(shape_c_ret)
        self.numjac_p = NumJac(shape_p + (1,))

        # Numerical Jacobians for thermochemical cross-coupling (non-isothermal)
        shape_p_ret = self._shapes["p_ret"]
        self.numjac_cT = NumJac(shape_in=shape_p_ret + (1,), shape_out=shape_c_ret)
        self.numjac_Tc = NumJac(shape_in=shape_c_ret, shape_out=shape_p_ret + (1,))
        self.numjac_TT_react = NumJac(shape_p_ret + (1,))

        # Summation matrix (concentration to total)
        self.sum_c = construct_coefficient_matrix(
            np.array([[[1.0]]]), shape=(shape_p + (1,), shape_c)
        )

        # Inlet flux contribution to residual
        self.g_c_in = (
            self.div_c_perm_ax[:, 0 : self.flux_perm_in.size]
            @ self.flux_perm_in.ravel()
        )
        if self.is_counter_current:
            self.g_c_in -= (
                self.div_c_ret_ax[:, -self.flux_ret_in.size :]
                @ self.flux_ret_in.ravel()
            )
        else:
            self.g_c_in += (
                self.div_c_ret_ax[:, 0 : self.flux_ret_in.size]
                @ self.flux_ret_in.ravel()
            )

        self._jac = None

    def _split_perm_and_ret(
        self, c: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Split full field into permeate and retentate views.

        Args:
            c: Field with shape (num_z, num_r, ...).

        Returns:
            Tuple of (c_perm, c_ret) views/slices.

        Raises:
            ValueError: If c does not have expected radial dimension.
        """
        c = np.asarray(c)
        if c.ndim > 1 and c.shape[1] == self.num_r:
            return self._mesh.split_perm_ret(c)
        raise ValueError(
            f"Expected field with shape (num_z, {self.num_r}, ...), "
            f"got shape {c.shape}"
        )

    def _get_axial_bcs_for_flow_direction(
        self,
    ) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict], Tuple[Dict, Dict]]:
        """Return boundary condition tuples based on flow direction.

        For co-current flow, inlet is at z=0 and outlet at z=L.
        For counter-current flow, retentate inlet is at z=L and outlet at z=0.

        Returns:
            Tuple of (bc_c_ret_ax, bc_p_ret_ax, bc_T_ret_ax) boundary conditions
            for concentration, pressure, and temperature on retentate side.
        """
        if self.is_counter_current:
            bc_c_ret_ax = (self.BC_NEUMANN_HOM, self.BC_NONE)
            bc_p_ret_ax = (self.BC_DIRICHLET, self.BC_NONE)
            bc_T_ret_ax = (self.BC_NEUMANN_HOM, self.BC_DIRICHLET)
        else:
            bc_c_ret_ax = (self.BC_NONE, self.BC_NEUMANN_HOM)
            bc_p_ret_ax = (self.BC_NONE, self.BC_DIRICHLET)
            bc_T_ret_ax = (self.BC_DIRICHLET, self.BC_NEUMANN_HOM)
        return bc_c_ret_ax, bc_p_ret_ax, bc_T_ret_ax

    def _construct_darcy_matrices(self, c=None, T=None, p=None):
        """Assemble permeability-weighted matrices for velocity (Darcy/Ergun).

        Uses Hagen-Poiseuille for laminar flow in permeate side and Ergun
        equation for pressure drop in the packed bed retentate side.

        Args:
            c: Concentration field. Defaults to self.cpT[..., :-2].
            T: Temperature field. Defaults to self.cpT[..., -1].
            p: Pressure field. Defaults to self.cpT[..., -2].
        """
        if c is None:
            c = self.cpT[..., :-2]
        if T is None:
            T = self.cpT[..., -1]
        if p is None:
            p = self.cpT[..., -2]
        c_perm, c_ret = self._split_perm_and_ret(c)
        T_perm, T_ret = self._split_perm_and_ret(T)
        p_perm, p_ret = self._split_perm_and_ret(p)

        # Permeate side: Hagen-Poiseuille
        viscosity_perm = self.correlation.viscosity(c_perm, T_perm)
        k_perm_ax, k_perm_rad = compute_permeate_permeability(
            self.r_max_perm, self.r_c_perm, viscosity_perm
        )
        k_field_perm_ax = interp_cntr_to_stagg(
            k_perm_ax, x_f=self.z_f, x_c=self.z_c, axis=0
        )
        self.k_matrix_perm_ax = construct_coefficient_matrix(
            k_field_perm_ax, (self.num_z, self.num_r_perm), axis=0
        )
        k_field_perm_rad = interp_cntr_to_stagg(
            k_perm_rad, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1
        )
        self.k_matrix_perm_rad = construct_coefficient_matrix(
            k_field_perm_rad, (self.num_z, self.num_r_perm), axis=1
        )

        # Retentate side: Ergun equation for packed bed
        viscosity_ret = self.correlation.viscosity(c_ret, T_ret)
        rho_ret = self.correlation.density(c_ret, T_ret, p_ret)
        u_ax_abs = np.abs(
            interp_stagg_to_cntr(self.u_ret_ax, self.z_f, self.z_c, axis=0)
        )
        k_ret = compute_packed_bed_permeability(
            viscosity_ret, rho_ret, u_ax_abs, self.eps, self.dp
        )
        shape_p_ret = (self.num_z, self.num_r_ret)
        k_field_ret_ax = interp_cntr_to_stagg(k_ret, x_f=self.z_f, x_c=self.z_c, axis=0)
        self.k_matrix_ret_ax = construct_coefficient_matrix(
            k_field_ret_ax, shape_p_ret, axis=0
        )
        k_field_ret_rad = interp_cntr_to_stagg(
            k_ret, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1
        )
        self.k_matrix_ret_rad = construct_coefficient_matrix(
            k_field_ret_rad, shape_p_ret, axis=1
        )

    def _construct_jac_darcy(self):
        """Assemble permeability-weighted matrices for velocity (Darcy / Ergun).

        Returns:
            csc_matrix|None: Pressure Jacobian contribution if compute_jac.
        """
        shape_p_perm = (self.num_z, self.num_r_perm, 1)
        shape_c_perm = (self.num_z, self.num_r_perm, self.num_c)
        ck_matrix = (
            construct_coefficient_matrix(
                self.c_perm_ax, shape=(shape_c_perm, shape_p_perm), axis=0
            )
            @ self.k_matrix_perm_ax
        )
        jac_darcy = self.div_c_perm_ax @ ((-ck_matrix) @ self.grad_p_perm_ax)
        ck_matrix = (
            construct_coefficient_matrix(
                self.c_perm_rad, shape=(shape_c_perm, shape_p_perm), axis=1
            )
            @ self.k_matrix_perm_rad
        )
        jac_darcy += self.div_c_perm_rad @ ((-ck_matrix) @ self.grad_p_perm_rad)
        shape_p_ret = (self.num_z, self.num_r_ret, 1)
        shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
        ck_matrix = (
            construct_coefficient_matrix(
                self.c_ret_ax, shape=(shape_c_ret, shape_p_ret), axis=0
            )
            @ self.k_matrix_ret_ax
        )
        jac_darcy += self.div_c_ret_ax @ ((-ck_matrix) @ self.grad_p_ret_ax)
        ck_matrix = (
            construct_coefficient_matrix(
                self.c_ret_rad, shape=(shape_c_ret, shape_p_ret), axis=1
            )
            @ self.k_matrix_ret_rad
        )
        jac_darcy += self.div_c_ret_rad @ ((-ck_matrix) @ self.grad_p_ret_rad)
        return jac_darcy

    def _construct_g_diff(self, c=None, T=None, p=None, compute_jac=False):
        """Assemble diffusion and membrane permeation residual.

        Args:
            c: Concentration field (num_z, num_r, num_c). Defaults to self.cpT.
            T: Temperature field (num_z, num_r). Defaults to self.cpT[..., -1].
            p: Pressure field (num_z, num_r). Defaults to self.cpT[..., -2].
            compute_jac: If True, compute and cache the Jacobian.

        Returns:
            Tuple of (g_diff, jac_diff) where jac_diff is None if compute_jac=False.
        """
        shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
        shape_c_perm = (self.num_z, self.num_r_perm, self.num_c)
        if c is None:
            c = self.cpT[..., :-2]
        if T is None:
            T = self.cpT[..., -1]
        if p is None:
            p = self.cpT[..., -2]
        c_perm, c_ret = self._split_perm_and_ret(c)
        T_perm, T_ret = self._split_perm_and_ret(T)
        p_perm, p_ret = self._split_perm_and_ret(p)

        # Retentate side
        g = np.empty(c.shape)
        g_vect = g.reshape((-1, 1))
        if compute_jac or not hasattr(self, "jac_c_diff"):
            c_tot_ret_diff = np.maximum(np.abs(np.sum(c_ret, axis=-1, keepdims=True)), 1e-10)
            y_ret = c_ret / c_tot_ret_diff  # Mole fractions
            diff_field_ret = 0.0*self.correlation.diffusion(y_ret, T_ret, p_ret)
            diff_field_ret_ax = interp_cntr_to_stagg(
                diff_field_ret, x_f=self.z_f, x_c=self.z_c, axis=0
            )
            diff_matrix_ret_ax = construct_coefficient_matrix(
                diff_field_ret_ax, shape_c_ret, axis=0
            )
            diff_field_ret_rad = interp_cntr_to_stagg(
                diff_field_ret, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1
            )
            diff_matrix_ret_rad = construct_coefficient_matrix(
                diff_field_ret_rad, shape_c_ret, axis=1
            )

            c_tot_perm_diff = np.maximum(np.abs(np.sum(c_perm, axis=-1, keepdims=True)), 1e-10)
            y_perm = c_perm / c_tot_perm_diff  # Mole fractions
            diff_field_perm = 0.0*self.correlation.diffusion(y_perm, T_perm, p_perm)
            diff_field_perm_ax = interp_cntr_to_stagg(
                diff_field_perm, x_f=self.z_f, x_c=self.z_c, axis=0
            )
            diff_matrix_perm_ax = construct_coefficient_matrix(
                diff_field_perm_ax, shape_c_perm, axis=0
            )
            diff_field_perm_rad = interp_cntr_to_stagg(
                diff_field_perm, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1
            )
            diff_matrix_perm_rad = construct_coefficient_matrix(
                diff_field_perm_rad, shape_c_perm, axis=1
            )

            self.jac_c_diff = (
                self.div_c_ret_ax @ (-diff_matrix_ret_ax) @ self.grad_c_ret_ax
                + self.div_c_ret_rad @ (-diff_matrix_ret_rad) @ self.grad_c_ret_rad
                + self.div_c_perm_ax @ (-diff_matrix_perm_ax) @ self.grad_c_perm_ax
                + self.div_c_perm_rad @ (-diff_matrix_perm_rad) @ self.grad_c_perm_rad
            )
            self.g_bc_c_diff = self.div_c_ret_ax @ (
                (-diff_matrix_ret_ax) @ self.grad_bc_c_ret_ax
            ) + self.div_c_perm_ax @ ((-diff_matrix_perm_ax) @ self.grad_bc_c_perm_ax)

            if T_perm.ndim > 1:
                T_perm_mem, _ = compute_boundary_values(
                    T_perm, self.r_f_perm, self.r_c_perm, bc=None, axis=1, bound_id=1
                )
                T_ret_mem, _ = compute_boundary_values(
                    T_ret, self.r_f_ret, self.r_c_ret, bc=None, axis=1, bound_id=0
                )
                P_matrix_perm_mem = construct_coefficient_matrix(
                    self.Rg * T_perm_mem[:, 0, np.newaxis] * self.perm
                )
                P_matrix_ret_mem = construct_coefficient_matrix(
                    self.Rg * T_ret_mem[:, 0, np.newaxis] * self.perm
                )
            else:
                P_perm_mem = np.broadcast_to(
                    (self.Rg * T_perm * self.perm).reshape((1, -1)),
                    (self.num_z, self.num_c),
                )
                P_ret_mem = np.broadcast_to(
                    (self.Rg * T_ret * self.perm).reshape((1, -1)),
                    (self.num_z, self.num_c),
                )
                P_matrix_perm_mem = construct_coefficient_matrix(P_perm_mem)
                P_matrix_ret_mem = construct_coefficient_matrix(P_ret_mem)

            flux_matrix_perm_mem = (
                P_matrix_perm_mem @ self.c_matrix_perm_mem
                - P_matrix_ret_mem @ self.c_matrix_ret_mem
            )
            factor_geom = (self.r_f_perm[-1] / self.r_f_ret[0]) ** self.nu
            flux_matrix_ret_mem = factor_geom * flux_matrix_perm_mem
            flux_matrix_perm_mem = update_csc_array_indices(
                flux_matrix_perm_mem,
                ((self.num_z, 1, self.num_c), None),
                ((self.num_z, self.num_r_perm + 1, self.num_c), None),
                offset=((0, self.num_r_perm, 0), None),
            )
            flux_matrix_ret_mem = update_csc_array_indices(
                flux_matrix_ret_mem,
                ((self.num_z, 1, self.num_c), None),
                ((self.num_z, self.num_r_ret + 1, self.num_c), None),
            )
            self.jac_c_diff += (
                self.div_c_ret_rad @ flux_matrix_ret_mem
                + self.div_c_perm_rad @ flux_matrix_perm_mem
            )
            self.flux_matrix_ret_mem = flux_matrix_ret_mem
            self.flux_matrix_perm_mem = flux_matrix_perm_mem

        g_vect[...] = self.g_bc_c_diff + self.jac_c_diff @ c.reshape((-1, 1))
        return g, self.jac_c_diff

    def _update_velocity_fields(self, p=None):
        """Update staggered velocity arrays from current pressure gradients.

        Returns:
            tuple: (u_perm_ax, u_perm_rad, u_ret_ax, u_ret_rad)
        """
        vel_matrix_perm_ax = (-self.k_matrix_perm_ax) @ self.grad_p_perm_ax
        vel_bc_perm_ax = (-self.k_matrix_perm_ax) @ self.grad_bc_p_perm_ax
        vel_matrix_perm_rad = (-self.k_matrix_perm_rad) @ self.grad_p_perm_rad
        vel_matrix_ret_ax = (-self.k_matrix_ret_ax) @ self.grad_p_ret_ax
        vel_bc_ret_out = -(self.k_matrix_ret_ax @ self.grad_bc_p_ret_ax)
        vel_matrix_ret_rad = (-self.k_matrix_ret_rad) @ self.grad_p_ret_rad
        if p is None:
            p_vec = self.cpT[..., -2].reshape((-1, 1))
        else:
            p_vec = p.reshape((-1, 1))
        self.u_perm_ax.reshape((-1, 1))[...] = (
            vel_matrix_perm_ax @ p_vec + vel_bc_perm_ax
        )
        self.u_perm_rad.reshape((-1, 1))[...] = vel_matrix_perm_rad @ p_vec
        self.u_ret_ax.reshape((-1, 1))[...] = vel_matrix_ret_ax @ p_vec + vel_bc_ret_out
        self.u_ret_rad.reshape((-1, 1))[...] = vel_matrix_ret_rad @ p_vec

        self.u_perm_ax[0, :] = self.u_perm_ax[1, :] - (self.z_f[1] - self.z_f[0]) / (
            self.z_f[2] - self.z_f[1]
        ) * (self.u_perm_ax[2, :] - self.u_perm_ax[1, :])
        if self.is_counter_current:
            self.u_ret_ax[-1, :] = self.u_ret_ax[-2, :] - (
                self.z_f[-2] - self.z_f[-1]
            ) / (self.z_f[-3] - self.z_f[-2]) * (
                self.u_ret_ax[-3, :] - self.u_ret_ax[-2, :]
            )
        else:
            self.u_ret_ax[0, :] = self.u_ret_ax[1, :] - (self.z_f[1] - self.z_f[0]) / (
                self.z_f[2] - self.z_f[1]
            ) * (self.u_ret_ax[2, :] - self.u_ret_ax[1, :])

        self.div_u = (
            self.div_p_perm_ax @ self.u_perm_ax.ravel()
            + self.div_p_perm_rad @ self.u_perm_rad.ravel()
            + self.div_p_ret_ax @ self.u_ret_ax.ravel()
            + self.div_p_ret_rad @ self.u_ret_rad.ravel()
        ).reshape(self.cpT[..., -1].shape)

        return self.u_perm_ax, self.u_perm_rad, self.u_ret_ax, self.u_ret_rad

    def _construct_g_conv(self, c=None, compute_jac=False):
        """Assemble convective species transport residual using upwind/TVD.

        Args:
            c (ndarray|None): Concentration field (optional, defaults to self.cpT).
            compute_jac (bool): If True, compute Jacobian sparsity pattern.

        Returns:
            tuple: (g, jac, jac_p) residual and Jacobian (or None if compute_jac=False)
        """
        if c is None:
            c = self.cpT[..., :-2]

        # Determine retentate axial BCs with inflow adjustment for reverse flow.
        # Use a local copy so get_axial_bcs_for_flow does not mutate self.u_ret_ax.
        inflow_conc = (
            self.p_ret_out / (self.Rg * self.T_ret_in) * np.array([[[0.0, 1.0, 0.0]]])
        )
        u_ret_ax_local = self.u_ret_ax.copy()
        bc_ret_ax = get_axial_bcs_for_flow(
            is_counter_current=self.is_counter_current,
            u_ax=u_ret_ax_local,
            bc_inlet=self.BC_NONE,
            inflow_value=inflow_conc,
        )

        g = np.empty(c.shape)
        g_vect = g.ravel()

        # Permeate region convection
        c_perm = c[:, : self.num_r_perm, :]
        u_perm_ax = self.u_perm_ax[..., np.newaxis]
        u_perm_rad = self.u_perm_rad[..., np.newaxis]
        bc_perm_ax = (self.BC_NONE, self.BC_NEUMANN_HOM)
        bc_perm_rad = (self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM)

        c_perm_ax, _ = interp_cntr_to_stagg_tvd(
            c_perm,
            self.z_f,
            self.z_c,
            bc=bc_perm_ax,
            v=u_perm_ax,
            tvd_limiter=upwind,
            axis=0,
        )
        flux_perm_ax = u_perm_ax * c_perm_ax
        g_vect[:] = self.div_c_perm_ax @ flux_perm_ax.ravel()

        c_perm_rad, _ = interp_cntr_to_stagg_tvd(
            c_perm,
            self.r_f_perm,
            self.r_c_perm,
            bc=bc_perm_rad,
            v=u_perm_rad,
            tvd_limiter=upwind,
            axis=1,
        )
        flux_perm_rad = u_perm_rad * c_perm_rad
        g_vect[:] += self.div_c_perm_rad @ flux_perm_rad.ravel()

        # Retentate region convection (use local copy which may have outlet zeroed)
        c_ret = c[:, self.num_r_perm :, :]
        u_ret_ax = u_ret_ax_local[..., np.newaxis]
        u_ret_rad = self.u_ret_rad[..., np.newaxis]
        bc_ret_rad = (self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM)

        c_ret_ax, _ = interp_cntr_to_stagg_tvd(
            c_ret,
            self.z_f,
            self.z_c,
            bc=bc_ret_ax,
            v=u_ret_ax,
            tvd_limiter=upwind,
            axis=0,
        )
        flux_ret_ax = u_ret_ax * c_ret_ax
        g_vect[:] += self.div_c_ret_ax @ flux_ret_ax.ravel()

        c_ret_rad, _ = interp_cntr_to_stagg_tvd(
            c_ret,
            self.r_f_ret,
            self.r_c_ret,
            bc=bc_ret_rad,
            v=u_ret_rad,
            tvd_limiter=upwind,
            axis=1,
        )
        flux_ret_rad = u_ret_rad * c_ret_rad
        g_vect[:] += self.div_c_ret_rad @ flux_ret_rad.ravel()

        if compute_jac:
            # Permeate Jacobian
            conv_matrix_perm_ax, _ = construct_convflux_upwind(
                c_perm.shape, self.z_f, self.z_c, bc=bc_perm_ax, v=u_perm_ax, axis=0
            )
            jac_perm = self.div_c_perm_ax @ conv_matrix_perm_ax
            
            ck_matrix = (
                construct_coefficient_matrix(
                    c_perm_ax, shape=(c_perm_ax.shape, c_perm_ax.shape[:-1]+(1,)))
                @ self.k_matrix_perm_ax
            )
            jac_darcy = self.div_c_perm_ax @ ((-ck_matrix) @ self.grad_p_perm_ax)
                        
            conv_matrix_perm_rad, _ = construct_convflux_upwind(
                c_perm.shape,
                self.r_f_perm,
                self.r_c_perm,
                bc=bc_perm_rad,
                v=u_perm_rad,
                axis=1,
            )
            jac_perm += self.div_c_perm_rad @ conv_matrix_perm_rad

            ck_matrix = (
                construct_coefficient_matrix(
                    c_perm_rad, shape=(c_perm_rad.shape, c_perm_rad.shape[:-1]+(1,)))
                @ self.k_matrix_perm_rad
            )
            jac_darcy += self.div_c_perm_rad @ ((-ck_matrix) @ self.grad_p_perm_rad)

            # Retentate Jacobian
            conv_matrix_ret_ax, _ = construct_convflux_upwind(
                c_ret.shape, self.z_f, self.z_c, bc=bc_ret_ax, v=u_ret_ax, axis=0
            )
            jac_ret = self.div_c_ret_ax @ conv_matrix_ret_ax
    
            ck_matrix = (
                construct_coefficient_matrix(
                    c_ret_ax, shape=(c_ret_ax.shape, c_ret_ax.shape[:-1]+(1,)))
                @ self.k_matrix_ret_ax
            )
            jac_darcy += self.div_c_ret_ax @ ((-ck_matrix) @ self.grad_p_ret_ax)

            conv_matrix_ret_rad, _ = construct_convflux_upwind(
                c_ret.shape,
                self.r_f_ret,
                self.r_c_ret,
                bc=bc_ret_rad,
                v=u_ret_rad,
                axis=1,
            )
            jac_ret += self.div_c_ret_rad @ conv_matrix_ret_rad

            ck_matrix = (
                construct_coefficient_matrix(
                    c_ret_rad, shape=(c_ret_rad.shape, c_ret_rad.shape[:-1]+(1,)))
                @ self.k_matrix_ret_rad
            )
            jac_darcy += self.div_c_ret_rad @ ((-ck_matrix) @ self.grad_p_ret_rad)


            # Remap to monolithic indices
            jac_perm = update_csc_array_indices(
                jac_perm, (None, c_perm.shape), (None, c.shape)
            )
            jac_ret = update_csc_array_indices(
                jac_ret,
                (None, c_ret.shape),
                (None, c.shape),
                offset=(None, (0, self.num_r_perm, 0)),
            )
            return g, jac_perm + jac_ret, jac_darcy

        return g, None, None

    def _construct_g_cpT(
        self, c_old, T_old, dt, compute_jac=False,
    ):
        """Construct coupled concentration-pressure residual and Jacobian.

        Args:
            c_old: Previous concentration field for transient term.
            T_old: Previous temperature field (unused, for API consistency).
            dt: Time step size.
            compute_jac: If True, compute the Jacobian matrix.

        Returns:
            Tuple of (g, jac) where g is the residual array (num_z, num_r, num_c+1)
            and jac is the sparse Jacobian (or None if compute_jac=False).
        """
        cpT = self.cpT
        c = cpT[..., :-2]
        p = cpT[..., -2]
        T = self.cpT[..., -1]
        shape_cpT = cpT.shape
        g = np.empty(shape_cpT)
        c_sum = np.sum(c, axis=-1)
        # Use safe denominator for mole fractions to handle negative concentrations
        c_sum_safe = np.maximum(np.abs(c_sum), 1e-10)
        y = c / c_sum_safe[..., np.newaxis]

        c_ret = c[:, self.num_r_perm :, :]
        c_tot_ret = np.sum(c_ret, axis=-1, keepdims=True)
        _, p_ret = self._split_perm_and_ret(p)
        # Use safe denominator for pressure/concentration ratio
        c_tot_ret_safe = np.maximum(np.abs(c_tot_ret), 1e-10)
        p_over_c_tot = p_ret[..., np.newaxis] / c_tot_ret_safe
        T_ret = T[:, self.num_r_perm :]        
        
        self._construct_darcy_matrices(c=cpT[..., :-2], p=cpT[..., -2])
        self._update_velocity_fields(p=self.cpT[..., -2])
        g_conv, jac_conv, jac_darcy = self._construct_g_conv(c, compute_jac=compute_jac)
        g_diff, jac_diff = self._construct_g_diff(c, compute_jac=compute_jac)

        # Handle isothermal mode: skip temperature solve if enabled
        if self.is_isothermal:
            g_T_conv = np.zeros_like(T)
            g_T_cond = np.zeros_like(T)
            jac_T_conv = None
            jac_T_darcy = None
            jac_T_cond = None
            # Use unit cp for energy scaling (not used in isothermal mode)
            cp_inv_mat = construct_coefficient_matrix(np.ones_like(T))
        else:
            g_T_conv, jac_T_conv, jac_T_darcy = self._construct_g_T_conv(T, compute_jac=compute_jac)
            g_T_cond, jac_T_cond, cp_inv_mat = self._construct_g_T_cond(T)
        if compute_jac:
            jac_cc = jac_conv + jac_diff
            if c_old is not None:
                jac_cc += (1.0 / dt) * self.jac_c_accum
            g_react, jac_react = self.numjac(
                lambda c: self.factor_react * self.kinetics(c * p_over_c_tot), c_ret
            )
            shape_c_ret = c_ret.shape
            offset = (0, self.num_r_perm, 0)
            jac_react = update_csc_array_indices(
                    jac_react, shape_c_ret, c.shape, offset=offset
            )
            jac_cc -= jac_react
            c_tot, dc_tot_dp_mat = self.numjac_p(
                lambda p: self.correlation.molar_density(y, T, p), p
            )
            c_tot, dc_tot_dT_mat = self.numjac_p(
                lambda T: self.correlation.molar_density(y, T, p), T, f_value=c_tot
            )
            #jac_darcy = self._construct_jac_darcy()
            jac_cp = jac_darcy
            jac_pp = self.factor_p * dc_tot_dp_mat
            jac_pc = -self.factor_p * self.sum_c
            jac_pT = self.factor_p * dc_tot_dT_mat
            # Handle isothermal mode: simplified temperature Jacobian
            jac_cT_ret = None
            jac_Tc_ret = None
            if self.is_isothermal:
                # Pin temperature to initial value: g_T = T - T_init = 0
                jac_TT = (1.0 / dt) * self.jac_T_accum
                jac_TP = None
            else:
                jac_TT = (1.0 / dt) * self.jac_T_accum + jac_T_conv + jac_T_cond
                jac_TP = jac_T_darcy

                # === Thermochemical coupling blocks ===
                enthalpies = self.correlation.species_enthalpies(T_ret)
                cp_inv = cp_inv_mat.data.reshape(T.shape)[:, self.num_r_perm:]
                shape_T_ret_loc = (self.num_z, self.num_r_ret, 1)

                # Task 1 (J_cT): d(g_react)/d(T_ret)
                # Maps T_ret perturbations → concentration residual changes
                self.kinetics(c_ret * p_over_c_tot, T_ret)
                _, jac_cT_ret = self.numjac_cT(
                    lambda T_var: self.factor_react * self.kinetics(
                        c_ret * p_over_c_tot, T_var[..., 0]
                    ),
                    T_ret[..., np.newaxis],
                )

                # Task 2 (J_Tc): d(heat_source)/d(c_ret)
                # Maps concentration perturbations → temperature residual changes
                self.kinetics(c_ret * p_over_c_tot, T_ret)
                _, jac_Tc_ret = self.numjac_Tc(
                    lambda c_var: (
                        np.sum(
                            self.factor_react
                            * self.kinetics(c_var * p_over_c_tot)
                            * enthalpies,
                            axis=-1,
                        )
                        * cp_inv
                    )[..., np.newaxis],
                    c_ret,
                )

                # Task 3 (J_TT_react): d(heat_source)/d(T_ret)
                # Adds Arrhenius temperature sensitivity to jac_TT
                self.kinetics(c_ret * p_over_c_tot, T_ret)
                _, jac_TT_react_ret = self.numjac_TT_react(
                    lambda T_var: (
                        np.sum(
                            self.factor_react
                            * self.kinetics(c_ret * p_over_c_tot, T_var[..., 0])
                            * self.correlation.species_enthalpies(T_var[..., 0]),
                            axis=-1,
                        )
                        * cp_inv
                    )[..., np.newaxis],
                    T_ret[..., np.newaxis],
                )
                # Map J_TT_react from retentate-T space to full-T space and add
                jac_TT += update_csc_array_indices(
                    jac_TT_react_ret,
                    shape_T_ret_loc,
                    (self.num_z, self.num_r, 1),
                    offset=(0, self.num_r_perm, 0),
                )
                # Restore kinetics to current (c_ret, T_ret) after all FD sweeps
                self.kinetics(c_ret * p_over_c_tot, T_ret)

            shape_c = c.shape
            shape_p = p.shape + (1,)
            offset_p = (0,) * (c.ndim - 1) + (shape_c[-1],)
            offset_T = (0,) * (c.ndim - 1) + (shape_c[-1]+1,)
            jac_cc = update_csc_array_indices(jac_cc, shape_c, shape_cpT)
            jac_pp = update_csc_array_indices(jac_pp, shape_p, shape_cpT, offset=offset_p)
            jac_cp = update_csc_array_indices(
                jac_cp, (shape_c, shape_p), shape_cpT, offset=(None, offset_p)
            )
            jac_pc = update_csc_array_indices(
                jac_pc, (shape_p, shape_c), shape_cpT, offset=(offset_p, None)
            )
            jac_pT = update_csc_array_indices(
                jac_pT, (shape_p, shape_p), shape_cpT, offset=(offset_p, offset_T)
            )
            jac_TT = update_csc_array_indices(jac_TT, shape_p, shape_cpT, offset=offset_T)
            # Task 4: map thermochemical blocks to cpT space and add to assembly
            if jac_cT_ret is not None:
                shape_c_ret_loc = (self.num_z, self.num_r_ret, self.num_c)
                shape_T_ret_loc = (self.num_z, self.num_r_ret, 1)
                offset_c_ret = (0, self.num_r_perm, 0)
                offset_T_ret = (0, self.num_r_perm, self.num_c + 1)
                jac_cT = update_csc_array_indices(
                    jac_cT_ret,
                    (shape_c_ret_loc, shape_T_ret_loc),
                    shape_cpT,
                    offset=(offset_c_ret, offset_T_ret),
                )
                jac_Tc = update_csc_array_indices(
                    jac_Tc_ret,
                    (shape_T_ret_loc, shape_c_ret_loc),
                    shape_cpT,
                    offset=(offset_T_ret, offset_c_ret),
                )
            else:
                jac_cT = jac_Tc = None

            base_jac = jac_cc + jac_pp + jac_cp + jac_pc + jac_pT + jac_TT
            if jac_TP is not None:
                jac_TP = update_csc_array_indices(
                    jac_TP, shape_p, shape_cpT, offset=(offset_T, offset_p)
                )
                base_jac = base_jac + jac_TP
            if jac_cT is not None:
                base_jac = base_jac + jac_cT + jac_Tc
            self._jac = base_jac
        else:
            c_tot = self.correlation.molar_density(y, T, p)
            # Pass T_ret explicitly so kinetics always uses the current temperature
            g_react = self.factor_react * self.kinetics(c_ret * p_over_c_tot, T_ret)

        g_c = self.g_c_in.reshape(c.shape) + g_conv + g_diff
        if c_old is not None:
            g_c += (self.jac_c_accum @ ((c - c_old).reshape((-1, 1)) / dt)).reshape(
                c.shape
            )
        g_ret = g_c[:, self.num_r_perm :, :]
        g_ret[...] += g_react
        g_p = self.factor_p * (c_tot - c_sum)

        # Temperature residual
        if self.is_isothermal:
            # In isothermal mode, constrain temperature to initial value
            # g_T = (T - T_init) / dt -> T held constant
            T_init = np.empty_like(T)
            T_init[:, :self.num_r_perm] = self.T_perm_in
            T_init[:, self.num_r_perm:] = self.T_ret_in
            g_T = (T - T_init) / dt
        else:
            g_T = g_T_conv + g_T_cond
            if T_old is not None:
                g_T += (self.jac_T_accum @ ((T - T_old).reshape((-1, 1)) / dt)).reshape(
                    T.shape
                )
            enthalpies = self.correlation.species_enthalpies(T_ret)
            dH_react = np.sum(g_react * enthalpies, axis=-1)
            g_T_ret = g_T[:, self.num_r_perm :]
            g_T_ret[...] += (
                dH_react * cp_inv_mat.data.reshape(T.shape)[:, self.num_r_perm :]
            )
        g[..., :-2] = g_c
        g[..., -2] = g_p
        g[..., -1] = g_T
        return g, self._jac

    def _construct_g_T_conv(self, T, compute_jac=False):
        """Assemble convective energy residual (includes -T∇·u term).

        Args:
            T (ndarray): Temperature field (z,r).
            compute_jac (bool): If True, compute Jacobian sparsity pattern.

        Returns:
            tuple: (g, jac) residual and Jacobian (or None if compute_jac=False)
        """
        # Build Dirichlet BCs with temperature values
        bc_ret_dirichlet = make_dirichlet_bc(self.T_ret_in)
        bc_perm_dirichlet = make_dirichlet_bc(self.T_perm_in)

        # Determine retentate axial BCs with inflow adjustment for reverse flow.
        # Use a local copy so get_axial_bcs_for_flow does not mutate self.u_ret_ax.
        inflow_T = self.T_ret_in  # scalar for temperature
        u_ret_ax_local = self.u_ret_ax.copy()
        bc_ret_ax = get_axial_bcs_for_flow(
            is_counter_current=self.is_counter_current,
            u_ax=u_ret_ax_local,
            bc_inlet=bc_ret_dirichlet,
            inflow_value=inflow_T,
        )

        g = np.empty(T.shape)
        g_vect = g.ravel()

        T_perm = T[:, 0 : self.num_r_perm]
        T_perm_ax, _ = interp_cntr_to_stagg_tvd(
            T_perm,
            self.z_f,
            self.z_c,
            bc=(bc_perm_dirichlet, self.BC_NEUMANN_HOM),
            v=self.u_perm_ax,
            tvd_limiter=upwind,
            axis=0,
        )
        flux_perm_ax = self.u_perm_ax * T_perm_ax
        g_vect[:] = self.div_p_perm_ax @ flux_perm_ax.ravel()
        T_perm_rad, _ = interp_cntr_to_stagg_tvd(
            T_perm,
            self.r_f_perm,
            self.r_c_perm,
            bc=(self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM),
            v=self.u_perm_rad,
            tvd_limiter=upwind,
            axis=1,
        )
        flux_perm_rad = self.u_perm_rad * T_perm_rad
        g_vect[:] += self.div_p_perm_rad @ flux_perm_rad.ravel()

        T_ret = T[:, self.num_r_perm :]
        T_ret_ax, _ = interp_cntr_to_stagg_tvd(
            T_ret,
            self.z_f,
            self.z_c,
            bc=bc_ret_ax,
            v=u_ret_ax_local,
            tvd_limiter=upwind,
            axis=0,
        )
        flux_ret_ax = u_ret_ax_local * T_ret_ax
        g_vect[:] += self.div_p_ret_ax @ flux_ret_ax.ravel()
        T_ret_rad, _ = interp_cntr_to_stagg_tvd(
            T_ret,
            self.r_f_ret,
            self.r_c_ret,
            bc=(self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM),
            v=self.u_ret_rad,
            tvd_limiter=upwind,
            axis=1,
        )
        flux_ret_rad = self.u_ret_rad * T_ret_rad
        g_vect[:] += self.div_p_ret_rad @ flux_ret_rad.ravel()
        g_vect[:] -= (T * self.div_u).ravel()

        if compute_jac:
            conv_matrix_perm_ax, _ = construct_convflux_upwind(
                T_perm.shape,
                self.z_f,
                self.z_c,
                bc=(bc_perm_dirichlet, self.BC_NEUMANN_HOM),
                v=self.u_perm_ax,
                axis=0,
            )
            jac_perm = self.div_p_perm_ax @ conv_matrix_perm_ax
            
            Tk_matrix = (
                construct_coefficient_matrix(T_perm_ax)
                @ self.k_matrix_perm_ax
            )
            jac_darcy = self.div_p_perm_ax @ ((-Tk_matrix) @ self.grad_p_perm_ax)
            jac_darcy_div = self.div_p_perm_ax @ (self.k_matrix_perm_ax @ self.grad_p_perm_ax)
            
            conv_matrix_perm_rad, _ = construct_convflux_upwind(
                T_perm.shape,
                self.r_f_perm,
                self.r_c_perm,
                bc=(self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM),
                v=self.u_perm_rad,
                axis=1,
            )
            jac_perm += self.div_p_perm_rad @ conv_matrix_perm_rad

            Tk_matrix = (
                construct_coefficient_matrix(T_perm_rad)
                @ self.k_matrix_perm_rad
            )
            jac_darcy += self.div_p_perm_rad @ ((-Tk_matrix) @ self.grad_p_perm_rad)
            jac_darcy_div += self.div_p_perm_rad @ (self.k_matrix_perm_rad @ self.grad_p_perm_rad)

            conv_matrix_ret_ax, _ = construct_convflux_upwind(
                T_ret.shape, self.z_f, self.z_c, bc=bc_ret_ax, v=u_ret_ax_local, axis=0
            )
            jac_ret = self.div_p_ret_ax @ conv_matrix_ret_ax
            
            Tk_matrix = (
                construct_coefficient_matrix(T_ret_ax)
                @ self.k_matrix_ret_ax
            )
            jac_darcy += self.div_p_ret_ax @ ((-Tk_matrix) @ self.grad_p_ret_ax)
            jac_darcy_div += self.div_p_ret_ax @ (self.k_matrix_ret_ax @ self.grad_p_ret_ax)
            
            conv_matrix_ret_rad, _ = construct_convflux_upwind(
                T_ret.shape,
                self.r_f_ret,
                self.r_c_ret,
                bc=(self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM),
                v=self.u_ret_rad,
                axis=1,
            )
            jac_ret += self.div_p_ret_rad @ conv_matrix_ret_rad

            Tk_matrix = (
                construct_coefficient_matrix(T_ret_rad)
                @ self.k_matrix_ret_rad
            )
            jac_darcy += self.div_p_ret_rad @ ((-Tk_matrix) @ self.grad_p_ret_rad)
            jac_darcy_div += self.div_p_ret_rad @ (self.k_matrix_ret_rad @ self.grad_p_ret_rad)
            jac_darcy += construct_coefficient_matrix(T) @ jac_darcy_div

            jac_perm = update_csc_array_indices(
                jac_perm, (None, T_perm.shape), (None, T.shape)
            )
            jac_ret = update_csc_array_indices(
                jac_ret,
                (None, T_ret.shape),
                (None, T.shape),
                offset=(None, (0, self.num_r_perm, 0)),
            )
            jac = jac_perm + jac_ret - construct_coefficient_matrix(self.div_u)
            return g, jac, jac_darcy
        else:
            return g, None, None

    def _construct_g_T_cond(self, T):
        """Assemble conductive + membrane interfacial heat transfer residual.

        Returns:
            tuple: (g_cond, jac_cond, cp_inv_matrix)
        """
        c = self.cpT[..., :-2]
        g = np.empty(T.shape)
        g_vect = g.reshape((-1, 1))

        c_tot_safe = np.maximum(np.abs(np.sum(c, axis=-1, keepdims=True)), 1e-10)
        y = c / c_tot_safe  # Mole fractions
        lmbda = self.correlation.thermal_conductivity(y, T)
        cp = self.correlation.specific_heat(c, T)

        lmbda_perm = lmbda[:, 0 : self.num_r_perm]
        lmbda_perm_ax = interp_cntr_to_stagg(lmbda_perm, self.z_f, self.z_c, axis=0)
        lmbda_perm_ax_mat = construct_coefficient_matrix(lmbda_perm_ax)
        jac_cond = self.div_p_perm_ax @ (-lmbda_perm_ax_mat @ self.grad_T_perm_ax)
        g_cond_bc = self.div_p_perm_ax @ (-lmbda_perm_ax_mat @ self.grad_bc_T_perm_ax)

        lmbda_perm_rad = interp_cntr_to_stagg(
            lmbda_perm, self.r_f_perm, self.r_c_perm, axis=1
        )
        lmbda_perm_rad_mat = construct_coefficient_matrix(lmbda_perm_rad)
        jac_cond += self.div_p_perm_rad @ (-lmbda_perm_rad_mat @ self.grad_T_perm_rad)

        lmbda_ret = lmbda[:, self.num_r_perm :]
        lmbda_ret_ax = interp_cntr_to_stagg(lmbda_ret, self.z_f, self.z_c, axis=0)
        lmbda_ret_ax_mat = construct_coefficient_matrix(lmbda_ret_ax)
        jac_cond += self.div_p_ret_ax @ (-lmbda_ret_ax_mat @ self.grad_T_ret_ax)
        g_cond_bc += self.div_p_ret_ax @ (-lmbda_ret_ax_mat @ self.grad_bc_T_ret_ax)

        lmbda_ret_rad = interp_cntr_to_stagg(
            lmbda_ret, self.r_f_ret, self.r_c_ret, axis=1
        )
        lmbda_ret_rad_mat = construct_coefficient_matrix(lmbda_ret_rad)
        jac_cond += self.div_p_ret_rad @ (-lmbda_ret_rad_mat @ self.grad_T_ret_rad)

        # Heat transfer through membrane - extract interface values
        bc_rad_hom = (self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM)
        T_perm = self.cpT[..., -1][:, 0 : self.num_r_perm]
        T_ret = self.cpT[..., -1][:, self.num_r_perm :]
        _, _, T_perm_i, _ = compute_boundary_values(
            T_perm, self.r_f_perm, self.r_c_perm, bc=bc_rad_hom, axis=1
        )
        T_ret_i, _, _, _ = compute_boundary_values(
            T_ret, self.r_f_ret, self.r_c_ret, bc=bc_rad_hom, axis=1
        )
        y_perm = y[:, 0 : self.num_r_perm, :]
        y_ret = y[:, self.num_r_perm :, :]
        _, _, y_perm_i, _ = compute_boundary_values(
            y_perm, self.r_f_perm, self.r_c_perm, bc=bc_rad_hom, axis=1
        )
        y_ret_i, _, _, _ = compute_boundary_values(
            y_ret, self.r_f_ret, self.r_c_ret, bc=bc_rad_hom, axis=1
        )
        c_perm = c[:, 0 : self.num_r_perm, :]
        c_ret = c[:, self.num_r_perm :, :]
        _, _, c_perm_i, _ = compute_boundary_values(
            c_perm, self.r_f_perm, self.r_c_perm, bc=bc_rad_hom, axis=1
        )
        c_ret_i, _, _, _ = compute_boundary_values(
            c_ret, self.r_f_ret, self.r_c_ret, bc=bc_rad_hom, axis=1
        )
        _, _, u_perm_ax_i, _ = compute_boundary_values(
            self.u_perm_ax, self.r_f_perm, self.r_c_perm, bc=bc_rad_hom, axis=1
        )
        u_perm_i = interp_stagg_to_cntr(u_perm_ax_i, self.z_f, self.z_c, axis=0)
        u_ret_ax_i, _, _, _ = compute_boundary_values(
            self.u_ret_ax, self.r_f_ret, self.r_c_ret, bc=bc_rad_hom, axis=1
        )
        u_ret_i = interp_stagg_to_cntr(u_ret_ax_i, self.z_f, self.z_c, axis=0)

        visc_ret = self.correlation.viscosity(y_ret_i, T_ret_i)
        rho_ret = self.correlation.molecular_weight(c_ret_i)
        cp_ret = self.correlation.specific_heat(c_ret_i, T_ret_i)
        Re_ret = np.abs(rho_ret * self.dp * u_ret_i / visc_ret)
        Pr_ret = np.abs(visc_ret * cp_ret / lmbda_ret_rad[:, [0]])
        Nu_ret = self.Nu_ret(Re_ret, Pr_ret)
        # Add minimum heat transfer coefficient to avoid division by zero
        h_min = 1.0  # W/(m^2 K) - natural convection lower bound
        h_ret = np.maximum(Nu_ret * lmbda_ret_rad[:, [0]] / self.dp, h_min)

        d_tube = 2.0 * self.r_f_perm[-1]
        visc_perm = self.correlation.viscosity(y_perm_i, T_perm_i)
        rho_perm = self.correlation.molecular_weight(c_perm_i)
        cp_perm = self.correlation.specific_heat(c_perm_i, T_perm_i)
        Re_perm = np.abs(rho_perm * d_tube * u_perm_i / visc_perm)
        Pr_perm = np.abs(visc_perm * cp_perm / lmbda_perm_rad[:, [-1]])
        Nu_perm = self.Nu_perm(Re_perm, Pr_perm)
        h_perm = np.maximum(Nu_perm * lmbda_perm_rad[:, [-1]] / d_tube, h_min)

        if self.nu == 1:
            resist_mem = (
                self.r_f_perm[-1] * np.log(self.r_f_ret[0] / self.r_f_perm[-1])
            ) / self.lambda_mem
            factor_geom = self.r_f_ret[0] / self.r_f_perm[-1]
            U = 1.0 / (1.0 / h_perm + resist_mem + 1.0 / (factor_geom * h_ret))
        else:
            resist_mem = (self.r_f_perm[-1] - self.r_f_ret[0]) / self.lambda_mem
            U = 1.0 / (1.0 / h_ret + resist_mem + 1.0 / h_perm)
            factor_geom = 1.0

        ic_1 = {"a": (lmbda_perm_rad[:, [-1]], 0), "b": (U, U)}
        ic_2 = {"a": (0, factor_geom * lmbda_ret_rad[:, [0]]), "b": (-U, U)}
        interf_mat_perm, _, interf_mat_ret, _ = construct_interface_matrices(
            (T_perm.shape, T_ret.shape),
            (self.r_f_perm, self.r_f_ret),
            ic=(ic_1, ic_2),
            axis=1,
        )
        jac_cond_ic_perm = self.div_p_perm_rad @ (
            -lmbda_perm_rad_mat @ self.grad_bc_T_perm_rad
        )
        jac_cond_ic_ret = self.div_p_ret_rad @ (
            -lmbda_ret_rad_mat @ self.grad_bc_T_ret_rad
        )

        jac_cond += (
            jac_cond_ic_perm @ interf_mat_perm + jac_cond_ic_ret @ interf_mat_ret
        )

        cp_inv_mat = construct_coefficient_matrix(1.0 / cp)
        g_vect[:] = cp_inv_mat @ (jac_cond @ T.reshape((-1, 1)) + g_cond_bc)
        jac_cond = cp_inv_mat @ jac_cond

        return g, jac_cond, cp_inv_mat

    def _solve_cpT(self, c_old, T_old, dt, verbose=0, use_line_search=False):
        """Newton/line-search solve for species concentrations.

        Args:
            c_old: Previous concentration field
            T_old: Previous temperature field
            dt: Time step
            verbose: Verbosity level
            use_line_search: If True, use Armijo backtracking line search

        Returns:
            tuple: (g_norm, g_norm_init, g_p_norm, g_p_norm_init, success)
        """
        success = True
        cpT = self.cpT
        shape_cpT = cpT.shape
        ord = self.ord_norm
        factor_norm_c = self.factor_norm_c
        factor_norm_p = self.factor_norm_p
        cpT_vec = cpT.ravel()

        # Define norm function for combined concentration + pressure residual
        def compute_norms(g):
            g_c = np.linalg.norm(g[..., :-2].ravel(), ord=ord) * factor_norm_c
            g_p = np.linalg.norm(g[..., -2].ravel(), ord=ord) * factor_norm_p
            return g_c, g_p

        # Wrapper for residual evaluation with side effects
        def eval_residual(x_vec):
            cpT_vec[:] = x_vec
            self._update_velocity_fields(p=cpT[..., -2])
            g, _ = self._construct_g_cpT(c_old, T_old, dt, compute_jac=False)
            return g

        g_norm_init = None
        g_p_norm_init = None

        for k in range(self.num_concentration_iterations):
            g, jac = self._construct_g_cpT(c_old, T_old, dt, compute_jac=True)
            g_norm, g_p_norm = compute_norms(g)

            if k == 0:
                g_norm_init = g_norm
                g_p_norm_init = g_p_norm

            # Compute Newton step
            try:
                dcpT = -sla.spsolve(jac, g.reshape((-1, 1)))
            except RuntimeError:
                # Factorization failure (singular/ill-conditioned Jacobian).
                # Return inf so the outer loop treats this as divergence
                # and reduces dt.
                return np.inf, g_norm_init, np.inf, g_p_norm_init, False
            # MatrixRankWarning path: spsolve may return NaN without raising.
            if not np.all(np.isfinite(dcpT)):
                return np.inf, g_norm_init, np.inf, g_p_norm_init, False
            self.cnt_num_solves_cpT += 1

            # Apply step with optional line search
            if use_line_search:
                # Use armijo_line_search from solvers.py
                def norm_fn(g_arr):
                    g_c, g_p = compute_norms(g_arr.reshape(shape_cpT))
                    return max(g_c, g_p)  # Combined norm for line search

                x_new, g_new_norm, alpha, ls_success = armijo_line_search(
                    x=cpT_vec.copy(),
                    dx=dcpT,
                    g_norm=max(g_norm, g_p_norm),
                    residual_fn=lambda x: eval_residual(x).ravel(),
                    norm_fn=norm_fn,
                    armijo_coeff=ARMIJO_COEFF,
                    min_alpha=MIN_LINE_SEARCH_ALPHA,
                )
                cpT_vec[:] = x_new

                if not ls_success:
                    success = False
                    if verbose > 0:
                        warnings.warn(
                            f"Line search failed at iteration {k}: "
                            f"residual {g_new_norm:.2e}",
                            RuntimeWarning,
                        )
            else:
                # Full Newton step (no line search)
                cpT_vec[:] = cpT_vec + dcpT

            # Enforce physical bounds to prevent NaN propagation
            # Note: Concentrations are allowed to go negative during Newton iterations
            # The kinetics use safe power functions that handle negative values
            # Temperature must be positive and within reasonable bounds
            cpT[..., -1] = np.clip(cpT[..., -1], 200.0, 2000.0)
            # Pressure must be positive
            cpT[..., -2] = np.maximum(cpT[..., -2], 1e3)

            # Update velocity fields after step
            self._update_velocity_fields(p=cpT[..., -2])

            # Recompute residual norms for convergence check
            g, _ = self._construct_g_cpT(c_old, T_old, dt, compute_jac=False)
            g_norm, g_p_norm = compute_norms(g)

            # Check convergence
            if g_norm < max(self.rtol_c * g_norm_init, self.atol_c) and g_p_norm < max(
                self.rtol_p * g_p_norm_init, self.atol_p
            ):
                break

        return g_norm, g_norm_init, g_p_norm, g_p_norm_init, success

    def _compute_dt_cfl(self, cfl=CFL_INIT):
        """Return minimum time step avoiding CFL condition violation."""
        dz_cell = np.min(self.z_f[1:] - self.z_f[:-1])
        u_max = np.maximum(
            np.max(np.abs(self.u_perm_ax)), np.max(np.abs(self.u_ret_ax))
        )
        dt_cfl = cfl * dz_cell / u_max
        return dt_cfl

    def _compute_dt_chem_min(self):
        """Return minimum explicit chemical time step avoiding negative c."""
        c_ret = self.cpT[:, self.num_r_perm :, :-2]
        c_tot_ret = np.sum(c_ret, axis=-1, keepdims=True)
        _, p_ret = self._split_perm_and_ret(self.cpT[..., -2])
        p_over_c_tot = p_ret[..., np.newaxis] / c_tot_ret
        rates = self.kinetics(c_ret * p_over_c_tot)
        eps = EPS_CHEM_TIMESTEP
        dt_chem_local = np.where(
            rates < 0, np.maximum(c_ret, eps) / (-rates + eps), np.inf
        )
        dt_chem_min = np.min(dt_chem_local)
        return dt_chem_min

    def _solve_step(
        self,
        c_old: NDArray[np.float64],
        T_old: NDArray[np.float64],
        dt: float,
    ) -> SegregatedSolveResult:
        """Perform segregated Newton iterations for fixed reaction factor.

        This is the "corrector" part of the continuation scheme, solving
        the coupled concentration-pressure and temperature equations.

        Args:
            c_old: Previous concentration field (num_z, num_r, num_c).
            T_old: Previous temperature field (num_z, num_r).
            dt: Pseudo-time step size.

        Returns:
            SegregatedSolveResult with convergence info.
        """
        g_norm_init = None
        g_p_norm_init = None
        success = False

        for j in range(self.num_newton_iterations):
            # Solve concentration-pressure system
            g_norm, g_norm_start, g_p_norm, g_p_norm_start, success = (
                self._solve_cpT(c_old, T_old, dt)
            )
            logger.debug("Newton iteration %d: g_norm = %.4e", j, g_norm)

            if j == 0:
                g_norm_init = g_norm_start
                g_p_norm_init = g_p_norm_start

            # Check for divergence
            if ~np.isfinite(g_norm) or (g_norm > 10 * g_norm_init):
                return SegregatedSolveResult(
                    converged=False,
                    num_iterations=j + 1,
                    g_norm=g_norm,
                    g_p_norm=g_p_norm,
                    g_norm_init=g_norm_init,
                    g_p_norm_init=g_p_norm_init,
                )

            # Check convergence
            if g_norm < max(self.rtol * g_norm_init, self.atol):
                return SegregatedSolveResult(
                    converged=success,
                    num_iterations=j + 1,
                    g_norm=g_norm,
                    g_p_norm=g_p_norm,
                    g_norm_init=g_norm_init,
                    g_p_norm_init=g_p_norm_init,
                )

        # Max iterations reached - return last success status
        return SegregatedSolveResult(
            converged=success,  # Original code returned success, not False
            num_iterations=self.num_newton_iterations,
            g_norm=g_norm,
            g_p_norm=g_p_norm,
            g_norm_init=g_norm_init,
            g_p_norm_init=g_p_norm_init,
        )

    def solve(
        self,
        num_timesteps: Optional[int] = None,
        dt_init: Optional[float] = None,
        use_adaptive_dt: bool = True,
        steady_state_tol: float = 1e-3,
        n_plateau_required: int = 20,
        plateau_tol: float = 0.02,
        verbose: int = 0,
        **kwargs: Any,
    ) -> bool:
        """Solve the steady-state reactor problem.

        Uses adaptive time-stepping for robust and efficient convergence.
        The pseudo-transient term (1/dt) regularizes the Newton iteration.
        Small dt ensures convergence but is slow; large dt is fast but may
        diverge. Adaptive dt starts small and increases as solution improves.

        Args:
            num_timesteps: Maximum number of pseudo-transient steps.
            dt_init: Initial time step. Defaults to self.dt_init.
            use_adaptive_dt: If True, adapt dt based on convergence quality.
            steady_state_tol: Relative tolerance for steady-state detection.
            verbose: Verbosity level (0=silent, 1=summary, 2=detailed).
            **kwargs: Additional arguments passed to solve methods.

        Returns:
            True if converged to steady state, False otherwise.
        """
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        if dt_init is None:
            dt_init = getattr(self, 'dt_init', 1e-3)

        self.cnt_num_solves_cpT = 0
        dt = dt_init
        dt_min = getattr(self, 'dt_min', 1e-16)
        dt_max = getattr(self, 'dt_max', 1e3)
        dt_increase = getattr(self, 'dt_increase_factor', 2.0)
        dt_decrease = getattr(self, 'dt_decrease_factor', 0.5)
        threshold = getattr(self, 'adaptive_dt_threshold', 0.5)

        g_ss_norm_prev = None
        g_ss_norm_0 = None   # initial SS residual, used for relative convergence
        g_ss_norm = None
        g_ss_norm_best = None   # lowest g_ss seen — best physical state
        cpT_best = None
        is_converged = False
        n_increasing = 0        # consecutive steps with g_ss growing
        n_plateau_steps = 0     # consecutive steps where |reduction-1| < plateau_tol

        # Compute baseline SS residual before any steps.
        # Used to detect if a first accepted step produced a corrupted state
        # (e.g. when dt_init is too large and the Newton step jumps to a
        # non-physical region without producing NaN).
        g_baseline, _ = self._construct_g_cpT(
            self.cpT[..., :-2].copy(), self.cpT[..., -1].copy(), 1e10, compute_jac=False
        )
        g_ss_norm_baseline  = np.linalg.norm(g_baseline)

        if verbose >= 1:
            print(f"Starting adaptive dt solve: dt_init={dt_init:.2e}, "
                  f"dt_range=[{dt_min:.2e}, {dt_max:.2e}]")

        for i in range(num_timesteps):
            T_old = self.cpT[..., -1].copy()
            c_old = self.cpT[..., :-2].copy()
            cpT_backup = self.cpT.copy()

            # Attempt solve with current dt
            result = self._solve_step(c_old, T_old, dt)
            g_norm = result.g_norm

            # --- Reject if factorization/NaN failure (g_norm=inf) ---
            if not np.isfinite(g_norm):
                self.cpT = cpT_backup
                self._construct_darcy_matrices()
                self._update_velocity_fields(self.cpT[..., -2])
                dt = max(dt * dt_decrease, dt_min)
                n_plateau_steps = 0
                if verbose >= 2:
                    print(f"  Step {i}: REJECTED (NaN/singular), dt -> {dt:.2e}")
                continue

            # Compute actual steady-state residual (without transient term)
            # dt=1e10 makes the transient term (c-c_old)/dt negligible.
            # Note: _construct_g_cpT reads self.cpT directly for concentrations/pressure/T;
            # the c_old/T_old args only affect the negligible transient term at dt=1e10.
            g_ss, _ = self._construct_g_cpT(c_old, T_old, 1e10, compute_jac=False)
            g_ss_norm = np.linalg.norm(g_ss)

            # --- Reject if SS residual increased dramatically vs previous accepted step ---
            # Do NOT compare against baseline for step 0: the baseline is computed with
            # cold (un-warmed) velocity fields, while after _solve_step the velocity fields
            # have been updated by many Picard steps. These are incomparable measurements
            # and the comparison causes spurious step-0 rejection even for tiny dt.
            if g_ss_norm_prev is not None and g_ss_norm > 10 * g_ss_norm_prev:
                self.cpT = cpT_backup
                self._construct_darcy_matrices()
                self._update_velocity_fields(self.cpT[..., -2])
                dt = max(dt * dt_decrease, dt_min)
                n_increasing = 0
                n_plateau_steps = 0
                if verbose >= 2:
                    print(f"  Step {i}: REJECTED (g_ss grew), dt -> {dt:.2e}, "
                          f"||g_ss||={g_ss_norm:.2e}")
                continue

            # Track best state so far (lowest g_ss)
            if g_ss_norm_best is None or g_ss_norm < g_ss_norm_best:
                g_ss_norm_best = g_ss_norm
                cpT_best = self.cpT.copy()

            # Record initial SS norm once (after first accepted step)
            if g_ss_norm_0 is None:
                g_ss_norm_0 = max(g_ss_norm, 1e-30)

            # Adapt dt based on steady-state residual progress, and track plateau
            if use_adaptive_dt and g_ss_norm_prev is not None:
                reduction = g_ss_norm / g_ss_norm_prev

                if reduction > 1.05:
                    # Residual growing fast — reduce dt.
                    dt = max(dt * dt_decrease, dt_min)
                    n_increasing += 1
                    if verbose >= 2:
                        print(f"  Step {i}: poor (red={reduction:.2f}), dt -> {dt:.2e}, "
                              f"||g_ss||={g_ss_norm:.2e}")
                elif reduction > 1.0:
                    # Residual drifting upward slightly — hold dt.
                    n_increasing += 1
                    if verbose >= 2:
                        print(f"  Step {i}: drift (red={reduction:.2f}), dt={dt:.2e}, "
                              f"||g_ss||={g_ss_norm:.2e}")
                elif reduction < threshold:
                    # Good reduction — increase dt aggressively (dt_increase²).
                    dt = min(dt * dt_increase ** 2, dt_max)
                    n_increasing = 0
                    if verbose >= 2:
                        print(f"  Step {i}: good (red={reduction:.2f}), dt -> {dt:.2e}, "
                              f"||g_ss||={g_ss_norm:.2e}")
                else:
                    # threshold ≤ reduction ≤ 1.0 — moderate progress, increase dt normally.
                    dt = min(dt * dt_increase, dt_max)
                    n_increasing = 0
                    if verbose >= 2:
                        print(f"  Step {i}: ok (red={reduction:.2f}), dt -> {dt:.2e}, "
                              f"||g_ss||={g_ss_norm:.2e}")

                # Track plateau: sole authority on n_plateau_steps.
                # |reduction - 1| < plateau_tol means the residual is essentially
                # stationary — the Picard fixed point has been reached.
                if abs(reduction - 1.0) < plateau_tol:
                    n_plateau_steps += 1
                else:
                    n_plateau_steps = 0

                # If consistently drifting away from best, restore best state.
                if n_increasing >= 10 and cpT_best is not None:
                    self.cpT = cpT_best.copy()
                    self._update_velocity_fields(self.cpT[..., -2])
                    g_ss_norm = g_ss_norm_best
                    g_ss_norm_prev = g_ss_norm_best
                    n_increasing = 0
                    n_plateau_steps = 0
                    if verbose >= 2:
                        print(f"  Step {i}: Restored best state, "
                              f"||g_ss||={g_ss_norm:.2e}")
            elif verbose >= 2:
                print(f"  Step {i}: dt={dt:.2e}, ||g_ss||={g_ss_norm:.2e}")

            # Check steady-state convergence.
            rel_residual = g_ss_norm / g_ss_norm_0

            # Relative-reduction criterion — only valid once dt is in the large-dt regime,
            # to prevent false convergence while the pseudo-transient term still dominates.
            if rel_residual < steady_state_tol and dt >= 0.1 * dt_max:
                is_converged = True
                if verbose >= 1:
                    print(f"  Step {i}: CONVERGED, dt={dt:.2e}, "
                          f"||g_ss||={g_ss_norm:.2e}, ||g_ss||/||g_ss_0||={rel_residual:.2e}")
                break

            # Plateau criterion — Picard fixed point: best achievable with one Picard
            # step per Newton iteration. Declare convergence once stable at large dt.
            if n_plateau_steps >= n_plateau_required and dt >= 0.1 * dt_max:
                is_converged = True
                if verbose >= 1:
                    print(f"  Step {i}: CONVERGED (Picard plateau), dt={dt:.2e}, "
                          f"||g_ss||={g_ss_norm:.2e}")
                break

            g_ss_norm_prev = g_ss_norm

        # Always restore the best (lowest g_ss) state found, even if the final
        # state drifted away from it after the minimum.
        if cpT_best is not None and (g_ss_norm is None or g_ss_norm_best < g_ss_norm):
            self.cpT = cpT_best
            self._update_velocity_fields(self.cpT[..., -2])
            g_ss_norm = g_ss_norm_best
            if verbose >= 1:
                print(f"  Restored best state: ||g_ss_best||={g_ss_norm_best:.2e}")

        if verbose >= 1 and not is_converged:
            print(f"  Did not converge after {num_timesteps} steps, "
                  f"final ||g_ss||={g_ss_norm:.2e}, dt={dt:.2e}")

        return is_converged

    def _solve_adaptive_react(self, dt=None, c_old=None, T_old=None, verbose=0):
        """Solve using adaptive continuation on reaction rate scaling factor.

        Uses predictor-corrector scheme with secant extrapolation. Step size
        adapts based on Newton convergence: increases after success, decreases
        after failure.

        Args:
            dt: Pseudo-time step size. Defaults to self.dt.
            c_old: Previous concentration field for transient term.
            T_old: Previous temperature field for transient term.
            verbose: Verbosity level (0=quiet, 1=warnings, 2=progress).

        Returns:
            True if converged at factor_react=1.0, False otherwise.
        """
        # --- Initialization ---
        if dt is None:
            dt = self.dt

        conv_factor_min = self.conv_factor_min
        g_norm, g_p_norm = None, None

        # History for predictor step (current, previous)
        cpT_prev = self.cpT.copy()
        cpT_prev_prev= None
        factor_react_prev, factor_react_prev_prev = None, None

        # --- Step 2: Main Continuation Loop ---
        is_first_step = True
        is_converged = False
        if self.factor_react is None:
            self.factor_react = 1.0
            factor_react_max = 1.0
        else:
            factor_react_max = self.factor_react
        dfactor_react = min(self.dfactor_react_init, factor_react_max)

        while True:
            g_norm, g_p_norm = None, None

            # --- Predictor Step ---
            if factor_react_prev_prev is not None:
                # Use a first-order (secant) predictor for a better initial guess
                d_factor_hist = factor_react_prev - factor_react_prev_prev
                if d_factor_hist > 1e-9:  # Avoid division by zero on retry
                    step_ratio = (self.factor_react - factor_react_prev) / d_factor_hist
                    self.cpT = cpT_prev + (cpT_prev - cpT_prev_prev) * step_ratio
#                    self.cpT = np.maximum(
#                        self.cpT, 0
#                    )  # Ensure concentrations are non-negative
                else:
                    self.cpT = cpT_prev.copy()
                self.kinetics.set_T_and_p(
                    T=self.cpT[..., -1][:, self.num_r_perm :], p=self.cpT[:, self.num_r_perm :, -2]
                )
                self._construct_darcy_matrices()
                self._update_velocity_fields(p=self.cpT[..., -2])
            elif not is_first_step:
                # Use a zero-order predictor (the last solution) for the first step
                self.cpT = cpT_prev.copy()
                self.kinetics.set_T_and_p(
                    T=self.cpT[..., -1][:, self.num_r_perm :], p=self.cpT[:, self.num_r_perm :, -2]
                )
                self._construct_darcy_matrices()
                self._update_velocity_fields(p=self.cpT[..., -2])

            # --- Corrector Step ---
            if verbose > 1:
                logger.info(
                    "Attempting factor_react = %.4f (step size = %.4f)...",
                    self.factor_react,
                    dfactor_react,
                )
            result = self._solve_step(c_old, T_old, dt)
            num_iters = result.num_iterations
            g_norm, g_p_norm = result.g_norm, result.g_p_norm
            is_converged = result.converged
            conv_factor, conv_factor_p = (
                result.convergence_factor,
                result.convergence_factor_p,
            )
            is_converging = np.isfinite(g_norm) and ((conv_factor < conv_factor_min) or is_converged)
            if is_converged and self.factor_react == factor_react_max:
                break
            # --- Adapt Step Size ---
            if is_converging:
                if verbose > 1:
                    logger.info(
                        "Converged in %d iterations: g_norm=%.4e, conv_factor=%.4e, "
                        "g_p_norm=%.4e, conv_factor_p=%.4e. Increasing step size.",
                        num_iters,
                        g_norm,
                        conv_factor,
                        g_p_norm,
                        conv_factor_p,
                    )
                # Update history for the next predictor step
                if not is_first_step:
                    cpT_prev_prev = cpT_prev
                    factor_react_prev_prev = factor_react_prev
                    cpT_prev = self.cpT.copy()
                    factor_react_prev = self.factor_react
                    # Increase step size
                    if is_converged:
                        dfactor_react *= self.dfactor_react_increase
                    self.factor_react = min(
                        self.factor_react + dfactor_react, factor_react_max
                    )
            else:
                if verbose > 1:
                    logger.warning(
                        "Failed after %d iterations. Restoring state and reducing step size.",
                        num_iters,
                    )

                # Restore previous is_convergedful state
                #self.cpT = cpT_prev
                if is_first_step:
                    self.factor_react = 0.0
                    is_first_step = False
                elif factor_react_prev is None:
                    if verbose > 0:
                        warnings.warn(
                            f"No convergence even with factor_react = {self.factor_react}",
                            RuntimeWarning,
                        )
                    break
                else:
                    self.factor_react = factor_react_prev
                    # Decrease step size and retry from the last good point
                    dfactor_react *= self.dfactor_react_decrease
                    if dfactor_react < self.dfactor_react_min:
                        break
                    self.factor_react = min(
                        self.factor_react + dfactor_react, factor_react_max
                    )
        if is_converged:
            if verbose > 1:
                logger.info(
                    "Continuation completed. Final solution at factor_react = 1.0 reached."
                )
        else:
            if verbose > 0:
                warnings.warn(
                    f"Continuation failed: final factor_react = {self.factor_react}",
                    RuntimeWarning,
                )
        return is_converged

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
        # --- Initialization ---
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

        # History for predictor step (current, previous)
        cpT_prev = self.cpT.copy()
        t = 0.0
        is_converged = False
        while t < t_final or not is_converged:
            try:
                result = self._solve_step(c_old, T_old, dt)
                is_converged = result.converged
                is_converging = is_converged
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
                    self.cpT = cpT_prev.copy()
                    dt = max(dt * dt_factor_decrease, dt_min)
                    self.kinetics.set_T_and_p(
                        T=self.cpT[..., -1][:, self.num_r_perm :],
                        p=self.cpT[:, self.num_r_perm :, -2],
                    )
                    self._construct_darcy_matrices()
                    self._update_velocity_fields(p=self.cpT[..., -2])
                    if verbose > 1:
                        logger.info("Reduced dt to %.4e", dt)
        if verbose > 1 and not is_converged:
            logger.warning("Failed: could not converge at t = %.4e", t)
        return is_converged

    def _compute_fluxes_diff(self, c=None, T=None, p=None):
        """Compute diffusive + permeation species fluxes (axis & radial)."""
        shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
        shape_c_perm = (self.num_z, self.num_r_perm, self.num_c)
        if c is None:
            c = self.cpT[..., :-2]
        if T is None:
            T = self.cpT[..., -1]
        if p is None:
            p = self.cpT[..., -2]
        c_perm, c_ret = self._split_perm_and_ret(c)
        T_perm, T_ret = self._split_perm_and_ret(T)
        p_perm, p_ret = self._split_perm_and_ret(p)
        c_vect = c.reshape((-1, 1))

        c_tot_ret_safe = np.maximum(np.abs(np.sum(c_ret, axis=-1, keepdims=True)), 1e-10)
        y_ret = c_ret / c_tot_ret_safe  # Mole fractions
        diff_field_ret = 0.0*self.correlation.diffusion(y_ret, T_ret, p_ret)
        diff_field_ret_ax = interp_cntr_to_stagg(
            diff_field_ret, x_f=self.z_f, x_c=self.z_c, axis=0
        )
        diff_matrix_ret_ax = construct_coefficient_matrix(
            diff_field_ret_ax, shape_c_ret, axis=0
        )
        diff_field_ret_rad = interp_cntr_to_stagg(
            diff_field_ret, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1
        )
        diff_matrix_ret_rad = construct_coefficient_matrix(
            diff_field_ret_rad, shape_c_ret, axis=1
        )

        c_tot_perm_safe = np.maximum(np.abs(np.sum(c_perm, axis=-1, keepdims=True)), 1e-10)
        y_perm = c_perm / c_tot_perm_safe  # Mole fractions
        diff_field_perm = 0.0*self.correlation.diffusion(y_perm, T_perm, p_perm)
        diff_field_perm_ax = interp_cntr_to_stagg(
            diff_field_perm, x_f=self.z_f, x_c=self.z_c, axis=0
        )
        diff_matrix_perm_ax = construct_coefficient_matrix(
            diff_field_perm_ax, shape_c_perm, axis=0
        )
        diff_field_perm_rad = interp_cntr_to_stagg(
            diff_field_perm, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1
        )
        diff_matrix_perm_rad = construct_coefficient_matrix(
            diff_field_perm_rad, shape_c_perm, axis=1
        )

        fluxes_ret_ax = (
            -diff_matrix_ret_ax @ (self.grad_c_ret_ax @ c_vect + self.grad_bc_c_ret_ax)
        ).reshape((self.num_z + 1, self.num_r_ret, self.num_c))
        fluxes_ret_rad = (
            -diff_matrix_ret_rad @ (self.grad_c_ret_rad @ c_vect)
        ).reshape((self.num_z, self.num_r_ret + 1, self.num_c))
        fluxes_perm_ax = (
            -diff_matrix_perm_ax
            @ (self.grad_c_perm_ax @ c_vect + self.grad_bc_c_perm_ax)
        ).reshape((self.num_z + 1, self.num_r_perm, self.num_c))
        fluxes_perm_rad = (
            -diff_matrix_perm_rad @ (self.grad_c_perm_rad @ c_vect)
        ).reshape((self.num_z, self.num_r_perm + 1, self.num_c))

        if T_perm.ndim > 1:
            T_perm_mem, _ = compute_boundary_values(
                T_perm, self.r_f_perm, self.r_c_perm, bc=None, axis=1, bound_id=1
            )
            T_ret_mem, _ = compute_boundary_values(
                T_ret, self.r_f_ret, self.r_c_ret, bc=None, axis=1, bound_id=0
            )
            P_perm_mem = self.Rg * T_perm_mem[:, 0, np.newaxis] * self.perm
            P_ret_mem = self.Rg * T_ret_mem[:, 0, np.newaxis] * self.perm
        else:
            P_perm_mem = np.broadcast_to(
                (self.Rg * T_perm * self.perm).reshape((1, 1, -1)),
                (self.num_z, self.num_c),
            )
            P_ret_mem = np.broadcast_to(
                (self.Rg * T_ret * self.perm).reshape((1, 1, -1)),
                (self.num_z, self.num_c),
            )

        c_perm_mem, _ = compute_boundary_values(
            c_perm, self.r_f_perm, self.r_c_perm, bc=None, axis=1, bound_id=1
        )
        c_ret_mem, _ = compute_boundary_values(
            c_ret, self.r_f_ret, self.r_c_ret, bc=None, axis=1, bound_id=0
        )

        fluxes_perm_rad[:, -1, :] = (
            P_perm_mem * c_perm_mem[:, 0, :] - P_ret_mem * c_ret_mem[:, 0, :]
        )
        factor_geom = (self.r_f_perm[-1] / self.r_f_ret[0]) ** self.nu
        fluxes_ret_rad[:, 0, :] = factor_geom * fluxes_perm_rad[:, -1, :]

        return fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad

    def _compute_fluxes_conv(self, c=None):
        """Compute convective species fluxes using upwind TVD reconstructions."""
        if c is None:
            c = self.cpT[..., :-2]

        if self.is_counter_current:
            bc_ret_ax = (self.BC_NEUMANN_HOM, self.BC_NONE)
        else:
            bc_ret_ax = (self.BC_NONE, self.BC_NEUMANN_HOM)

        c_perm = c[:, 0 : self.num_r_perm, :]
        u_perm_ax = self.u_perm_ax[..., np.newaxis]
        u_perm_rad = self.u_perm_rad[..., np.newaxis]
        c_perm_ax, _ = interp_cntr_to_stagg_tvd(
            c_perm,
            self.z_f,
            self.z_c,
            bc=(self.BC_NONE, self.BC_NEUMANN_HOM),
            v=u_perm_ax,
            tvd_limiter=upwind,
            axis=0,
        )
        fluxes_perm_ax = u_perm_ax * c_perm_ax
        c_perm_rad, _ = interp_cntr_to_stagg_tvd(
            c_perm,
            self.r_f_perm,
            self.r_c_perm,
            bc=(self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM),
            v=u_perm_rad,
            tvd_limiter=upwind,
            axis=1,
        )
        fluxes_perm_rad = u_perm_rad * c_perm_rad

        c_ret = c[:, self.num_r_perm :, :]
        u_ret_ax = self.u_ret_ax[..., np.newaxis]
        u_ret_rad = self.u_ret_rad[..., np.newaxis]
        c_ret_ax, _ = interp_cntr_to_stagg_tvd(
            c_ret,
            self.z_f,
            self.z_c,
            bc=bc_ret_ax,
            v=u_ret_ax,
            tvd_limiter=upwind,
            axis=0,
        )
        fluxes_ret_ax = u_ret_ax * c_ret_ax
        c_ret_rad, _ = interp_cntr_to_stagg_tvd(
            c_ret,
            self.r_f_ret,
            self.r_c_ret,
            bc=(self.BC_NEUMANN_HOM, self.BC_NEUMANN_HOM),
            v=u_ret_rad,
            tvd_limiter=upwind,
            axis=1,
        )
        fluxes_ret_rad = u_ret_rad * c_ret_rad

        fluxes_perm_ax[0, :, :] = self.flux_perm_in[0, :, :]
        if self.is_counter_current:
            fluxes_ret_ax[-1, :, :] = -self.flux_ret_in[0, :, :]
        else:
            fluxes_ret_ax[0, :, :] = self.flux_ret_in[0, :, :]

        return fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad

    def compute_flows(self):
        """Integrate fluxes to obtain net axial inlet/outlet & membrane transfer."""
        fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad = (
            self._compute_fluxes_diff()
        )
        (
            fluxes_conv_ret_ax,
            fluxes_conv_ret_rad,
            fluxes_conv_perm_ax,
            fluxes_conv_perm_rad,
        ) = self._compute_fluxes_conv()
        fluxes_ret_ax += fluxes_conv_ret_ax
        fluxes_ret_rad += fluxes_conv_ret_rad
        fluxes_perm_ax += fluxes_conv_perm_ax
        fluxes_perm_rad += fluxes_conv_perm_rad

        # Compute cross-sectional areas for radial and axial directions
        dr_sq_ret = (self.r_f_ret[1:] ** 2 - self.r_f_ret[:-1] ** 2).reshape((-1, 1))
        dr_sq_perm = (self.r_f_perm[1:] ** 2 - self.r_f_perm[:-1] ** 2).reshape((-1, 1))
        dz = (self.z_f[1:] - self.z_f[:-1]).reshape((-1, 1))

        flows_ret_ax = np.pi * np.sum(fluxes_ret_ax * dr_sq_ret, axis=1)
        flows_ret_mem = (
            2.0 * np.pi * self.r_f_ret[0] * np.sum(fluxes_ret_rad[:, 0, :] * dz, axis=0)
        )
        flows_perm_ax = np.pi * np.sum(fluxes_perm_ax * dr_sq_perm, axis=1)
        flows_perm_mem = (
            2.0
            * np.pi
            * self.r_f_perm[-1]
            * np.sum(fluxes_perm_rad[:, -1, :] * dz, axis=0)
        )

        return flows_ret_ax, flows_ret_mem, flows_perm_ax, flows_perm_mem

    def info(self):
        """
        Print information about the membrane reactor.
        """
        vol_ret = (
            np.pi
            * (self.r_f_ret[-1] ** 2 - self.r_f_ret[0] ** 2)
            * (self.z_f[-1] - self.z_f[0])
        )
        vol_perm = (
            np.pi
            * (self.r_f_perm[-1] ** 2 - self.r_f_perm[0] ** 2)
            * (self.z_f[-1] - self.z_f[0])
        )
        flow_vol_ret = self.F_ret_in * self.Rg * self.T_ret_in / self.p_ret_out
        flow_vol_perm = self.F_perm_in * self.Rg * self.T_perm_in / self.p_perm_out
        logger.info("Residence time retentate side: %.4f s", vol_ret / flow_vol_ret)
        logger.info("Residence time permeate side: %.4f s", vol_perm / flow_vol_perm)
