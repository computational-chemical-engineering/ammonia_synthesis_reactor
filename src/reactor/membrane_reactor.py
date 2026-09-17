"""2D axisymmetric membrane reactor model for ammonia synthesis.

Defines :class:`MembraneReactor`, which assembles the coupled
concentration/pressure/temperature residual and Jacobian on a non-uniform
(z, r) grid using pymrm operators, together with the Newton and adaptive
pseudo-transient solvers (plus escalation strategies such as temperature
continuation) used to reach steady state, and the result types
:class:`SolveResult` and :class:`SteadyStateSolveStatus`.
"""

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

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

from .ammonia_synthesis_kinetics import AmmoniaSynthesisKinetics
from .config import ReactorConfig, get_membrane_permeances
from .gas_mixture_correlations import GasMixtureCorrelations
from .mesh import ReactorMesh
from .physics import (
    BC_DIRICHLET,
    BC_DIRICHLET_HOM,
    BC_NEUMANN,
    BC_NEUMANN_HOM,
    BC_NONE,
    compute_inlet_flux_permeate,
    compute_inlet_flux_retentate,
    compute_packed_bed_permeability,
    compute_permeate_permeability,
    get_axial_bcs_for_flow,
    make_dirichlet_bc,
)
from .solvers import ContinuationConfig, NewtonConfig, armijo_line_search
from .numerical_safety import RecoverableNumericalError
from .convergence import (
    Classification,
    ConvergenceMonitor,
    steady_state_weights,
    weighted_rms,
)

# Module-level logger
logger = logging.getLogger(__name__)

# =============================================================================
# Solver result types
# =============================================================================


@dataclass
class SolveResult:
    """Result from the coupled concentration-pressure Newton solve."""

    converged: bool
    num_iterations: int
    g_norm: float  # Newton residual norm
    g_norm_init: float  # Initial residual norm

    @property
    def convergence_factor(self) -> float:
        """Unified Newton convergence factor."""
        return self.g_norm / self.g_norm_init if self.g_norm_init > 0 else 0.0

    def as_tuple(self) -> Tuple[int, float, float, bool, float, float]:
        """Return legacy tuple format for backwards compatibility."""
        return (
            self.num_iterations,
            self.g_norm,
            self.converged,
            self.convergence_factor,
        )

    def __iter__(self):
        """Allow legacy tuple unpacking in diagnostics and ad hoc scripts."""
        return iter(self.as_tuple())


@dataclass
class SteadyStateSolveStatus:
    """Structured status from the adaptive pseudo-transient steady-state solve."""

    converged: bool
    num_steps_attempted: int
    num_steps_accepted: int
    final_dt: float
    steady_state_norm: Optional[float]
    initial_steady_state_norm: Optional[float]
    best_steady_state_norm: Optional[float]
    baseline_steady_state_norm: Optional[float]
    last_failure_message: Optional[str]
    # Steady-state stopping machinery (weighted norm + classifier) -----
    # Which norm steady_state_norm was measured in ("absolute"|"weighted").
    norm_kind: str = "absolute"
    # One of reactor.convergence.OUTCOMES. "converged" and "floored" are
    # both accepted results; "floored" means the iteration stagnated at its
    # residual floor with stagnant KPIs (the threshold was unreachable, the
    # state stopped changing).
    outcome: str = "failed"
    # Evidence dict from the classifier for floored/oscillatory/diverging.
    outcome_evidence: Optional[dict] = None
    # Pseudo-transient oscillation evidence (limit-cycle orbit) if one was
    # observed. Pseudo-time is NOT physical time: this classifies, it does
    # not quantify the physical cycle.
    oscillation: Optional[dict] = None
    # The state was reached by the temperature-continuation ladder after the
    # march orbited it: the steady state is (very likely) dynamically
    # unstable. Confirm with reactor.stability.leading_eigenvalues.
    used_temperature_continuation: bool = False
    dynamically_unstable: bool = False


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
STEADY_STATE_DT = 1e10  # Large dt used to evaluate the steady-state residual
STEADY_STATE_PLATEAU_REQUIRED = 20
STEADY_STATE_PLATEAU_TOL = 0.02

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
        config: Optional[ReactorConfig] = None,
        c: Optional[NDArray[np.float64]] = None,
        p: Optional[NDArray[np.float64]] = None,
        T: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> None:
        """Construct reactor, load defaults, apply overrides, allocate fields.

        Args:
            config_file: Optional JSON file overriding defaults.
            config: Optional pre-built ReactorConfig. When provided, config_file
                and kwargs must be omitted.
            c: Initial concentration field (num_z, num_r, num_c).
            p: Initial pressure field (num_z, num_r).
            T: Initial temperature field (num_z, num_r).
            **kwargs: Explicit overrides of default parameters when config is
                not provided.

        Side Effects:
            Creates spatial discretization, initializes kinetics & Jacobian
            matrices, performs a non-reactive pressure/velocity initialization.
        """
        self._init_config(config_file=config_file, config=config, kwargs=kwargs)
        self._init_derived_parameters()
        self._init_fields(c, p, T)
        _, T_ret = self._split_perm_and_ret(self.cpT[..., -1])
        _, p_ret = self._split_perm_and_ret(self.cpT[..., -2])

        self.kinetics = AmmoniaSynthesisKinetics(
            self.species, T=T_ret, p=p_ret, rho_b=self.rho_b, rho_c=self.rho_c
        )
        self._init_jac()
        self.factor_norm = (self.cpT.size) ** (-1.0 / self.ord_norm)
        self.last_solve_status = None

        dt_cfl = self._compute_dt_cfl(cfl=CFL_INIT)
        factor_react = self.factor_react
        self.factor_react = 0.0
        self.solve(dt=dt_cfl, num_timesteps=2)
        self.factor_react = factor_react

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


    def _init_config(
        self,
        config_file: Optional[str],
        config: Optional[ReactorConfig],
        kwargs: Dict[str, Any],
    ):
        """Initialize configuration from either object injection or merge path.

        Two mutually exclusive initialization modes are supported:
        1. Injected config object (`config` argument)
        2. Merged config from defaults/file/kwargs

        Args:
            config_file (str|None): Path to JSON configuration file.
            config (ReactorConfig|None): Pre-built validated config object.
            kwargs (dict): Explicit parameter overrides.

        Side Effects:
            Sets all configuration parameters as instance attributes.
            Stores merged dict as self.param_dict for serialization.
        """
        if config is not None:
            if not isinstance(config, ReactorConfig):
                raise TypeError(
                    "config must be an instance of ReactorConfig when provided"
                )
            if config_file is not None:
                raise ValueError(
                    "Provide either config or config_file/kwargs, not both"
                )
            if kwargs:
                raise ValueError(
                    "Provide either config or kwargs, not both"
                )
            self._config = config
        else:
            # Create ReactorConfig (handles merging and validation)
            self._config = ReactorConfig.from_defaults(config_file, **kwargs)

        # Store param_dict for serialization (legacy)
        self.param_dict = self._config.to_dict()

    def _init_derived_parameters(self):
        """Compute geometry-dependent counts, densities, permeances & inlets.

        Populates:
            Mesh object, Arrhenius membrane permeance arrays (self.P0, self.EA),
            inlet flux distributions, Reynolds/Schmidt groups.
        """
        self.correlation = GasMixtureCorrelations(self.species, self.database)

        # Create mesh (attributes accessed via __getattr__)
        self._mesh = ReactorMesh(self._config)

        # ----- membrane permeances -----
        self.P0, self.EA = get_membrane_permeances(
            species=self.species,
            config=self._config,
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

        # Inflow composition used when backward flow occurs at the retentate outlet.
        # Nearly pure N2 with a trace of NH3 to prevent kinetics singularities.
        _nh3_trace = 1e-4  # mole fraction
        self.inflow_conc_ret_backflow = (
            self.p_ret_out
            / (self.Rg * self.T_ret_in)
            * np.array([[[0.0, 1.0 - _nh3_trace, _nh3_trace]]])
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

    def _reaction_partial_pressures(self, c_ret, T_ret, p_ret):
        """Return thermodynamic partial pressures used by the kinetics model."""
        if self.kinetics_uses_auxiliary_pressure:
            c_tot_ret = np.sum(c_ret, axis=-1, keepdims=True)
            c_tot_ret_safe = np.maximum(np.abs(c_tot_ret), 1e-10)
            return c_ret * (p_ret[..., np.newaxis] / c_tot_ret_safe)
        return c_ret * (self.Rg * T_ret)[..., np.newaxis]

    def _evaluate_reaction_heat_source(
        self,
        c_ret,
        T_ret,
        p_ret,
        enthalpies=None,
        cp_inv=None,
    ):
        """Return the retentate reaction heat source contribution to g_T."""
        if enthalpies is None:
            enthalpies = self.correlation.species_enthalpies(T_ret)
        if cp_inv is None:
            _, _, cp_inv_mat = self._construct_g_T_cond(self.cpT[..., -1])
            cp_inv = cp_inv_mat.data.reshape(self.cpT[..., -1].shape)[:, self.num_r_perm :]
        return (
            np.sum(
                self.factor_react
                * self.kinetics(
                    self._reaction_partial_pressures(c_ret, T_ret, p_ret),
                    T_ret,
                )
                * enthalpies,
                axis=-1,
            )
            * cp_inv
        )[..., np.newaxis]

    def diagnose_jac_Tc_react(self, seed=0, fd_scale=1e-7):
        """Diagnose the J_Tc reaction-heat Jacobian on the current reactor state."""
        c = self.cpT[..., :-2]
        p = self.cpT[..., -2]
        T = self.cpT[..., -1]
        c_ret = c[:, self.num_r_perm :, :].copy()
        _, p_ret = self._split_perm_and_ret(p)
        T_ret = T[:, self.num_r_perm :]

        enthalpies = self.correlation.species_enthalpies(T_ret)
        _, _, cp_inv_mat = self._construct_g_T_cond(T)
        cp_inv = cp_inv_mat.data.reshape(T.shape)[:, self.num_r_perm :]

        def heat_source(c_var):
            """Evaluate the reaction heat source for perturbed concentrations with the other fields frozen."""
            return self._evaluate_reaction_heat_source(
                c_var,
                T_ret,
                p_ret,
                enthalpies=enthalpies,
                cp_inv=cp_inv,
            )

        linear_weights = np.arange(1.0, self.num_c + 1.0).reshape((1, 1, -1))
        linear_fun = lambda c_var: np.sum(c_var * linear_weights, axis=-1, keepdims=True)
        _, jac_linear = self.numjac_Tc(linear_fun, c_ret)
        jac_linear_exact = construct_coefficient_matrix(
            np.broadcast_to(linear_weights, c_ret.shape),
            shape=((self.num_z, self.num_r_ret, 1), c_ret.shape),
        )

        rng = np.random.default_rng(seed)
        direction = rng.standard_normal(c_ret.size)
        direction /= np.linalg.norm(direction)
        step = fd_scale * max(1.0, float(np.mean(np.abs(c_ret))))

        heat_base, jac_numjac = self.numjac_Tc(heat_source, c_ret)
        heat_base = heat_base.ravel()
        jac_action = np.asarray(jac_numjac @ direction).ravel()

        c_plus = c_ret + step * direction.reshape(c_ret.shape)
        c_minus = c_ret - step * direction.reshape(c_ret.shape)
        central_action = (heat_source(c_plus).ravel() - heat_source(c_minus).ravel()) / (2.0 * step)

        eps_jac = self.numjac_Tc.eps_jac
        dc = -eps_jac * np.abs(c_ret)
        dc[dc > (-eps_jac)] = eps_jac
        dc = (c_ret + dc) - c_ret
        serial_action = np.zeros_like(heat_base)
        for flat_idx, coeff in enumerate(direction):
            if coeff == 0.0:
                continue
            c_pert = c_ret.copy()
            c_pert.ravel()[flat_idx] += dc.ravel()[flat_idx]
            serial_action += coeff * (
                heat_source(c_pert).ravel() - heat_base
            ) / dc.ravel()[flat_idx]

        def summarize(reference, candidate):
            """Return norm, relative-error, and cosine comparison statistics for two vectors."""
            ref_norm = np.linalg.norm(reference)
            cand_norm = np.linalg.norm(candidate)
            diff_norm = np.linalg.norm(reference - candidate)
            scale = max(ref_norm, cand_norm, 1e-30)
            if ref_norm <= 1e-30 or cand_norm <= 1e-30:
                cos = None
            else:
                cos = float(np.dot(reference, candidate) / (ref_norm * cand_norm))
            return {
                "reference_norm": float(ref_norm),
                "candidate_norm": float(cand_norm),
                "diff_norm": float(diff_norm),
                "rel_error": float(diff_norm / scale),
                "cosine": cos,
            }

        return {
            "kinetics_uses_auxiliary_pressure": bool(self.kinetics_uses_auxiliary_pressure),
            "numjac_shape_in": tuple(self.numjac_Tc.shape_in),
            "numjac_shape_out": tuple(self.numjac_Tc.shape_out),
            "numjac_num_groups": int(self.numjac_Tc.num_gr),
            "numjac_dependencies": [tuple(dep) for dep in self.numjac_Tc.dependencies],
            "linear_shape_check": summarize(jac_linear_exact.data, jac_linear.data),
            "directional_grouped_vs_central": summarize(central_action, jac_action),
            "directional_serial_vs_central": summarize(central_action, serial_action),
            "directional_grouped_vs_serial": summarize(serial_action, jac_action),
        }

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
            diff_field_ret = self.correlation.diffusion(y_ret, T_ret, p_ret)
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
            diff_field_perm = self.correlation.diffusion(y_perm, T_perm, p_perm)
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
                Rg = self.Rg
                perm = self.P0 * np.exp(-self.EA / (Rg * T_ret_mem[:, 0, np.newaxis]))  # (nz,nc)
                P_matrix_perm_mem = construct_coefficient_matrix(
                    Rg * T_perm_mem[:, 0, np.newaxis] * perm
                )
                P_matrix_ret_mem = construct_coefficient_matrix(
                    Rg * T_ret_mem[:, 0, np.newaxis] * perm
                )
            else:
                P_perm_mem = np.broadcast_to(
                    (Rg * T_perm * perm).reshape((1, -1)),
                    (self.num_z, self.num_c),
                )
                P_ret_mem = np.broadcast_to(
                    (Rg * T_ret * perm).reshape((1, -1)),
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
        u_ret_ax_local = self.u_ret_ax.copy()
        bc_ret_ax = get_axial_bcs_for_flow(
            is_counter_current=self.is_counter_current,
            u_ax=u_ret_ax_local,
            bc_inlet=self.BC_NONE,
            inflow_value=self.inflow_conc_ret_backflow,
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

    def _continuity_pressure_row(
        self,
        c,
        y,
        T,
        p,
        c_sum_safe,
        c_old,
        T_old,
        dt,
        c_tot=None,
        dc_tot_dp_mat=None,
        dc_tot_dT_mat=None,
        compute_jac=False,
    ):
        """Total-continuity pressure row with the EOS inserted (cf. pymrm-book L8).

        Instead of the algebraic constraint c_tot(y, T, p) - sum(c) = 0, the
        pressure row is the sum of the species balances evaluated at the
        EOS-consistent concentrations c_p = y * c_tot(y, T, p): convection with
        the Darcy velocities, diffusion and membrane permeation, reaction, and
        accumulation against the previous state. The EOS then enters through
        the density used in total continuity, so pressure is solved such that
        the velocity field satisfies total molar continuity rather than
        enforcing the EOS pointwise.

        Returns:
            Tuple of (g_p, blocks) where g_p has shape p.shape and blocks is
            (jac_pc, jac_pp, jac_pT) in (p x c), (p x p), (p x p) index space
            (or None when compute_jac is False). jac_pT columns are remapped to
            the temperature slot by the caller.
        """
        shape_c = c.shape
        shape_p1 = p.shape + (1,)
        if c_tot is None:
            c_tot = self.correlation.molar_density(y, T, p)
        c_p = y * c_tot[..., np.newaxis]
        c_p_ret = c_p[:, self.num_r_perm :, :]
        T_ret = T[:, self.num_r_perm :]
        _, p_ret = self._split_perm_and_ret(p)

        g_conv_p, jac_conv_p, jac_darcy_p = self._construct_g_conv(
            c_p, compute_jac=compute_jac
        )
        # Diffusion/permeation is affine with matrices depending on (y, T, p)
        # only, which are identical for c and c_p, so the cached operator applies.
        g_diff_p, _ = self._construct_g_diff(c_p)
        g_c_p = self.g_c_in.reshape(shape_c) + g_conv_p + g_diff_p

        if compute_jac:
            g_react_p, jac_react_p = self.numjac(
                lambda c_var: self.factor_react
                * self.kinetics(
                    self._reaction_partial_pressures(c_var, T_ret, p_ret), T_ret
                ),
                c_p_ret,
            )
        else:
            g_react_p = self.factor_react * self.kinetics(
                self._reaction_partial_pressures(c_p_ret, T_ret, p_ret), T_ret
            )
            jac_react_p = None
        g_c_p[:, self.num_r_perm :, :] -= g_react_p

        if c_old is not None:
            p_old = getattr(self, "_p_old", None)
            if p_old is None:
                p_old = p
            c_sum_old_safe = np.maximum(np.abs(np.sum(c_old, axis=-1)), 1e-10)
            y_old = c_old / c_sum_old_safe[..., np.newaxis]
            c_tot_old = self.correlation.molar_density(y_old, T_old, p_old)
            c_p_old = y_old * c_tot_old[..., np.newaxis]
            g_c_p += (
                self.jac_c_accum @ ((c_p - c_p_old).reshape((-1, 1)) / dt)
            ).reshape(shape_c)

        g_p = (self.sum_c @ g_c_p.reshape((-1, 1))).reshape(p.shape)

        if not compute_jac:
            return g_p, None

        # --- Exact chain-rule Jacobian blocks ---
        # A_p = d(species operators)/d(c_p): same approximation level as the
        # species rows (frozen diffusion/permeation and Darcy mobility matrices).
        shape_c_ret = c_p_ret.shape
        jac_react_p = update_csc_array_indices(
            jac_react_p, shape_c_ret, shape_c, offset=(0, self.num_r_perm, 0)
        )
        A_p = jac_conv_p + self.jac_c_diff - jac_react_p
        if c_old is not None:
            A_p = A_p + (1.0 / dt) * self.jac_c_accum

        # d(c_p)/d(c): c_tot/c_sum * (delta_ij - y_i * 1_j)
        ratio = c_tot / c_sum_safe
        diag_ratio = construct_coefficient_matrix(
            np.broadcast_to(ratio[..., np.newaxis], shape_c).copy(), shape=shape_c
        )
        outer = construct_coefficient_matrix(
            y * ratio[..., np.newaxis], shape=(shape_c, shape_p1)
        )
        M_c = diag_ratio - outer @ self.sum_c
        # d(c_p)/dp = y_i * dc_tot/dp ; d(c_p)/dT = y_i * dc_tot/dT
        dcdp = dc_tot_dp_mat.diagonal().reshape(p.shape)
        dcdT = dc_tot_dT_mat.diagonal().reshape(p.shape)
        M_p = construct_coefficient_matrix(
            y * dcdp[..., np.newaxis], shape=(shape_c, shape_p1)
        )
        M_T = construct_coefficient_matrix(
            y * dcdT[..., np.newaxis], shape=(shape_c, shape_p1)
        )

        jac_pc = self.sum_c @ (A_p @ M_c)
        jac_pp = self.sum_c @ (A_p @ M_p + jac_darcy_p)
        jac_pT = self.sum_c @ (A_p @ M_T)

        if not self.is_isothermal:
            # Direct Arrhenius temperature dependence of the reaction term,
            # mirroring jac_cT_react of the species rows.
            _, jac_cT_react_p = self.numjac_cT(
                lambda T_var: self.factor_react
                * self.kinetics(
                    self._reaction_partial_pressures(c_p_ret, T_var[..., 0], p_ret),
                    T_var[..., 0],
                ),
                T_ret[..., np.newaxis],
            )
            shape_p_ret1 = (self.num_z, self.num_r_ret, 1)
            if not hasattr(self, "_sum_c_ret"):
                self._sum_c_ret = construct_coefficient_matrix(
                    np.array([[[1.0]]]), shape=(shape_p_ret1, shape_c_ret)
                )
            react_T_term = self._sum_c_ret @ (-jac_cT_react_p)
            offset_ret = (0, self.num_r_perm, 0)
            jac_pT = jac_pT + update_csc_array_indices(
                react_T_term,
                (shape_p_ret1, shape_p_ret1),
                (shape_p1, shape_p1),
                offset=(offset_ret, offset_ret),
            )

        return g_p, (jac_pc, jac_pp, jac_pT)

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
        factor_p = self.factor_p
        factor_T = self.factor_T
        shape_cpT = cpT.shape
        g = np.empty(shape_cpT)
        c_sum = np.sum(c, axis=-1)
        # Use safe denominator for mole fractions to handle negative concentrations
        c_sum_safe = np.maximum(np.abs(c_sum), 1e-10)
        y = c / c_sum_safe[..., np.newaxis]

        c_ret = c[:, self.num_r_perm :, :]
        _, p_ret = self._split_perm_and_ret(p)
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
        else:
            g_T_conv, jac_T_conv, jac_T_darcy = self._construct_g_T_conv(T, compute_jac=compute_jac)
            g_T_cond, jac_T_cond, cp_inv_mat = self._construct_g_T_cond(T)
            enthalpies = self.correlation.species_enthalpies(T_ret)
            cp_inv = cp_inv_mat.data.reshape(T.shape)[:, self.num_r_perm:]
        if compute_jac:
            jac_cc = jac_conv + jac_diff
            if c_old is not None:
                jac_cc += (1.0 / dt) * self.jac_c_accum
            g_react, jac_react = self.numjac(
                lambda c_var: self.factor_react
                * self.kinetics(self._reaction_partial_pressures(c_var, T_ret, p_ret), T_ret),
                c_ret,
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
            if self.pressure_equation == "continuity":
                g_p, (jac_pc, jac_pp, jac_pT) = self._continuity_pressure_row(
                    c, y, T, p, c_sum_safe, c_old, T_old, dt,
                    c_tot=c_tot,
                    dc_tot_dp_mat=dc_tot_dp_mat,
                    dc_tot_dT_mat=dc_tot_dT_mat,
                    compute_jac=True,
                )
            else:
                jac_pp = dc_tot_dp_mat
                jac_pc = -self.sum_c
                jac_pT = dc_tot_dT_mat
            # Handle isothermal mode: simplified temperature Jacobian
            jac_cT_react = None
            jac_Tc_react = None
            if self.is_isothermal:
                # Pin temperature to initial value: g_T = T - T_init = 0
                jac_TT = self.jac_T_accum
                jac_TP = None
            else:
                jac_TT = (1.0 / dt) * self.jac_T_accum + jac_T_conv + jac_T_cond
                jac_TP = jac_T_darcy

                # === Thermochemical coupling blocks ===
                
                #cp_inv = cp_inv_mat.data.reshape(T.shape)[:, self.num_r_perm:]
                shape_T_ret_loc = (self.num_z, self.num_r_ret, 1)

                # Task 1 (J_cT): d(g_react)/d(T_ret)
                # Maps T_ret perturbations → concentration residual changes
                # self.kinetics(self._reaction_partial_pressures(c_ret, T_ret, p_ret), T_ret)
                _, jac_cT_react = self.numjac_cT(
                    lambda T_var: self.factor_react * self.kinetics(
                        self._reaction_partial_pressures(c_ret, T_var[..., 0], p_ret),
                        T_var[..., 0],
                    ),
                    T_ret[..., np.newaxis],
                )

                # Task 2 (J_Tc): d(heat_source)/d(c_ret)
                # Maps concentration perturbations → temperature residual changes
                #self.kinetics(self._reaction_partial_pressures(c_ret, T_ret, p_ret), T_ret)
                _, jac_Tc_react = self.numjac_Tc(
                    lambda c_var: self._evaluate_reaction_heat_source(
                        c_var,
                        T_ret,
                        p_ret,
                        enthalpies=enthalpies,
                        cp_inv=cp_inv,
                    ),
                    c_ret,
                )

                # Task 3 (J_TT_react): d(heat_source)/d(T_ret)
                # Adds Arrhenius temperature sensitivity to jac_TT
                # self.kinetics(self._reaction_partial_pressures(c_ret, T_ret, p_ret), T_ret)
                g_T_react, jac_TT_react_ret = self.numjac_TT_react(
                    lambda T_var: (
                        np.sum(
                            self.factor_react
                            * self.kinetics(
                                self._reaction_partial_pressures(c_ret, T_var[..., 0], p_ret),
                                T_var[..., 0],
                            )
                            * self.correlation.species_enthalpies(T_var[..., 0]),
                            axis=-1,
                        )
                        * cp_inv
                    )[..., np.newaxis],
                    T_ret[..., np.newaxis],
                )
                g_T_react = g_T_react.reshape(T_ret.shape)
                # Map J_TT_react from retentate-T space to full-T space and add
                jac_TT += update_csc_array_indices(
                    jac_TT_react_ret,
                    shape_T_ret_loc,
                    (self.num_z, self.num_r, 1),
                    offset=(0, self.num_r_perm, 0),
                )
                # Restore kinetics to current (c_ret, T_ret) after all FD sweeps
                #self.kinetics(self._reaction_partial_pressures(c_ret, T_ret, p_ret), T_ret)
                #self._construct_darcy_matrices(c=c, T=T, p=p)
                #self._update_velocity_fields(p=p)

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
            if jac_cT_react is not None:
                shape_c_ret_loc = (self.num_z, self.num_r_ret, self.num_c)
                shape_T_ret_loc = (self.num_z, self.num_r_ret, 1)
                offset_c_ret = (0, self.num_r_perm, 0)
                offset_T_ret = (0, self.num_r_perm, self.num_c + 1)
                jac_cT = update_csc_array_indices(
                    -jac_cT_react,
                    (shape_c_ret_loc, shape_T_ret_loc),
                    shape_cpT,
                    offset=(offset_c_ret, offset_T_ret),
                )
                jac_Tc = update_csc_array_indices(
                    jac_Tc_react,
                    (shape_T_ret_loc, shape_c_ret_loc),
                    shape_cpT,
                    offset=(offset_T_ret, offset_c_ret),
                )
            else:
                jac_cT = jac_Tc = None

            base_jac = jac_cc + factor_p*jac_pp + jac_cp + factor_p*jac_pc + factor_p*jac_pT + factor_T*jac_TT
            if jac_TP is not None:
                jac_TP = update_csc_array_indices(
                    jac_TP, shape_p, shape_cpT, offset=(offset_T, offset_p)
                )
                base_jac = base_jac + factor_T*jac_TP
            if jac_cT is not None:
                base_jac = base_jac + jac_cT + factor_T*jac_Tc
            self._jac = base_jac
        else:
            if self.pressure_equation == "continuity":
                g_p, _ = self._continuity_pressure_row(
                    c, y, T, p, c_sum_safe, c_old, T_old, dt, compute_jac=False
                )
            else:
                c_tot = self.correlation.molar_density(y, T, p)
            # Pass T_ret explicitly so kinetics always uses the current temperature
            g_react = self.factor_react * self.kinetics(
                self._reaction_partial_pressures(c_ret, T_ret, p_ret), T_ret
            )
            if (not self.is_isothermal):
                g_T_react = (
                    np.sum(g_react * enthalpies, axis=-1) * cp_inv
                )  # Convert to temperature residual contribution

        g_c = self.g_c_in.reshape(c.shape) + g_conv + g_diff
        if c_old is not None:
            g_c += (self.jac_c_accum @ ((c - c_old).reshape((-1, 1)) / dt)).reshape(
                c.shape
            )
        g_ret = g_c[:, self.num_r_perm :, :]
        g_ret[...] -= g_react
        if self.pressure_equation == "eos":
            g_p = c_tot - c_sum

        # Temperature residual
        if self.is_isothermal:
            # In isothermal mode, constrain temperature to initial value
            # g_T = (T - T_init) / dt -> T held constant
            T_init = np.empty_like(T)
            T_init[:, :self.num_r_perm] = self.T_perm_in
            T_init[:, self.num_r_perm:] = self.T_ret_in
            g_T = (T - T_init)
        else:
            g_T = g_T_conv + g_T_cond
            if T_old is not None:
                g_T += (self.jac_T_accum @ ((T - T_old).reshape((-1, 1)) / dt)).reshape(
                    T.shape
                )
            enthalpies = self.correlation.species_enthalpies(T_ret)
            g_T_ret = g_T[:, self.num_r_perm :]
            g_T_ret[...] += g_T_react
        g[..., :-2] = g_c
        g[..., -2] = factor_p * g_p
        g[..., -1] = factor_T * g_T
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
        rho_ret = self.correlation.molecular_weight(c_ret_i) # This is actually a mass density when called with concentrations instead of mole fractions
        cp_ret = self.correlation.specific_heat(c_ret_i, T_ret_i)
        Re_ret = np.abs(rho_ret * self.dp * u_ret_i / visc_ret)
        Pr_ret = np.abs(visc_ret/rho_ret * cp_ret / lmbda_ret_rad[:, [0]])
        Nu_ret = self.Nu_ret(Re_ret, Pr_ret)
        # Add minimum heat transfer coefficient to avoid division by zero
        h_min = 1.0  # W/(m^2 K) - natural convection lower bound
        h_ret = np.maximum(Nu_ret * lmbda_ret_rad[:, [0]] / self.dp, h_min)

        d_tube = 2.0 * self.r_f_perm[-1]
        visc_perm = self.correlation.viscosity(y_perm_i, T_perm_i)
        rho_perm = self.correlation.molecular_weight(c_perm_i) # This is actually a mass density when called with concentrations instead of mole fractions
        cp_perm = self.correlation.specific_heat(c_perm_i, T_perm_i)
        Re_perm = np.abs(rho_perm * d_tube * u_perm_i / visc_perm)
        Pr_perm = np.abs(visc_perm/rho_perm * cp_perm / lmbda_perm_rad[:, [-1]])
        Nu_perm = self.Nu_perm(Re_perm, Pr_perm)
        h_perm = np.maximum(Nu_perm * lmbda_perm_rad[:, [-1]] / d_tube, h_min)

        if self.nu == 1:
            resist_mem = (
                self.r_f_perm[-1] * np.log(self.r_f_ret[0] / self.r_f_perm[-1])
            ) / self.lambda_mem(T_ret_i)
            factor_geom = self.r_f_ret[0] / self.r_f_perm[-1]
            U = 1.0 / (1.0 / h_perm + resist_mem + 1.0 / (factor_geom * h_ret))
        else:
            resist_mem = (self.r_f_perm[-1] - self.r_f_ret[0]) / self.lambda_mem(T_ret_i)
            U = 1.0 / (1.0 / h_ret + resist_mem + 1.0 / h_perm)
            factor_geom = 1.0

        ic_1 = {"a": (lmbda_perm_rad[:, [-1]], 0), "b": (U, -U)}
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

    def _solve_cpT(self, c_old, T_old, dt, verbose=0, use_line_search=False,
                   max_iter=None):
        """Run the monolithic nonlinear correction for a fixed pseudo-time step.

        Args:
            c_old: Previous concentration field (None solves the steady
                equations directly, without the accumulation term)
            T_old: Previous temperature field
            dt: Time step
            verbose: Verbosity level
            use_line_search: If True, use Armijo backtracking line search
            max_iter: Newton iteration budget; defaults to
                self.max_newton_iterations

        Returns:
            SolveResult with correction diagnostics.
        """
        if max_iter is None:
            max_iter = self.max_newton_iterations
        cpT = self.cpT
        shape_cpT = cpT.shape
        ord = self.ord_norm
        factor_norm = self.factor_norm
        cpT_vec = cpT.ravel()
        self.last_solver_failure_message = None

        def mark_retryable_failure(exc, stage):
            """Record a retryable solver-failure message for the given stage."""
            message = f"{stage} failed at dt={dt:.2e}: {exc}"
            self.last_solver_failure_message = message
            logger.debug(message)

        # Define normalized component residuals and a unified Newton norm.
        def compute_norm(g):
            """Return the scaled ord-norm of a residual vector."""
            g_norm = np.linalg.norm(g.ravel(), ord=ord) * factor_norm
            return g_norm

        # Wrapper for residual evaluation with side effects
        def eval_residual(x_vec):
            """Write the trial state into cpT, refresh velocity fields, and return the residual."""
            cpT_vec[:] = x_vec
            self._update_velocity_fields(p=cpT[..., -2])
            g, _ = self._construct_g_cpT(c_old, T_old, dt, compute_jac=False)
            return g

        g_norm_init = None
        g_norm = np.inf
        # (line-search failures no longer latch; convergence is judged on g_norm)

        for j in range(max_iter):
            try:
                g, jac = self._construct_g_cpT(c_old, T_old, dt, compute_jac=True)
            except RecoverableNumericalError as exc:
                mark_retryable_failure(exc, stage="residual/Jacobian assembly")
                return SolveResult(
                    converged=False,
                    num_iterations=j + 1,
                    g_norm=np.inf,
                    g_norm_init=g_norm_init if g_norm_init is not None else np.inf,
                )

            g_norm = compute_norm(g)
            logger.debug(
                "Newton iteration %d: g_norm = %.4e",
                j,
                g_norm,
            )

            if j == 0:
                g_norm_init = g_norm

            if (not np.isfinite(g_norm)) or (g_norm_init is not None and g_norm > 10 * g_norm_init):
                return SolveResult(
                    converged=False,
                    num_iterations=j + 1,
                    g_norm=g_norm,
                    g_norm_init=g_norm_init,
                )

            if g_norm <= max(self.rtol * g_norm_init, self.atol):
                return SolveResult(
                    converged=True,
                    num_iterations=j + 1,
                    g_norm=g_norm,
                    g_norm_init=g_norm_init,
                )

            try:
                dcpT = -sla.spsolve(jac, g.reshape((-1, 1)))
            except RuntimeError:
                return SolveResult(
                    converged=False,
                    num_iterations=j + 1,
                    g_norm=np.inf,
                    g_norm_init=g_norm_init if g_norm_init is not None else np.inf,
                )
            if not np.all(np.isfinite(dcpT)):
                return SolveResult(
                    converged=False,
                    num_iterations=j + 1,
                    g_norm=np.inf,
                    g_norm_init=g_norm_init if g_norm_init is not None else np.inf,
                )
            self.cnt_num_solves_cpT += 1

            if use_line_search:
                try:
                    x_new, g_new_norm, alpha, ls_success = armijo_line_search(
                        x=cpT_vec.copy(),
                        dx=dcpT,
                        g_norm=g_norm,
                        residual_fn=lambda x: eval_residual(x).ravel(),
                        norm_fn=compute_norm,
                        armijo_coeff=ARMIJO_COEFF,
                        min_alpha=MIN_LINE_SEARCH_ALPHA,
                    )
                except RecoverableNumericalError as exc:
                    mark_retryable_failure(exc, stage="line search residual evaluation")
                    return SolveResult(
                        converged=False,
                        num_iterations=j + 1,
                        g_norm=np.inf,
                        g_norm_init=g_norm_init if g_norm_init is not None else np.inf,
                    )
                if not ls_success and alpha == 0.0:
                    # Every trial point overflowed: the Newton direction is
                    # unusable. Abort gracefully with the state unchanged and
                    # a finite residual so the caller's dt control can react.
                    mark_retryable_failure(
                        "line search found no finite trial point", stage="line search"
                    )
                    return SolveResult(
                        converged=False,
                        num_iterations=j + 1,
                        g_norm=g_norm,
                        g_norm_init=g_norm_init,
                    )
                cpT_vec[:] = x_new

                if not ls_success and verbose > 0:
                    warnings.warn(
                        f"Line search: no sufficient decrease at iteration {j}, "
                        f"continuing from best trial (residual {g_new_norm:.2e})",
                        RuntimeWarning,
                    )
            else:
                cpT_vec[:] = cpT_vec + dcpT

            self._update_velocity_fields(p=cpT[..., -2])

        try:
            g, _ = self._construct_g_cpT(c_old, T_old, dt, compute_jac=False)
        except RecoverableNumericalError as exc:
            mark_retryable_failure(exc, stage="post-step residual evaluation")
            return SolveResult(
                converged=False,
                num_iterations=max_iter,
                g_norm=np.inf,
                g_norm_init=g_norm_init if g_norm_init is not None else np.inf,
            )

        g_norm = compute_norm(g)
        if (not np.isfinite(g_norm)) or (g_norm_init is not None and g_norm > 10 * g_norm_init):
            return SolveResult(
                converged=False,
                num_iterations=max_iter,
                g_norm=g_norm,
                g_norm_init=g_norm_init if g_norm_init is not None else np.inf                  ,
            )

        g_norm = compute_norm(g)
        return SolveResult(
            converged=(g_norm <= max(self.rtol * g_norm_init, self.atol)),
            num_iterations=max_iter,
            g_norm=g_norm,
            g_norm_init=g_norm_init if g_norm_init is not None else np.inf,
        )

    def _compute_dt_cfl(self, cfl=CFL_INIT):
        """Return minimum time step avoiding CFL condition violation."""
        dz_cell = np.min(self.z_f[1:] - self.z_f[:-1])
        u_max = np.maximum(
            np.max(np.abs(self.u_perm_ax)), np.max(np.abs(self.u_ret_ax))
        )
        dt_cfl = cfl * dz_cell / u_max
        return dt_cfl

    def _compute_steady_state_residual(self):
        """Evaluate the steady-state residual on the current reactor state."""
        c_current = self.cpT[..., :-2].copy()
        T_current = self.cpT[..., -1].copy()
        self._p_old = self.cpT[..., -2].copy()
        g_ss, _ = self._construct_g_cpT(
            c_current,
            T_current,
            STEADY_STATE_DT,
            compute_jac=False,
        )
        return g_ss

    def _steady_state_reference_density(self):
        """Reference total molar density for the weighted norm's absolute floors."""
        return self.p_ret_out / (self.Rg * self.T_ret_in)

    def _compute_steady_state_weights(self):
        """Error weights for the WRMS norm, from the CURRENT state (never frozen)."""
        return steady_state_weights(
            self.cpT,
            rtol_c=self.wrms_rtol_c,
            atol_c_rel=self.wrms_atol_c_rel,
            rtol_p=self.wrms_rtol_p,
            atol_p_rel=self.wrms_atol_p_rel,
            rtol_T=self.wrms_rtol_T,
            atol_T=self.wrms_atol_T,
            Rg=self.Rg,
            c_ref=self._steady_state_reference_density(),
        )

    def _compute_steady_state_norm(self, norm_kind=None):
        """Return the norm of the steady-state residual on the current state.

        ``norm_kind`` overrides ``self.norm_kind`` (used to log both norms
        side by side during development; see
        CVODE-style error-weighted RMS norm). The residual's
        pressure and temperature rows carry the assembly factors
        ``factor_p``/``factor_T``; the weighted norm divides them out so the
        per-block tolerances weight the physical residuals.
        """
        norm_kind = self.norm_kind if norm_kind is None else norm_kind
        g_ss = self._compute_steady_state_residual()
        if norm_kind == "absolute":
            return np.linalg.norm(g_ss.ravel(), ord=self.ord_norm) * self.factor_norm
        if norm_kind != "weighted":
            raise ValueError(f"unknown norm_kind {norm_kind!r}")
        g_phys = g_ss.copy()
        if self.factor_p != 0.0:
            g_phys[..., -2] /= self.factor_p
        if self.factor_T != 0.0:
            g_phys[..., -1] /= self.factor_T
        return weighted_rms(g_phys, self._compute_steady_state_weights())

    def _kpi_monitor_vector(self):
        """Scalar monitor of the published quantities, for stagnation stopping.

        Outlet molar flows per species on both sides plus the peak
        temperature — every published KPI (conversion, production, recovery,
        purity, DeltaT) is a function of these, so their stagnation implies
        KPI stagnation. O(N) cost against a sparse LU per Newton step.
        """
        flows_ret_ax, _, flows_perm_ax, _ = self.compute_flows()
        return np.concatenate([
            flows_ret_ax[-1],
            flows_perm_ax[-1],
            [float(np.max(self.cpT[..., -1]))],
        ])

    def _refresh_transport_state(self, c=None, T=None, p=None):
        """Rebuild transport operators and velocity fields for the given state."""
        self._construct_darcy_matrices(c=c, T=T, p=p)
        pressure = self.cpT[..., -2] if p is None else p
        self._update_velocity_fields(p=pressure)

    def _restore_state(self, cpT_state):
        """Restore a saved state and refresh derived transport fields."""
        self.cpT = cpT_state.copy()
        self._refresh_transport_state(p=self.cpT[..., -2])

    def solve(
        self,
        num_timesteps: Optional[int] = None,
        dt: Optional[float] = None,
        dt_init: Optional[float] = None,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
        dt_increase_factor: Optional[float] = None,
        dt_decrease_factor: Optional[float] = None,
        adaptive_dt_threshold: Optional[float] = None,
        use_adaptive_dt: bool = True,
        steady_state_atol: Optional[float] = None,
        steady_state_rtol: Optional[float] = None,
        verbose: int = 0,
        return_status: bool = False,
        **kwargs: Any,
    ) -> Union[bool, SteadyStateSolveStatus]:
        """Solve the steady-state reactor problem.

        Uses adaptive time-stepping for robust and efficient convergence.
        The pseudo-transient term (1/dt) regularizes the Newton iteration.
        Small dt ensures convergence but is slow; large dt is fast but may
        diverge. Adaptive dt starts small and increases as solution improves.

        Args:
            num_timesteps: Maximum number of pseudo-transient steps.
            dt: Fixed pseudo-time step size. When provided, adaptive dt is disabled.
            dt_init: Initial adaptive time step. Defaults to self.dt_init.
            dt_min: Minimum adaptive time step override for this solve call.
            dt_max: Maximum adaptive time step override for this solve call.
            dt_increase_factor: Adaptive dt growth factor override.
            dt_decrease_factor: Adaptive dt shrink factor override.
            adaptive_dt_threshold: Residual reduction threshold for increasing dt.
            use_adaptive_dt: If True, adapt dt based on convergence quality.
            steady_state_atol: Absolute tolerance on ||g_ss||.
            steady_state_rtol: Relative tolerance on ||g_ss|| referenced to the
                first accepted steady-state residual.
            verbose: Verbosity level (0=silent, 1=summary, 2=detailed).
            return_status: If True, return a structured solve-status object.
            **kwargs: Unsupported keyword arguments. Raises TypeError.

        Returns:
            True/False by default, or a SteadyStateSolveStatus when return_status=True.
        """
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"solve() got unexpected keyword argument(s): {unknown}")

        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        if dt is not None and dt_init is not None:
            raise ValueError("Provide either dt or dt_init, not both.")

        if dt is not None:
            dt_init = dt
            use_adaptive_dt = False
        elif dt_init is None:
            dt_init = getattr(self, 'dt_init', 1e-3)

        self.cnt_num_solves_cpT = 0
        dt = dt_init
        dt_min = getattr(self, 'dt_min', 1e-16) if dt_min is None else dt_min
        dt_max = getattr(self, 'dt_max', 1e3) if dt_max is None else dt_max
        dt_increase = (
            getattr(self, 'dt_increase_factor', 2.0)
            if dt_increase_factor is None
            else dt_increase_factor
        )
        dt_decrease = (
            getattr(self, 'dt_decrease_factor', 0.5)
            if dt_decrease_factor is None
            else dt_decrease_factor
        )
        threshold = (
            getattr(self, 'adaptive_dt_threshold', 0.5)
            if adaptive_dt_threshold is None
            else adaptive_dt_threshold
        )
        steady_state_atol = (
            getattr(self, 'steady_state_atol', 0.0)
            if steady_state_atol is None
            else steady_state_atol
        )
        steady_state_rtol = (
            getattr(self, 'steady_state_rtol', 1e-3)
            if steady_state_rtol is None
            else steady_state_rtol
        )
        if steady_state_atol < 0.0:
            raise ValueError("steady_state_atol must be non-negative.")
        if steady_state_rtol is not None and steady_state_rtol < 0.0:
            raise ValueError("steady_state_rtol must be non-negative when provided.")
        if steady_state_atol == 0.0 and steady_state_rtol is None:
            raise ValueError(
                "Enable at least one steady-state convergence criterion."
            )
        self.last_solver_failure_message = None

        g_ss_norm_prev = None
        g_ss_norm_0 = None   # first accepted SS residual, used for relative convergence
        g_ss_norm = None
        g_ss_norm_best = None   # lowest g_ss seen — best physical state
        cpT_best = None
        is_converged = False
        accepted_steps = 0
        n_increasing = 0        # consecutive steps with g_ss growing
        n_plateau_steps = 0     # consecutive steps where |reduction-1| stays small
        n_drift = 0             # consecutive drift-hold steps (limit-cycle symptom)
        n_restores = 0          # times the best state was restored after drifting away
        steady_attempts = 0     # steady-Newton escalation attempts used
        steady_cooldown = 0     # steps to wait before the next escalation attempt
        t_contin_tried = False  # temperature continuation fired (at most once)
        # Opportunistic-jump re-arm gate: after a failed ripe attempt, require
        # another decade of residual drop before trying again.
        ripe_gate = self.steady_jump_rel_drop
        MAX_STEADY_ATTEMPTS = 5

        # Stagnation stopping and outcome classification: monitor the
        # published quantities and classify the outcome instead of pass/fail.
        outcome = "failed"
        outcome_evidence = None
        oscillation_evidence = None
        force_stall_escalation = False  # oscillatory verdict: ladder, now
        t_contin_used = False
        monitor = None
        if self.kpi_stop:
            flow_floor = 1e-6 * (self.F_ret_in + self.F_perm_in)
            monitor = ConvergenceMonitor(
                check_every=self.kpi_check_every,
                kpi_rtol=self.kpi_rtol,
                n_stagnant=self.kpi_n_stagnant,
                dt_max=dt_max,
                soft_step_cap=self.steady_soft_step_cap,
                kpi_floor=np.concatenate([
                    np.full(2 * self.num_c, flow_floor), [1.0]
                ]),
            )

        def finalize(converged: bool, steps_attempted: int):
            """Build the SteadyStateSolveStatus, store it on self, and return it (or its converged flag)."""
            final_outcome = outcome
            if converged:
                final_outcome = "converged"
            status = SteadyStateSolveStatus(
                converged=converged or final_outcome == "floored",
                num_steps_attempted=steps_attempted,
                num_steps_accepted=accepted_steps,
                final_dt=dt,
                steady_state_norm=g_ss_norm,
                initial_steady_state_norm=g_ss_norm_0,
                best_steady_state_norm=g_ss_norm_best,
                baseline_steady_state_norm=g_ss_norm_baseline if 'g_ss_norm_baseline' in locals() else None,
                last_failure_message=self.last_solver_failure_message,
                norm_kind=self.norm_kind,
                outcome=final_outcome,
                outcome_evidence=outcome_evidence,
                oscillation=oscillation_evidence,
                used_temperature_continuation=t_contin_used,
                dynamically_unstable=bool(t_contin_used and oscillation_evidence is not None),
            )
            self.last_solve_status = status
            return status if return_status else status.converged

        # Compute baseline SS residual before any steps.
        # Used to detect if a first accepted step produced a corrupted state
        # (e.g. when dt_init is too large and the Newton step jumps to a
        # non-physical region without producing NaN).
        try:
            g_ss_norm_baseline = self._compute_steady_state_norm()
        except RecoverableNumericalError as exc:
            self.last_solver_failure_message = f"initial steady-state residual evaluation failed: {exc}"
            if verbose >= 1:
                print(self.last_solver_failure_message)
            return finalize(False, 0)

        if verbose >= 1:
            print(f"Starting adaptive dt solve: dt_init={dt_init:.2e}, "
                  f"dt_range=[{dt_min:.2e}, {dt_max:.2e}]")

        def check_steady_state_convergence(current_norm: float):
            """Check current_norm against the absolute and relative steady-state
            tolerances; return (is_ok, abs_ok, rel_ok).
            """
            abs_ok = steady_state_atol > 0.0 and current_norm <= steady_state_atol
            rel_ok = (
                steady_state_rtol is not None
                and g_ss_norm_0 is not None
                and current_norm / g_ss_norm_0 <= steady_state_rtol
            )
            criteria = []
            if steady_state_atol > 0.0:
                criteria.append(abs_ok)
            if steady_state_rtol is not None:
                criteria.append(rel_ok)
            is_ok = bool(criteria) and all(criteria)
            return is_ok, abs_ok, rel_ok

        for i in range(num_timesteps):
            T_old = self.cpT[..., -1].copy()
            c_old = self.cpT[..., :-2].copy()
            self._p_old = self.cpT[..., -2].copy()
            cpT_backup = self.cpT.copy()

            # Attempt solve with current dt
            result = self._solve_cpT(c_old, T_old, dt)
            g_norm = result.g_norm

            # --- Reject if factorization/NaN failure (g_norm=inf) ---
            if not np.isfinite(g_norm):
                self._restore_state(cpT_backup)
                dt = max(dt * dt_decrease, dt_min)
                n_plateau_steps = 0
                if verbose >= 2:
                    reason = self.last_solver_failure_message or "NaN/singular"
                    print(f"  Step {i}: REJECTED ({reason}), dt -> {dt:.2e}")
                continue

            # Compute actual steady-state residual (without transient term)
            try:
                g_ss_norm = self._compute_steady_state_norm()
            except RecoverableNumericalError as exc:
                self.last_solver_failure_message = (
                    f"steady-state residual evaluation failed at dt={dt:.2e}: {exc}"
                )
                self._restore_state(cpT_backup)
                dt = max(dt * dt_decrease, dt_min)
                n_plateau_steps = 0
                if verbose >= 2:
                    print(f"  Step {i}: REJECTED ({self.last_solver_failure_message}), dt -> {dt:.2e}")
                continue

            # --- Reject if SS residual increased dramatically vs previous accepted step ---
            # Do NOT compare against baseline for step 0: the baseline is computed with
            # cold (un-warmed) velocity fields, while after a first solve the velocity fields
            # have been updated by many Picard steps. These are incomparable measurements
            # and the comparison causes spurious step-0 rejection even for tiny dt.
            if g_ss_norm_prev is not None and g_ss_norm > 10 * g_ss_norm_prev:
                self._restore_state(cpT_backup)
                dt = max(dt * dt_decrease, dt_min)
                n_increasing = 0
                n_plateau_steps = 0
                if verbose >= 2:
                    print(f"  Step {i}: REJECTED (g_ss grew), dt -> {dt:.2e}, "
                          f"||g_ss||={g_ss_norm:.2e}, ||g||={g_norm:.2e}")
                continue

            accepted_steps += 1

            # Track best state so far (lowest g_ss)
            if g_ss_norm_best is None or g_ss_norm < g_ss_norm_best:
                g_ss_norm_best = g_ss_norm
                cpT_best = self.cpT.copy()

            # Record initial SS norm once (after first accepted step)
            if g_ss_norm_0 is None:
                g_ss_norm_0 = max(g_ss_norm, 1e-30)

            # Adapt dt based on steady-state residual progress, and track plateau.
            if use_adaptive_dt and g_ss_norm_prev is not None:
                reduction = g_ss_norm / g_ss_norm_prev

                if reduction > 1.05:
                    # Residual growing fast — reduce dt.
                    dt = max(dt * dt_decrease, dt_min)
                    n_increasing += 1
                    if verbose >= 2:
                        print(f"  Step {i}: poor (red={reduction:.2f}), dt -> {dt:.2e}, "
                              f"||g_ss||={g_ss_norm:.2e}, ||g||={g_norm:.2e}")
                elif reduction > 1.0:
                    # Residual drifting upward slightly — hold dt.
                    n_increasing += 1
                    n_drift += 1
                    if verbose >= 2:
                        print(f"  Step {i}: drift (red={reduction:.2f}), dt={dt:.2e}, "
                              f"||g_ss||={g_ss_norm:.2e}, ||g||={g_norm:.2e}")
                elif reduction < threshold:
                    # Good reduction — increase dt aggressively (dt_increase²).
                    dt = min(dt * dt_increase ** 2, dt_max)
                    n_increasing = 0
                    n_drift = 0
                    if verbose >= 2:
                        print(f"  Step {i}: good (red={reduction:.2f}), dt -> {dt:.2e}, "
                              f"||g_ss||={g_ss_norm:.2e}, ||g||={g_norm:.2e}")
                else:
                    # threshold ≤ reduction ≤ 1.0 — moderate progress, increase dt normally.
                    dt = min(dt * dt_increase, dt_max)
                    n_increasing = 0
                    n_drift = 0
                    if verbose >= 2:
                        print(f"  Step {i}: ok (red={reduction:.2f}), dt -> {dt:.2e}, "
                              f"||g_ss||={g_ss_norm:.2e}, ||g||={g_norm:.2e}")

                # Track plateau: sole authority on n_plateau_steps.
                # |reduction - 1| < STEADY_STATE_PLATEAU_TOL means the steady-state residual is
                # essentially stationary.
                if abs(reduction - 1.0) < STEADY_STATE_PLATEAU_TOL:
                    n_plateau_steps += 1
                else:
                    n_plateau_steps = 0

                # If consistently drifting away from best, restore best state.
                if n_increasing >= 10 and cpT_best is not None:
                    self._restore_state(cpT_best)
                    g_ss_norm = g_ss_norm_best
                    g_ss_norm_prev = g_ss_norm_best
                    n_increasing = 0
                    n_plateau_steps = 0
                    n_restores += 1
                    if verbose >= 2:
                        print(f"  Step {i}: Restored best state, "
                              f"||g_ss||={g_ss_norm:.2e}, ||g||={g_norm:.2e}")
            elif verbose >= 2:
                print(f"  Step {i}: dt={dt:.2e}, ||g_ss||={g_ss_norm:.2e}")

            # --- KPI monitor: sample the published quantities -----------------
            # Sampled BEFORE the residual test: the classifier's stagnation
            # state also gates the residual-threshold break below.
            verdict = None
            if monitor is not None and monitor.due(accepted_steps):
                try:
                    kpi_vec = self._kpi_monitor_vector()
                except RecoverableNumericalError:
                    kpi_vec = None
                if kpi_vec is not None:
                    verdict = monitor.observe(
                        step=accepted_steps, g_ss_norm=g_ss_norm, dt=dt,
                        kpi=kpi_vec,
                    )

            is_converged, abs_ok, rel_ok = check_steady_state_convergence(g_ss_norm)
            rel_residual = None if g_ss_norm_0 is None else g_ss_norm / g_ss_norm_0
            # Item 3: "converged" needs the residual under tolerance AND the
            # published quantities stagnant. A threshold crossed while the
            # KPIs are still moving (measured on the low-GHSV cases, where
            # X_H2 keeps climbing for ~60 steps after the crossing) is not
            # convergence. The stagnation must be measured while the state
            # is actually evolving (dt at dt_max) — checks accumulated with
            # dt trapped small are stale: the state barely moves per step,
            # so the KPIs look stagnant regardless of convergence (caught by
            # a 2.4% certificate drift on draft G1_50). Steady-Newton jump
            # states are exempt (below) — a converged direct steady solve is
            # its own certificate.
            kpi_confirmed = (
                monitor is None
                or (monitor.stagnant_checks >= self.kpi_n_stagnant
                    and dt >= 0.99 * dt_max)
            )
            if is_converged and not kpi_confirmed:
                is_converged = False
                if verbose >= 2:
                    print(f"  Step {i}: residual under tolerance but KPIs "
                          f"still moving — continuing")
            if is_converged:
                if verbose >= 1:
                    message = f"  Step {i}: CONVERGED, dt={dt:.2e}, ||g_ss||={g_ss_norm:.2e}"
                    if steady_state_atol > 0.0:
                        message += (
                            f", abs={'yes' if abs_ok else 'no'}"
                            f" (target {steady_state_atol:.2e})"
                        )
                    if rel_residual is not None and steady_state_rtol is not None:
                        message += (
                            f", rel={rel_residual:.2e}"
                            f" (target {steady_state_rtol:.2e})"
                        )
                    print(message)
                break

            if verbose >= 2 and n_plateau_steps == STEADY_STATE_PLATEAU_REQUIRED:
                print(
                    f"  Step {i}: STAGNATING, dt={dt:.2e}, "
                    f"||g_ss||={g_ss_norm:.2e}, plateau_steps={n_plateau_steps}"
                )

            # --- Outcome classification (Items 2+3) --------------------------
            if verdict is not None:
                if verdict.outcome == "diverging":
                    outcome = "diverging"
                    outcome_evidence = verdict.evidence
                    self.last_solver_failure_message = (
                        f"diverging: {verdict.evidence.get('reason')}"
                    )
                    if verbose >= 1:
                        print(f"  Step {i}: DIVERGING "
                              f"({verdict.evidence.get('reason')})")
                    break
                if verdict.outcome == "floored":
                    # The iteration converged: the residual stopped
                    # decreasing and the KPIs stopped moving. The threshold
                    # was simply unreachable. Accept.
                    outcome = "floored"
                    outcome_evidence = verdict.evidence
                    if verbose >= 1:
                        print(f"  Step {i}: FLOORED (accepted), "
                              f"||g_ss||={g_ss_norm:.2e}, "
                              f"kpi_drift={verdict.evidence.get('kpi_rel_change'):.2e}")
                    break
                if verdict.outcome == "oscillatory":
                    # Limit-cycle orbit: marching will never converge. Hand
                    # to the escalation ladder immediately instead of
                    # burning the step budget.
                    oscillation_evidence = verdict.evidence
                    force_stall_escalation = True
                    if verbose >= 1:
                        print(f"  Step {i}: OSCILLATORY "
                              f"(amplitude={verdict.evidence.get('amplitude_rel'):.2e}, "
                              f"{verdict.evidence.get('sign_changes')} reversals) "
                              f"-> escalation ladder")

            # --- Steady-Newton escalation -----------------------------------
            # Newton on the steady equations (dt -> infinity) does not care
            # about the dynamic stability of the steady state, so it lands
            # solutions that pseudo-transient marching orbits forever (drift /
            # limit-cycle stall). Two triggers:
            #   stalled: sustained drift, a long plateau, or repeated
            #            best-state restores;
            #   ripe:    the residual has already dropped by
            #            steady_jump_rel_drop -- a direct solve from here is
            #            usually much cheaper than marching on.
            if steady_cooldown > 0:
                steady_cooldown -= 1
            if (self.try_direct_steady
                    and steady_cooldown == 0
                    and steady_attempts < MAX_STEADY_ATTEMPTS
                    and cpT_best is not None):
                # A stall is a limit cycle that traps the dt controller at
                # small dt. Grinding near the scheme's accuracy floor looks
                # similar (plateau) but happens AT dt_max, where the heavy
                # escalation rungs cannot help -- so require dt to be trapped.
                stalled = (
                    dt < 0.01 * dt_max
                    and (
                        n_drift >= self.drift_escalate_steps
                        or n_plateau_steps >= STEADY_STATE_PLATEAU_REQUIRED
                        or n_restores >= 2
                    )
                # A classified oscillation is a stall by definition — the
                # march orbits a dynamically unstable steady state.
                ) or force_stall_escalation
                ripe = (
                    g_ss_norm_0 is not None
                    and g_ss_norm_best <= ripe_gate * g_ss_norm_0
                    # Probe only when marching has slowed: while the residual
                    # is still dropping fast, plain stepping wins the race.
                    and (g_ss_norm_prev is None
                         or g_ss_norm > threshold * g_ss_norm_prev)
                )
                if stalled or ripe:
                    steady_attempts += 1
                    t_contin_ok = False
                    cpT_backup_jump = self.cpT.copy()
                    # Rung 1. Opportunistic (ripe): one direct steady solve
                    # from the best state -- cheap, and either it finishes the
                    # job or marching continues unharmed. Stalled: a dt-ramp
                    # homotopy toward the steady equations, holding the base
                    # state fixed with dt increasing x10 per stage -- the
                    # (1/dt) accumulation term keeps the Jacobian
                    # well-conditioned early on and the final stage is the
                    # steady problem itself.
                    self._restore_state(cpT_best)
                    c_base = cpT_best[..., :-2].copy()
                    T_base = cpT_best[..., -1].copy()
                    self._p_old = cpT_best[..., -2].copy()
                    dt_esc = dt_max if not stalled else max(10.0 * dt, 10.0)
                    try:
                        while True:
                            stage_backup = self.cpT.copy()
                            at_infinity = dt_esc >= dt_max
                            result_esc = self._solve_cpT(
                                None if at_infinity else c_base,
                                None if at_infinity else T_base,
                                STEADY_STATE_DT if at_infinity else dt_esc,
                                use_line_search=True,
                                # Opportunistic probes fail fast; stall rescues
                                # get the doubled budget.
                                max_iter=(2 * self.max_newton_iterations
                                          if stalled
                                          else self.max_newton_iterations),
                            )
                            if not (result_esc.converged
                                    and np.isfinite(result_esc.g_norm)):
                                self._restore_state(stage_backup)
                                break
                            if at_infinity:
                                break
                            dt_esc *= 10.0
                        ss_after = self._compute_steady_state_norm()
                    except RecoverableNumericalError:
                        ss_after = np.inf
                    if (stalled
                            and not (np.isfinite(ss_after)
                                     and ss_after < g_ss_norm_best)):
                        # Rung 2: reaction continuation from the best state.
                        # Weakening the reaction stabilizes the steady branch;
                        # walking factor_react back up tracks it into the
                        # dynamically unstable region that time marching
                        # orbits, with Newton starting in-basin at every stage.
                        self._restore_state(cpT_best)
                        if verbose >= 2:
                            print(f"  Step {i}: STEADY JUMP escalating to "
                                  f"reaction continuation")
                        try:
                            self._solve_adaptive_react(dt=dt_max, verbose=0)
                            ss_after = self._compute_steady_state_norm()
                        except RecoverableNumericalError:
                            ss_after = np.inf
                    if (stalled
                            and not (np.isfinite(ss_after)
                                     and ss_after < g_ss_norm_best)
                            and self.try_temperature_continuation
                            # An oscillatory verdict fast-tracks the ladder:
                            # rungs 1-2 cannot rescue a limit-cycle orbit
                            # that has already been classified.
                            and (steady_attempts >= 3 or force_stall_escalation)
                            and not t_contin_tried):
                        t_contin_tried = True
                        # Rung 3 (once, it is the expensive one):
                        # temperature continuation from a stable neighbor, for
                        # the multiplicity window where the steady state is
                        # dynamically unstable and unreachable by marching.
                        if verbose >= 1:
                            print(f"  Step {i}: escalating to temperature "
                                  f"continuation")
                        try:
                            for T_off in (25.0, -25.0):
                                if self._solve_by_temperature_continuation(
                                    T_offset=T_off, verbose=verbose,
                                ):
                                    t_contin_ok = True
                                    break
                            ss_after = self._compute_steady_state_norm()
                        except RecoverableNumericalError:
                            ss_after = np.inf
                    if np.isfinite(ss_after) and ss_after < g_ss_norm_best:
                        # Polish the accepted steady state with one tight
                        # direct Newton pass (rtol 1e-3 / atol 1e-5). The
                        # default Newton tolerances leave up to ~0.5%
                        # element imbalance in a jump-landed state; from an
                        # in-basin state the polish costs a few quadratic
                        # iterations and lands the tight-Newton answer that
                        # a marched solve only reaches at its floor
                        # (measured: X_H2 51.884% vs 51.46% on G2_50).
                        polish_backup = self.cpT.copy()
                        rtol_saved, atol_saved = self.rtol, self.atol
                        self.rtol = min(rtol_saved, 1e-3)
                        self.atol = min(atol_saved, 1e-5)
                        try:
                            self._solve_cpT(None, None, STEADY_STATE_DT,
                                            use_line_search=True,
                                            max_iter=self.max_newton_iterations)
                            ss_polished = self._compute_steady_state_norm()
                        except RecoverableNumericalError:
                            ss_polished = np.inf
                        finally:
                            self.rtol, self.atol = rtol_saved, atol_saved
                        if np.isfinite(ss_polished) and ss_polished < ss_after:
                            ss_after = ss_polished
                            if verbose >= 2:
                                print(f"  Step {i}: steady polish accepted, "
                                      f"||g_ss||={ss_polished:.2e}")
                        else:
                            self._restore_state(polish_backup)
                        g_ss_norm = ss_after
                        g_ss_norm_best = ss_after
                        cpT_best = self.cpT.copy()
                        n_drift = n_plateau_steps = n_restores = 0
                        steady_cooldown = 10
                        t_contin_used = t_contin_used or t_contin_ok
                        if force_stall_escalation:
                            # The state jumped: the monitor's trailing history
                            # no longer describes this trajectory.
                            force_stall_escalation = False
                            if monitor is not None:
                                monitor.history.clear()
                                monitor.stagnant_checks = 0
                        if verbose >= 1:
                            print(f"  Step {i}: STEADY JUMP accepted, "
                                  f"||g_ss||={ss_after:.2e}")
                        is_converged, abs_ok, rel_ok = (
                            check_steady_state_convergence(g_ss_norm)
                        )
                        if is_converged:
                            if verbose >= 1:
                                print(f"  Step {i}: CONVERGED after steady jump, "
                                      f"||g_ss||={g_ss_norm:.2e}")
                            break
                    else:
                        # Failed: restore the pre-attempt state. Only cut dt
                        # when the trigger was a stall (to break the limit
                        # cycle); a failed opportunistic attempt must not
                        # sabotage healthy marching, it just raises the bar
                        # for the next attempt by a decade.
                        self._restore_state(cpT_backup_jump)
                        if stalled:
                            dt = max(dt * 0.25, dt_min)
                        else:
                            ripe_gate *= 0.1
                        n_drift = n_plateau_steps = 0
                        steady_cooldown = 20
                        if verbose >= 2:
                            print(f"  Step {i}: STEADY JUMP failed "
                                  f"(||g_ss||={ss_after:.2e}), dt -> {dt:.2e}")
                        if force_stall_escalation and t_contin_tried:
                            # A classified limit cycle and the whole ladder —
                            # including temperature continuation — failed:
                            # more marching cannot help. Stop now instead of
                            # burning the remaining step budget.
                            outcome = "oscillatory"
                            outcome_evidence = oscillation_evidence
                            self.last_solver_failure_message = (
                                "oscillatory (limit-cycle orbit); escalation "
                                "ladder incl. temperature continuation failed"
                            )
                            if verbose >= 1:
                                print(f"  Step {i}: OSCILLATORY and ladder "
                                      f"exhausted — stopping")
                            break

            g_ss_norm_prev = g_ss_norm

        # Always restore the best (lowest g_ss) state found, even if the final
        # state drifted away from it after the minimum.
        if cpT_best is not None and (g_ss_norm is None or g_ss_norm_best < g_ss_norm):
            self._restore_state(cpT_best)
            g_ss_norm = g_ss_norm_best
            if verbose >= 1:
                print(f"  Restored best state: ||g_ss_best||={g_ss_norm_best:.2e}")

        if verbose >= 1 and not is_converged and outcome in ("floored", "oscillatory", "diverging"):
            print(f"  Stopped by classifier: outcome={outcome}, "
                  f"||g_ss||={g_ss_norm:.2e} ({self.norm_kind} norm)")
        elif verbose >= 1 and not is_converged:
            if g_ss_norm is None:
                reason = self.last_solver_failure_message or "no accepted timestep"
                print(f"  Did not converge after {num_timesteps} steps, "
                      f"no accepted step, dt={dt:.2e}, reason={reason}")
            else:
                print(f"  Did not converge after {num_timesteps} steps, "
                      f"final ||g_ss||={g_ss_norm:.2e}, dt={dt:.2e}")

        return finalize(is_converged, i + 1 if num_timesteps > 0 else 0)

    def _solve_by_temperature_continuation(self, T_offset=25.0, verbose=0):
        """Deterministic fallback for the multiplicity / dynamic-instability window.

        Some operating points (e.g. the 598 K temperature-sweep cases) have a
        steady state that exists but is dynamically unstable: pseudo-transient
        marching orbits it forever, and reaction continuation folds back before
        reaching factor_react = 1. The steady branch is, however, smoothly
        connected in inlet temperature to easily-converged neighbors. This
        method solves the same case at T_in + T_offset (cold, standard solver),
        then walks the inlet temperatures back to the target through a fixed
        ladder of warm-started steady Newton solves with line search. Newton is
        blind to dynamic stability, so it tracks the branch into the unstable
        region as long as each stage starts within its basin.

        The procedure is fully deterministic and self-contained: every stage is
        defined by the target configuration alone.

        Returns:
            True if the target-temperature stage converged; self.cpT then holds
            the solution of the ORIGINAL configuration. False otherwise
            (self.cpT is left unchanged).
        """
        import copy

        cfg0 = self._config
        cpT_entry = self.cpT.copy()
        fractions = (1.0, 0.6, 0.32, 0.16, 0.08, 0.04, 0.0)

        def stage_reactor(frac):
            """Build a fresh MembraneReactor with inlet/init temperatures offset by
            frac*T_offset and tightened per-stage tolerances.
            """
            cfg = copy.deepcopy(cfg0)
            dT = T_offset * frac
            cfg.T_ret_in = cfg0.T_ret_in + dT
            cfg.T_ret_init = cfg0.T_ret_init + dT
            cfg.T_perm_in = cfg0.T_perm_in + dT
            cfg.T_perm_init = cfg0.T_perm_init + dT
            # Tight per-stage Newton tolerances: a sloppily converged stage
            # poisons the warm start of the next one.
            cfg.rtol = min(cfg.rtol, 1e-5)
            cfg.atol = min(cfg.atol, 1e-4)
            cfg.try_direct_steady = True
            cfg.try_temperature_continuation = False  # no recursion
            return MembraneReactor(config=cfg)

        try:
            r_stage = stage_reactor(fractions[0])
            # status.converged also covers a "floored" outcome: a cold stage
            # stuck marginally above the target at its residual floor is a
            # perfectly good warm start (defect of 2026-08-16: a cold stage
            # at ss=1.01e-3 against a 1e-3 target was thrown away).
            status = r_stage.solve(return_status=True, verbose=0)
            if not status.converged:
                if verbose >= 1:
                    print(f"  T-continuation: cold stage at +{T_offset:.0f} K "
                          f"did not converge (ss={status.steady_state_norm:.2e})")
                return False
            prev_cpT = r_stage.cpT.copy()
            if verbose >= 2:
                print(f"  T-continuation: +{T_offset:.0f} K cold stage converged "
                      f"(ss={status.steady_state_norm:.2e}, outcome={status.outcome})")
            # Walk the ladder with a worklist so a failed stage can be
            # bisected toward the last good stage instead of aborting the
            # whole continuation (defect of 2026-08-16: the +4.0 K stage was
            # judged against 0.1x the target, "failed" at g=9.23e-4 with a
            # 1e-3 target, and every good state was discarded).
            remaining = list(fractions[1:])
            last_good_frac = fractions[0]
            bisect_budget = 8
            while remaining:
                frac = remaining[0]
                r_stage = stage_reactor(frac)
                r_stage.cpT[...] = prev_cpT
                r_stage._refresh_transport_state(p=r_stage.cpT[..., -2])
                res = r_stage._solve_cpT(
                    None, None, STEADY_STATE_DT,
                    use_line_search=True,
                    max_iter=5 * self.max_newton_iterations,
                )
                # Judge the stage against the caller's ACTUAL steady-state
                # target (in the configured norm), not Newton's own criterion
                # and not an arbitrary fraction of the target: what poisons
                # the next warm start is a sloppy state, not an unpolished
                # one, and a state at the target is by definition not sloppy.
                # A stage that missed the target but reduced its warm-start
                # residual substantially is also kept — it sits at (or near)
                # the scheme's floor for that temperature and is a better
                # warm start than the state it started from. Only a stage
                # whose Newton went sideways (branch fold, divergence) is a
                # failure worth bisecting.
                ss_stage = np.inf
                if np.isfinite(res.g_norm):
                    try:
                        ss_stage = r_stage._compute_steady_state_norm()
                    except RecoverableNumericalError:
                        ss_stage = np.inf
                stage_ok = (
                    res.converged
                    or ss_stage <= self.steady_state_atol
                    or (np.isfinite(ss_stage)
                        and res.g_norm_init > 0.0
                        and res.g_norm <= 0.3 * res.g_norm_init)
                )
                if stage_ok:
                    prev_cpT = r_stage.cpT.copy()
                    last_good_frac = frac
                    remaining.pop(0)
                    if verbose >= 2:
                        print(f"  T-continuation: stage +{T_offset * frac:.1f} K "
                              f"converged (g={res.g_norm:.2e}, ss={ss_stage:.2e})")
                    continue
                if bisect_budget > 0 and (last_good_frac - frac) > 0.005:
                    midpoint = 0.5 * (last_good_frac + frac)
                    remaining.insert(0, midpoint)
                    bisect_budget -= 1
                    if verbose >= 2:
                        print(f"  T-continuation: stage +{T_offset * frac:.1f} K "
                              f"failed (g={res.g_norm:.2e}, ss={ss_stage:.2e}), "
                              f"bisecting to +{T_offset * midpoint:.1f} K")
                    continue
                if verbose >= 1:
                    print(f"  T-continuation: stage +{T_offset * frac:.1f} K "
                          f"failed (g={res.g_norm:.2e}, ss={ss_stage:.2e})")
                return False
        except RecoverableNumericalError as exc:
            if verbose >= 1:
                print(f"  T-continuation: aborted ({exc})")
            self._restore_state(cpT_entry)
            return False

        # Final stage ran at the original temperatures: adopt its solution.
        self.cpT[...] = prev_cpT
        self._refresh_transport_state(p=self.cpT[..., -2])
        return True

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
        g_norm = None

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
            g_norm = None

            # --- Predictor Step ---
            if factor_react_prev_prev is not None:
                # Use a first-order (secant) predictor for a better initial guess
                d_factor_hist = factor_react_prev - factor_react_prev_prev
                if d_factor_hist > 1e-9:  # Avoid division by zero on retry
                    step_ratio = (self.factor_react - factor_react_prev) / d_factor_hist
                    self.cpT = cpT_prev + (cpT_prev - cpT_prev_prev) * step_ratio
                    # A long secant extrapolation can leave the physical
                    # region (e.g. negative temperatures) and crash the
                    # guarded kinetics. Fall back to the zero-order
                    # predictor when the extrapolated state is unphysical.
                    if (not np.all(np.isfinite(self.cpT))
                            or np.min(self.cpT[..., -1]) <= 100.0
                            or np.min(self.cpT[..., -2]) <= 0.0):
                        self.cpT = cpT_prev.copy()
                else:
                    self.cpT = cpT_prev.copy()
                self.kinetics.set_T_and_p(
                    T=self.cpT[..., -1][:, self.num_r_perm :], p=self.cpT[:, self.num_r_perm :, -2]
                )
                self._refresh_transport_state(p=self.cpT[..., -2])
            elif not is_first_step:
                # Use a zero-order predictor (the last solution) for the first step
                self.cpT = cpT_prev.copy()
                self.kinetics.set_T_and_p(
                    T=self.cpT[..., -1][:, self.num_r_perm :], p=self.cpT[:, self.num_r_perm :, -2]
                )
                self._refresh_transport_state(p=self.cpT[..., -2])

            # --- Corrector Step ---
            if verbose > 1:
                logger.info(
                    "Attempting factor_react = %.4f (step size = %.4f)...",
                    self.factor_react,
                    dfactor_react,
                )
            result = self._solve_cpT(
                c_old, T_old, dt,
                use_line_search=True,
                max_iter=2 * self.max_newton_iterations,
            )
            num_iters = result.num_iterations
            g_norm = result.g_norm
            is_converged = result.converged
            conv_factor = result.convergence_factor
            is_converging = np.isfinite(g_norm) and ((conv_factor < conv_factor_min) or is_converged)
            if is_converged and self.factor_react == factor_react_max:
                break
            # --- Adapt Step Size ---
            if is_converging:
                if verbose > 1:
                    logger.info(
                        "Converged in %d iterations: g_norm=%.4e, conv_factor=%.4e, ",
                        num_iters,
                        g_norm,
                        conv_factor,
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
            # Exit hygiene: leave the reactor at the last state that converged
            # (a valid solution of a reduced-reaction problem, useful as a warm
            # start) rather than a failed iterate.
            self._restore_state(cpT_prev)
        # Never leak a reduced factor_react: subsequent solves must target the
        # configured reaction strength regardless of how far continuation got.
        self.factor_react = factor_react_max
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
        diff_field_ret = self.correlation.diffusion(y_ret, T_ret, p_ret)
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
        diff_field_perm = self.correlation.diffusion(y_perm, T_perm, p_perm)
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

        Rg = self.Rg
        if T_perm.ndim > 1:
            T_perm_mem, _ = compute_boundary_values(
                T_perm, self.r_f_perm, self.r_c_perm, bc=None, axis=1, bound_id=1
            )
            T_ret_mem, _ = compute_boundary_values(
                T_ret, self.r_f_ret, self.r_c_ret, bc=None, axis=1, bound_id=0
            )
            perm = self.P0 * np.exp(-self.EA / (Rg * T_ret_mem[:, 0, np.newaxis]))  # (nz,nc)
            P_perm_mem = Rg * T_perm_mem[:, 0, np.newaxis] * perm
            P_ret_mem = Rg * T_ret_mem[:, 0, np.newaxis] * perm
        else:
            perm = self.P0 * np.exp(-self.EA / (Rg * T_ret))  # (nz,nc)
            P_perm_mem = np.broadcast_to(
                (Rg * T_perm * perm).reshape((1, -1)),
                (self.num_z, self.num_c),
            )
            P_ret_mem = np.broadcast_to(
                (Rg * T_ret * perm).reshape((1, -1)),
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

        # Use the same backward-flow BC as _construct_g_c so that compute_flows
        # diagnostics are consistent with the residual: backward-flow outlet cells
        # receive self.inflow_conc_ret_backflow, not the extrapolated interior value.
        u_ret_ax_local = self.u_ret_ax.copy()
        bc_ret_ax = get_axial_bcs_for_flow(
            is_counter_current=self.is_counter_current,
            u_ax=u_ret_ax_local,
            bc_inlet=self.BC_NONE,
            inflow_value=self.inflow_conc_ret_backflow,
        )

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
        u_ret_ax = u_ret_ax_local[..., np.newaxis]
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

