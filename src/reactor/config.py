"""
ReactorConfig: Validated configuration for the membrane reactor model.

Merges parameters from class defaults, JSON files, and explicit overrides.
"""

from dataclasses import dataclass, field, fields
from typing import List, Callable, Optional, Any
import json
import os

import numpy as np
from numpy.typing import NDArray
import scipy.constants as const


@dataclass
class ReactorConfig:
    """Validated reactor configuration parameters.

    Configuration is applied in order of increasing priority:
    1. Default values declared on ReactorConfig
    2. Values from JSON config file (if provided)
    3. Explicit kwargs (highest priority)
    """

    # Species and database
    species: List[str] = field(default_factory=lambda: ["H2", "N2", "NH3"])
    database: str = "data/properties_database.json"

    # Grid settings
    dim: int = 2
    num_r: int = 30
    num_z: int = 100

    # Physical constants
    Rg: float = const.R

    # Reactor geometry
    L: float = 1.0
    Lsealing: float = 0.05
    nu: int = 1  # 1 for cylindrical, 0 for plate
    r_min: float = 0.5e-2
    r_max: float = 1.65e-2
    r_min_perm: float = 0.0
    r_max_perm: float = 0.35e-2

    # Membrane characteristics
    # Pre-exponential factors and activation energies for Arrhenius equations are based on literature values for ammonia synthesis membranes, but can be overridden by users.
    # Perm = P0 * exp(-EA / (Rg * T_ret)) where P0 is the pre-exponential factor and EA is the activation energy. The values for NH3, H2, and N2 are set to typical values for ceramic membranes used in ammonia synthesis, but users can adjust these based on specific membrane materials or experimental data.
    #Perm_NH3: float = 4e-7 
    P0_NH3: float = 5.36e-8         #pre-exponential factor of Arrhenius equation
    EA_NH3: float = -6517.26        #activation energy of Arrhenius equation
    P0_H2: float = 5.36e-9
    EA_H2: float = -6517.26
    P0_N2: float = 5.36e-11
    EA_N2: float = -6517.26
    #Sel_am_hy: float = 50.0
    #Sel_am_ni: float = 1000.0
    #Nu_ret: Callable = field(default_factory=lambda: lambda Re, Pr: 0.017 * Re**0.79)
    Nu_ret: Callable = field(default_factory=lambda: lambda Re_particle, Pr: 0.19 * Re_particle**0.9 * Pr**0.33)  #correlation De Wasch & Fermont
    Nu_perm: Callable = field(
        default_factory=lambda: lambda Re, Pr: 0.023 * Re**0.8 * Pr**0.4
    )
    #lambda_mem: float = 16.0
    #%Conductivity of ceramic membrane (Aluminum oxide) 
    #lamda_mem=78.67*exp(-0.003*T)     ;             %Thermal conductivity of membrane (W/(m K))
    lambda_mem: Callable = field(default_factory=lambda: lambda T: 78.67 * np.exp(-0.003 * T))

    # Reactor properties
    Nm: int = 1
    eps: float = 0.4
    Dcat: float = 1.0 / 3.0
    rho_c: float = 590.0
    dp: float = 2.5e-4

    # Solver settings
    dt: float = 1e6
    dt_init: float = 1e-3
    dt_min: float = 1e-6
    dt_max: float = 1e3
    dt_increase_factor: float = 2.0
    dt_decrease_factor: float = 0.5
    adaptive_dt_threshold: float = 0.5
    steady_state_atol: float = 1e-6
    steady_state_rtol: Optional[float] = 1e-3
    # Steady-state residual norm (error-weighted RMS, CVODE convention).
    #   "absolute": unscaled RMS of the raw residual vector (historical).
    #   "weighted": error-weighted RMS (CVODE/IDA convention). Each residual
    #     row is divided by w_i = rtol_block*scale_i + atol_block with
    #     per-block tolerances for the concentration / pressure / temperature
    #     rows and weights recomputed from the current state; converged when
    #     the norm is <= steady_state_atol (canonically 1.0).
    # The two norms are NOT comparable: steady_state_atol changes meaning
    # with norm_kind. See reactor.convergence.steady_state_weights for the
    # block scales (atol_c/atol_p are relative to the reference density
    # p_ret_out/(Rg*T_ret_in); atol_T is Kelvin).
    #
    # The block tolerances are calibrated (2026-08-17 floor probes: G2_50,
    # G8_0.06, G7_0.02, G8_0.02 at 40x100; G1_50, G2_50, G2_3000 at 24x60)
    # so that the scheme's per-case accuracy floors land at wrms ~ 0.3-1.1
    # across grids and cases — i.e. wrms <= 1 with steady_state_atol = 1.0
    # is the canonical converged test, and one tolerance means the same
    # thing for every case. The temperature block is 10x looser than the
    # concentration block (relative): the T-rows' floor is grid-noise-
    # dominated and would otherwise dominate the norm for hot fast cases.
    norm_kind: str = "absolute"
    wrms_rtol_c: float = 1e-4
    wrms_atol_c_rel: float = 1e-7
    wrms_rtol_p: float = 1e-4
    wrms_atol_p_rel: float = 1e-7
    wrms_rtol_T: float = 1e-5
    wrms_atol_T: float = 1e-3
    # Items 2+3: KPI-stagnation stopping and outcome classification. When
    # enabled, "converged" additionally requires the outlet KPIs to be
    # stagnant (kpi_n_stagnant consecutive checks), a residual floor with
    # stagnant KPIs is accepted as "floored", and limit-cycle orbits are
    # classified "oscillatory" and handed to the escalation ladder early.
    # False (default) recovers the pure residual-threshold behavior.
    kpi_stop: bool = False
    kpi_check_every: int = 10
    kpi_rtol: float = 1e-3
    kpi_n_stagnant: int = 3
    # Budget policy: soft cap on accepted steps for the main march; past it a
    # KPI-stagnant case is accepted as "floored" even if dt has not reached
    # dt_max. num_timesteps stays the hard cap.
    steady_soft_step_cap: int = 400
    num_timesteps: int = 100
    max_newton_iterations: int = 50
    num_timesteps_max: int = 10
    rtol: float = 1e-6
    atol: float = 1e-4
    ord_norm: int = 2

    # Continuation settings
    newton_conv_rate_min: float = 3.0
    factor_react: float = 1.0
    dfactor_react_init: float = 1e-2
    dfactor_react_min: float = 1e-4
    dfactor_react_increase: float = 1.5
    dfactor_react_decrease: float = 0.5
    factor_p: float = 1.0
    factor_T: float = 1.0
    kinetics_uses_auxiliary_pressure: bool = False
    # Steady-Newton escalation of the pseudo-transient solver.
    # try_direct_steady: attempt a damped Newton solve of the steady equations
    #   (dt -> infinity, Armijo line search) either when pseudo-transient
    #   marching stalls (drift/plateau/limit cycle -- e.g. a dynamically
    #   unstable steady state that time marching orbits forever) or
    #   opportunistically once the steady-state residual has dropped by
    #   steady_jump_rel_drop, which is usually much cheaper than marching on.
    try_direct_steady: bool = True
    steady_jump_rel_drop: float = 1e-2
    drift_escalate_steps: int = 8
    # Last-resort fallback for operating points whose steady state is
    # dynamically unstable (marching orbits it): solve at a shifted inlet
    # temperature and walk back to the target with warm-started steady solves.
    try_temperature_continuation: bool = True
    # Sherwood-CP closure of the corrected 1D model (membrane_reactor_1d_corrected):
    # Sh = sh_cp_coeff * kappa**sh_cp_exp with kappa = r_mem/r_max (< 1).
    # The defaults are the manuscript's fit re-expressed provisionally; they
    # are superseded by the refit against the certified 2D dataset
    # (reactor.paper.closures) — always pass the fitted values explicitly
    # for paper runs.
    sh_cp_coeff: float = 6.13
    sh_cp_exp: float = -0.859
    # CP closure mode of the corrected 1D model:
    #   "sh_kappa"  fitted power law Sh(kappa) (geometry-only, chemistry-
    #               specific — must be refit for other kinetics);
    #   "screened"  mechanistic reaction-screening + conduction closure
    #               (reactor.cp_closure), evaluated from LOCAL bulk
    #               properties along z; screen_c1/screen_c2 are its two O(1)
    #               calibrated shape constants (screening resp. conduction).
    cp_closure: str = "sh_kappa"
    screen_c1: float = 1.0
    screen_c2: float = 1.0
    # Pressure-row formulation:
    #   "eos"        - algebraic EOS constraint c_tot(y, T, p) - sum(c) = 0 (original)
    #   "continuity" - total species continuity with the EOS-consistent
    #                  concentrations c_p = y * c_tot(y, T, p) inserted, so the
    #                  EOS enters through the density in total continuity rather
    #                  than as a separate algebraic equation (cf. pymrm-book L8).
    pressure_equation: str = "eos"

    # Flow rates
    F_ret_in: float = 0.1
    F_perm_in: float = 0.02
    is_counter_current: bool = False

    # Pressure settings
    p_ret_out: float = 29.83e5
    p_perm_out: float = 1e5

    # Temperature settings
    is_isothermal: bool = False
    T_ret_in: float = 273 + 380.0
    T_perm_in: float = 273 + 380.0 - 100
    T_ret_init: float = 273 + 380.0
    T_perm_init: float = 273 + 380.0 - 100

    # Gas compositions (will be converted to numpy arrays)
    y_ret_init: Any = field(default_factory=lambda: [0.666, 0.334, 0.0])
    y_perm_init: Any = field(default_factory=lambda: [0, 1, 0.0])
    y_ret_in: Any = field(default_factory=lambda: [0.666, 0.334, 0.0])
    y_perm_in: Any = field(default_factory=lambda: [0, 1, 0.0])

    def __post_init__(self):
        """Convert composition lists to numpy arrays and validate."""
        self.y_ret_init = np.asarray(self.y_ret_init, dtype=np.float64).reshape(
            (1, 1, -1)
        )
        self.y_perm_init = np.asarray(self.y_perm_init, dtype=np.float64).reshape(
            (1, 1, -1)
        )
        self.y_ret_in = np.asarray(self.y_ret_in, dtype=np.float64).reshape((1, 1, -1))
        self.y_perm_in = np.asarray(self.y_perm_in, dtype=np.float64).reshape(
            (1, 1, -1)
        )
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        if self.r_min >= self.r_max:
            raise ValueError(
                f"r_min ({self.r_min}) must be less than r_max ({self.r_max})"
            )
        if self.r_max_perm >= self.r_min:
            raise ValueError(
                f"r_max_perm ({self.r_max_perm}) must be less than r_min ({self.r_min})"
            )
        if self.L <= 0:
            raise ValueError(f"L must be positive, got {self.L}")
        if self.Lsealing < 0:
           raise ValueError(f"Lsealing must be non-negative, got {self.Lsealing}")
        if self.Lsealing >= self.L:
           raise ValueError(
        f"Lsealing ({self.Lsealing}) must be smaller than total length L ({self.L})"
    )
        if self.pressure_equation not in ("eos", "continuity"):
            raise ValueError(
                f"pressure_equation must be 'eos' or 'continuity', got {self.pressure_equation!r}"
            )
        if self.num_r < 4:
            raise ValueError(f"num_r must be at least 4, got {self.num_r}")
        if self.num_z < 4:
            raise ValueError(f"num_z must be at least 4, got {self.num_z}")
        if self.max_newton_iterations < 1:
            raise ValueError(
                f"max_newton_iterations must be at least 1, got {self.max_newton_iterations}"
            )
        if self.rtol < 0.0:
            raise ValueError(f"rtol must be non-negative, got {self.rtol}")
        if self.atol < 0.0:
            raise ValueError(f"atol must be non-negative, got {self.atol}")
        if self.steady_state_atol < 0.0:
            raise ValueError(
                f"steady_state_atol must be non-negative, got {self.steady_state_atol}"
            )
        if self.steady_state_rtol is not None and self.steady_state_rtol < 0.0:
            raise ValueError(
                "steady_state_rtol must be non-negative when provided, "
                f"got {self.steady_state_rtol}"
            )
        if self.norm_kind not in ("absolute", "weighted"):
            raise ValueError(
                f"norm_kind must be 'absolute' or 'weighted', got {self.norm_kind!r}"
            )
        for name in ("wrms_rtol_c", "wrms_atol_c_rel", "wrms_rtol_p",
                     "wrms_atol_p_rel", "wrms_rtol_T", "wrms_atol_T"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.norm_kind == "weighted":
            for rtol_name, atol_name in (
                ("wrms_rtol_c", "wrms_atol_c_rel"),
                ("wrms_rtol_p", "wrms_atol_p_rel"),
                ("wrms_rtol_T", "wrms_atol_T"),
            ):
                if getattr(self, rtol_name) == 0.0 and getattr(self, atol_name) == 0.0:
                    raise ValueError(
                        f"weighted norm needs {rtol_name} or {atol_name} > 0"
                    )
        if self.cp_closure not in ("sh_kappa", "screened"):
            raise ValueError(
                f"cp_closure must be 'sh_kappa' or 'screened', got {self.cp_closure!r}"
            )
        if self.kpi_check_every < 1:
            raise ValueError("kpi_check_every must be >= 1")
        if self.kpi_n_stagnant < 1:
            raise ValueError("kpi_n_stagnant must be >= 1")
        if len(self.species) != self.y_ret_in.shape[-1]:
            raise ValueError(
                f"Species count ({len(self.species)}) must match composition length ({self.y_ret_in.shape[-1]})"
            )

    @property
    def num_c(self) -> int:
        """Number of chemical species."""
        return len(self.species)

    @property
    def rho_b(self) -> float:
        """Bed density [kg_cat/m³_reactor]."""
        return self.rho_c * self.Dcat * (1 - self.eps)

    @classmethod
    def from_dict(cls, param_dict: dict) -> "ReactorConfig":
        """Create config from a dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in param_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_defaults(
        cls, config_file: Optional[str] = None, **kwargs
    ) -> "ReactorConfig":
        """Create config by merging defaults, JSON file, and kwargs.

        Args:
            config_file: Optional path to JSON configuration file.
            **kwargs: Explicit parameter overrides (highest priority).

        Returns:
            ReactorConfig with merged parameters.
        """
        # Start from dataclass defaults (single source of truth).
        param_dict = cls.default_dict()

        # Override with JSON file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                user_config = json.load(f)
            param_dict.update(user_config)

        # Override with explicit kwargs
        param_dict.update(kwargs)

        return cls.from_dict(param_dict)

    @classmethod
    def default_dict(cls) -> dict:
        """Return serializable class defaults as a plain dictionary."""
        return cls().to_dict()

    def to_dict(self) -> dict:
        """Export configuration to dictionary (for serialization)."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Convert numpy arrays back to lists for JSON compatibility
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # Skip callables (not serializable)
            elif callable(value):
                continue
            result[f.name] = value
        return result


# Backward-compatible dictionary-style access used by scripts.
# This now mirrors ReactorConfig dataclass defaults, not a separate defaults module.
DEFAULTS = ReactorConfig.default_dict()

def get_membrane_permeances(
    species: list,
    config: ReactorConfig,
    z_c: NDArray,
    Lsealing: float,
) -> NDArray:
    """Extract Arrhenius pre-exponential factors and activation energies for membrane permeance.

    Permeance follows an Arrhenius relation: Perm(T) = P0 * exp(-EA / (Rg * T)).
    The sealing region (z <= Lsealing) has zero pre-exponential factor (P0 = 0),
    effectively disabling permeation there.

    Args:
        species: List of species names. Each species must have corresponding
            ``P0_<sp>`` and ``EA_<sp>`` attributes on the config object.
        config: ReactorConfig instance supplying P0 and EA values per species.
        z_c: Axial cell centers, shape (num_z,).
        Lsealing: Length of sealing region [m] where permeation is suppressed (P0 = 0).

    Returns:
        P0: Pre-exponential factors, shape (num_z, num_species) [mol/(m².s.Pa)].
        EA: Activation energies, shape (1, num_species) [J/mol].
    """
    num_z = len(z_c)
    num_c = len(species)

    P0 = np.empty((1,num_c))
    EA = np.empty((1,num_c))
    for i, sp in enumerate(species):
        attr_name = f"P0_{sp}"
        P0[0,i] = getattr(config, attr_name)
        attr_name = f"EA_{sp}"
        EA[0,i] = getattr(config, attr_name)

    # Broadcast to full grid
    P0 = np.broadcast_to(P0, (num_z, num_c))

    # Zero out permeability in sealing region
    sealing_mask = z_c <= Lsealing
    if np.any(sealing_mask):
        P0 = P0.copy()
        P0[sealing_mask, :] = 0.0

    return P0, EA



