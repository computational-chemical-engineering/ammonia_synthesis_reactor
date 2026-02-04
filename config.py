"""
ReactorConfig: Validated configuration for the membrane reactor model.

Merges parameters from defaults, JSON files, and explicit overrides.
"""

from dataclasses import dataclass, field, fields
from typing import List, Callable, Optional, Any
import json
import os

import numpy as np
import scipy.constants as const

import defaults


@dataclass
class ReactorConfig:
    """Validated reactor configuration parameters.

    Configuration is applied in order of increasing priority:
    1. Default values from defaults.DEFAULTS
    2. Values from JSON config file (if provided)
    3. Explicit kwargs (highest priority)
    """

    # Species and database
    species: List[str] = field(default_factory=lambda: ["H2", "N2", "NH3"])
    database: str = "properties_database.json"

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
    Perm_NH3: float = 4e-7
    Sel_am_hy: float = 50.0
    Sel_am_ni: float = 1000.0
    Nu_ret: Callable = field(default_factory=lambda: lambda Re, Pr: 0.017 * Re**0.79)
    Nu_perm: Callable = field(
        default_factory=lambda: lambda Re, Pr: 0.023 * Re**0.8 * Pr**0.4
    )
    lambda_mem: float = 16.0

    # Reactor properties
    Nm: int = 1
    eps: float = 0.4
    Dcat: float = 1.0 / 3.0
    rho_c: float = 590.0
    dp: float = 2.5e-4

    # Solver settings
    dt: float = 1e6
    num_timesteps: int = 10
    num_newton_iterations: int = 10
    num_pressure_iterations: int = 5
    num_concentration_iterations: int = 1
    num_timesteps_max: int = 1
    rtol: float = 1e-6
    atol: float = 0.0
    rtol_p: float = 1e-6
    atol_p: float = 0.0
    rtol_c: float = 0.0
    atol_c: float = 0.0
    ord_norm: int = 2

    # Continuation settings
    newton_conv_rate_min: float = 3.0
    factor_react: float = 1.0
    dfactor_react_init: float = 1e-2
    dfactor_react_min: float = 1e-4
    dfactor_react_increase: float = 1.5
    dfactor_react_decrease: float = 0.5
    factor_p: float = 1.0

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
    y_ret_init: Any = field(default_factory=lambda: [0.75, 0.25, 0.0])
    y_perm_init: Any = field(default_factory=lambda: [0.0, 1.0, 0.0])
    y_ret_in: Any = field(default_factory=lambda: [0.75, 0.25, 0.0])
    y_perm_in: Any = field(default_factory=lambda: [0.0, 1.0, 0.0])

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
        if self.num_r < 4:
            raise ValueError(f"num_r must be at least 4, got {self.num_r}")
        if self.num_z < 4:
            raise ValueError(f"num_z must be at least 4, got {self.num_z}")
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
        # Start with defaults
        param_dict = defaults.DEFAULTS.copy()

        # Override with JSON file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                user_config = json.load(f)
            param_dict.update(user_config)

        # Override with explicit kwargs
        param_dict.update(kwargs)

        return cls.from_dict(param_dict)

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
