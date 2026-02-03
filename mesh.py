"""
ReactorMesh: 2D axisymmetric grid for membrane reactor with retentate/permeate regions.

Handles grid generation and provides helpers for accessing region-specific data.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from pymrm import non_uniform_grid
from config import ReactorConfig


class ReactorMesh:
    """2D axisymmetric mesh with separate retentate and permeate regions.

    The mesh consists of:
    - Axial direction (z): shared between both regions
    - Radial direction (r): split into permeate (inner) and retentate (outer)

    Grid layout in radial direction:
        [0, r_max_perm] = permeate region (num_r_perm points)
        [r_min, r_max] = retentate region (num_r_ret points)

    The full radial array has num_r = num_r_perm + num_r_ret points,
    with permeate indices [0:num_r_perm] and retentate indices [num_r_perm:].
    """

    def __init__(self, config: ReactorConfig):
        """Create mesh from configuration.

        Args:
            config: Reactor configuration with geometry parameters.
        """
        self.config = config

        # Compute radial point distribution
        self.num_r_perm = int(np.round(config.r_max_perm / config.r_max * config.num_r)) + 3
        self.num_r_ret = config.num_r - self.num_r_perm
        if self.num_r_ret < 2:
            self.num_r_ret = 2
            self.num_r_perm = config.num_r - self.num_r_ret

        self.num_r = config.num_r
        self.num_z = config.num_z

        self._create_grids()

    def _create_grids(self):
        """Generate non-uniform grids for both regions."""
        config = self.config

        # Retentate radial grid (outer region)
        dr_ret = (config.r_max - config.r_min) / max(self.num_r_ret - 10, 0.8 * self.num_r_ret)
        self.r_f_ret = non_uniform_grid(
            config.r_min, config.r_max, self.num_r_ret + 1, dr_ret, 1.2
        )
        self.r_c_ret = 0.5 * (self.r_f_ret[:-1] + self.r_f_ret[1:])

        # Permeate radial grid (inner region)
        dr_perm = (config.r_max_perm - config.r_min_perm) / max(self.num_r_perm - 10, 0.8 * self.num_r_perm)
        self.r_f_perm = non_uniform_grid(
            config.r_min_perm, config.r_max_perm, self.num_r_perm + 1, dr_perm, 1.0 / 1.2
        )
        self.r_c_perm = 0.5 * (self.r_f_perm[:-1] + self.r_f_perm[1:])

        # Axial grid (shared): uniform in sealing region, non-uniform after
        num_z_sealing = int(np.round(config.Lsealing / config.L * config.num_z))
        z_f_uniform = np.linspace(0, config.Lsealing, num_z_sealing + 1)
        dz_nonuniform = (config.L - config.Lsealing) / max(
            config.num_z - num_z_sealing - 8, 0.8 * (config.num_z - num_z_sealing)
        )
        z_f_non_uniform = non_uniform_grid(
            config.Lsealing, config.L, config.num_z + 1 - num_z_sealing, dz_nonuniform, 1.2
        )
        self.z_f = np.concatenate((z_f_uniform, z_f_non_uniform[1:]), axis=0)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])

    # =========================================================================
    # Region extraction helpers
    # =========================================================================

    def get_permeate_slice(self) -> slice:
        """Return slice for permeate region in radial axis."""
        return slice(0, self.num_r_perm)

    def get_retentate_slice(self) -> slice:
        """Return slice for retentate region in radial axis."""
        return slice(self.num_r_perm, self.num_r)

    def get_permeate_data(self, full_field: NDArray) -> NDArray:
        """Extract permeate region from a full field.

        Args:
            full_field: Array with shape (num_z, num_r, ...).

        Returns:
            View of permeate region with shape (num_z, num_r_perm, ...).
        """
        return full_field[:, :self.num_r_perm, ...]

    def get_retentate_data(self, full_field: NDArray) -> NDArray:
        """Extract retentate region from a full field.

        Args:
            full_field: Array with shape (num_z, num_r, ...).

        Returns:
            View of retentate region with shape (num_z, num_r_ret, ...).
        """
        return full_field[:, self.num_r_perm:, ...]

    def split_perm_ret(self, full_field: NDArray) -> Tuple[NDArray, NDArray]:
        """Split full field into permeate and retentate views.

        Args:
            full_field: Array with shape (num_z, num_r, ...).

        Returns:
            Tuple of (permeate_view, retentate_view).
        """
        return self.get_permeate_data(full_field), self.get_retentate_data(full_field)

    # =========================================================================
    # Shape helpers
    # =========================================================================

    def shape_full(self, num_components: int = 1) -> Tuple[int, ...]:
        """Shape for full field with optional component dimension."""
        if num_components == 1:
            return (self.num_z, self.num_r)
        return (self.num_z, self.num_r, num_components)

    def shape_retentate(self, num_components: int = 1) -> Tuple[int, ...]:
        """Shape for retentate-only field."""
        if num_components == 1:
            return (self.num_z, self.num_r_ret)
        return (self.num_z, self.num_r_ret, num_components)

    def shape_permeate(self, num_components: int = 1) -> Tuple[int, ...]:
        """Shape for permeate-only field."""
        if num_components == 1:
            return (self.num_z, self.num_r_perm)
        return (self.num_z, self.num_r_perm, num_components)

    # =========================================================================
    # Index offset helpers (for monolithic matrix assembly)
    # =========================================================================

    def retentate_offset(self, num_components: int = 1) -> Tuple[int, ...]:
        """Offset tuple for shifting retentate indices in monolithic arrays."""
        if num_components == 1:
            return (0, self.num_r_perm)
        return (0, self.num_r_perm, 0)

    @property
    def r_membrane(self) -> float:
        """Membrane radius (interface between regions)."""
        return self.config.r_min
