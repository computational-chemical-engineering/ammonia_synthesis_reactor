"""
Physics module for membrane reactor residual and Jacobian assembly.

Contains stateless functions for computing transport residuals (convection,
diffusion, reaction) and their Jacobians. Methods take explicit state and
operator objects rather than relying on class attributes.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array

from pymrm import (
    construct_coefficient_matrix,
    construct_convflux_upwind,
    interp_cntr_to_stagg,
    interp_cntr_to_stagg_tvd,
    upwind,
    update_csc_array_indices,
)


# =============================================================================
# Boundary condition templates
# =============================================================================
# Robin-type BCs: a*grad(u) + b*u = d

BC_NONE = {'a': 0, 'b': 0, 'd': 0}           # No boundary (internal face)
BC_DIRICHLET_HOM = {'a': 0, 'b': 1, 'd': 0}  # Homogeneous Dirichlet: u = 0
BC_NEUMANN_HOM = {'a': 1, 'b': 0, 'd': 0}    # Homogeneous Neumann: grad(u) = 0
BC_DIRICHLET = {'a': 0, 'b': 1, 'd': 1}      # Dirichlet with placeholder value
BC_NEUMANN = {'a': 1, 'b': 0, 'd': 1}        # Neumann with placeholder value


def make_dirichlet_bc(value: float) -> Dict[str, Any]:
    """Create a Dirichlet BC with specified value."""
    return {'a': 0, 'b': 1, 'd': value}


def make_neumann_bc(flux: float) -> Dict[str, Any]:
    """Create a Neumann BC with specified flux."""
    return {'a': 1, 'b': 0, 'd': flux}


def get_axial_bcs_for_flow(
    is_counter_current: bool,
    u_ax: NDArray,
    bc_inlet: Dict,
    bc_outlet: Dict,
    inflow_value: Optional[NDArray] = None,
) -> Tuple[Dict, Dict]:
    """Determine axial boundary conditions based on flow direction.

    For counter-current: inlet at z=L, outlet at z=0
    For co-current: inlet at z=0, outlet at z=L

    Handles reverse flow at outlet by switching to Dirichlet if needed.

    Args:
        is_counter_current: True if retentate flows opposite to permeate
        u_ax: Axial velocity at faces, shape (num_z+1, num_r)
        bc_inlet: BC dict for inlet boundary
        bc_outlet: BC dict for outlet boundary
        inflow_value: Optional value for inflow BC at outlet (reverse flow).
                      Can be scalar or array with shape matching field.

    Returns:
        Tuple of (bc_left, bc_right) for z=0 and z=L boundaries
    """
    # Determine extra dimensions for reshaping based on inflow_value
    inflow_arr = np.asarray(inflow_value) if inflow_value is not None else None
    extra_dims = max(0, inflow_arr.ndim - 2) if inflow_arr is not None else 0

    if is_counter_current:
        # Inlet at z=L (right), outlet at z=0 (left)
        bc_left = bc_outlet  # outlet
        bc_right = bc_inlet  # inlet
        # Check for reverse flow at outlet (z=0)
        is_inflow = u_ax[0, :] > 0
        if np.any(is_inflow) and inflow_value is not None:
            b_out = (is_inflow * 1.0).reshape((1, -1) + (1,) * extra_dims)
            a_out = 1.0 - b_out
            d_out = b_out * inflow_value
            bc_left = {'a': a_out, 'b': b_out, 'd': d_out}
    else:
        # Inlet at z=0 (left), outlet at z=L (right)
        bc_left = bc_inlet  # inlet
        bc_right = bc_outlet  # outlet
        # Check for reverse flow at outlet (z=L)
        is_inflow = u_ax[-1, :] < 0
        if np.any(is_inflow) and inflow_value is not None:
            b_out = (is_inflow * 1.0).reshape((1, -1) + (1,) * extra_dims)
            a_out = 1.0 - b_out
            d_out = b_out * inflow_value
            bc_right = {'a': a_out, 'b': b_out, 'd': d_out}

    return bc_left, bc_right


# =============================================================================
# Convection residual assembly
# =============================================================================

def assemble_convection_residual(
    c: NDArray,
    u_ax: NDArray,
    u_rad: NDArray,
    div_ax: csc_array,
    div_rad: csc_array,
    z_f: NDArray,
    z_c: NDArray,
    r_f: NDArray,
    r_c: NDArray,
    bc_ax: Tuple[Dict, Dict],
    bc_rad: Tuple[Dict, Dict] = (BC_NEUMANN_HOM, BC_NEUMANN_HOM),
    compute_jac: bool = False,
) -> Tuple[NDArray, Optional[csc_array]]:
    """Assemble convective transport residual for a single region.

    Args:
        c: Concentration field, shape (num_z, num_r, num_species)
        u_ax: Axial velocity at faces, shape (num_z+1, num_r)
        u_rad: Radial velocity at faces, shape (num_z, num_r+1)
        div_ax: Axial divergence operator
        div_rad: Radial divergence operator
        z_f, z_c: Axial face and cell-center coordinates
        r_f, r_c: Radial face and cell-center coordinates
        bc_ax: Axial boundary conditions (inlet, outlet)
        bc_rad: Radial boundary conditions (inner, outer)
        compute_jac: Whether to compute Jacobian

    Returns:
        Tuple of (residual, jacobian). Jacobian is None if compute_jac=False.
    """
    g = np.empty(c.shape)
    g_vect = g.ravel()

    # Expand velocity for species dimension
    u_ax_exp = u_ax[..., np.newaxis]
    u_rad_exp = u_rad[..., np.newaxis]

    # Axial convection with TVD/upwind
    c_ax, _ = interp_cntr_to_stagg_tvd(
        c, z_f, z_c, bc=bc_ax, v=u_ax_exp, tvd_limiter=upwind, axis=0
    )
    flux_ax = u_ax_exp * c_ax
    g_vect[:] = div_ax @ flux_ax.ravel()

    # Radial convection with TVD/upwind
    c_rad, _ = interp_cntr_to_stagg_tvd(
        c, r_f, r_c, bc=bc_rad, v=u_rad_exp, tvd_limiter=upwind, axis=1
    )
    flux_rad = u_rad_exp * c_rad
    g_vect[:] += div_rad @ flux_rad.ravel()

    if compute_jac:
        conv_matrix_ax, _ = construct_convflux_upwind(
            c.shape, z_f, z_c, bc=bc_ax, v=u_ax_exp, axis=0
        )
        jac = div_ax @ conv_matrix_ax

        conv_matrix_rad, _ = construct_convflux_upwind(
            c.shape, r_f, r_c, bc=bc_rad, v=u_rad_exp, axis=1
        )
        jac = jac + div_rad @ conv_matrix_rad
        return g, jac

    return g, None


def assemble_diffusion_residual(
    c: NDArray,
    T: NDArray,
    p: NDArray,
    correlation,  # GasMixtureCorrelations
    div_ax: csc_array,
    div_rad: csc_array,
    grad_ax: csc_array,
    grad_rad: csc_array,
    grad_bc_ax: NDArray,
    z_f: NDArray,
    z_c: NDArray,
    r_f: NDArray,
    r_c: NDArray,
    compute_jac: bool = False,
) -> Tuple[NDArray, Optional[csc_array], NDArray]:
    """Assemble diffusive transport residual for a single region.

    Args:
        c: Concentration field, shape (num_z, num_r, num_species)
        T: Temperature field, shape (num_z, num_r)
        p: Pressure field, shape (num_z, num_r)
        correlation: Gas mixture property correlator
        div_ax, div_rad: Divergence operators
        grad_ax, grad_rad: Gradient operators
        grad_bc_ax: Gradient boundary contribution
        z_f, z_c, r_f, r_c: Grid coordinates
        compute_jac: Whether to compute Jacobian

    Returns:
        Tuple of (residual, jacobian, bc_contribution).
    """
    shape_c = c.shape

    # Compute mole fractions and diffusivities
    c_tot = np.sum(c, axis=-1, keepdims=True)
    y = c / c_tot
    diff_field = correlation.diffusion(y, T, p)

    # Interpolate diffusivity to faces
    diff_ax = interp_cntr_to_stagg(diff_field, x_f=z_f, x_c=z_c, axis=0)
    diff_ax_mat = construct_coefficient_matrix(diff_ax, shape_c, axis=0)

    diff_rad = interp_cntr_to_stagg(diff_field, x_f=r_f, x_c=r_c, axis=1)
    diff_rad_mat = construct_coefficient_matrix(diff_rad, shape_c, axis=1)

    # Assemble diffusion Jacobian: div(-D*grad(c))
    jac_diff = (
        div_ax @ (-diff_ax_mat) @ grad_ax
        + div_rad @ (-diff_rad_mat) @ grad_rad
    )

    # Boundary contribution
    g_bc = div_ax @ ((-diff_ax_mat) @ grad_bc_ax)

    return jac_diff, g_bc


# =============================================================================
# Temperature residual helpers
# =============================================================================

def assemble_temperature_convection(
    T: NDArray,
    u_ax: NDArray,
    u_rad: NDArray,
    div_u: NDArray,
    div_ax: csc_array,
    div_rad: csc_array,
    z_f: NDArray,
    z_c: NDArray,
    r_f: NDArray,
    r_c: NDArray,
    bc_ax: Tuple[Dict, Dict],
    bc_rad: Tuple[Dict, Dict] = (BC_NEUMANN_HOM, BC_NEUMANN_HOM),
    compute_jac: bool = False,
) -> Tuple[NDArray, Optional[csc_array]]:
    """Assemble convective energy residual including -T*div(u) term.

    Args:
        T: Temperature field, shape (num_z, num_r)
        u_ax, u_rad: Velocity components at faces
        div_u: Velocity divergence at cell centers
        div_ax, div_rad: Divergence operators
        z_f, z_c, r_f, r_c: Grid coordinates
        bc_ax, bc_rad: Boundary conditions
        compute_jac: Whether to compute Jacobian

    Returns:
        Tuple of (residual, jacobian).
    """
    g = np.empty(T.shape)
    g_vect = g.ravel()

    # Axial convection
    T_ax, _ = interp_cntr_to_stagg_tvd(
        T, z_f, z_c, bc=bc_ax, v=u_ax, tvd_limiter=upwind, axis=0
    )
    flux_ax = u_ax * T_ax
    g_vect[:] = div_ax @ flux_ax.ravel()

    # Radial convection
    T_rad, _ = interp_cntr_to_stagg_tvd(
        T, r_f, r_c, bc=bc_rad, v=u_rad, tvd_limiter=upwind, axis=1
    )
    flux_rad = u_rad * T_rad
    g_vect[:] += div_rad @ flux_rad.ravel()

    # Velocity divergence term
    g_vect[:] -= (T * div_u).ravel()

    if compute_jac:
        conv_mat_ax, _ = construct_convflux_upwind(
            T.shape, z_f, z_c, bc=bc_ax, v=u_ax, axis=0
        )
        jac = div_ax @ conv_mat_ax

        conv_mat_rad, _ = construct_convflux_upwind(
            T.shape, r_f, r_c, bc=bc_rad, v=u_rad, axis=1
        )
        jac = jac + div_rad @ conv_mat_rad

        # Subtract diagonal for -T*div(u) term
        jac = jac - construct_coefficient_matrix(div_u)
        return g, jac

    return g, None
