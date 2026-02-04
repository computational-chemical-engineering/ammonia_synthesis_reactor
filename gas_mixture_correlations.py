"""
This module provides the GasMixtureCorrelations class and utility functions to compute various properties of gas mixtures.

The GasMixtureCorrelations class includes methods to compute properties such as density, viscosity,
thermal conductivity, specific heat, enthalpy, and diffusivity of a gas mixture using the Peng-Robinson
equation of state and Wilke correlation.

Classes:
--------
GasMixtureCorrelations:
    A class to compute various properties using correlations for gas mixtures.

Functions:
----------
compute_roots(coeffs):
    Compute roots for a whole ND array of polynomials.

get_tri_shapes(shapes_species=None, shapes_nonspecies=None, axis=-1):
    Compute reshaped forms of input shapes for consistent three-dimensional processing.
"""

import math
import numpy as np
from scipy import constants
from scipy.interpolate import CubicSpline
from mixture_property_database import MixturePropertyDatabase


class GasMixtureCorrelations:
    """
    A class to compute various properties using correlations for gas mixture.

    This class provides methods to compute properties such as density, viscosity,
    thermal conductivity, specific heat, enthalpy, and diffusivity of a gas mixture
    using the Peng-Robinson equation of state and Wilke correlation.

    Attributes:
    -----------
    species : list
        List of species names.
    num_species : int
        Number of species in the mixture.
    Mw : numpy.ndarray
        Molecular weights of the species.
    Tc : numpy.ndarray
        Critical temperatures of the species.
    Pc : numpy.ndarray
        Critical pressures of the species.
    omega : numpy.ndarray
        Acentric factors of the species.
    mi : numpy.ndarray
        Peng-Robinson parameter for the species.
    aip_sqrt : numpy.ndarray
        Square root of the Peng-Robinson parameter 'a' for the species.
    bi : numpy.ndarray
        Peng-Robinson parameter 'b' for the species.
    wilke_kij : numpy.ndarray
        Binary interaction parameters for the Wilke correlation.
    wilke_C1, wilke_C2, wilke_C3, wilke_C4 : numpy.ndarray
        Coefficients for the Wilke correlation.
    therm_cond_C1, therm_cond_C2, therm_cond_C3, therm_cond_C4 : numpy.ndarray
        Coefficients for the thermal conductivity correlation.
    c_p_C1, c_p_C2, c_p_C3, c_p_C4, c_p_C5 : numpy.ndarray
        Coefficients for the specific heat capacity correlation.
    Dij_inv : numpy.ndarray
        Inverse of the binary diffusion coefficients.
    dH_f : numpy.ndarray
        Standard enthalpies of formation for the species.

    Methods:
    --------
    density(y, T, p, axis=-1):
        Compute the density of a gas mixture using the Peng-Robinson (PR) equation of state.
    viscosity(y, T, axis=-1):
        Compute the viscosity of a gas mixture using the Wilke correlation.
    thermal_conductivity(y, T, axis=-1):
        Compute the thermal conductivity of a gas mixture.
    get_species_specific_heat_spline(T, dT=5.0, T_ref=None):
        Get a cubic spline for the specific heat capacity of the species.
    species_specific_heat(T, axis=-1):
        Compute the specific heat capacity of each species.
    specific_heat(y, T, axis=-1):
        Compute the specific heat capacity of the gas mixture.
    species_enthalpies(T, T_ref=298.15, dT=5.0, axis=-1):
        Compute the enthalpies of the species.
    enthalpy(y, T, axis=-1):
        Compute the enthalpy of the gas mixture.
    diffusion(y, T, p, axis=-1):
        Compute the species diffusivities in a gas mixture.
    """

    def __init__(self, species, filename):
        """
        Initialize the GasMixtureProperties class.

        Parameters:
        - species (list): List of species names.
        - filename (str): Path to the file containing species properties.
        """
        self.species = species
        self.num_species = len(species)
        db = MixturePropertyDatabase(filename)
        properties = db.get_species_properties("species_properties", species)
        shape_t = (1, self.num_species, 1)
        shape_matrix = (1, self.num_species, self.num_species, 1)
        self.Mw = properties["Mw"]
        self.Tc = properties["Tc"]
        self.Pc = properties["Pc"]
        Tc_t = self.Tc.reshape(shape_t)
        Pc_t = self.Pc.reshape(shape_t)
        self.omega = properties["omega"].reshape(shape_t)
        self.mi = np.where(
            self.omega < 0.491,
            0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2,
            0.379642
            + 1.48503 * self.omega
            - 0.164423 * self.omega**2
            + 0.016666 * self.omega**3,
        )
        R = constants.R
        self.aip_sqrt = np.sqrt(0.457235529 * R**2 * Tc_t**2 / Pc_t)
        self.bi = 0.0777960739 * R * Tc_t / Pc_t
        binary_properties = db.get_species_properties("binary_properties", species)
        self.wilke_kij = binary_properties["wilke_kij"].reshape(shape_matrix)
        self.wilke_kij[np.isnan(self.wilke_kij)] = 0
        self.wilke_C1 = properties["wilke_C1"].reshape(shape_t)
        self.wilke_C2 = properties["wilke_C2"].reshape(shape_t)
        self.wilke_C3 = properties["wilke_C3"].reshape(shape_t)
        self.wilke_C4 = properties["wilke_C4"].reshape(shape_t)
        self.therm_cond_C1 = properties["therm_cond_C1"].reshape(shape_t)
        self.therm_cond_C2 = properties["therm_cond_C2"].reshape(shape_t)
        self.therm_cond_C3 = properties["therm_cond_C3"].reshape(shape_t)
        self.therm_cond_C4 = properties["therm_cond_C4"].reshape(shape_t)
        self.c_p_C1 = properties["c_p_C1"].reshape((1, -1))
        self.c_p_C2 = properties["c_p_C2"].reshape((1, -1))
        self.c_p_C3 = properties["c_p_C3"].reshape((1, -1))
        self.c_p_C4 = properties["c_p_C4"].reshape((1, -1))
        self.c_p_C5 = properties["c_p_C5"].reshape((1, -1))
        Dij = binary_properties["Dij"].reshape(shape_matrix)
        self.Dij_inv = 1.0 / Dij
        self.Dij_inv[np.isnan(self.Dij_inv)] = 0
        self.dH_f = properties["dH_f"].reshape(shape_t)

    def molar_density(self, y, T, p, axis=-1):
        """
        Compute the molar density of a gas mixture using the Peng-Robinson (PR) equation of state.

        Parameters:
        -----------
        y : numpy.ndarray
            Mole fraction array of each species.
        T : numpy.ndarray
            Temperature array. [K]
        p : numpy.ndarray
            Pressure array. [Pa]
        axis : int, optional
            The axis corresponding to the species index in `y`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Molar density of the gas mixture.

        The method follows these steps:
        1. Convert inputs to numpy arrays and adjust their dimensions if necessary.
        2. Compute the temperature and pressure dependent parameters.
        3. Calculate the coefficients for the cubic equation of state.
        4. Solve the cubic equation to find the compressibility factor (Z).
        5. Compute the molar density using the compressibility factor and the molecular weights of the species.
        """
        y = np.asarray(y)
        T = np.asarray(T)
        p = np.asarray(p)
        shape_t_y, shapes_t, _, shape_out = get_tri_shapes(
            shapes_species=y.shape, shapes_nonspecies=[T.shape, p.shape], axis=axis
        )
        y.reshape(shape_t_y)
        T_t = T.reshape(shapes_t[0])
        p_t = p.reshape(shapes_t[1])
        self.Tc.reshape((1, -1, 1))

        # y_sqrt_ai = y_t*self.aip_sqrt*np.abs(1.0 + self.mi*(1.0 - np.sqrt(T_t/Tc_t)))
        # y_sqrt_ai_j = np.expand_dims(y_sqrt_ai, axis=1)
        # a = np.sum(y_sqrt_ai*np.sum((1.0-self.wilke_kij)*y_sqrt_ai_j, axis=2), axis=1, keepdims=True)
        # b = np.sum(y_t*self.bi, axis=1, keepdims=True)
        R = constants.R
        # A = (a * p_t) / ((R * T_t)**2)
        # B = (b * p_t) / (R * T_t)
        # coeffs = np.stack([-((A - B - B**2) * B), A - 2 * B - 3 * B**2, -(1 - B), np.ones_like(B)],axis=-1)
        # roots = compute_roots(coeffs)
        # roots = np.real_if_close(roots, tol=1e-6)
        # real_roots = np.where(np.isreal(roots), np.real(roots), -np.inf)
        # Z = np.max(real_roots, axis=-1)
        # c = p_t / (R * T_t * Z)
        c = p_t / (R * T_t)  # Initial molar density without compressibility factor
        return c.reshape(shape_out)

    def molecular_weight(self, y, axis=-1):
        """
        Compute the molecular weight of a gas mixture.

        Parameters:
        -----------
        y : numpy.ndarray
            Mole fraction array of each species.
        axis : int, optional
            The axis corresponding to the species index in `y`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Molecular weight of the gas mixture.

        """
        shape_Mw = [1] * y.ndim
        shape_Mw[axis] = self.num_species
        Mw = np.sum(y * self.Mw.reshape(shape_Mw), axis=axis) / np.sum(
            y + 1e-13, axis=axis
        )
        return Mw

    def density(self, y, T, p, axis=-1):
        """
        Compute the density of a gas mixture using the Peng-Robinson (PR) equation of state.

        Parameters:
        -----------
        y : numpy.ndarray
            Mole fraction array of each species.
        T : numpy.ndarray
            Temperature array. [K]
        p : numpy.ndarray
            Pressure array. [Pa]
        axis : int, optional
            The axis corresponding to the species index in `y`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Density of the gas mixture.

        """
        c = self.molar_density(y, T, p, axis)
        Mw = self.molecular_weight(y, axis)
        rho = c * Mw
        return rho

    def viscosity(self, y, T, axis=-1):
        """
        Compute the viscosity of a gas mixture using the Wilke correlation.

        Parameters:
        -----------
        y : numpy.ndarray
            Mole fraction array of each species.
        T : numpy.ndarray
            Temperature array. [K]
        axis : int, optional
            The axis corresponding to the species index in `y`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Mixture viscosity (Pa·s).
        """
        y = np.asarray(y)
        T = np.asarray(T)
        shape_t_y, shape_t_T, _, shape_out = get_tri_shapes(
            shapes_species=y.shape, shapes_nonspecies=T.shape, axis=axis
        )
        y_t = y.reshape(shape_t_y)
        T_t = T.reshape(shape_t_T)

        # Compute pure component viscosities (mu_i) using the Wilke correlation
        mu_i = (
            self.wilke_C1
            * T_t**self.wilke_C2
            * (1 + self.wilke_C3 / T_t + self.wilke_C4 / T_t**2)
        )  # Shape: same as y

        # Expand dimensions to compute interaction terms phi_ij
        mu_i_exp = np.expand_dims(mu_i, 2)  # Shape: (..., num_species, 1)
        mu_j_exp = np.expand_dims(mu_i, 1)  # Shape: (..., 1, num_species)

        Mw_t = self.Mw.reshape((1, -1, 1))
        Mw_i_exp = np.expand_dims(Mw_t, 2)  # Shape: (..., num_species, 1)
        Mw_j_exp = np.expand_dims(Mw_t, 1)  # Shape: (..., 1, num_species)

        # Compute the interaction parameter φ_ij (broadcasted properly)
        phi_ij = (
            (1 + (mu_i_exp / mu_j_exp) ** 0.5 * (Mw_i_exp / Mw_j_exp) ** 0.25) ** 2
        ) / ((8 * (1 + Mw_i_exp / Mw_j_exp)) ** 0.5)

        # Compute summation term in Wilke's formula (denominator)
        y_j = np.expand_dims(y_t, 1)  # Shape: (..., num_species, 1)
        phi_sum = np.sum(phi_ij * y_j, axis=2)  # Sum over species axis

        # Compute final mixture viscosity
        mu_mix = np.sum(y_t * mu_i / phi_sum, axis=1)  # Sum over species axis
        return mu_mix.reshape(shape_out)

    def thermal_conductivity(self, y, T, axis=-1):
        """
        Compute the thermal conductivity of a gas mixture.

        Parameters:
        -----------
        y : numpy.ndarray
            Mole fraction array of each species.
        T : numpy.ndarray
            Temperature array.
        axis : int, optional
            The axis corresponding to the species index in `y`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Mixture thermal conductivity (W/m·K).
        """
        y = np.asarray(y)
        T = np.asarray(T)
        shape_t_y, shape_t_T, _, shape_out = get_tri_shapes(
            shapes_species=y.shape, shapes_nonspecies=T.shape, axis=axis
        )
        y_t = y.reshape(shape_t_y)
        T_t = T.reshape(shape_t_T)

        # Compute pure component viscosities (mu_i) using the Wilke correlation
        mu_i = (
            self.wilke_C1
            * T_t**self.wilke_C2
            * (1 + self.wilke_C3 / T_t + self.wilke_C4 / T_t**2)
        )  # Shape: same as y

        # Expand dimensions to compute interaction terms phi_ij
        mu_i_exp = np.expand_dims(mu_i, 2)  # Shape: (..., num_species, 1)
        mu_j_exp = np.expand_dims(mu_i, 1)  # Shape: (..., 1, num_species)

        Mw_t = self.Mw.reshape((1, -1, 1))
        Mw_i_exp = np.expand_dims(Mw_t, 2)  # Shape: (..., num_species, 1)
        Mw_j_exp = np.expand_dims(Mw_t, 1)  # Shape: (..., 1, num_species)

        # Compute the interaction parameter φ_ij (broadcasted properly)
        phi_ij = (
            (1 + (mu_i_exp / mu_j_exp) ** 0.5 * (Mw_i_exp / Mw_j_exp) ** 0.25) ** 2
        ) / ((8 * (1 + Mw_i_exp / Mw_j_exp)) ** 0.5)

        # Compute summation term in Wilke's formula (denominator)
        y_j = np.expand_dims(y_t, 1)  # Shape: (..., num_species, 1)
        phi_sum = np.sum(phi_ij * y_j, axis=2)  # Sum over species axis

        kg_i = (
            self.therm_cond_C1
            * T_t**self.therm_cond_C2
            * (1 + self.therm_cond_C3 / T_t + self.therm_cond_C4 / T_t**2)
        )
        kg_mix = np.sum(y_t * kg_i / phi_sum, axis=1)  # Sum over species axis
        return kg_mix.reshape(shape_out)

    def get_species_specific_heat_spline(self, T, dT=5.0, T_ref=None):
        """
        Get a cubic spline for the specific heat capacity of the species.

        Parameters:
        -----------
        T : numpy.ndarray
            Temperature array.
        dT : float, optional
            Temperature step for spline generation. Defaults to 5.0.
        T_ref : float, optional
            Reference temperature. Defaults to None.

        Returns:
        --------
        CubicSpline
            Cubic spline for the specific heat capacity of the species.
        """
        T_lin = np.asarray(T).ravel()
        T_min = np.min(T_lin)
        T_max = np.max(T_lin)
        if T_ref is not None:
            T_min = min(T_min, T_ref)
            T_max = max(T_max, T_ref)

        if (
            hasattr(self, "c_p_spline")
            and T_min >= self.c_p_spline.x[0]
            and T_max <= self.c_p_spline.x[-1]
        ):
            return self.c_p_spline

        def c_p_func(T):
            T_lin = T.reshape((-1, 1))
            return 1e-3 * (
                self.c_p_C1
                + self.c_p_C2
                * ((self.c_p_C3 / T_lin) / np.sinh(self.c_p_C3 / T_lin)) ** 2
                + self.c_p_C4
                * ((self.c_p_C5 / T_lin) / np.cosh(self.c_p_C5 / T_lin)) ** 2
            )

        if T_min == T_max:
            T_min -= dT
            T_max += dT
        num_T = max(6, int(np.ceil(T_max - T_min) / dT))
        range_T = np.linspace(T_min, T_max, num_T)
        self.c_p_spline = CubicSpline(
            range_T, c_p_func(range_T), axis=0, bc_type="natural", extrapolate=True
        )
        return self.c_p_spline

    def species_specific_heat(self, T, axis=-1):
        """
        Compute the specific heat capacity of each species.

        Parameters:
        -----------
        T : numpy.ndarray
            Temperature array.
        axis : int, optional
            The axis corresponding to the species index in `T`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Specific heat capacity of each species (J/mol·K).
        """
        T = np.asarray(T)
        shape_t_T, shape_out, _ = get_tri_shapes(shapes_nonspecies=T.shape, axis=axis)
        T_t = T.reshape(shape_t_T)
        axis = axis if axis >= 0 else len(shape_out) + axis
        shape_out = shape_out[:axis] + (self.num_species,) + shape_out[axis + 1 :]

        # Ensure correct broadcasting for species properties

        c_p_spline = self.get_species_specific_heat_spline(T_t.ravel())
        c_p = c_p_spline(T_t.ravel()).reshape(
            (shape_t_T[0], shape_t_T[2], self.num_species)
        )
        c_p = np.permute_dims(c_p, (0, 2, 1)).reshape(shape_out)
        return c_p

    def specific_heat(self, c, T, axis=-1):
        """
        Compute the specific heat capacity of the gas mixture.

        Parameters:
        -----------
        c : numpy.ndarray
            Molar concentration of each species.
        T : numpy.ndarray
            Temperature array.
        axis : int, optional
            The axis corresponding to the species index in `c`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Specific heat capacity of the gas mixture (J/m^3·K).
        """
        c = np.asarray(c)
        T = np.asarray(T)
        shape_t_c, shape_t_T, _, shape_out = get_tri_shapes(
            shapes_species=c.shape, shapes_nonspecies=T.shape, axis=axis
        )
        c_t = c.reshape(shape_t_c)
        T_t = T.reshape(shape_t_T)
        c_p_s = self.species_specific_heat(T_t, axis=axis).reshape(
            (shape_t_T[0], self.num_species, shape_t_T[2])
        )
        c_p_mix = np.sum(c_t * c_p_s, axis=1)
        return c_p_mix.reshape(shape_out)

    def species_enthalpies(self, T, T_ref=298.15, dT=5.0, axis=-1):
        """
        Compute the enthalpies of the species.

        Parameters:
        -----------
        T : numpy.ndarray
            Temperature array.
        T_ref : float, optional
            Reference temperature. Defaults to 298.15 K.
        dT : float, optional
            Temperature step for spline generation. Defaults to 5.0.
        axis : int, optional
            The axis corresponding to the species index in `T`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Enthalpies of the species (J/kg).
        """
        T = np.asarray(T)
        shape_t_T, _, shape_out = get_tri_shapes(shapes_nonspecies=T.shape, axis=axis)
        c_p_spline = self.get_species_specific_heat_spline(T, dT=dT, T_ref=T_ref)
        H_spline = c_p_spline.antiderivative()
        H = H_spline(T.ravel()).reshape((shape_t_T[0], shape_t_T[2], -1))
        H = (
            np.permute_dims(H, (0, 2, 1))
            - H_spline(T_ref).reshape((1, -1, 1))
            + self.dH_f
        )
        axis = axis if axis > 0 else self.num_species + axis
        shape_out = shape_out[:axis] + (-1,) + shape_out[axis:]
        return H.reshape(shape_out)

    def enthalpy(self, y, T, axis=-1):
        """
        Compute the enthalpy of the gas mixture.

        Parameters:
        -----------
        y : numpy.ndarray
            Mole fraction array of each species.
        T : numpy.ndarray
            Temperature array.
        axis : int, optional
            The axis corresponding to the species index in `y`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Enthalpy of the gas mixture (J/kg).
        """
        y = np.asarray(y)
        T = np.asarray(T)
        shape_t_y, shape_t_T, _, shape_out = get_tri_shapes(
            shapes_species=y.shape, shapes_nonspecies=T.shape, axis=axis
        )
        y_t = y.reshape(shape_t_y)
        T.reshape(shape_t_T)
        H_s = self.species_enthalpies(T, axis=axis).reshape(
            (shape_t_T[0], -1, shape_t_T[2])
        )
        H = np.sum(y_t * H_s, axis=1)
        return H.reshape(shape_out)

    def diffusion(self, y, T, p, axis=-1):
        """
        Compute the species diffusivities in a gas mixture.

        Parameters:
        -----------
        y : numpy.ndarray
            Mole fraction array of each species.
        T : numpy.ndarray
            Temperature array.
        p : numpy.ndarray
            Pressure array.
        axis : int, optional
            The axis corresponding to the species index in `y`. Defaults to `-1` (last axis).

        Returns:
        --------
        numpy.ndarray
            Species diffusivities (m²/s).
        """
        y = np.asarray(y)
        T = np.asarray(T)
        p = np.asarray(p)
        shape_t_y, shapes_t, shape_out, _ = get_tri_shapes(
            shapes_species=y.shape, shapes_nonspecies=[T.shape, p.shape], axis=axis
        )
        y_t = y.reshape(shape_t_y)
        T_t = T.reshape(shapes_t[0])
        p_t = p.reshape(shapes_t[1])

        y_t = np.where(y_t < 1e-10, 1e-10, y_t)
        y_t = y_t / np.sum(y_t, axis=1, keepdims=True)
        y_j = np.expand_dims(y_t, axis=1)
        D_sc = (1.0 - y_t) / np.sum(y_j * self.Dij_inv, axis=2)
        D_i = D_sc * (T_t / 273.0) ** 1.75 * (1e5 / p_t)

        return D_i.reshape(shape_out)


def compute_roots(coeffs):
    """
    Compute roots for a whole ND array of polynomials.

    Args:
        coeffs (ndarray): ND array where the last axis contains polynomial coefficients.

    Returns:
        ndarray: ND array of roots, with an extra dimension for roots.
    """
    shape = coeffs.shape[:-1]  # Shape of the ND grid
    degree = coeffs.shape[-1] - 1  # Polynomial degree
    roots = np.full(shape + (degree,), np.nan, dtype=np.complex128)  # Store roots

    for index in np.ndindex(shape):  # Loop over all indices except last axis
        poly_coeffs = coeffs[index]  # Extract 1D polynomial
        poly_roots = np.polynomial.polynomial.polyroots(poly_coeffs)
        roots[index] = poly_roots  # Store roots, pad with NaN if needed

    return roots


def get_tri_shapes(shapes_species=None, shapes_nonspecies=None, axis=-1):
    """
    Compute reshaped forms of input shapes for consistent three-dimensional processing.

    Parameters:
    -----------
    axis : int, optional
        The axis corresponding to the species index in `shapes_species`. Defaults to `-1` (last axis).

    shapes_species : tuple, list of tuples, or object, optional
        A single shape tuple or a list of shape tuples for species-dependent variables.
        If not provided, it will be ignored in the output.

    shapes_nonspecies : tuple, list of tuples, or object, optional
        A single shape tuple or a list of shape tuples for nonspecies-dependent variables.
        If not provided, it will be ignored in the output.

    Returns:
    --------
    If no input is provided for `shapes_species` or `shapes_nonspecies`, the corresponding return values are omitted.

    """

    # Detect if input was provided
    has_species = shapes_species is not None
    has_nonspecies = shapes_nonspecies is not None

    # If neither input was provided, return nothing
    if not has_species and not has_nonspecies:
        return

    # Convert single shape tuples to lists internally
    is_single_species = isinstance(shapes_species, tuple)
    is_single_nonspecies = isinstance(shapes_nonspecies, tuple)

    if is_single_species:
        shapes_species = [shapes_species]
    elif not has_species or shapes_species is None:
        shapes_species = []

    if is_single_nonspecies:
        shapes_nonspecies = [shapes_nonspecies]
    elif not has_nonspecies or shapes_nonspecies is None:
        shapes_nonspecies = []

    shapes_t_species = []
    shapes_t_nonspecies = []

    # Determine max number of dimensions among input shapes
    ndim = max(
        max((len(shape) for shape in shapes_species), default=0) if has_species else 0,
        max((len(shape) + 1 for shape in shapes_nonspecies), default=0)
        if has_nonspecies
        else 0,
    )

    # Ensure axis is valid
    axis = axis if axis >= 0 else ndim + axis
    if not (0 <= axis < ndim):
        raise ValueError(f"Invalid axis {axis} for shapes with ndim {ndim}")

    shape_out = np.zeros(ndim, dtype=int)

    # Process species shapes if provided
    if has_species:
        for shape in shapes_species:
            np.maximum.at(shape_out, slice(None, len(shape)), shape)
            shape_t = (
                (math.prod(shape), 1, 1)
                if axis >= len(shape)
                else (
                    math.prod(shape[:axis]),
                    shape[axis],
                    math.prod(shape[axis + 1 :]),
                )
            )
            shapes_t_species.append(shape_t)

    # Process nonspecies shapes if provided
    if has_nonspecies:
        for shape in shapes_nonspecies:
            if axis >= len(shape):
                shape_t = (math.prod(shape), 1, 1)
                np.maximum.at(shape_out, slice(None, len(shape)), shape)
            else:
                shape_t = (math.prod(shape[:axis]), 1, math.prod(shape[axis + 1 :]))
                np.maximum.at(shape_out, slice(None, axis), shape[:axis])
                np.maximum.at(shape_out, slice(axis + 1, len(shape)), shape[axis + 1 :])
            shapes_t_nonspecies.append(shape_t)

    # Convert output back to a tuple if the input was a single shape
    if is_single_species:
        shapes_t_species = shapes_t_species[0]
    if is_single_nonspecies:
        shapes_t_nonspecies = shapes_t_nonspecies[0]

    shape_out_nonspecies = tuple(np.delete(shape_out, axis))
    # Return only the outputs that were provided
    if has_species and has_nonspecies:
        return (
            shapes_t_species,
            shapes_t_nonspecies,
            tuple(shape_out),
            shape_out_nonspecies,
        )
    elif has_species:
        return shapes_t_species, tuple(shape_out), shape_out_nonspecies
    elif has_nonspecies:
        return shapes_t_nonspecies, tuple(shape_out), shape_out_nonspecies
