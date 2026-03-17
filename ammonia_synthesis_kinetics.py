import numpy as np
from scipy import constants

from numerical_safety import (
    finite_minmax_context,
    guarded_compute,
    require_positive,
)

C_SMALL = 1e-4  # Small concentration to avoid division by zero (mol/m^3)
A_SMALL = 1e-4  # Small activity to avoid division by zero (bar)


class AmmoniaSynthesisKinetics:
    """
    A class to compute the reaction rates for ammonia synthesis.

    This class provides methods to set temperature and pressure, compute equilibrium constants,
    kinetic constants, adsorption constants, and fugacity_coeffs, and to calculate the reaction rates
    for ammonia synthesis.

    Attributes:
    -----------
    rate_constant : float
        The rate constant for the reaction.
    Rc : float
        Universal gas constant in cal/(mol·K).
    Ra : float
        Universal gas constant in J/(mol·K).
    axis : int
        The axis corresponding to the species index.
    rho_b : float
        Bulk density.
    rho_c : float
        Catalyst density.
    T : numpy.ndarray
        Temperature array.
    p : numpy.ndarray
        Pressure array.
    K_eq : numpy.ndarray
        Equilibrium constant array.
    k_f : numpy.ndarray
        Forward reaction rate constant array.
    K_H2 : numpy.ndarray
        Adsorption constant for H2.
    K_NH3 : numpy.ndarray
        Adsorption constant for NH3.
    fugacity_coeffs : numpy.ndarray
        Fugacity coefficients for the species.
    stoichiometry : numpy.ndarray
        Stoichiometric coefficients for the reaction.

    Methods:
    --------
    __init__(self, species=["H2", "N2", "NH3"], T=None, p=None, rho_b=None, rho_c=None, axis=-1):
        Initialize the class with species, temperature, pressure, and other parameters.

    pow(self, c, n):
        Compute the power of concentrations with a small constant to avoid division by zero.

    set_T_and_p(self, T=None, p=None):
        Set the temperature and pressure, and compute related constants.

    compute_K_eq(self, T):
        Compute the equilibrium constant for the reaction.

    compute_kinetic_constant(self, T):
        Compute the forward reaction rate constant.

    compute_adsorption_constants(self, T):
        Compute the adsorption constants for H2 and NH3.

    compute_fugacity_coeffs(self, T, p):
        Compute the fugacity coefficients for the species.

    __call__(self, c, T=None, p=None):
        Compute the reaction rates for ammonia synthesis.
    """

    def __init__(
        self,
        species=["H2", "N2", "NH3"],
        T=None,
        p=None,
        rho_b=None,
        rho_c=None,
        axis=-1
    ):
        """
        Initialize the class with species, temperature, pressure, and other parameters.

        Parameters:
        -----------
        species : list, optional
            List of species names. Defaults to ["H2", "N2", "NH3"].
        T : numpy.ndarray, optional
            Temperature array. Defaults to None.
        p : numpy.ndarray, optional
            Pressure array. Defaults to None.
        rho_b : float, optional
            Bulk density. Defaults to None.
        rho_c : float, optional
            Catalyst density. Defaults to None.
        axis : int, optional
            The axis corresponding to the species index. Defaults to -1.
        """
        self.rate_constant = None
        self.Rc = (
            constants.R / constants.calorie
        )  # Universal gas constant in cal/(mol·K)
        self.Ra = constants.R
        self.axis = axis
        self.rho_b = rho_b
        self.rho_c = rho_c
        self.set_T_and_p(T, p)
        self.index_H2 = species.index("H2")
        self.index_N2 = species.index("N2")
        self.index_NH3 = species.index("NH3")
        self.stoichiometry = np.zeros(len(species))
        self.stoichiometry[self.index_H2] = -3
        self.stoichiometry[self.index_N2] = -1
        self.stoichiometry[self.index_NH3] = 2

    def pow(self, c, n):
        """
        Compute the power of concentrations with a small constant to avoid division by zero.

        Parameters:
        -----------
        c : numpy.ndarray
            Concentration array.
        n : float
            Exponent.

        Returns:
        --------
        numpy.ndarray
            Resulting array after applying the power operation.
        """
        return c * (np.abs(c + C_SMALL) + C_SMALL) ** (n - 1)

    def set_T_and_p(self, T=None, p=None):
        """
        Set the temperature and pressure, and compute related constants.

        Parameters:
        -----------
        T : numpy.ndarray, optional
            Temperature array. Defaults to None.
        p : numpy.ndarray, optional
            Pressure array. Defaults to None.
        """
        if T is None and p is None:
            return
        if T is not None:
            self.T = np.array(T)
            self.compute_K_eq(self.T)
            self.compute_kinetic_constant(self.T)
            self.compute_adsorption_constants(self.T)
        if p is not None:
            self.p = np.array(p)
        if self.T is not None and self.p is not None:
            self.compute_fugacity_coeffs(self.T, self.p)
        self.rate_constant = self.k_f * 1000.0 * self.rho_b / 3600.0 / self.rho_c

    def compute_K_eq(self, T):
        """
        Compute the equilibrium constant for the reaction.

        Parameters:
        -----------
        T : numpy.ndarray
            Temperature array.

        Returns:
        --------
        numpy.ndarray
            Equilibrium constant array.
        """
        # self.K_eq = T**(-2.691122) * 10**(-5.519265e-5 * T + 1.848863e-7 * T**2 + 2001.6 / T + 2.6899)
        require_positive("temperature", T, context=finite_minmax_context("T", T))
        self.K_eq = guarded_compute(
            "computing K_eq",
            lambda: np.exp(
                -2.691122 * np.log(T)
                + np.log(10.0)
                * ((-5.519265e-5 + 1.848863e-7 * T) * T + 2001.6 / T + 2.6899)
            ),
            **finite_minmax_context("T", T),
        )
        return self.K_eq

    def compute_kinetic_constant(self, T):
        """
        Compute the forward reaction rate constant.

        Parameters:
        -----------
        T : numpy.ndarray
            Temperature array.

        Returns:
        --------
        numpy.ndarray
            Forward reaction rate constant array.
        """
        require_positive("temperature", T, context=finite_minmax_context("T", T))
        self.k_f = guarded_compute(
            "computing kinetic constant",
            lambda: 9.02e8 * np.exp(-23000.0 / (self.Rc * T)),
            **finite_minmax_context("T", T),
        )
        return self.k_f

    def compute_adsorption_constants(self, T):
        """
        Compute the adsorption constants for H2 and NH3.

        Parameters:
        -----------
        T : numpy.ndarray
            Temperature array.

        Returns:
        --------
        tuple
            Adsorption constants for H2 and NH3.
        """
        require_positive("temperature", T, context=finite_minmax_context("T", T))
        self.K_H2 = guarded_compute(
            "computing H2 adsorption constant",
            lambda: np.exp(-13.6 / self.Rc + 9000.0 / (self.Rc * T)),
            **finite_minmax_context("T", T),
        )
        self.K_NH3 = guarded_compute(
            "computing NH3 adsorption constant",
            lambda: np.exp(-8.3 / self.Rc + 7000.0 / (self.Rc * T)),
            **finite_minmax_context("T", T),
        )
        return self.K_H2, self.K_NH3

    def compute_fugacity_coeffs(self, T, p):
        """
        Compute the fugacity coefficients for the species.

        Parameters:
        -----------
        T : numpy.ndarray
            Temperature array.
        p : numpy.ndarray
            Pressure array.

        Returns:
        --------
        numpy.ndarray
            Fugacity coefficients for the species.
        """
        ndim = max(T.ndim, p.ndim, self.axis) + 1
        axis = self.axis if self.axis >= 0 else ndim + self.axis
        T_loc = T.reshape(T.shape + (1,) * (ndim - 1 - T.ndim))
        p_loc = 1e-5 * p.reshape(
            p.shape + (1,) * (ndim - 1 - p.ndim)
        )  # local pressures in bar
        shape = tuple([max(s1, s2) for s1, s2 in zip(T_loc.shape, p_loc.shape)]) + (3,)
        self.fugacity_coeffs = np.empty(shape)
        slices = [slice(None)] * ndim
        slices[axis] = 0
        context = {**finite_minmax_context("T", T), **finite_minmax_context("p_bar", p_loc)}
        require_positive("temperature", T, context=context)
        require_positive("pressure", p, context=context)
        self.fugacity_coeffs[tuple(slices)] = guarded_compute(
            "computing H2 fugacity coefficients",
            lambda: np.exp(
                np.exp(-3.8402 * T_loc**0.125 + 0.541) * p_loc
                - np.exp(-0.1263 * np.sqrt(T_loc) - 15.98) * p_loc**2
                + 300.0
                * (np.exp(-0.011901 * T_loc - 5.941))
                * (np.exp(-p_loc / 300.0) - 1.0)
            ),
            **context,
        )
        slices[axis] = 1
        self.fugacity_coeffs[tuple(slices)] = (
            0.93431737
            + 0.3101804e-3 * T_loc
            + 0.295896e-3 * p_loc
            - 0.2707279e-6 * T_loc**2
            + 0.4775207e-6 * p_loc**2
        )
        slices[axis] = 2
        self.fugacity_coeffs[tuple(slices)] = (
            0.1438996
            + 0.2028538e-2 * T_loc
            - 0.4487672e-3 * p_loc
            - 0.1142945e-5 * T_loc**2
            + 0.2761216e-6 * p_loc**2
        )
        return self.fugacity_coeffs

    def __call__(self, p_partial, T=None):
        """
        Compute the reaction rates for ammonia synthesis.

        Parameters:
        -----------
        p_partial : numpy.ndarray
            Partial pressures [Pa]
        T : numpy.ndarray, optional
            Temperature array. Defaults to None.

        Returns:
        --------
        numpy.ndarray
            Reaction rates for ammonia synthesis. [mol/m3]
        """
        # slices = [slice(None)] * c.ndim
        # slices[self.axis] = [self.index_H2, self.index_N2, self.index_NH3]
        p_loc = np.take(
            p_partial, (self.index_H2, self.index_N2, self.index_NH3), axis=self.axis
        )
        p = np.sum(p_loc, axis=self.axis)
        self.set_T_and_p(T, p)

        activities = 1e-5 * p_loc * self.fugacity_coeffs
        a_H2 = np.take(activities, 0, axis=self.axis)
        a_N2 = np.take(activities, 1, axis=self.axis)
        a_NH3 = np.take(activities, 2, axis=self.axis)
        rate = guarded_compute(
            "computing ammonia synthesis rate",
            lambda: (
                self.rate_constant
                * (
                    self.pow(a_N2, 0.5)
                    * self.pow(a_H2, 0.375)
                    / (A_SMALL + np.maximum(a_NH3, 0.0)) ** 0.25
                    - (1.0 / self.K_eq)
                    * self.pow(a_NH3, 0.75)
                    / (A_SMALL + np.maximum(a_H2, 0.0)) ** 1.125
                )
                / (
                    1.0
                    + self.K_H2 * np.abs(a_H2) ** 0.3
                    + self.K_NH3 * np.abs(a_NH3) ** 0.2
                )
            ),
            T_min=float(np.nanmin(self.T)),
            T_max=float(np.nanmax(self.T)),
            p_min=float(np.nanmin(p_loc)),
            p_max=float(np.nanmax(p_loc)),
        )
        shape = [1] * p_partial.ndim
        shape[self.axis] = -1
        rates = np.expand_dims(rate, self.axis) * self.stoichiometry.reshape(shape)
        return rates
