# defaults.py

import numpy as np
import math
import scipy.constants as const

DEFAULTS = {
    # Species and database
    "species": ["H2", "N2", "NH3"],  # Chemical species in the system
    "database": "properties_database.json",  # Path to properties database

    # Grid settings
    "dim": 2,  # Number of dimensions
    "num_r": 30,  # Number of radial grid points
    "num_z": 100,  # Number of axial grid points

    # Physical constants
    "Rg": const.R,  # Universal gas constant [J/(mol·K)]

    "L": 1.0,  # Reactor length [m]
    "Lsealing": 0.05,  # Sealed section length [m]

    # Reactor dimensions
    "nu": 1, # geometry parameter: for cylindrical geometry nu=1, plate nu=0
    "r_min": 0.5e-2,  # Membrane outer radius [m]
    "r_max": 1.65e-2,  # Reactor outer radius [m]
    "r_min_perm": 0,  # Minimum permeate side radius [m]
    "r_max_perm": 0.35e-2,  # Maximum permeate side radius [m]

    # Membrane characteristics
    "Perm_NH3": 4e-7,  # NH3 permeability [mol/m²·s·Pa]
    "Sel_am_hy": 50.0,  # Selectivity NH3/H2
    "Sel_am_ni": 1000.0,  # Selectivity NH3/N2
    "Nu_ret": (lambda Re, Pr: 0.017 * Re**0.79), # Nusselt correlation retentate side
    "Nu_perm": (lambda Re, Pr: 0.023*Re**0.8 * Pr**0.4), # Nusselt correlation permeate side
    "lambda_mem": 16.0, # Thermal conductivity membrane    

    # Reactor properties
    "Nm": 1,  # Number of membranes
    "eps": 0.4,  # Bed voidage (m³ void/m³ reactor)
    "Dcat": 1.0/3.0,  # Catalyst dilution factor (with inerts like SiC)
    "rho_c": 590.0,  # Catalyst density [kg/m³ catalyst]
    "dp": 2.5e-4,  # Catalyst particle diameter [m]
    
    # Solver settings
    "dt": 1e6,  # Time step size
    "num_timesteps": 10,  # Number of time steps / outer iterations
    "num_newton_iterations": 10,  # Maximum number of outer iterations
    "num_pressure_iterations": 5,  # Maximum number of inner iterations in pressure solver
    "num_concentration_iterations": 1,  # Maximum number of inner iterations in concentration solver
    "rtol": 1e-6,  # Convergence relative tolerance for Newton's method
    "atol": 0.0,  # Convergence absolute tolerance for Newton's method
    "rtol_p": 1e-6,  # Relative tolerance for pressure convergence
    "atol_p": 0.0,  # Absolute tolerance for pressure convergence
    "rtol_c": 0.0,  # Relative tolerance for concentration convergence
    "atol_c": 0.0,  # Absolute tolerance for concentration convergence
    "ord_norm": 2,  # Order of norm for convergence criteria

    # factor_react is the continuation factor for the reaction rate
    # This factor is adaptively cocntrolled
    "newton_conv_rate_min": 3.0,
    "factor_react": 1.0,  # Reaction rate factor
    "dfactor_react_init": 1.0,  # Initial reaction rate factor increment
    "dfactor_react_min": 1e-4,  # Minimum value for dfactor_react
    "dfactor_react_increase": 1.5,  # Factor by which to increase dfactor_react
    "dfactor_react_decrease": 0.5,  # Factor by which to decrease dfactor_react
    "penalty_p": 0.0,
    # Molar flow rates
    "F_ret_in": 0.1,   # Inlet molar flow rate [mol/s]
    "F_perm_in": 0.02, # Inlet molar flow rate [mol/s]
    "is_counter_current": False,  # Counter-current flow if True, co-current if False
    
    # Pressure settings
    "p_ret_out": 29.83e5, # Retentate side outlet pressure [Pa]
    "p_perm_out": 1e5,    # Permeate side outlet pressure [Pa]
    
    # Temperature and pressure settings
    "is_isothermal": False,  # If True, the reactor is isothermal
    "T_ret_in": 273 + 380.0,  # Inlet temperature [K]
    "T_perm_in": 273 + 380.0 - 100,  # Permeate side inlet temperature [K]
    "T_ret_init": 273 + 380.0,  # Inlet temperature [K]
    "T_perm_init": 273 + 380.0 - 100,  # Permeate side inlet temperature [K]

    # Gas concentrations
    "y_ret_init": [0.75, 0.25, 0.0],
    "y_perm_init": [0.0, 1.0, 0.0],
    "y_ret_in": [0.75, 0.25, 0.0],   # Inlet mole fractions in retentate
    "y_perm_in": [0.0, 1.0, 0.0],  # Initial mole fractions in retentate

}

