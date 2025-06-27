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
    "num_r": 50,  # Number of radial grid points
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

    # Reactor properties
    "Nm": 1,  # Number of membranes
    "eps": 0.4,  # Bed voidage (m³ void/m³ reactor)
    "Dcat": 1.0/3.0,  # Catalyst dilution factor (with inerts like SiC)
    "rho_c": 590.0,  # Catalyst density [kg/m³ catalyst]
    "dp": 2.5e-4,  # Catalyst particle diameter [m]
    
    # Solver settings
    "dt": np.inf,  # Time step size
    "num_timesteps": 1,  # Number of time steps
    "num_newton_iterations": 5,  # Maximum number of inner iterations
    "num_pressure_iterations": 2,
    "rtol": 1e-3,  # Convergence relative tolerance for Newton's method
    "atol": 0.0,  # Convergence absolute tolerance for Newton's method
    "rtol_p": 1e-4,  # Relative tolerance for pressure convergence
    "atol_p": 0.0,  # Absolute tolerance for pressure convergence
    "rtol_dc": 0.0,  # Relative tolerance for determining steady state
    "atol_dc": 0.0,  # Absolute tolerance for determining steady state
    
    # Molar flow rates
    "F_ret_in": 0.1,   # Inlet molar flow rate [mol/s]
    "F_perm_in": 0.02, # Inlet molar flow rate [mol/s]
    "is_counter_current": True,  # Counter-current flow if True, co-current if False
    
    # Pressure settings
    "p_ret_out": 29.83e5, # Retentate side outlet pressure [Pa]
    "p_perm_out": 1e5,    # Permeate side outlet pressure [Pa]
    
    # Temperature and pressure settings
    "T_ret_in": 273 + 380.0,  # Inlet temperature [K]
    "T_perm_in": 273 + 380.0 - 100,  # Permeate side inlet temperature [K]
    "T_ret_init": 273 + 380.0,  # Inlet temperature [K]
    "T_perm_init": 273 + 380.0 - 100,  # Permeate side inlet temperature [K]
    
    # Gas concentrations
    "y_ret_init": [0.333, 0.333, 0.333],
    "y_perm_init": [0.333, 0.333, 0.333],
    "y_ret_in": [0.6, 0.4, 0],  # Inlet mole fractions in retentate
    "y_perm_in": [0.6, 0.4, 0],  # Initial mole fractions in retentate    

}

