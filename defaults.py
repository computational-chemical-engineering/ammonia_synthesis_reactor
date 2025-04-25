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

    # Temperature and pressure settings
    "T_in": 273 + 380.0,  # Inlet temperature [K]
    "T_P_in": 273 + 380.0 - 100,  # Permeate side inlet temperature [K]
    "p_out": 29.83e5,  # Outlet pressure [Pa]
    "p_P_out": 1e5,  # Permeate side outlet pressure [Pa]
    "L": 1.0,  # Reactor length [m]
    "Lsealing": 0.05,  # Sealed section length [m]

    # Reactor dimensions
    "nu": 1, # geometry parameter: for cylindrical geometry nu=1, plate nu=0
    "r_min": 0.5e-2,  # Membrane outer radius [m]
    "r_max": 1.65e-2,  # Reactor outer radius [m]
    "r_min_P": 0,  # Minimum permeate side radius [m]
    "r_max_P": 0.35e-2,  # Maximum permeate side radius [m]

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

    # Gas properties
    "q": 1.5,  # Gas mixture ratio factor
    "yNH3_init": 0.045 / 1e3,  # Initial NH3 mole fraction

    # Inlet flow conditions
    "Fmass_NH3_target": 0.1408 * 150 / 148.52,  # NH3 mass flow target [kg/hr]
    "guess_conv": 0.2,  # Initial guess for NH3 conversion
    "SW": 0.05,  # Ratio permeate/retentate mole flow

    # Time stepping
    "dt": 0.1,  # Time step size
    "num_timesteps": 100,  # Number of time steps
    "num_inner_iter": 1,  # Number of inner iterations
    "is_restart": True  # Restart flag
}

