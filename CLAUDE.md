# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2D axisymmetric membrane reactor model for ammonia synthesis (N2 + 3H2 ⇌ 2NH3). Simulates coupled mass, momentum, and energy transport with catalytic reaction and selective permeation between retentate and permeate regions.

**Current state**: Monolithic implementation on `feat/monolithic` branch, with planned refactoring to modular architecture documented in `refactor.md`.

## Running the Code

```bash
# Single reactor simulation
python -c "
from membrane_reactor import MembraneReactor
reactor = MembraneReactor(L=1.0, r_max=0.0165, p_ret_out=29.83e5, T_ret_in=643, F_ret_in=0.1)
reactor.solve(num_timesteps=100)
"

# Run all case studies (reads case_studies.csv)
python run_case_studies.py

# Interactive development/testing
jupyter notebook membrane_reactor.ipynb
```

## Architecture

### Core Modules

- **membrane_reactor.py**: Main `MembraneReactor` class (~1300 lines). Handles discretization, field initialization, residual assembly, Newton solver with line search, and adaptive time-stepping.
- **ammonia_synthesis_kinetics.py**: `AmmoniaSynthesisKinetics` class for Langmuir-Hinshelwood reaction kinetics and equilibrium calculations.
- **gas_mixture_correlations.py**: `GasMixtureCorrelations` for H2-N2-NH3 mixture properties (Peng-Robinson EOS, Wilke viscosity, thermal conductivity, diffusion).
- **mixture_property_database.py**: JSON database loader for species thermodynamic properties.
- **defaults.py**: Central parameter dictionary (geometry, solver settings, inlet/outlet conditions).

### Spatial Domain

- 2D axisymmetric (z-radial) non-uniform grid
- Retentate: annular region (r_min to r_max) with packed catalyst bed
- Permeate: cylindrical region (0 to r_max_perm) with sweep gas
- Membrane interface couples regions via permeation flux

### Solution Strategy

1. Initialize with non-reactive pressure/velocity calculation
2. Pseudo-transient continuation with adaptive time-stepping
3. Newton-Raphson with Armijo line search for concentration-pressure (coupled)
4. Separate Newton solve for temperature
5. Adaptive reaction factor ramp (0→1) for stability

### Key Physics

- **Pressure**: Ergun equation for packed bed, Darcy law
- **Mass**: Species convection-diffusion with reaction source
- **Membrane**: Solution-diffusion permeation model (NH3 selective)
- **Energy**: Convection-conduction with exothermic reaction heat

## Dependencies

- numpy, scipy (sparse matrices, linear algebra)
- pandas (case studies, property database)
- matplotlib (visualization in notebooks)
- pymrm (custom library for non-uniform grids and FVM operators)

## Key Files

- `properties_database.json`: Species thermodynamic data (critical properties, cp polynomials, binary diffusion)
- `case_studies.csv`: 11 validation test cases (sweep ratios, GHSV, geometry variations)
- `ammonia_synthesis_data_rossetti_et_al.csv`: Experimental validation data
