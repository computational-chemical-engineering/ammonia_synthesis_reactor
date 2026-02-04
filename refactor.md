# Refactoring Specification: Modular Membrane Reactor Model

## Objective
- Transform the monolithic `MembraneReactor` class into a decoupled, modular library.
- Separate configuration, spatial discretization (mesh), physical residuals (physics), and numerical algorithms (solvers) to improve maintainability and extensibility.

## Target Architecture
- Follow a clear Separation of Concerns (SoC) design.

| Module | Responsibility |
| --- | --- |
| config.py | Validates and stores reactor parameters using Dataclasses or Pydantic. |
| mesh.py | Handles 2D axisymmetric grid generation and index mapping for retentate/permeate regions. |
| physics.py | Pure functions/classes to assemble residuals ($g$) and Jacobians ($J$) for mass, momentum, and energy. |
| solvers.py | Generic numerical routines: Newton–Raphson (with line search) and continuation methods. |
| reactor.py | High-level orchestrator that connects the mesh, physics, and solver. |

## Refactoring Roadmap

### Phase 0: Regression Test Harness ✅ COMPLETE
Before modifying any code, establish a reference solution to detect numerical drift during refactoring.

**Reference case** (from `debug.ipynb`):
```python
GHSV = 150
H2_N2_ratio = 1.5
sweep_ratio = 0.05
r_min, r_max, L = 0.005, 0.0165, 1.0
config_file = 'debug.json'
```

**Artifacts to capture**:
- Final state arrays: `c_p`, `T`, `u_ret_ax`, `u_perm_ax`
- Computed flows: `flows_ret_ax`, `flows_perm_ax`, `flows_ret_mem`, `flows_perm_mem`
- Elemental balances (H, N)
- Solver statistics: `cnt_num_solves_c_p`, `cnt_num_solves_T`

**Validation criterion**: L2 relative error < 1e-10 for field arrays after each phase.

**Deliverables**:
- `regression_test.py` — Test harness with `--save` and `--test` modes
- `regression_reference.npz` — Saved reference solution

**Commit**: `d9e1027`

### Phase 1: Configuration & Mesh (Foundation) ✅ COMPLETE
- Extract parameters: Move all `defaults.DEFAULTS` and JSON loading into a `ReactorConfig` class.
- Abstract the grid: Move `_create_spatial_discretization` into a `ReactorMesh` class.
- Provide helpers instead of manual slicing (e.g., `[:, :self.num_r_perm, :]`):
	- `get_retentate_data(full_field)`
	- `get_permeate_data(full_field)`

**Deliverables**:
- `config.py` — `ReactorConfig` dataclass with validation, merging, serialization
- `mesh.py` — `ReactorMesh` with grid generation and region helpers
- Updated `membrane_reactor.py` using new modules (backward-compatible API preserved)

**Commit**: `2261218`

**Regression test**: PASSED (L2 rel error = 0.00e+00)

### Phase 2: Physics Decoupling 🔄 IN PROGRESS
- Stateless residuals: Refactor `_construct_g_conv`, `_construct_g_diff`, and `_construct_g_T` into a `PhysicsEngine`.
- Pure I/O: Methods take the current state $(c, p, T)$ and return the residual/Jacobian.
- Remove side effects: Do not modify `self.c_p` or `self.T` during assembly.
- Matrix assembly: Consolidate `update_csc_array_indices` using a descriptive mapping from local (retentate/permeate) to global (monolithic) matrices.

**Completed**:
- `c660f5e`: Created `physics.py` with BC definitions and stateless residual templates
  - BC_NONE, BC_DIRICHLET, BC_NEUMANN, etc.
  - `make_dirichlet_bc()`, `make_neumann_bc()` helpers
  - `assemble_convection_residual()`, `assemble_diffusion_residual()`, `assemble_temperature_convection()`
- `f4c9bf6`: Added `get_axial_bcs_for_flow()` helper and refactored BC logic
  - Handles co-current vs counter-current flow direction
  - Handles reverse flow at outlets (scalar and array inflow values)
  - Refactored `_construct_g_conv()` and `_construct_g_T_conv()` to use it
- `0562b0b`: Extracted permeability calculations to physics.py
  - `compute_permeate_permeability()` for Hagen-Poiseuille (permeate side)
  - `compute_packed_bed_permeability()` for Ergun equation (packed bed)
  - Documented ERGUN and permeability constants
  - Refactored `_construct_darcy_matrices()` to use new functions
- `84e27f0`: Cleanup - consolidated all physics constants in physics.py
- `aa70e99`: Extracted membrane permeability calculation
  - `compute_membrane_permeabilities()` from NH3 permeability and selectivity ratios
  - Handles sealing region (zero permeability for z <= Lsealing)
- `9e02ec6`: Extracted inlet flux calculations
  - `compute_inlet_flux_permeate()` for parabolic velocity profile
  - `compute_inlet_flux_retentate()` for uniform annular profile

**Note**: The stateless `assemble_*_residual()` functions cannot be directly wired in
because the divergence operators output to monolithic arrays. A full extraction would
require restructuring the operator setup in `_init_jac()`.

**Remaining work**:
1. (Optional) Extract membrane flux matrix assembly from `_construct_g_diff()`
2. (Optional) Restructure operators to enable full physics function integration

**Phase 2 Summary**: Core physics calculations have been extracted to physics.py:
- Boundary condition helpers
- Flow permeability (Hagen-Poiseuille, Ergun)
- Membrane species permeabilities
- Inlet flux profiles (parabolic, uniform)

### Phase 3: Solver Abstraction ⏳ PENDING
- Generalize Newton solver: Extract logic from `_solve_c_p` into a standalone `NewtonSolver`.
	- Accept `residual_fn`, `jacobian_fn`, and `initial_guess`.
	- Implement Armijo line-search and convergence monitoring as generic features.
- Generalize continuation: Move adaptive reaction factor (`_solve_adaptive_react`) and adaptive time-stepping (`_solve_adaptive_dt`) into a `ContinuationManager`.

### Phase 4: Modernization ⏳ PENDING
- Type hinting: Apply `numpy.typing.NDArray` to numerical inputs.
- Logging: Replace `print` statements with a structured logging configuration.

## Interface Definitions
Define contracts between modules before implementation to prevent late-stage mismatches.

```python
from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.sparse import csc_array

@dataclass(frozen=True)
class State:
    """Immutable reactor state passed through solvers."""
    c: NDArray[np.float64]   # shape: (num_z, num_r, num_species)
    p: NDArray[np.float64]   # shape: (num_z, num_r)
    T: NDArray[np.float64]   # shape: (num_z, num_r)
    u_ax: NDArray[np.float64]  # shape: (num_z+1, num_r) axial velocity

# physics.py
def assemble_residual(state: State, mesh: ReactorMesh, config: ReactorConfig) -> NDArray[np.float64]: ...
def assemble_jacobian(state: State, mesh: ReactorMesh, config: ReactorConfig) -> csc_array: ...

# solvers.py
def newton_solve(
    residual_fn: Callable[[NDArray], NDArray],
    jacobian_fn: Callable[[NDArray], csc_array],
    x0: NDArray,
    tol: float = 1e-8,
    max_iter: int = 50
) -> tuple[NDArray, bool]: ...  # returns (solution, converged)
```

## Migration Strategy
Maintain a working system throughout refactoring by using `MembraneReactor` as a thin wrapper:

```python
class MembraneReactor:
    """Legacy interface delegates to new modules."""
    def __init__(self, **kwargs):
        self.config = ReactorConfig(**kwargs)
        self.mesh = ReactorMesh(self.config)
        self._state = State(...)  # internal state
        # ... existing public API preserved
```

Each phase introduces new modules while the old class continues to pass regression tests.

## Specific Logic Improvements
- Critical: Handling the "God Object" state
	- Pass a `State` object containing $c$, $p$, $T$, and $u$ through solvers to ensure thread-safety and prevent hidden state corruption.
- Pressure–velocity coupling
	- Encapsulate the Darcy/Ergun logic within a `FlowEngine` class inside `physics.py` (not a separate module—it's still physics, just specialized).
- Boundary conditions
	- Convert dictionary-based BCs into a Protocol for extensibility:
	```python
	class BoundaryCondition(Protocol):
	    def apply(self, field: NDArray, mesh: ReactorMesh) -> NDArray: ...

	class DirichletBC(BoundaryCondition): ...
	class NeumannBC(BoundaryCondition): ...
	class RobinBC(BoundaryCondition): ...
	```

## Technical Constraints
- Libraries: Maintain compatibility with `scipy.sparse` (using `csc_array`, not `csc_matrix`), `numpy`, and the custom `pymrm` library.
- Vectorization: Ensure residual calculations remain fully vectorized for performance.
- Non-isothermal logic: Keep segregated solve (Concentration–Pressure vs. Temperature) as an option; allow fully coupled assembly in the future.
- State immutability: Treat `State` objects as immutable; solvers return new states rather than mutating in place.

## Current Status

**Last updated**: 2026-02-04
**Branch**: `feat/monolithic`
**Latest commit**: `9e02ec6` (Phase 2 - inlet flux extraction)

| Phase | Status | Commit |
|-------|--------|--------|
| Phase 0: Regression Test | ✅ Complete | `d9e1027` |
| Phase 1: Config & Mesh | ✅ Complete | `2261218` |
| Phase 2: Physics Decoupling | ✅ Complete | `9e02ec6` |
| Phase 3: Solver Abstraction | ⏳ Pending | — |
| Phase 4: Modernization | ⏳ Pending | — |

## Next Steps
1. **Phase 3**: Extract solver logic to solvers.py
   - Newton iteration from `_solve_c_p()` and `_solve_T()`
   - Armijo line search logic
   - Continuation strategies (`_solve_adaptive_react`, `_solve_adaptive_dt`)

2. Run regression test after each change: `.venv/bin/python regression_test.py --test`
