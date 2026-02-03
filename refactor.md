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

### Phase 0: Regression Test Harness
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

### Phase 1: Configuration & Mesh (Foundation)
- Extract parameters: Move all `defaults.DEFAULTS` and JSON loading into a `ReactorConfig` class.
- Abstract the grid: Move `_create_spatial_discretization` into a `ReactorMesh` class.
- Provide helpers instead of manual slicing (e.g., `[:, :self.num_r_perm, :]`):
	- `get_retentate_data(full_field)`
	- `get_permeate_data(full_field)`

### Phase 2: Physics Decoupling
- Stateless residuals: Refactor `_construct_g_conv`, `_construct_g_diff`, and `_construct_g_T` into a `PhysicsEngine`.
- Pure I/O: Methods take the current state $(c, p, T)$ and return the residual/Jacobian.
- Remove side effects: Do not modify `self.c_p` or `self.T` during assembly.
- Matrix assembly: Consolidate `update_csr_array_indices` using a descriptive mapping from local (retentate/permeate) to global (monolithic) matrices.

### Phase 3: Solver Abstraction
- Generalize Newton solver: Extract logic from `_solve_c_p` into a standalone `NewtonSolver`.
	- Accept `residual_fn`, `jacobian_fn`, and `initial_guess`.
	- Implement Armijo line-search and convergence monitoring as generic features.
- Generalize continuation: Move adaptive reaction factor (`_solve_adaptive_react`) and adaptive time-stepping (`_solve_adaptive_dt`) into a `ContinuationManager`.

### Phase 4: Modernization
- Type hinting: Apply `numpy.typing.NDArray` to numerical inputs.
- Logging: Replace `print` statements with a structured logging configuration.

## Interface Definitions
Define contracts between modules before implementation to prevent late-stage mismatches.

```python
from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.sparse import csr_array

@dataclass(frozen=True)
class State:
    """Immutable reactor state passed through solvers."""
    c: NDArray[np.float64]   # shape: (num_z, num_r, num_species)
    p: NDArray[np.float64]   # shape: (num_z, num_r)
    T: NDArray[np.float64]   # shape: (num_z, num_r)
    u_ax: NDArray[np.float64]  # shape: (num_z+1, num_r) axial velocity

# physics.py
def assemble_residual(state: State, mesh: ReactorMesh, config: ReactorConfig) -> NDArray[np.float64]: ...
def assemble_jacobian(state: State, mesh: ReactorMesh, config: ReactorConfig) -> csr_array: ...

# solvers.py
def newton_solve(
    residual_fn: Callable[[NDArray], NDArray],
    jacobian_fn: Callable[[NDArray], csr_array],
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
- Libraries: Maintain compatibility with `scipy.sparse` (using `csr_array`, not `csc_matrix`), `numpy`, and the custom `pymrm` library.
- Vectorization: Ensure residual calculations remain fully vectorized for performance.
- Non-isothermal logic: Keep segregated solve (Concentration–Pressure vs. Temperature) as an option; allow fully coupled assembly in the future.
- State immutability: Treat `State` objects as immutable; solvers return new states rather than mutating in place.

## Next Steps
- Create initial modules: [config.py](config.py) and [mesh.py](mesh.py).
- Validate mesh indexing; then migrate transport equations from `MembraneReactor` to [physics.py](physics.py).
