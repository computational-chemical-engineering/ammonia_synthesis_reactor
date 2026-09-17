"""Linear stability of a converged steady state (generalized eigenproblem).

For operating points reached only by continuation (the 598 K multiplicity
window) the steady state exists but pseudo-transient marching orbits it —
strong evidence of dynamic instability. This module makes that rigorous:
assemble the steady Jacobian ``J`` and the accumulation (mass) matrix ``M``
(zero block for the algebraic pressure row) and solve the generalized pencil

    -J v = lambda * M v

near the origin by shift-invert Arnoldi (which tolerates the singular
``M``). A complex-conjugate pair with positive real part confirms the
Hopf-type character: the physical attractor is a limit cycle, not the
steady state: a complex eigenvalue pair with positive real part marks a
Hopf-type dynamic instability of the computed steady state.

Pseudo-transient time is *not* physical time (the momentum balance is
quasi-static), but the eigenvalues of this pencil are those of the model's
actual accumulation terms, so signs and frequencies are meaningful within
the model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from pymrm import update_csc_array_indices


def assemble_mass_matrix(reactor) -> sp.csc_array:
    """Accumulation matrix M in the monolithic cpT ordering.

    Rows match the residual assembly of ``_construct_g_cpT``: concentration
    rows carry ``jac_c_accum`` (with the bed porosity on the retentate
    side), temperature rows carry ``factor_T * jac_T_accum`` (the residual's
    T-row is scaled by ``factor_T``), and the algebraic pressure row is
    zero.
    """
    shape_cpT = reactor.cpT.shape
    shape_c = reactor.cpT[..., :-2].shape
    shape_p = reactor.cpT[..., -2].shape + (1,)
    offset_T = (0,) * (reactor.cpT.ndim - 1) + (shape_c[-1] + 1,)
    M = update_csc_array_indices(reactor.jac_c_accum.copy(), shape_c, shape_cpT)
    M = M + reactor.factor_T * update_csc_array_indices(
        reactor.jac_T_accum.copy(), shape_p, shape_cpT, offset=offset_T
    )
    return sp.csc_array(M)


def leading_eigenvalues(reactor, k: int = 10) -> dict[str, Any]:
    """Leading eigenvalues of the steady state currently held in ``reactor``.

    Returns a JSON-friendly certificate::

        {"eigenvalues_real": [...], "eigenvalues_imag": [...],
         "max_real_part": ..., "n_unstable": ...,
         "has_unstable_complex_pair": ..., "hopf_frequency_rad_s": ...}

    ``has_unstable_complex_pair`` True is the Hopf signature. Call this on a
    converged state only — the linearization is meaningless elsewhere.
    """
    from .membrane_reactor import STEADY_STATE_DT

    _, J = reactor._construct_g_cpT(None, None, STEADY_STATE_DT, compute_jac=True)
    A = -sp.csc_array(J)
    M = assemble_mass_matrix(reactor)
    vals = sla.eigs(A, k=k, M=M, sigma=0.0, return_eigenvectors=False)
    order = np.argsort(-vals.real)
    vals = vals[order]

    real = vals.real
    imag = vals.imag
    unstable = vals[real > 0.0]
    # An unstable complex pair (Hopf signature): positive real part with a
    # genuinely nonzero frequency.
    tol_imag = 1e-12 * max(1.0, float(np.max(np.abs(vals))))
    unstable_complex = unstable[np.abs(unstable.imag) > tol_imag]
    hopf_freq = float(np.max(np.abs(unstable_complex.imag))) if unstable_complex.size else None

    return {
        "k": int(k),
        "eigenvalues_real": [float(v) for v in real],
        "eigenvalues_imag": [float(v) for v in imag],
        "max_real_part": float(np.max(real)),
        "n_unstable": int(unstable.size),
        "has_unstable_complex_pair": bool(unstable_complex.size >= 2),
        "hopf_frequency_rad_s": hopf_freq,
    }
