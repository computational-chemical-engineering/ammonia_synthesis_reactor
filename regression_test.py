"""
Phase 0: Regression Test Harness for Membrane Reactor Refactoring

Captures reference solutions and validates numerical equivalence after code changes.
Reference case: GHSV=150, H2:N2=1.5, sweep_ratio=0.05 (from debug.ipynb)
"""

import numpy as np
import scipy.constants as const
from pathlib import Path

REFERENCE_FILE = Path(__file__).parent / "regression_reference.npz"


def calculate_flows(GHSV, vol_reactor, sweep_ratio, H2_N2_ratio, T_STP=273.15, P_STP=101325.0):
    """Converts GHSV (1/h) and geometry to F_ret_in and F_perm_in (mol/s)."""
    vol_flow_std = (GHSV * vol_reactor) / 3600.0
    F_ret_in = (P_STP * vol_flow_std) / (const.R * T_STP)
    y_N2_in = 1.0 / (H2_N2_ratio + 1.0)
    y_H2_in = 1.0 - y_N2_in
    F_perm_in = F_ret_in * sweep_ratio
    return F_ret_in, F_perm_in, y_H2_in, y_N2_in


def get_reference_config():
    """Returns the reference case configuration."""
    GHSV = 150
    H2_N2_ratio = 1.5
    sweep_ratio = 0.05
    r_min = 0.005
    r_max = 0.0165
    L = 1.0
    vol_reactor = np.pi * (r_max**2 - r_min**2) * L

    F_ret_in, F_perm_in, y_H2_in, y_N2_in = calculate_flows(
        GHSV, vol_reactor, sweep_ratio, H2_N2_ratio
    )
    y_ret_in = [y_H2_in, y_N2_in, 0.0]

    return dict(
        config_file="debug.json",
        L=L,
        r_min=r_min,
        r_max=r_max,
        F_ret_in=F_ret_in,
        F_perm_in=F_perm_in,
        y_ret_in=y_ret_in,
    )


def extract_artifacts(reactor):
    """Extracts all artifacts needed for regression comparison."""
    flows_ret_ax, flows_ret_mem, flows_perm_ax, flows_perm_mem = reactor.compute_flows()

    # Elemental balances
    flows_tot = (
        flows_ret_ax[0, :] + flows_perm_ax[0, :]
        - flows_ret_ax[-1, :] - flows_perm_ax[-1, :]
    )
    H_balance = 2 * flows_tot[0] + 3 * flows_tot[2]
    N_balance = 2 * flows_tot[1] + flows_tot[2]

    return dict(
        # State arrays
        c_p=reactor.c_p.copy(),
        T=reactor.T.copy(),
        u_ret_ax=reactor.u_ret_ax.copy(),
        u_perm_ax=reactor.u_perm_ax.copy(),
        # Computed flows
        flows_ret_ax=flows_ret_ax,
        flows_ret_mem=flows_ret_mem,
        flows_perm_ax=flows_perm_ax,
        flows_perm_mem=flows_perm_mem,
        # Balances
        H_balance=np.array([H_balance]),
        N_balance=np.array([N_balance]),
        # Solver stats
        cnt_num_solves_c_p=np.array([reactor.cnt_num_solves_c_p]),
        cnt_num_solves_T=np.array([reactor.cnt_num_solves_T]),
    )


def save_reference(reactor, filepath=REFERENCE_FILE):
    """Saves reference artifacts to file."""
    artifacts = extract_artifacts(reactor)
    np.savez_compressed(filepath, **artifacts)
    print(f"Reference saved to {filepath}")
    print(f"  c_p shape: {artifacts['c_p'].shape}")
    print(f"  T shape: {artifacts['T'].shape}")
    print(f"  H balance: {artifacts['H_balance'][0]:.2e}")
    print(f"  N balance: {artifacts['N_balance'][0]:.2e}")
    print(f"  Solves (c_p): {artifacts['cnt_num_solves_c_p'][0]}")
    print(f"  Solves (T): {artifacts['cnt_num_solves_T'][0]}")


def load_reference(filepath=REFERENCE_FILE):
    """Loads reference artifacts from file."""
    data = np.load(filepath)
    return {key: data[key] for key in data.files}


def compare_artifacts(current, reference, rtol=1e-10):
    """
    Compares current artifacts against reference.
    Returns (passed, report) tuple.
    """
    field_keys = ["c_p", "T", "u_ret_ax", "u_perm_ax"]
    flow_keys = ["flows_ret_ax", "flows_ret_mem", "flows_perm_ax", "flows_perm_mem"]

    report = []
    all_passed = True

    # Check field arrays with L2 relative error
    for key in field_keys:
        cur = current[key]
        ref = reference[key]
        ref_norm = np.linalg.norm(ref)
        if ref_norm > 0:
            rel_err = np.linalg.norm(cur - ref) / ref_norm
        else:
            rel_err = np.linalg.norm(cur - ref)
        passed = rel_err < rtol
        status = "PASS" if passed else "FAIL"
        report.append(f"  {key}: L2 rel error = {rel_err:.2e} [{status}]")
        all_passed = all_passed and passed

    # Check flow arrays
    for key in flow_keys:
        cur = current[key]
        ref = reference[key]
        ref_norm = np.linalg.norm(ref)
        if ref_norm > 0:
            rel_err = np.linalg.norm(cur - ref) / ref_norm
        else:
            rel_err = np.linalg.norm(cur - ref)
        passed = rel_err < rtol
        status = "PASS" if passed else "FAIL"
        report.append(f"  {key}: L2 rel error = {rel_err:.2e} [{status}]")
        all_passed = all_passed and passed

    return all_passed, "\n".join(report)


def run_regression_test(reactor_class, rtol=1e-10):
    """
    Runs the reference case and compares against saved reference.
    Returns True if all checks pass.
    """
    if not REFERENCE_FILE.exists():
        print(f"ERROR: Reference file not found: {REFERENCE_FILE}")
        print("Run 'python regression_test.py --save' first to create reference.")
        return False

    config = get_reference_config()
    reactor = reactor_class(**config)
    reactor.solve(verbose=0, dt_min=1e-2)

    current = extract_artifacts(reactor)
    reference = load_reference()

    passed, report = compare_artifacts(current, reference, rtol=rtol)

    print(f"Regression test {'PASSED' if passed else 'FAILED'} (rtol={rtol:.0e})")
    print(report)

    return passed


def create_reference():
    """Creates and saves the reference solution."""
    from membrane_reactor import MembraneReactor

    config = get_reference_config()
    print("Running reference case...")
    print(f"  F_ret_in: {config['F_ret_in']:.6e} mol/s")
    print(f"  F_perm_in: {config['F_perm_in']:.6e} mol/s")
    print(f"  y_ret_in: {config['y_ret_in']}")

    reactor = MembraneReactor(**config)
    reactor.solve(verbose=1, dt_min=1e-2)

    save_reference(reactor)
    return reactor


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        create_reference()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        from membrane_reactor import MembraneReactor
        success = run_regression_test(MembraneReactor)
        sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  python regression_test.py --save   # Create reference solution")
        print("  python regression_test.py --test   # Run regression test")
