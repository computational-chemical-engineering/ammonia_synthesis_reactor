"""
Post-processing utilities for the membrane reactor model.

Provides :func:`compute_dimensionless_numbers` to compute Re, Sc, Pe and
(optionally) Da fields for both the retentate and permeate regions of a
solved :class:`MembraneReactor`, and :func:`save_dimensionless` to persist
the result to an HDF5 file that is easy to load with *h5py* or *xarray*.

Extended with membrane-transport indicators on the retentate side:
    - CP_NH3   : concentration polarisation = y_NH3_wall / y_NH3_avg
    - theta_NH3: axial convection / NH3 permeation competition
    - DaPe_NH3 : theta_NH3 * Da_conv_NH3

Notes
-----
* Both the **retentate** (packed-bed annulus) and **permeate** (inner tube)
  regions are post-processed.
* Sherwood number is omitted: no mass-transfer coefficient is computed
  inside the solver and no correlation has been selected.
* Diffusivity-based numbers (Sc, Pe, Da) are **per species**:
  H2 (index 0), N2 (index 1), NH3 (index 2).
* Damköhler numbers are only computed for the retentate (reaction side).
* NH3 permeance is Arrhenius-based and evaluated locally at the retentate
  membrane wall temperature.

Characteristic lengths
~~~~~~~~~~~~~~~~~~~~~~
Retentate (packed bed):
    - Re  : particle diameter ``dp``
    - Pe  : annular gap ``r_max − r_min``

Permeate (open tube):
    - Re  : tube diameter ``2 · r_max_perm``
    - Pe  : tube diameter ``2 · r_max_perm``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import h5py

from .config import get_membrane_permeances


# Small number to guard against division by zero
_EPS = 1e-30

#: Species names matching the cpT axis ordering
SPECIES = ("H2", "N2", "NH3")
IH2 = 0
IN2 = 1
INH3 = 2


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DimensionlessResult:
    """Container returned by :func:`compute_dimensionless_numbers`.

    Attributes
    ----------
    z : ndarray, shape (Nz,)
        Axial cell-centre coordinates [m].
    r_ret : ndarray, shape (Nr_ret,)
        Retentate radial cell-centre coordinates [m].
    r_perm : ndarray, shape (Nr_perm,)
        Permeate radial cell-centre coordinates [m].
    fields_2d_ret : dict[str, ndarray]
        2-D (or 3-D for per-species) retentate fields on the
        ``(Nz, Nr_ret[, 3])`` grid.
        Keys: ``Re``, ``Sc``, ``Pe``, ``u``, ``T``, ``p``,
        and (if requested) ``Da_conv``, ``Da_diff``.
    fields_2d_perm : dict[str, ndarray]
        Same set of fields (without Da) on the ``(Nz, Nr_perm[, 3])`` grid.
    axial_ret : dict[str, ndarray]
        Cylindrically radial-averaged (``*_avg``) and membrane-wall
        (``*_wall``) retentate profiles, shape ``(Nz,)`` or ``(Nz, 3)``.
        Also includes CP_NH3, theta_NH3, DaPe_NH3.
    axial_perm : dict[str, ndarray]
        Same profiles for the permeate region.
    meta : dict[str, Any]
        Scalar metadata.
    """

    z:              np.ndarray
    r_ret:          np.ndarray
    r_perm:         np.ndarray
    fields_2d_ret:  Dict[str, np.ndarray]
    fields_2d_perm: Dict[str, np.ndarray]
    axial_ret:      Dict[str, np.ndarray]
    axial_perm:     Dict[str, np.ndarray]
    meta:           Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _radial_avg_cyl(arr: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Cylindrical radial average: ``<phi>(z) = ∫ phi·r dr / ∫ r dr``.

    Parameters
    ----------
    arr : ndarray, shape (Nz, Nr) or (Nz, Nr, Ns)
    r   : ndarray, shape (Nr,)

    Returns
    -------
    ndarray, shape (Nz,) or (Nz, Ns)
    """
    denom = np.trapezoid(r, r) + _EPS
    if arr.ndim == 2:
        return np.array([
            np.trapezoid(arr[i] * r, r) / denom
            for i in range(arr.shape[0])
        ])
    else:  # (Nz, Nr, Ns)
        return np.array([
            np.trapezoid(arr[i] * r[:, np.newaxis], r, axis=0) / denom
            for i in range(arr.shape[0])
        ])


def _compute_region_fields(
    correlation,
    c: np.ndarray,
    T: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    L_Re: float,
    L_Pe: float,
) -> Dict[str, np.ndarray]:
    """Compute dimensionless fields for one region (retentate or permeate)."""
    c_tot = np.sum(c, axis=-1, keepdims=True)
    y     = c / np.maximum(c_tot, _EPS)

    mu  = correlation.viscosity(c, T)
    rho = correlation.density(c, T, p)
    D   = correlation.diffusion(y, T, p)

    Re = rho * u * L_Re / (mu + _EPS)
    Sc = mu[..., np.newaxis] / (rho[..., np.newaxis] * D + _EPS)
    Pe = u[..., np.newaxis] * L_Pe / (D + _EPS)

    return {"Re": Re, "Sc": Sc, "Pe": Pe, "u": u, "T": T, "p": p}


def _axial_profiles(
    fields: Dict[str, np.ndarray],
    r: np.ndarray,
    iwall: int,
) -> Dict[str, np.ndarray]:
    """Build ``*_avg`` and ``*_wall`` axial profiles for every field."""
    axial: Dict[str, np.ndarray] = {}
    for name, arr in fields.items():
        axial[f"{name}_avg"] = _radial_avg_cyl(arr, r)
        axial[f"{name}_wall"] = arr[:, iwall] if arr.ndim == 2 else arr[:, iwall, :]
    return axial


def _compute_membrane_transport(
    reactor,
    c_ret: np.ndarray,
    T_ret: np.ndarray,
    p_ret: np.ndarray,
    u_ret: np.ndarray,
    r_ret: np.ndarray,
    c_perm: np.ndarray,
    p_perm: np.ndarray,
    r_perm: np.ndarray,
    Da_conv_NH3_avg: np.ndarray,
    iwall_ret: int = 0,
    iwall_perm: int = -1,
) -> Dict[str, np.ndarray]:
    """Compute CP_NH3, theta_NH3 and DaPe_NH3 as axial profiles.

    Definitions
    -----------
    CP_NH3(z)   = y_NH3,wall / y_NH3,avg

    theta_NH3(z) = F_conv(z) / F_perm(z)

    with
        F_conv(z) = <u * c_tot>_r * A_ret
        F_perm(z) = J_NH3(z) * A_mem,cell
                  = Q_NH3(z) * Δp_NH3(z) * (2π r_mem Δz)

    DaPe_NH3(z) = theta_NH3(z) * Da_conv_NH3_avg(z)
    """
    z = reactor.z_c
    dz = reactor.dz if np.isscalar(reactor.dz) else np.asarray(reactor.dz).reshape(-1)

    # ---- retentate NH3 mole fraction: wall and average
    ctot_ret = np.sum(c_ret, axis=-1)
    yret = c_ret / np.maximum(ctot_ret[..., np.newaxis], _EPS)

    yNH3_wall = yret[:, iwall_ret, INH3]
    yNH3_avg  = _radial_avg_cyl(yret[:, :, INH3], r_ret)

    CP_NH3 = yNH3_wall / np.maximum(yNH3_avg, _EPS)

    # ---- NH3 partial-pressure difference across membrane
    pNH3_ret_wall = p_ret[:, iwall_ret] * yNH3_wall

    ctot_perm = np.sum(c_perm, axis=-1)
    yperm = c_perm / np.maximum(ctot_perm[..., np.newaxis], _EPS)
    yNH3_perm_wall = yperm[:, iwall_perm, INH3]
    pNH3_perm_wall = p_perm[:, iwall_perm] * yNH3_perm_wall

    delta_p_NH3 = np.maximum(pNH3_ret_wall - pNH3_perm_wall, _EPS)

    # ---- local NH3 permeance from Arrhenius relation
    P0, EA = get_membrane_permeances(
        reactor.config.species,
        reactor.config,
        z,
        reactor.config.Lsealing,
    )
    P0_NH3 = P0[:, INH3]
    EA_NH3 = EA[0, INH3]

    T_wall = T_ret[:, iwall_ret]
    Q_NH3 = P0_NH3 * np.exp(-EA_NH3 / (reactor.config.Rg * T_wall + _EPS))

    # ---- convection term: radial-averaged molar flux density × retentate area
    r_in  = float(reactor.config.r_min)
    r_out = float(reactor.config.r_max)
    A_ret = np.pi * (r_out**2 - r_in**2)

    axial_molar_flux = _radial_avg_cyl(u_ret * ctot_ret, r_ret)   # mol / m² / s
    F_conv = axial_molar_flux * A_ret                             # mol / s

    # ---- permeation term: local flux × local membrane area in each axial cell
    r_mem = float(reactor.config.r_min)
    perimeter_mem = 2.0 * np.pi * r_mem

    if np.isscalar(dz):
        A_mem_cell = perimeter_mem * float(dz)
    else:
        A_mem_cell = perimeter_mem * dz

    J_NH3 = Q_NH3 * delta_p_NH3                                   # mol / m² / s
    F_perm = J_NH3 * A_mem_cell                                   # mol / s

    theta_NH3 = F_conv / np.maximum(F_perm, _EPS)
    DaPe_NH3 = theta_NH3 * Da_conv_NH3_avg

    return {
        "CP_NH3": CP_NH3,
        "theta_NH3": theta_NH3,
        "DaPe_NH3": DaPe_NH3,
        "Q_NH3": Q_NH3,
        "J_NH3": J_NH3,
        "delta_p_NH3": delta_p_NH3,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_dimensionless_numbers(
    reactor,
    include_damkohler: bool = True,
    include_membrane_transport: bool = True,
) -> DimensionlessResult:
    """Compute 2-D dimensionless number fields for both reactor regions."""
    nr_perm = reactor.num_r_perm
    cpT     = reactor.cpT
    z       = reactor.z_c

    # ------------------------------------------------------------------
    # Retentate region  [ nr_perm : ]
    # ------------------------------------------------------------------
    c_ret = cpT[:, nr_perm:, :-2]
    p_ret = cpT[:, nr_perm:, -2]
    T_ret = cpT[:, nr_perm:, -1]
    r_ret = reactor.r_c_ret

    u_ret = 0.5 * (reactor.u_ret_ax[:-1, :] + reactor.u_ret_ax[1:, :])

    dp  = reactor.config.dp
    gap = reactor.config.r_max - reactor.config.r_min

    fields_ret = _compute_region_fields(
        reactor.correlation, c_ret, T_ret, p_ret, u_ret,
        L_Re=dp, L_Pe=gap,
    )

    # Membrane on retentate side is at r_min = first radial node
    IWALL_RET = 0

    # Damköhler numbers (retentate only)
    if include_damkohler:
        pp    = reactor._reaction_partial_pressures(c_ret, T_ret, p_ret)
        r_rxn = reactor.kinetics(pp, T_ret)
        r_abs = np.abs(r_rxn)
        c_ref = np.maximum(np.abs(c_ret), _EPS)

        D_ret = reactor.correlation.diffusion(
            c_ret / np.maximum(np.sum(c_ret, axis=-1, keepdims=True), _EPS),
            T_ret, p_ret,
        )

        fields_ret["Da_conv"] = r_abs * gap / (np.abs(u_ret[..., np.newaxis]) * c_ref + _EPS)
        fields_ret["Da_diff"] = r_abs * gap**2 / (D_ret * c_ref + _EPS)

    axial_ret = _axial_profiles(fields_ret, r_ret, iwall=IWALL_RET)

    # ------------------------------------------------------------------
    # Permeate region  [ : nr_perm ]
    # ------------------------------------------------------------------
    c_perm = cpT[:, :nr_perm, :-2]
    p_perm = cpT[:, :nr_perm, -2]
    T_perm = cpT[:, :nr_perm, -1]
    r_perm = reactor.r_c_perm

    u_perm = 0.5 * (reactor.u_perm_ax[:-1, :] + reactor.u_perm_ax[1:, :])

    D_tube = 2.0 * reactor.config.r_max_perm

    fields_perm = _compute_region_fields(
        reactor.correlation, c_perm, T_perm, p_perm, u_perm,
        L_Re=D_tube, L_Pe=D_tube,
    )

    # Permeate membrane side is the outer edge of permeate domain
    IWALL_PERM = -1
    axial_perm = _axial_profiles(fields_perm, r_perm, iwall=IWALL_PERM)

    # ------------------------------------------------------------------
    # Membrane transport numbers
    # ------------------------------------------------------------------
    if include_membrane_transport:
        if not include_damkohler:
            raise ValueError(
                "include_membrane_transport=True requires include_damkohler=True "
                "because DaPe_NH3 uses Da_conv_NH3."
            )

        extra = _compute_membrane_transport(
            reactor=reactor,
            c_ret=c_ret,
            T_ret=T_ret,
            p_ret=p_ret,
            u_ret=u_ret,
            r_ret=r_ret,
            c_perm=c_perm,
            p_perm=p_perm,
            r_perm=r_perm,
            Da_conv_NH3_avg=axial_ret["Da_conv_avg"][:, INH3],
            iwall_ret=IWALL_RET,
            iwall_perm=IWALL_PERM,
        )
        axial_ret.update(extra)

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    meta = {
        "dp_m": float(dp),
        "gap_ret_m": float(gap),
        "D_tube_perm_m": float(D_tube),
        "r_min_m": float(reactor.config.r_min),
        "r_max_m": float(reactor.config.r_max),
        "r_max_perm_m": float(reactor.config.r_max_perm),
        "species": list(SPECIES),
    }

    if include_membrane_transport:
        active_mask = z > reactor.config.Lsealing
        if np.any(active_mask):
            meta.update({
                "theta_NH3_avg": float(np.mean(axial_ret["theta_NH3"][active_mask])),
                "theta_NH3_min": float(np.min(axial_ret["theta_NH3"][active_mask])),
                "DaPe_NH3_avg": float(np.mean(axial_ret["DaPe_NH3"][active_mask])),
                "CP_NH3_avg": float(np.mean(axial_ret["CP_NH3"][active_mask])),
                "CP_NH3_min": float(np.min(axial_ret["CP_NH3"][active_mask])),
            })

    return DimensionlessResult(
        z=z,
        r_ret=r_ret,
        r_perm=r_perm,
        fields_2d_ret=fields_ret,
        fields_2d_perm=fields_perm,
        axial_ret=axial_ret,
        axial_perm=axial_perm,
        meta=meta,
    )


def save_dimensionless(result: DimensionlessResult, path: str) -> None:
    """Save a :class:`DimensionlessResult` to an HDF5 file."""
    _UNITS = {
        "Re": "-", "Sc": "-", "Pe": "-",
        "Da_conv": "-", "Da_diff": "-",
        "CP_NH3": "-", "theta_NH3": "-", "DaPe_NH3": "-",
        "Q_NH3": "mol/m^2/s/Pa",
        "J_NH3": "mol/m^2/s",
        "delta_p_NH3": "Pa",
        "u": "m/s", "T": "K", "p": "Pa",
    }
    _DESC = {
        "Re": "Reynolds number",
        "Sc": "Schmidt number per species [H2, N2, NH3]",
        "Pe": "Péclet number per species [H2, N2, NH3]",
        "Da_conv": "Convective Damköhler number per species [H2, N2, NH3]",
        "Da_diff": "Diffusive Damköhler number per species [H2, N2, NH3]",
        "CP_NH3": "NH3 concentration polarisation = y_NH3_wall / y_NH3_avg",
        "theta_NH3": "Convection-to-permeation ratio for NH3",
        "DaPe_NH3": "theta_NH3 * Da_conv_NH3_avg",
        "Q_NH3": "NH3 membrane permeance from Arrhenius relation",
        "J_NH3": "NH3 permeation flux = Q_NH3 * delta_p_NH3",
        "delta_p_NH3": "NH3 partial-pressure difference across membrane",
        "u": "Cell-centre axial velocity",
        "T": "Temperature",
        "p": "Pressure",
    }

    def _write_region(grp, fields_2d, axial):
        """Write one region's 2D fields and axial profiles into HDF5 subgroups,
        attaching units and description attributes.
        """
        grp2 = grp.create_group("fields_2d")
        for name, arr in fields_2d.items():
            ds = grp2.create_dataset(name, data=arr.astype(np.float64))
            ds.attrs["units"] = _UNITS.get(name, "-")
            ds.attrs["description"] = _DESC.get(name, name)

        grp_ax = grp.create_group("axial")
        for name, arr in axial.items():
            ds = grp_ax.create_dataset(name, data=arr.astype(np.float64))
            base_name = name
            if name.endswith("_avg"):
                base_name = name[:-4]
            elif name.endswith("_wall"):
                base_name = name[:-5]
            ds.attrs["units"] = _UNITS.get(base_name, "-")
            ds.attrs["description"] = _DESC.get(base_name, name)

    with h5py.File(path, "w") as f:
        for key, val in result.meta.items():
            if isinstance(val, (int, float, np.floating)):
                f.attrs[key] = val
            elif isinstance(val, (list, tuple)):
                f.attrs[key] = ",".join(str(v) for v in val)

        grp_c = f.create_group("coords")
        ds = grp_c.create_dataset("z", data=result.z.astype(np.float64))
        ds.attrs["units"] = "m"

        ds = grp_c.create_dataset("r_ret", data=result.r_ret.astype(np.float64))
        ds.attrs["units"] = "m"
        ds.attrs["description"] = "retentate radial cell-centres"

        ds = grp_c.create_dataset("r_perm", data=result.r_perm.astype(np.float64))
        ds.attrs["units"] = "m"
        ds.attrs["description"] = "permeate radial cell-centres"

        _write_region(f.create_group("retentate"), result.fields_2d_ret, result.axial_ret)
        _write_region(f.create_group("permeate"), result.fields_2d_perm, result.axial_perm)

        print(f"Saved dimensionless numbers → {path}")
    # ← close save_dimensionless here, no more indentation


def print_regime_summary(result: DimensionlessResult) -> None:   # ← top-level, no indent
    """Print a concise regime-map summary from ``result.meta``."""
    m = result.meta
    print("\n" + "═" * 52)
    print("  DIMENSIONLESS REGIME SUMMARY")
    print("═" * 52)
    print(f"  θ_NH3  mean : {m.get('theta_NH3_avg', float('nan')):.3f}")
    print(f"  θ_NH3  min  : {m.get('theta_NH3_min', float('nan')):.3f}  ← worst case (reactor exit)")
    print(f"  DaPe_NH3 mean: {m.get('DaPe_NH3_avg', float('nan')):.3f}")
    print(f"  CP_NH3 mean : {m.get('CP_NH3_avg',   float('nan')):.4f}")
    print(f"  CP_NH3 min  : {m.get('CP_NH3_min',   float('nan')):.4f}")
    print("═" * 52 + "\n")