"""Dimensionless-number extraction from the certified 2D cache.

Feeds Figures 12-14 (local wall-Sherwood vs Re_dh and the Graetz
coordinate) and the SI regime figures S.2-S.5 (radial Peclet / diffusive
Damkoehler scaling and regime maps).

The heavy lifting is :func:`reactor.postprocessing.compute_dimensionless_numbers`,
which needs a live reactor; the injection of cached fields into an
un-solved instance is the pattern of ``scripts/recompute_dimensionless.py``.
Per-case profiles are cached as npz under
``results/paper/<resolution>/dimless/`` so figure cells re-read in
milliseconds; anything here is derived data, reproducible from the case
caches at any time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reactor import MembraneReactor, ReactorConfig
from reactor.postprocessing import compute_dimensionless_numbers

from reactor.paper import cache, closures, kpis as kpi_mod, settings

INH3 = 2

#: Manuscript Figure 12 local-point filters (caption: z > 0.10 m, CP < 0.95).
Z_MIN_SH = 0.10
CP_MAX_SH = 0.95


def dimless_dir(resolution_name: str) -> Path:
    """Directory of the cached per-case dimensionless npz profiles."""
    return settings.cache_root(resolution_name) / "dimless"


def _inject_fields(reactor: MembraneReactor, fields: dict[str, np.ndarray]) -> None:
    """Inject cached field arrays into an un-solved reactor instance."""
    nr_perm = reactor.num_r_perm
    nz = reactor.num_z
    num_c = reactor.config.num_c
    nr_ret = fields["c_ret"].shape[1]

    cpT = np.zeros((nz, nr_perm + nr_ret, num_c + 2), dtype=np.float64)
    cpT[:, :nr_perm, :-2] = fields["c_perm"]
    cpT[:, :nr_perm, -2] = fields["p_perm_bar"] * 1e5
    cpT[:, :nr_perm, -1] = fields["T_perm"]
    cpT[:, nr_perm:, :-2] = fields["c_ret"]
    cpT[:, nr_perm:, -2] = fields["p_ret_bar"] * 1e5
    cpT[:, nr_perm:, -1] = fields["T_ret"]
    reactor.cpT = cpT
    reactor.u_ret_ax = fields["u_ret_ax"]
    reactor.u_perm_ax = fields["u_perm_ax"]
    # compute_dimensionless_numbers reads reactor.dz, which the current
    # MembraneReactor no longer defines; reconstruct it from the face grid.
    if not hasattr(reactor, "dz"):
        reactor.dz = np.diff(np.asarray(reactor.z_f).reshape(-1))


def case_dimless(
    resolution_name: str,
    case_id: str,
    *,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Axial dimensionless profiles for one cached, accepted 2D case.

    Returns (and caches) z-profiles of the cross-section-averaged numbers
    plus the wall-based Sherwood profile and the pieces of the Graetz
    coordinate:

    ``z``, ``Re_avg`` (particle Reynolds), ``Re_dh`` (hydraulic-diameter
    Reynolds, = Re_avg * d_h/d_p with d_h = 2*(r_max-r_min)),
    ``Pe_avg`` (nz, 3), ``Da_conv_avg`` (nz, 3), ``Da_diff_avg`` (nz, 3),
    ``u_avg``, ``D_m`` (nz, 3), ``Sh_wall`` (nz, 3), ``CP_NH3``,
    ``z_star`` (Graetz coordinate D_m,NH3 * z / (u l_c^2)), and scalars
    ``kappa``, ``l_c``, ``d_h``, ``dp``, ``L``.
    """
    out_path = dimless_dir(resolution_name) / f"{case_id}.npz"
    if out_path.exists() and not force:
        return dict(np.load(out_path))

    case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id)
    fields = cache.load_fields(case_dir)
    config = cache.read_json(Path(case_dir) / "config.json")
    cfg = ReactorConfig.from_dict(config)

    reactor = MembraneReactor(config=cfg)
    _inject_fields(reactor, fields)
    res = compute_dimensionless_numbers(
        reactor, include_damkohler=True, include_membrane_transport=True
    )

    ax = res.axial_ret
    l_c = float(cfg.r_max - cfg.r_min)
    d_h = 2.0 * l_c

    # Wall-based Sherwood and the mean-state mixture diffusivity, on the
    # same conventions as the Figure 12-15 pipeline (gap-width basis).
    cl = closures.case_closure(case_dir, kind="wall")
    with np.errstate(divide="ignore", invalid="ignore"):
        D_m = np.where(cl["sh"] > 0, cl["kcp"] * l_c / cl["sh"], np.nan)
    # D_m from the closure is undefined where kcp is; recompute directly
    # for a gap-free profile.
    from reactor.gas_mixture_correlations import GasMixtureCorrelations

    corr = GasMixtureCorrelations(config["species"], config["database"])
    w = kpi_mod.area_weights(fields["r_f_ret"])
    y_mean = np.tensordot(fields["y_ret"], w, axes=([1], [0]))
    T_mean = np.tensordot(fields["T_ret"], w, axes=([1], [0]))
    p_mean = np.tensordot(fields["p_ret_bar"] * 1e5, w, axes=([1], [0]))
    nz, nc = y_mean.shape
    D_m = np.asarray(corr.diffusion(
        y_mean.reshape(nz, 1, nc), T_mean.reshape(nz, 1), p_mean.reshape(nz, 1)
    ))
    while D_m.ndim > 2:
        D_m = D_m.squeeze(axis=1)
    D_m = np.maximum(D_m, 1e-30)

    u_avg = np.asarray(ax["u_avg"], dtype=float)
    z = np.asarray(res.z, dtype=float)
    z_star = D_m[:, INH3] * z / np.maximum(np.abs(u_avg) * l_c**2, 1e-30)

    data: dict[str, np.ndarray] = {
        "z": z,
        "Re_avg": np.asarray(ax["Re_avg"], dtype=float),
        "Re_dh": np.asarray(ax["Re_avg"], dtype=float) * (d_h / float(cfg.dp)),
        "Pe_avg": np.asarray(ax["Pe_avg"], dtype=float),
        "Da_conv_avg": np.asarray(ax["Da_conv_avg"], dtype=float),
        "Da_diff_avg": np.asarray(ax["Da_diff_avg"], dtype=float),
        "u_avg": u_avg,
        "D_m": D_m,
        "Sh_wall": np.asarray(cl["sh"], dtype=float),
        "CP_NH3": np.asarray(ax["CP_NH3"], dtype=float),
        "z_star": z_star,
        "kappa": np.float64(cfg.r_min / cfg.r_max),
        "l_c": np.float64(l_c),
        "d_h": np.float64(d_h),
        "dp": np.float64(cfg.dp),
        "L": np.float64(z.max()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return data


def _accepted_case_ids(case_table: pd.DataFrame, resolution_name: str) -> list[str]:
    """Case IDs with a complete, non-failed 2D cache entry that has fields."""
    res = settings.resolution(resolution_name)
    ids = []
    for case_id in case_table["Case_ID"]:
        case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, str(case_id))
        if not cache.is_complete(case_dir, res):
            continue
        if cache.load_meta(case_dir).get("status") == "failed":
            continue
        if not (Path(case_dir) / "fields.npz").exists():
            continue
        ids.append(str(case_id))
    return ids


def sweep_scalars(
    case_table: pd.DataFrame,
    resolution_name: str,
    *,
    z_min: float = 0.05,
    include_598k: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Per-case dimensionless scalars over the membrane-active length.

    Means and maxima of the radial Peclet and Damkoehler numbers per
    species, Re_dh statistics, geometry and operating scalars — the
    input to SI Figures S.2-S.5. Cached as one CSV per resolution.
    """
    from reactor.paper import cases as cases_mod

    out_path = dimless_dir(resolution_name) / "dimless_scalars.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    table = cases_mod.select_cases(case_table, include_598k=include_598k)
    rows: list[dict[str, Any]] = []
    for case_id in _accepted_case_ids(table, resolution_name):
        d = case_dimless(resolution_name, case_id, force=force)
        mask = d["z"] > z_min
        row: dict[str, Any] = {
            "Case_ID": case_id,
            "family": case_id.split(" ")[0],
            "kappa": float(d["kappa"]),
            "Re_dh_mean": float(np.nanmean(d["Re_dh"][mask])),
            "Re_dh_min": float(np.nanmin(d["Re_dh"][mask])),
            "Re_dh_max": float(np.nanmax(d["Re_dh"][mask])),
        }
        for i, sp in enumerate(("H2", "N2", "NH3")):
            row[f"Pe_{sp}_mean"] = float(np.nanmean(d["Pe_avg"][mask, i]))
            row[f"Pe_{sp}_max"] = float(np.nanmax(d["Pe_avg"][mask, i]))
            row[f"Da_diff_{sp}_mean"] = float(np.nanmean(d["Da_diff_avg"][mask, i]))
            row[f"Da_diff_{sp}_max"] = float(np.nanmax(d["Da_diff_avg"][mask, i]))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Attach operating scalars from the case table.
    meta_cols = [c for c in ("GHSV_h", "p_ret_bar", "T_ret_K", "r_max_m") if c in case_table.columns]
    if meta_cols:
        df = df.merge(case_table[["Case_ID", *meta_cols]].astype({"Case_ID": str}),
                      on="Case_ID", how="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def sh_local_table(
    case_table: pd.DataFrame,
    resolution_name: str,
    *,
    z_min: float = Z_MIN_SH,
    cp_max: float = CP_MAX_SH,
    include_598k: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Local wall-Sherwood points for Figure 12 (manuscript filters).

    One row per (case, z-cell) passing the caption's filters
    (z > 0.10 m, CP < 0.95, finite positive Sh): Sh_wall_NH3, Re_dh,
    kappa, family, WHSV group value.
    """
    from reactor.paper import cases as cases_mod

    out_path = dimless_dir(resolution_name) / "sh_local_points.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    table = cases_mod.select_cases(case_table, include_598k=include_598k)
    frames = []
    for case_id in _accepted_case_ids(table, resolution_name):
        d = case_dimless(resolution_name, case_id, force=force)
        sh = d["Sh_wall"][:, INH3]
        ok = ((d["z"] > z_min) & (d["CP_NH3"] < cp_max)
              & np.isfinite(sh) & (sh > 0.0))
        if not np.any(ok):
            continue
        frames.append(pd.DataFrame({
            "Case_ID": case_id,
            "family": case_id.split(" ")[0],
            "kappa": float(d["kappa"]),
            "z_m": d["z"][ok],
            "Sh": sh[ok],
            "Re_dh": d["Re_dh"][ok],
            "z_star": d["z_star"][ok],
        }))

    df = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
