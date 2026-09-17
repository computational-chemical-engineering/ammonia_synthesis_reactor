"""Supplementary-material figures (S.1-S.57), one function per figure.

Layout and content mirror the supplementary document of the accompanying
paper; data comes only from the certified cache via
``reactor.paper.dimensionless`` and the summary CSVs — updated dataset,
same figures.

Figure map (from the SI captions):

* S.1   NH3 diffusivity vs P and vs T — correlation against the digitized
        Chapman et al. data (``settings.S1_DIFFUSION_DATA_CSV``).
* S.2   Pe_rad and Da_diff power-law scaling with GHSV (G1 / G2).
* S.3   Pe_rad and Da_diff scaling with tube radius (G7 / G8).
* S.4   <Pe_NH3> vs <Pe_H2> over all cases.
* S.5   Regime map Pe_H2 vs Da_NH3.
* S.6   Rossetti validation, remaining conditions
        (``figures.figure_s6_rossetti_rest``).
* S.7   1D vs 2D KPI sweep across temperature (G5 / G6).
* S.8-S.55  Per-case 2D fields (NH3 mole fraction + temperature).
* S.56  CP profiles across temperature (G5 / G6).
* S.57  Parity of the three 1D variants against the certified 2D KPIs
        (mechanistic-closure section).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reactor.paper import cache, cases as cases_mod, kpis as kpi_mod, settings
from reactor.paper.figures import (
    CP_YLABEL, CP_YLIM, FAMILY_COLORS, FAMILY_MARKERS, FONT_FAMILY,
    LEG_KWARGS, Z_MIN_PLOT, kpi_sweep_figure, load_cp, save as save_figure,
)

INH3 = kpi_mod.INH3


def si_dir(resolution_name: str) -> Path:
    """Directory of the supplementary-material figures for a resolution."""
    return settings.figures_dir(resolution_name) / "si"


def save_si(fig, name: str, resolution_name: str) -> list[Path]:
    """Write an SI figure as PNG and PDF into the resolution's ``si`` directory."""
    out_dir = si_dir(resolution_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300 if ext == "pdf" else 200, bbox_inches="tight")
        written.append(path)
    return written


# ── S.1: NH3 diffusivity correlation vs Chapman et al. ──────────────

def figure_s1_diffusivity(data_csv: str | Path | None = None,
                          *, T_fixed: float = 273.5, p_fixed_bar: float = 1.013):
    """NH3 diffusivity correlation vs the Chapman et al. data: D vs P (a), D vs T (b).

    The digitized source data (see the CSV header) is the NH3-H2 *binary*
    diffusivity at the source-figure conditions (P panel at ~273 K, T panel
    at ~1 atm) — identified by the pipeline's own Chapman-Enskog machinery
    reproducing the points to <1% for trace NH3 in pure H2, versus a factor
    ~3 off for NH3 in N2. Solid lines are therefore the pipeline correlation
    (``GasMixtureCorrelations.diffusion``) evaluated at trace NH3 in H2; the
    same binaries feed the reactor's mixture-averaged diffusivities.
    Experimental symbols are drawn only when ``data_csv`` provides them
    (columns: ``p_bar``/``T_K``/``D_m2_s``, a ``panel`` column ``P``|``T``,
    and optionally a ``series`` column of which only ``Chapman`` is
    plotted); never fake the points.
    """
    from reactor import ReactorConfig
    from reactor.gas_mixture_correlations import GasMixtureCorrelations

    cfg = ReactorConfig.from_defaults()
    corr = GasMixtureCorrelations(list(cfg.species), cfg.database)
    # trace NH3 in pure H2 — the composition of the Chapman source data
    y_in = np.array([1.0, 1e-6, 1e-6])
    y_in = y_in / y_in.sum()

    def d_nh3(T: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
        """NH3 mixture diffusivity at the inlet composition for arrays of T and p."""
        n = len(np.atleast_1d(T))
        y = np.broadcast_to(y_in, (n, 1, 3))
        D = np.asarray(corr.diffusion(y, np.atleast_1d(T).reshape(n, 1),
                                      np.atleast_1d(p_pa).reshape(n, 1)))
        while D.ndim > 2:
            D = D.squeeze(axis=1)
        return D[:, INH3]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    p_arr = np.geomspace(0.9, 40, 120) * 1e5
    axes[0].plot(p_arr / 1e5, d_nh3(np.full(120, T_fixed), p_arr) * 1e4, "-",
                 color="#1f77b4", lw=2, label="correlation")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("$P$  [bar]", fontsize=14)
    axes[0].set_title(rf"(a) $D_{{\mathrm{{NH_3}}}}$ vs $P$ at $T$ = {T_fixed:.0f} K",
                      fontsize=13)
    T_arr = np.linspace(250, 720, 120)
    axes[1].plot(T_arr, d_nh3(T_arr, np.full(120, p_fixed_bar * 1e5)) * 1e4, "-",
                 color="#1f77b4", lw=2, label="correlation")
    axes[1].set_xlabel("$T$  [K]", fontsize=14)
    axes[1].set_title(rf"(b) $D_{{\mathrm{{NH_3}}}}$ vs $T$ at $P$ = {p_fixed_bar:.0f} bar",
                      fontsize=13)

    if data_csv is not None:
        exp = pd.read_csv(data_csv, comment="#")
        if "series" in exp.columns:
            # The CSV also carries the correlation curve digitized from the
            # same source figure (a cross-check, never plotted here).
            exp = exp[exp["series"] == "Chapman"]
        for ax, panel, xcol, xf in ((axes[0], "P", "p_bar", 1.0),
                                    (axes[1], "T", "T_K", 1.0)):
            sub = exp[exp["panel"] == panel]
            ax.plot(sub[xcol] * xf, sub["D_m2_s"] * 1e4, "o", ms=6, mfc="none",
                    color="#d62728", label="Chapman et al.")
    else:
        for ax in axes:
            ax.text(0.5, 0.12, "experimental points pending\n"
                    "(settings.S1_DIFFUSION_DATA_CSV)", transform=ax.transAxes,
                    ha="center", fontsize=11, color="#b45309")
    for ax in axes:
        ax.set_ylabel(r"$D_{\mathrm{NH_3-H_2}}$  [cm$^2$ s$^{-1}$]", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, ls=":", alpha=0.25)
        ax.legend(fontsize=11, **LEG_KWARGS)
    fig.tight_layout()
    return fig


# ── S.2 / S.3: Pe and Da power-law scaling ───────────────────────────

def _powerfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(prefactor, exponent, R^2) of y = a x^b in log-log space."""
    lx, ly = np.log(x), np.log(y)
    b, la = np.polyfit(lx, ly, 1)
    r = np.corrcoef(lx, ly)[0, 1]
    return float(np.exp(la)), float(b), float(r**2)


def figure_s2_pe_da_whsv(scalars: pd.DataFrame,
                         families: Sequence[str] = ("G1", "G2")):
    """S.2 — max Pe_rad (H2) and max Da_diff (NH3) power laws vs GHSV.

    The SI's convention: the Peclet number of the limiting reactant H2
    (1D bulk validity) against the Damkoehler number of the permeating
    product NH3 (membrane boundary-condition validity).
    """
    fig, axes = plt.subplots(1, len(families), figsize=(6.2 * len(families), 4.8))
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families):
        sub = scalars[scalars["family"] == fam].sort_values("GHSV_h")
        ghsv = sub["GHSV_h"].to_numpy()
        note = []
        for col, color, marker, label in (
                ("Pe_H2_max", "#2980b9", "o", r"$Pe_{\mathrm{rad,H_2}}^{\max}$"),
                ("Da_diff_NH3_max", "#d62728", "s", r"$Da_{\mathrm{diff,NH_3}}^{\max}$")):
            y = sub[col].to_numpy()
            ok = np.isfinite(y) & (y > 0)
            a, b, r2 = _powerfit(ghsv[ok], y[ok])
            ax.loglog(ghsv[ok], y[ok], marker, ms=7, mfc="none", color=color,
                      label=label + rf": $\propto$ GHSV$^{{{b:.2f}}}$ ($R^2$={r2:.3f})")
            xs = np.array([ghsv[ok].min() * 0.7, ghsv[ok].max() * 1.4])
            ax.loglog(xs, a * xs**b, "--", lw=1.3, color=color)
            crit = (1.0 / a) ** (1.0 / b)
            note.append((label, crit))
        ax.axhline(1.0, color="k", lw=1.0, ls=":")
        crit_pe = note[0][1]
        ax.annotate(rf"GHSV$_{{crit}}$({note[0][0]}=1) $\approx$ {crit_pe:,.0f}",
                    xy=(0.03, 0.05), xycoords="axes fraction", fontsize=11)
        r_m = sub.iloc[0]["r_max_m"] if "r_max_m" in sub.columns else float("nan")
        ax.set_title(rf"{fam}:  $r$ = {r_m:g} m", fontsize=13)
        ax.set_xlabel(r"GHSV  [h$^{-1}$]", fontsize=13.5)
        ax.set_ylabel(r"$Pe_{\mathrm{rad}}$, $Da_{\mathrm{diff}}$  [–]", fontsize=13.5)
        ax.tick_params(labelsize=12)
        ax.grid(True, which="both", ls=":", alpha=0.22)
        ax.legend(fontsize=10.5, loc="upper left", **LEG_KWARGS)
    fig.suptitle("Power-law scaling of radial Péclet and diffusive Damköhler"
                 " numbers with GHSV", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_s3_pe_da_radius(scalars: pd.DataFrame):
    """S.3 — Pe_rad,max (G7, low GHSV) and Da_diff,max (G8) vs tube radius."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    panels = (
        ("G7", "Pe_H2_max", "#2980b9", "o",
         r"$Pe_{\mathrm{rad,H_2}}^{\max}$", "GHSV = 100"),
        ("G8", "Da_diff_NH3_max", "#d62728", "s",
         r"$Da_{\mathrm{diff,NH_3}}^{\max}$", "GHSV = 10,000"),
    )
    for ax, (fam, col, color, marker, label, whsv_lbl) in zip(axes, panels):
        sub = scalars[scalars["family"] == fam].sort_values("r_max_m")
        r = sub["r_max_m"].to_numpy()
        y = sub[col].to_numpy()
        ok = np.isfinite(y) & (y > 0)
        a, b, r2 = _powerfit(r[ok], y[ok])
        ax.loglog(r[ok], y[ok], marker, ms=8, mfc="none", color=color,
                  label=label + rf": $\propto r^{{{b:.2f}}}$ ($R^2$={r2:.3f})")
        xs = np.array([r[ok].min() * 0.9, r[ok].max() * 1.1])
        ax.loglog(xs, a * xs**b, "--", lw=1.4, color=color)
        ax.set_title(f"{fam}  ({whsv_lbl}"
                     r" mLn g$_{cat}^{-1}$ h$^{-1}$)", fontsize=13)
        ax.set_xlabel(r"$r$  [m]", fontsize=13.5)
        ax.set_ylabel(label + "  [–]", fontsize=13.5)
        ax.tick_params(labelsize=12)
        ax.grid(True, which="both", ls=":", alpha=0.22)
        ax.legend(fontsize=10.5, loc="upper left", **LEG_KWARGS)
    fig.suptitle("Power-law scaling of radial Péclet and diffusive Damköhler"
                 " numbers with tube radius", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ── S.4 / S.5: species comparison and regime map ─────────────────────

def figure_s4_pe_ratio(scalars: pd.DataFrame):
    """S.4 — <Pe_NH3> vs <Pe_H2> for all cases, with constant-ratio lines."""
    ratio = scalars["Pe_NH3_mean"] / scalars["Pe_H2_mean"]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    lim = [min(scalars["Pe_H2_mean"].min(), scalars["Pe_NH3_mean"].min()) * 0.6,
           max(scalars["Pe_H2_mean"].max(), scalars["Pe_NH3_mean"].max()) * 1.6]
    for rr, ls in ((1.48, "--"), (ratio.mean(), "-.")):
        ax.plot(lim, [v * rr for v in lim], ls, color="#7f8c8d", lw=1.2,
                label=rf"ratio = {rr:.2f}")
    for fam, sub in scalars.groupby("family"):
        ax.plot(sub["Pe_H2_mean"], sub["Pe_NH3_mean"], FAMILY_MARKERS.get(fam, "o"),
                ms=7, mfc="none", color=FAMILY_COLORS.get(fam, "k"), label=fam)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\langle Pe_{\mathrm{rad,H_2}}\rangle$  [–]", fontsize=14)
    ax.set_ylabel(r"$\langle Pe_{\mathrm{rad,NH_3}}\rangle$  [–]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(rf"{len(scalars)} cases — mean ratio "
                 rf"{ratio.mean():.2f} ± {ratio.std():.2f}", fontsize=13)
    ax.grid(True, which="both", ls=":", alpha=0.22)
    ax.legend(fontsize=10, ncol=2, loc="upper left", **LEG_KWARGS)
    fig.tight_layout()
    return fig


def figure_s5_regime_map(scalars: pd.DataFrame):
    """S.5 — regime map <Pe_H2> vs <Da_diff_NH3> with unity thresholds."""
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for fam, sub in scalars.groupby("family"):
        ax.plot(sub["Da_diff_NH3_mean"], sub["Pe_H2_mean"],
                FAMILY_MARKERS.get(fam, "o"), ms=8, mfc="none",
                color=FAMILY_COLORS.get(fam, "k"), label=fam)
    ax.axhline(1.0, color="k", ls="--", lw=1.1)
    ax.axvline(1.0, color="k", ls="--", lw=1.1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\langle Da_{\mathrm{diff,NH_3}}\rangle$  [–]", fontsize=14)
    ax.set_ylabel(r"$\langle Pe_{\mathrm{rad,H_2}}\rangle$  [–]", fontsize=14)
    ax.tick_params(labelsize=12)
    n_1d_ok = int(((scalars["Pe_H2_mean"] < 1) & (scalars["Da_diff_NH3_mean"] < 1)).sum())
    ax.set_title(f"Regime map — {len(scalars)} cases; "
                 f"{n_1d_ok} satisfy Pe < 1 and Da < 1 (1D-valid)", fontsize=13)
    ax.grid(True, which="both", ls=":", alpha=0.22)
    ax.legend(fontsize=10.5, ncol=2, loc="upper left", **LEG_KWARGS)
    fig.tight_layout()
    return fig


# ── S.7: 1D vs 2D across temperature ─────────────────────────────────

def figure_s7_temperature(summary_2d: pd.DataFrame,
                          summary_1d: pd.DataFrame | None = None):
    """S.7 — 1D vs 2D KPI sweep across temperature (G5 / G6)."""
    return kpi_sweep_figure(
        summary_2d, summary_1d,
        families={
            "G5": dict(color="#d4a017", marker="o", label=r"G5: $r=0.06$ m"),
            "G6": dict(color="#2471a3", marker="s", label=r"G6: $r=0.03$ m"),
        },
        x_column="T_ret_K",
        x_label=r"$T$ [K]",
        suptitle="1D (dashed) vs 2D (solid): temperature group\n"
                 "G5: r=0.06 m,  G6: r=0.03 m,  P=80 bar,  "
                 r"GHSV=5,000 h$^{-1}$",
    )


# ── S.8-S.55: per-case 2D field pairs ────────────────────────────────

def figure_s_fields(resolution_name: str, case_id: str, *, ghsv: float):
    """One SI field figure: (a) NH3 mole fraction, (b) temperature."""
    plt.rcParams["font.family"] = FONT_FAMILY
    fields = cache.load_fields(
        settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id))
    config = cache.read_json(
        settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id) / "config.json")
    z = fields["z_c"]
    mask = z > Z_MIN_PLOT
    r_f_cm = fields["r_f_ret"] * 100.0
    r_c_cm = 0.5 * (r_f_cm[:-1] + r_f_cm[1:])
    Z, R = np.meshgrid(z[mask], r_c_cm, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    panels = (
        (fields["y_ret"][mask, :, INH3], "viridis", r"$y_{\mathrm{NH_3}}$ [–]",
         r"(a) NH$_3$ mole fraction"),
        (fields["T_ret"][mask, :], "plasma", "$T$ [K]", "(b) Temperature"),
    )
    for ax, (data, cmap, cb_label, title) in zip(axes, panels):
        mesh = ax.pcolormesh(Z, R, data, cmap=cmap, shading="nearest")
        cb = plt.colorbar(mesh, ax=ax, pad=0.03)
        cb.set_label(cb_label, fontsize=13.5)
        cb.ax.tick_params(labelsize=12)
        ax.axhline(r_f_cm[0], color="white", lw=1.2, ls="--", alpha=0.85)
        ax.set_xlabel("$z$ [m]", fontsize=14)
        ax.set_ylabel("$r$ [cm]", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(z[mask].min(), z[mask].max())
        ax.set_ylim(r_f_cm[0], r_f_cm[-1])
    family = case_id.split(" ")[0]
    fig.suptitle(
        rf"{family}:  $r$ = {float(config['r_max']):g} m,  "
        rf"$T$ = {float(config['T_ret_in']):.0f} K,  "
        rf"$P$ = {float(config['p_ret_out'])/1e5:.0f} bar,  "
        rf"GHSV = {ghsv:,.0f} h$^{{-1}}$",
        fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


#: Cases the SI document leaves out of its field-figure series: the 548 K
#: pair (G5_548 is main-text Figure 9's field case). With them the count
#: would be 50, not the SI's 48 (S.8-S.55).
SI_FIELD_SKIP = ("G5 — Temperature sweep_548", "G6 — Temperature sweep_548")


def si_field_sequence(case_table: pd.DataFrame,
                      include_598k: str | None = None,
                      skip: Sequence[str] = SI_FIELD_SKIP) -> list[str]:
    """SI field-figure order (S.8 onward), mirroring the SI document:

    G1/G2 interleaved per GHSV value, then G3/G4 per pressure, G5/G6 per
    temperature, then all of G7, then all of G8.
    """
    table = cases_mod.select_cases(case_table, include_598k=include_598k)
    table = table.assign(Case_ID=table["Case_ID"].astype(str))
    table = table[~table["Case_ID"].isin(skip)]

    def fam(name: str) -> pd.DataFrame:
        """Rows of the case table whose Case_ID starts with this family name."""
        return table[table["Case_ID"].str.startswith(name + " ")]

    seq: list[str] = []
    for a, b, key in (("G1", "G2", "GHSV_h"), ("G3", "G4", "p_ret_bar"),
                      ("G5", "G6", "T_ret_K")):
        fa, fb = fam(a).sort_values(key), fam(b).sort_values(key)
        for value in sorted(set(fa[key]) | set(fb[key])):
            for f in (fa, fb):
                match = f[f[key] == value]
                seq.extend(match["Case_ID"].tolist())
    for g in ("G7", "G8"):
        seq.extend(fam(g).sort_values("r_max_m")["Case_ID"].tolist())
    return seq


def render_si_fields(resolution_name: str, case_table: pd.DataFrame,
                     *, start_number: int = 8,
                     include_598k: str | None = None,
                     verbose: int = 1) -> pd.DataFrame:
    """Render every SI field figure; returns the number-to-case map."""
    rows = []
    for offset, case_id in enumerate(si_field_sequence(case_table, include_598k)):
        number = start_number + offset
        ghsv = float(case_table.loc[case_table["Case_ID"].astype(str) == case_id,
                                    "GHSV_h"].iloc[0])
        try:
            fig = figure_s_fields(resolution_name, case_id, ghsv=ghsv)
        except (FileNotFoundError, OSError) as exc:
            if verbose:
                print(f"S.{number} SKIPPED {case_id}: {exc}", flush=True)
            rows.append({"figure": f"S.{number}", "Case_ID": case_id, "status": "skipped"})
            continue
        save_si(fig, f"figS{number:02d}_fields", resolution_name)
        plt.close(fig)
        rows.append({"figure": f"S.{number}", "Case_ID": case_id, "status": "ok"})
        if verbose:
            print(f"S.{number}: {case_id}", flush=True)
    mapping = pd.DataFrame(rows)
    out = si_dir(resolution_name) / "si_field_map.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out, index=False)
    return mapping


# ── S.56: CP profiles across temperature ─────────────────────────────

def figure_s56_cp_temperature(resolution_name: str,
                              temps: Sequence[int] = (573, 623, 648)):
    """S.56 — axial CP profiles, G5 vs G6 across operating temperature."""
    plt.rcParams["font.family"] = FONT_FAMILY
    col_g5, col_g6 = "#0d6e8a", "#c85a1e"
    fig, axes = plt.subplots(1, len(temps), figsize=(18, 7), sharey=True)
    fig.subplots_adjust(wspace=0.10, left=0.07, right=0.97, top=0.86, bottom=0.13)
    for ax, T in zip(np.atleast_1d(axes), temps):
        ax.axhline(1.0, color="k", lw=1.0, ls=":", alpha=1.0)
        for group, color, ls in (("G5", col_g5, "-"), ("G6", col_g6, "--")):
            z, cp = load_cp(resolution_name, f"{group} — Temperature sweep_{T}")
            ax.plot(z, cp, color=color, lw=2.2, ls=ls)
        ax.set_title(f"$T$ = {T} K", fontsize=16, pad=6)
        ax.set_xlabel("$z$ [m]", fontsize=16)
        ax.set_xlim(Z_MIN_PLOT, None)
        ax.set_ylim(*CP_YLIM)
        ax.grid(True, ls=":", alpha=0.22)
    np.atleast_1d(axes)[0].set_ylabel(CP_YLABEL, fontsize=16)
    fig.legend(handles=[
        mlines.Line2D([], [], color=col_g5, lw=2.2, ls="-", label=r"G5 — $r = 0.06$ m"),
        mlines.Line2D([], [], color=col_g6, lw=2.2, ls="--", label=r"G6 — $r = 0.03$ m"),
        mlines.Line2D([], [], color="k", lw=1.0, ls=":", label="CP = 1  (1D assumption)"),
    ], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99), fontsize=14, **LEG_KWARGS)
    fig.suptitle(
        r"Axial $CP_{\mathrm{NH_3}}$ profiles: G5 ($r=0.06$ m) vs G6 ($r=0.03$ m)"
        "\n"
        r"GHSV = 5,000 h$^{-1}$, $P = 80$ bar",
        fontsize=14, y=1.06)
    return fig


def figure_s57_parity(summary_2d: pd.DataFrame, summary_1d: pd.DataFrame,
                      summary_corrected: pd.DataFrame,
                      summary_screened: pd.DataFrame):
    """S.57 — parity of the three 1D variants against the certified 2D KPIs."""
    plt.rcParams["font.family"] = FONT_FAMILY
    drop = set(settings.CASES_598K)
    s2 = summary_2d[~summary_2d["Case_ID"].isin(drop)]

    def merged(kpi):
        m = s2[["Case_ID", kpi]].rename(columns={kpi: "v2"})
        for tag, df in (("p", summary_1d), ("c", summary_corrected),
                        ("s", summary_screened)):
            m = m.merge(df[["Case_ID", kpi]].rename(columns={kpi: f"v{tag}"}),
                        on="Case_ID")
        return m

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, kpi, lbl in ((axes[0], "X_H2_out", r"$X_{\mathrm{H_2}}$ [%]"),
                         (axes[1], "NH3_rec_out", r"NH$_3$ recovery [%]")):
        m = merged(kpi)
        lim = [0, float(max(m["v2"].max(), m["vp"].max()) * 1.06)]
        ax.plot(lim, lim, "k-", lw=0.8)
        ax.plot(m["v2"], m["vp"], "o", ms=6, mfc="none", color="#888888",
                label="plain 1D (CP = 1)")
        ax.plot(m["v2"], m["vc"], "s", ms=6, mfc="none", color="#1f77b4",
                label=r"$Sh_{opt}(\kappa)$-corrected 1D")
        ax.plot(m["v2"], m["vs"], "^", ms=6, mfc="none", color="#d62728",
                label="mechanistic-corrected 1D")
        ax.set_xlabel(f"2D {lbl}", fontsize=15)
        ax.set_ylabel(f"1D {lbl}", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.legend(fontsize=11.5, loc="upper left", **LEG_KWARGS)
    axes[0].set_title("(a)", loc="left", fontsize=13)
    axes[1].set_title("(b)", loc="left", fontsize=13)
    fig.tight_layout()
    return fig
