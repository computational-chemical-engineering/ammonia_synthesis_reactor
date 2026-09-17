"""Publication figures, one function per figure.

Every function reads only from the cache and the summary CSVs, so a figure
cell retreads in seconds. Logic is ported from ``scripts/paper_figures/*.py``
— the *logic*, never the hardcoded ``C:/Users/20214256/...`` paths.

Figure numbering and layout follow the manuscript; captions there are the
authority on content.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.lines as mlines
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reactor.paper import cache, kpis as kpi_mod, settings

# ── Shared style ─────────────────────────────────────────────────────
# Sizes and fonts reproduce the manuscript figures so the output drops
# into the paper unchanged.

LEG_KWARGS = dict(framealpha=1.0, facecolor="white", edgecolor="black", fancybox=False)
FONT_FAMILY = "DejaVu Sans"

KPI_LABELS = {
    "X_H2_out": r"$X_{\mathrm{H_2}}$ [%]",
    "NH3_prod_out": "NH$_3$ production rate\n"
                    r"[mmol g$_\mathrm{cat}^{-1}$ h$^{-1}$]",
    "NH3_rec_out": r"NH$_3$ recovery [%]",
    "NH3_yield_out": r"NH$_3$ yield [%]",
    "NH3_purity_out": r"NH$_3$ purity [%]",
    "DeltaT_max": r"$\Delta T_\mathrm{max}$ [K]",
}

CP_YLABEL = (r"$CP_{\mathrm{NH_3}}(z) = y_{\mathrm{NH_3,mw}} / "
             r"\langle y_{\mathrm{NH_3}} \rangle$ [–]")

Z_MIN_PLOT = 0.05   # skip the sealing region, where there is no permeation
CP_YLIM = (0.0, 1.05)


def _pending_1d_note(fig, *, y: float = 0.01) -> None:
    """Mark a figure as awaiting the 1D sweeps (footer note)."""
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.05)
    fig.text(
        0.5, y,
        "1D series pending — run the 1D sweeps (scripts.run_paper_sweep --model 1d)",
        ha="center", va="bottom", fontsize=11, color="#b45309",
    )


def save(fig, name: str, resolution_name: str, *, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    """Write a figure into the resolution's figure directory."""
    out_dir = settings.figures_dir(resolution_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300 if ext == "pdf" else 200, bbox_inches="tight")
        written.append(path)
    return written


# ── Figures 6-8: 1D vs 2D KPI sweeps ─────────────────────────────────
# Ported from scripts/paper_figures/plot_1d_2d_{whsv,pressure,radius}.py

def _accepted(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only solver-accepted (or, lacking that column, non-failed) rows."""
    if "solver_accepted" in df.columns:
        return df[df["solver_accepted"].astype(bool)]
    return df[df["status"] != "failed"]


def kpi_sweep_figure(
    summary_2d: pd.DataFrame,
    summary_1d: pd.DataFrame | None,
    *,
    families: dict[str, dict[str, Any]],
    x_column: str,
    x_label: str,
    x_scale: str = "linear",
    x_ticks: Sequence[float] | None = None,
    x_transform=None,
    suptitle: str = "",
    kpis: Sequence[str] = kpi_mod.SWEEP_KPIS,
    ylims: dict[str, tuple[float, float] | None] | None = None,
    figsize: tuple[float, float] = (14.0, 8.2),
    legend: str = "shared",
    legend_fontsize: int = 12,
):
    """The 2x3 KPI panel shared by Figures 6, 7, 8 and S.7.

    2D solid, 1D dashed, one colour/marker per family. ``summary_1d`` may be
    None, in which case only the 2D series is drawn and the figure is marked
    as awaiting the 1D sweeps.

    ``legend="shared"`` (default, all multi-panel figures per co-author
    review 2026-08-25) draws one legend below the grid;
    ``legend="panel"`` repeats it in every panel.
    """
    plt.rcParams["font.family"] = FONT_FAMILY
    df2 = _accepted(summary_2d).copy()
    df1 = _accepted(summary_1d).copy() if summary_1d is not None else None

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.subplots_adjust(hspace=0.42, wspace=0.38, left=0.08, right=0.97,
                        top=0.88, bottom=0.18 if legend == "shared" else 0.10)

    handles: list[Any] = []
    labels: list[str] = []

    for idx, (ax, kpi) in enumerate(zip(axes.flat, kpis)):
        # DeltaT_max: compare like with like. A 1D model represents the
        # cross-sectional mean temperature, so its DeltaT is compared against
        # the 2D radially averaged profile's maximum (DeltaT_max_avg, within
        # ~2% for the corrected 1D); the 2D pointwise hot-spot value is shown
        # separately as a dotted series (median 8.5% above the mean — an
        # observable no 1D closure can reproduce by construction).
        deltaT_split = kpi == "DeltaT_max" and "DeltaT_max_avg" in df2.columns
        kpi_2d = "DeltaT_max_avg" if deltaT_split else kpi
        for family, style in families.items():
            sub2 = df2[df2["family"] == family].sort_values(x_column)
            if sub2.empty:
                continue
            x2 = x_transform(sub2[x_column]) if x_transform else sub2[x_column]
            label_2d = (rf"2D $\langle T\rangle$ {style['label']}"
                        if deltaT_split else f"2D {style['label']}")
            h2, = ax.plot(x2, sub2[kpi_2d], ls="-", lw=2.0, color=style["color"],
                          marker=style["marker"], ms=6,
                          label=label_2d, zorder=4)
            if idx == 0:
                handles.append(h2)
                labels.append(f"2D {style['label']}")
            if deltaT_split:
                ax.plot(x2, sub2["DeltaT_max"], ls=":", lw=1.3,
                        color=style["color"], alpha=0.55,
                        label=f"2D hot-spot {style['label']}", zorder=2)

            if df1 is None:
                continue
            # Align the 1D series to the 2D x-values case by case, so a case
            # that converged in 2D but not in 1D simply drops out.
            sub1 = sub2[["Case_ID", x_column]].merge(
                df1[["Case_ID", kpi]], on="Case_ID").sort_values(x_column)
            if sub1.empty:
                continue
            x1 = x_transform(sub1[x_column]) if x_transform else sub1[x_column]
            h1, = ax.plot(x1, sub1[kpi], ls="--", lw=1.5, color=style["color"],
                          marker=style["marker"], ms=5, alpha=0.70,
                          label=f"1D {style['label']}", zorder=3)
            if idx == 0:
                handles.append(h1)
                labels.append(f"1D {style['label']}")

        if x_scale == "log":
            ax.set_xscale("log")
        if x_ticks is not None:
            ax.set_xticks(list(x_ticks))
            ax.set_xticklabels([str(t) for t in x_ticks], fontsize=14)
        if ylims and ylims.get(kpi):
            ax.set_ylim(*ylims[kpi])
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(KPI_LABELS.get(kpi, kpi), fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True, which="both", ls=":", alpha=0.25)
        if legend == "panel":
            leg = ax.legend(fontsize=legend_fontsize, loc="best", **LEG_KWARGS)
            leg.get_frame().set_linewidth(1.0)
        ax.set_title(f"({'abcdef'[idx]})", fontsize=13, fontweight="normal", pad=6)

    shared = legend == "shared" and bool(handles)
    if shared:
        if "DeltaT_max" in kpis and "DeltaT_max_avg" in df2.columns:
            # The DeltaT panel's extra series: hot-spot dotted (the solid 2D
            # curve of panel (f) is the radially averaged profile).
            handles.append(mlines.Line2D([], [], color="#666666", ls=":", lw=1.3))
            labels.append("2D hot-spot (f)")
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(labels), fontsize=legend_fontsize,
                   bbox_to_anchor=(0.5, 0.01), **LEG_KWARGS)

    fig.suptitle(suptitle, fontsize=13, y=0.99)
    if df1 is None:
        # Keep the note clear of the shared legend, which occupies the strip
        # the note would otherwise sit in.
        _pending_1d_note(fig, y=0.005 if shared else 0.01)
    return fig


def figure_6_whsv(summary_2d: pd.DataFrame, summary_1d: pd.DataFrame | None = None):
    """Figure 6 — 1D vs 2D across GHSV (G1: r=0.06 m, G2: r=0.03 m)."""
    return kpi_sweep_figure(
        summary_2d, summary_1d,
        families={
            "G1": dict(color="#2471a3", marker="o", label=r"G1: $r=0.06$ m"),
            "G2": dict(color="#c0392b", marker="s", label=r"G2: $r=0.03$ m"),
        },
        x_column="GHSV_h",
        x_label=r"GHSV [h$^{-1}$]",
        x_scale="log",
        suptitle="1D (dashed) vs 2D (solid): GHSV group\n"
                 "G1: r=0.06 m,  G2: r=0.03 m,  P=80 bar,  T=623 K",
    )


def figure_7_pressure(summary_2d: pd.DataFrame, summary_1d: pd.DataFrame | None = None):
    """Figure 7 — 1D vs 2D across retentate pressure (G3, G4)."""
    return kpi_sweep_figure(
        summary_2d, summary_1d,
        families={
            "G3": dict(color="#1e8449", marker="D", label=r"G3: $r=0.06$ m"),
            "G4": dict(color="#7d3c98", marker="P", label=r"G4: $r=0.03$ m"),
        },
        x_column="p_ret_bar",
        x_label="Pressure [bar]",
        # The archived script forces x-ticks 30..80 and clamps recovery to
        # (0, 50); the manuscript figure shows neither, so both are left to
        # autoscale here. Flag at the figure-map review.
        suptitle="1D (dashed) vs 2D (solid): Pressure group\n"
                 r"G3: $r=0.06$ m,  G4: $r=0.03$ m,  "
                 r"GHSV$=10\,000$ h$^{-1}$",
    )


def figure_8_radius(summary_2d: pd.DataFrame, summary_1d: pd.DataFrame | None = None):
    """Figure 8 — 1D vs 2D across tube radius (G7 low GHSV, G8 high GHSV)."""
    return kpi_sweep_figure(
        summary_2d, summary_1d,
        families={
            "G7": dict(color="#117a65", marker="<",
                       label=r"G7: GHSV$=100$ h$^{-1}$"),
            "G8": dict(color="#b03a2e", marker=">",
                       label=r"G8: GHSV$=10,000$ h$^{-1}$"),
        },
        x_column="r_max_m",
        x_label=r"Radius $r$  [m]$\times 10^{-2}$",
        x_ticks=[2, 3, 4, 5, 6],
        x_transform=lambda s: s * 100.0,
        legend="shared",
        legend_fontsize=10,
        suptitle="1D (dashed) vs 2D (solid): Radius group\n"
                 r"$P=80$ bar,  $T=623$ K  |  "
                 r"G7: GHSV$=100$ h$^{-1}$,  "
                 r"G8: GHSV$=10\,000$ h$^{-1}$",
    )


# ── CP helpers ───────────────────────────────────────────────────────

def load_cp(resolution_name: str, case_id: str, *, z_min: float = Z_MIN_PLOT):
    """Axial CP_NH3 profile for one cached case, past the sealing region."""
    fields = cache.load_fields(settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id))
    cp = kpi_mod.cp_profile(fields["y_ret"], fields["r_f_ret"])
    z = fields["z_c"]
    mask = z > z_min
    return z[mask], cp[mask]


def _unusable(resolution_name: str, case_ids: Iterable[str]) -> dict[str, str]:
    """Cases a field-reading figure cannot draw, and why.

    A failed case is cached — completely and legitimately, with its solve
    status and diagnosis — but carries no ``fields.npz``. Completeness alone
    is therefore not enough to plot from.
    """
    res = settings.resolution(resolution_name)
    reasons: dict[str, str] = {}
    for case_id in case_ids:
        case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id)
        if not cache.is_complete(case_dir, res):
            reasons[case_id] = "not solved at this resolution"
        elif cache.load_meta(case_dir).get("status") == "failed":
            reasons[case_id] = "solve failed — no fields to plot"
    return reasons


def _require(resolution_name: str, case_ids: Iterable[str]) -> None:
    """Raise FileNotFoundError when any of these cases cannot be plotted from."""
    reasons = _unusable(resolution_name, case_ids)
    if reasons:
        detail = "; ".join(f"{c} ({why})" for c, why in reasons.items())
        raise FileNotFoundError(
            f"cannot draw this figure at resolution {resolution_name!r}: {detail}"
            f"\nrun: python -m scripts.run_paper_sweep --resolution {resolution_name}"
        )


# ── Figure 9: 2D field evidence for CP ───────────────────────────────
# Ported from scripts/paper_figures/plot_field_cp.py

FIG9_FIELD_CASE = "G5 — Temperature sweep_548"
FIG9_CP_CASES = (
    ("G5 — Temperature sweep_548", r"G5, $T=548$ K (worst)", "#e74c3c", "-"),
    ("G1 — GHSV sweep_10000",
     r"G1, GHSV=10,000 h$^{-1}$", "#2980b9", "--"),
    ("G3 — Pressure sweep_30", r"G3, $P=30$ bar (best)", "#27ae60", "-."),
)


def figure_9_field_cp(resolution_name: str, *, field_case: str = FIG9_FIELD_CASE,
                      cp_cases: Sequence[tuple[str, str, str, str]] = FIG9_CP_CASES):
    """Figure 9 — NH3 field, temperature field and axial CP for three cases.

    The archived script hardcoded ``CP_min`` in the legend labels; here they
    are computed from the same data the curve is drawn from.
    """
    _require(resolution_name, [field_case, *(c[0] for c in cp_cases)])
    plt.rcParams["font.family"] = FONT_FAMILY
    lfs, tfs, tifs, cbfs = 18, 16, 14, 18

    fields = cache.load_fields(
        settings.case_cache_dir(resolution_name, settings.MODEL_2D, field_case))
    z = fields["z_c"]
    mask = z > Z_MIN_PLOT
    r_f_cm = fields["r_f_ret"] * 100.0
    r_c_cm = 0.5 * (r_f_cm[:-1] + r_f_cm[1:])
    Z, R = np.meshgrid(z[mask], r_c_cm, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.97, top=0.88, bottom=0.13)

    panels = (
        (fields["y_ret"][mask, :, kpi_mod.INH3], "viridis",
         r"$y_{\mathrm{NH_3}}$ [–]", r"(a) NH$_3$ mole fraction field"),
        (fields["T_ret"][mask, :], "plasma",
         "$T$ [K]", "(b) Temperature field"),
    )
    for ax, (data, cmap, cb_label, title) in zip(axes[:2], panels):
        mesh = ax.pcolormesh(Z, R, data, cmap=cmap, shading="nearest")
        cb = plt.colorbar(mesh, ax=ax, pad=0.03)
        cb.set_label(cb_label, fontsize=cbfs)
        cb.ax.tick_params(labelsize=tfs - 1)
        ax.axhline(r_f_cm[0], color="white", lw=1.5, ls="--", alpha=0.85)
        ax.set_xlabel("$z$ [m]", fontsize=lfs)
        ax.set_ylabel("$r$ [cm]", fontsize=lfs)
        ax.set_title(title, fontsize=tifs + 2, pad=6)
        ax.tick_params(labelsize=tfs)
        ax.set_xlim(z[mask].min(), z[mask].max())
        ax.set_ylim(r_f_cm[0], r_f_cm[-1])
        ax.grid(False)
    axes[0].text(z[mask][-1] * 0.98, r_f_cm[0] + 0.12, r"membrane wall ($r=R_1$)",
                 fontsize=13, color="white", ha="right", va="bottom",
                 path_effects=[patheffects.withStroke(linewidth=2.2,
                                                      foreground="#00000088")])

    ax = axes[2]
    for case_id, label, color, ls in cp_cases:
        z_cp, cp = load_cp(resolution_name, case_id)
        ax.plot(z_cp, cp, color=color, lw=2.2, ls=ls,
                label=rf"{label} ($CP_{{\min}}={cp.min():.2f}$)")
    ax.axhline(1.0, color="k", lw=1.0, ls=":", alpha=0.45, label="CP = 1 (1D assumption)")
    ax.set_xlabel("$z$ [m]", fontsize=lfs)
    ax.set_ylabel(CP_YLABEL, fontsize=lfs)
    ax.set_title(r"(c) Axial $CP_{\mathrm{NH_3}}$ profiles — 3 representative cases",
                 fontsize=tifs + 2, pad=6)
    ax.tick_params(labelsize=tfs)
    ax.grid(True, ls=":", alpha=0.22)
    ax.set_ylim(*CP_YLIM)
    ax.legend(fontsize=12, loc="lower left", **LEG_KWARGS)

    cfg = cache.read_json(
        settings.case_cache_dir(resolution_name, settings.MODEL_2D, field_case)
        / "config.json")
    fig.suptitle(
        r"2D field evidence for NH$_3$ concentration polarisation — most severe"
        rf" CP case: {field_case.split(' ')[0]}, $T$ = {float(cfg['T_ret_in']):.0f} K,"
        rf" $r$ = {float(cfg['r_max']):g} m, $P$ = {float(cfg['p_ret_out'])/1e5:.0f} bar",
        fontsize=15, y=1.01)
    return fig


# ── Figures 10-11: axial CP profiles ─────────────────────────────────
# Ported from scripts/paper_figures/plot_CP_profiles.py and plot_CP_tuberadius.py

def figure_10_cp_pressure(resolution_name: str, pressures: Sequence[int] = (30, 50, 80)):
    """Figure 10 — axial CP profiles, G3 vs G4 across operating pressures."""
    cases = {(g, p): f"{g} — Pressure sweep_{p}" for g in ("G3", "G4") for p in pressures}
    _require(resolution_name, cases.values())
    plt.rcParams["font.family"] = FONT_FAMILY
    lfs = tfs = tifs = 18
    col_g3, col_g4 = "#0d6e8a", "#c85a1e"

    fig, axes = plt.subplots(1, len(pressures), figsize=(18, 7), sharey=True)
    fig.subplots_adjust(wspace=0.10, left=0.07, right=0.97, top=0.88, bottom=0.13)

    for ax, p in zip(np.atleast_1d(axes), pressures):
        ax.axhline(1.0, color="k", lw=1.0, ls=":", alpha=1.0)
        for group, color, ls in (("G3", col_g3, "-"), ("G4", col_g4, "--")):
            z, cp = load_cp(resolution_name, cases[(group, p)])
            ax.plot(z, cp, color=color, lw=2.2, ls=ls)
        ax.set_title(f"$P$ = {p} bar", fontsize=tifs, pad=6)
        ax.set_xlabel("$z$ [m]", fontsize=lfs)
        ax.set_xlim(Z_MIN_PLOT, None)
        ax.set_ylim(*CP_YLIM)
        ax.tick_params(labelsize=tfs)
        ax.grid(True, ls=":", alpha=0.22)
    np.atleast_1d(axes)[0].set_ylabel(CP_YLABEL, fontsize=lfs)

    fig.legend(
        handles=[
            mlines.Line2D([], [], color=col_g3, lw=2.2, ls="-", label=r"G3 — $r = 0.06$ m"),
            mlines.Line2D([], [], color=col_g4, lw=2.2, ls="--", label=r"G4 — $r = 0.03$ m"),
            mlines.Line2D([], [], color="k", lw=1.0, ls=":", label="CP = 1  (1D assumption)"),
        ],
        loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), fontsize=15, **LEG_KWARGS,
    )
    fig.suptitle(
        r"Axial $CP_{\mathrm{NH_3}}$ profiles: G3 ($r=0.06$ m) vs G4 ($r=0.03$ m)"
        "\n"
        r"GHSV = 10,000 h$^{-1}$, $T = 623$ K",
        fontsize=15, y=1.08,
    )
    return fig


FIG11_GROUPS = {"G7": "Radius sweep (low GHSV)", "G8": "Radius sweep (high GHSV)"}
FIG11_PANEL_TITLES = {
    "G7": r"GHSV $= 100\ \mathrm{h^{-1}}$",
    "G8": r"GHSV $= 10\,000\ \mathrm{h^{-1}}$",
}
FIG11_COLORS = {0.02: "#2c7bb6", 0.045: "#1a9641", 0.06: "#d7191c"}


def figure_11_cp_radius(resolution_name: str, radii: Sequence[float] = (0.02, 0.045, 0.06)):
    """Figure 11 — axial CP profiles, G7 vs G8 across tube radius."""
    cases = {(g, r): f"{g} — {FIG11_GROUPS[g]}_{r}" for g in FIG11_GROUPS for r in radii}
    _require(resolution_name, cases.values())
    plt.rcParams["font.family"] = FONT_FAMILY
    lfs = tfs = tifs = 18

    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
    fig.subplots_adjust(wspace=0.10, left=0.08, right=0.97, top=0.78, bottom=0.13)

    cp_top = CP_YLIM[1]
    for ax, group in zip(axes, ("G7", "G8")):
        ax.axhline(1.0, color="k", lw=1.0, ls=":", alpha=1.0)
        for r in radii:
            z, cp = load_cp(resolution_name, cases[(group, r)])
            # CP can exceed 1 downstream at the smallest radius (wall
            # enrichment, discussed in the text) — never clip it.
            cp_top = max(cp_top, float(np.nanmax(cp)) * 1.03)
            ax.plot(z, cp, color=FIG11_COLORS[r], lw=2.2,
                    ls="-" if group == "G7" else "--")
        ax.set_title(FIG11_PANEL_TITLES[group], fontsize=tifs, pad=6)
        ax.set_xlabel(r"$z$ [m]", fontsize=lfs)
        ax.set_xlim(Z_MIN_PLOT, None)
        ax.set_ylim(CP_YLIM[0], cp_top)
        ax.tick_params(labelsize=tfs)
        ax.grid(True, ls=":", alpha=0.22)
    axes[0].set_ylabel(CP_YLABEL, fontsize=lfs)

    handles = [mlines.Line2D([], [], color=FIG11_COLORS[r], lw=2.2, ls="-",
                             label=rf"$r = {r}$ m") for r in radii]
    handles.append(mlines.Line2D([], [], color="k", lw=1.0, ls=":",
                                 label="CP = 1  (1D assumption)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.90),
               fontsize=13, handlelength=2.5, handleheight=1.4, labelspacing=0.8,
               handletextpad=0.8, borderpad=0.8, columnspacing=1.5, **LEG_KWARGS)
    fig.suptitle(
        r"Axial $CP_{\mathrm{NH_3}}$ profiles: G7 (GHSV $= 100$) vs "
        r"G8 (GHSV $= 10\,000\ \mathrm{h^{-1}}$)"
        "\n"
        r"$T = 623$ K, $P = 80$ bar"
        "\n"
        r"$r \in \{0.02,\,0.045,\,0.06\}$ m",
        fontsize=15, y=0.99,
    )
    return fig



# ── Figure 2: Weisz-Prater criterion ─────────────────────────────────
# No archived script existed. Manuscript formatting (linear axes, r_p in
# mm, C_WP on the y-axis, H2 black / N2 blue / NH3 magenta); content is
# the certified-envelope evaluation of reactor.paper.validation (worst
# local |R_i|/(D_eff,i c_i) over every accepted cached 2D case) — updated
# dataset relative to the manuscript's nominal-inlet evaluation.

WP_COLORS = {"H2": "black", "N2": "#1414c8", "NH3": "#e6007e"}


def figure_2_weisz_prater(scan: dict, *, r_p_max_mm: float = 5.0):
    """C_WP(r_p) per species — manuscript convention, honest states.

    The reactants (H2, N2) are evaluated at the nominal worst operating
    state of the caption ("maximum pressure, optimal temperature, inlet
    composition"). NH3 is a trace at the inlet, where its modulus is
    undefined; it is evaluated at the certified state of maximum NH3
    content instead (recorded in the scan). Solid: D_eff = f_eff*D;
    dotted: the molecular-diffusivity bound.
    """
    r_p = np.linspace(1e-4, r_p_max_mm, 400) * 1e-3        # [m]
    f_eff = scan.get("eff_diff_factor", 0.1)
    nominal = scan.get("nominal", {})
    state = scan["worst"].get("_state_at_max_yNH3")

    q_map = dict(nominal.get("phi_over_feff_rp2", {}))
    if state is not None:
        q_map["NH3"] = state["phi_over_feff_rp2"].get("NH3")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    pretty = {"H2": "H$_2$", "N2": "N$_2$", "NH3": "NH$_3$"}
    for sp in ("H2", "N2", "NH3"):
        q = q_map.get(sp)
        if q is None:
            continue
        ax.plot(r_p * 1e3, q * r_p**2 / f_eff, "-", lw=2.4,
                color=WP_COLORS[sp], label=pretty[sp])
        ax.plot(r_p * 1e3, q * r_p**2, ":", lw=1.3, color=WP_COLORS[sp], alpha=0.7)
    ax.axhline(0.3, color="#555", ls="--", lw=1.0)
    ax.text(0.12, 0.48, r"$C_{WP}=0.3$", fontsize=13, color="#444",
            transform=ax.get_yaxis_transform(),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    dp_mm = scan["dp_m"] * 1e3 / 2.0
    ax.axvline(dp_mm, color="#999", ls="-", lw=1.0)
    if nominal:
        title = (rf"H$_2$/N$_2$ at inlet composition, $T$ = {nominal['T_K']:.0f} K, "
                 rf"$P$ = {nominal['p_bar']:.0f} bar")
        if state is not None:
            title += (f"\nNH$_3$ at the max-NH$_3$ certified state "
                      f"({state['case_id'].split(' — ')[0]}, "
                      rf"$y_{{\mathrm{{NH_3}}}}$={state['y_NH3']:.2f}, "
                      rf"$T$ = {state['T_K']:.0f} K)"
                      f"\nsolid: $D_\\mathrm{{eff}}={f_eff:g}D$, dotted: $D_\\mathrm{{eff}}=D$")
        ax.set_title(title, fontsize=11.5)
    ax.set_xlabel(r"$r_P$  [mm]", fontsize=15)
    ax.set_ylabel(r"$C_{WP}$  [–]", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_xlim(0, r_p_max_mm)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=13, loc="upper left", **LEG_KWARGS)
    ax.grid(True, ls=":", alpha=0.25)
    fig.tight_layout()
    return fig


# ── Figures 3-4: Rossetti kinetics validation ────────────────────────
# The runner half existed (scripts/run_rossetti_1d.py); the plotting half
# is new. Layout matches the manuscript: Figure 3 shows four
# representative tests on linear GHSV axes; Figure S.6 (SI) carries the
# remaining conditions with the same panel function.

#: Number of tests in Figure 3. The manuscript's "four representative
#: operating conditions" are the four BEST-fitting tests by mean relative
#: error (the manuscript's selection rule);
#: Figure S.6 carries the rest. Selected from the table at plot time so
#: the updated dataset picks its own best four.
FIG3_N_BEST = 4


def _rossetti_best_tests(rossetti: pd.DataFrame, n: int = FIG3_N_BEST) -> tuple[int, ...]:
    """The ``n`` best-fitting Rossetti tests by mean relative NH3 error."""
    err = (rossetti["NH3_model_volpct"] / rossetti["NH3_exp_volpct"] - 1.0).abs()
    order = err.groupby(rossetti["Test"]).mean().sort_values()
    return tuple(int(t) for t in order.head(n).index)

_ROSSETTI_TEST_COLORS = {
    22: "#e878c8", 23: "#8c8c8c", 20: "#8464c8", 18: "#2ca02c",
    1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 7: "#9467bd",
    8: "#8c564b", 10: "#e377c2", 11: "#7f7f7f", 13: "#bcbd22",
    15: "#17becf", 16: "#1f77b4", 17: "#ff7f0e", 19: "#d62728",
    21: "#9467bd", 24: "#8c564b",
}

def _rossetti_model_label(rossetti: pd.DataFrame) -> str:
    """Legend/title label matching the model that produced the table.

    The cached tables carry a ``model`` column ("1d" | "2d"); the
    manuscript figures use the 2D table (``run_rossetti(model="2d")``,
    24 radial cells — the 1D table agrees to <=0.3% and remains as a
    cross-check).
    """
    models = set(rossetti["model"]) if "model" in rossetti.columns else set()
    if models == {"2d"}:
        return "2D Model"
    if models == {"1d"}:
        return "1D Model"
    return "Model"


def _rossetti_panel(ax, sub: pd.DataFrame, color: str, title: str) -> None:
    """Draw one Rossetti panel: model line and experimental points vs GHSV."""
    sub = sub.sort_values("GHSV_h")
    ax.plot(sub["GHSV_h"] / 1e5, sub["NH3_model_volpct"], "-", lw=1.8, color=color)
    ax.plot(sub["GHSV_h"] / 1e5, sub["NH3_exp_volpct"], "o", ms=6, color=color)
    ax.set_title(title, fontsize=12.5)
    ax.set_ylabel(r"$y_{\mathrm{NH_3}}$  [mol %]", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls=":", alpha=0.2)


def _rossetti_title(row: pd.Series) -> str:
    """Panel title with the test number and its operating conditions."""
    return (f"T{int(row['Test'])}: {row['T_C']:.0f}°C, "
            f"{row['Pressure_bar']:.0f} bar, "
            rf"H$_2$/N$_2$ = {row['H2_N2_ratio']:g}")


def figure_3_rossetti_ghsv(rossetti: pd.DataFrame,
                           tests: Sequence[int] | None = None):
    """NH3 outlet mole fraction vs GHSV, the four best-fitting tests."""
    if tests is None:
        tests = _rossetti_best_tests(rossetti)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))
    axes = axes.ravel()
    for ax, test in zip(axes, tests):
        sub = rossetti[rossetti["Test"] == test]
        if sub.empty:
            ax.set_visible(False)
            continue
        _rossetti_panel(ax, sub, _ROSSETTI_TEST_COLORS.get(test, "k"),
                        _rossetti_title(sub.iloc[0]))
    for ax in axes[2:]:
        ax.set_xlabel(r"GHSV  [h$^{-1}$] $\times 10^5$", fontsize=13)
    handles = [
        mlines.Line2D([], [], color="#555", marker="o", ls="none", label="Experiment"),
        mlines.Line2D([], [], color="#555", lw=1.8, label=_rossetti_model_label(rossetti)),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, -0.02), **LEG_KWARGS)
    fig.suptitle(f"Rossetti kinetics – {_rossetti_model_label(rossetti).lower()} validation\n"
                 r"NH$_3$ outlet mole fraction vs. GHSV",
                 fontsize=13.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return fig


def figure_s6_rossetti_rest(rossetti: pd.DataFrame,
                            exclude: Sequence[int] | None = None):
    """SI Figure S.6: the remaining Rossetti conditions, same panel style."""
    if exclude is None:
        exclude = _rossetti_best_tests(rossetti)
    tests = [t for t in sorted(rossetti["Test"].unique()) if t not in exclude]
    ncol = 3
    nrow = int(np.ceil(len(tests) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, test in zip(axes, tests):
        sub = rossetti[rossetti["Test"] == test]
        _rossetti_panel(ax, sub, _ROSSETTI_TEST_COLORS.get(int(test), "k"),
                        _rossetti_title(sub.iloc[0]))
        ax.title.set_fontsize(11.5)
        ax.yaxis.label.set_fontsize(11.5)
        ax.tick_params(labelsize=10.5)
    for ax in axes[len(tests):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(tests) - ncol):len(tests)]:
        ax.set_xlabel(r"GHSV  [h$^{-1}$] $\times 10^5$", fontsize=11.5)
    handles = [
        mlines.Line2D([], [], color="#555", marker="o", ls="none", label="Experiment"),
        mlines.Line2D([], [], color="#555", lw=1.8, label=_rossetti_model_label(rossetti)),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, -0.01), **LEG_KWARGS)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def figure_4_rossetti_parity(rossetti: pd.DataFrame):
    """Parity plot over all experimental points (manuscript layout)."""
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    lim = [0.0, max(rossetti["NH3_exp_volpct"].max(),
                    rossetti["NH3_model_volpct"].max()) * 1.10]
    ax.plot(lim, lim, "k-", lw=2.0, label="Ideal (y = x)")
    ax.plot(lim, [v * 1.1 for v in lim], "--", color="#8c8c8c", lw=1.1, label="+10%")
    ax.plot(lim, [v / 1.1 for v in lim], ":", color="#8c8c8c", lw=1.1, label="−10%")
    for test, sub in rossetti.groupby("Test"):
        ax.plot(sub["NH3_exp_volpct"], sub["NH3_model_volpct"], "o", ms=7,
                color=_ROSSETTI_TEST_COLORS.get(int(test), "k"), alpha=0.9,
                label=_rossetti_title(sub.iloc[0]))
    # R^2 about the identity line (coefficient of determination of the
    # parity, the definition quoted in the manuscript text) — NOT the
    # Pearson correlation squared, which is blind to a systematic offset.
    x = rossetti["NH3_exp_volpct"].to_numpy()
    y = rossetti["NH3_model_volpct"].to_numpy()
    r2 = 1.0 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)
    ax.text(0.97, 0.05, rf"$R^2$ = {r2:.4f}", transform=ax.transAxes,
            ha="right", fontsize=14,
            bbox=dict(facecolor="white", edgecolor="black"))
    ax.set_xlabel(r"$y_{\mathrm{NH_3}}^{\,\mathrm{exp}}$  [mol %]", fontsize=15)
    ax.set_ylabel(r"$y_{\mathrm{NH_3}}^{\,\mathrm{model}}$  [mol %]", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.2)
    ax.legend(fontsize=8, ncol=2, loc="upper left", **LEG_KWARGS)
    ax.set_title("Parity Plot – Rossetti kinetics validation\n"
                 f"{_rossetti_model_label(rossetti)}", fontsize=12)
    fig.tight_layout()
    return fig


# ── Figures 12-14: Sherwood correlation and profiles ─────────────────
# Wall-based local Sherwood analysis (Figures 12-14).
# Wall-based film Sherwood throughout (the manuscript's CP definition),
# with the refit laws of the certified dataset: the wall/film fit
# (sh_wall_fit.json) replaces "Sh_field", the flux-matched fit
# (sh_cp_fit.json — what the corrected 1D consumes) replaces "Sh_optimal".

FAMILY_COLORS = {"G1": "#2980b9", "G2": "#e74c3c", "G3": "#27ae60",
                 "G4": "#f39c12", "G5": "#8e44ad", "G6": "#16a085",
                 "G7": "#e67e22", "G8": "#34495e"}
FAMILY_MARKERS = {"G1": "o", "G2": "s", "G3": "D", "G4": "P",
                  "G5": "^", "G6": "v", "G7": "<", "G8": ">"}

#: Classical annular-duct Sherwood range quoted by the manuscript.
SH_CLASSICAL_RANGE = (3.5, 6.2)

#: Reference correlations of the manuscript's Figure 12(a), at Sc = 0.51.
SC_REF = 0.51


def _sh_re_reference_lines(ax, re_arr: np.ndarray) -> None:
    """Draw the Poto et al. and Wakao & Funazkri Sh(Re) reference correlations."""
    ax.plot(re_arr, 0.4338 * re_arr**0.3583 * SC_REF**(1 / 3),
            "k--", lw=1.6, zorder=2)
    ax.plot(re_arr, 2 + 1.1 * re_arr**0.6 * SC_REF**(1 / 3),
            color="#7f8c8d", lw=1.4, ls=":", zorder=2)


def figure_12_sh_correlation(resolution_name: str, case_table: pd.DataFrame):
    """Figure 12 — three-panel wall-Sherwood correlation: Sh vs Re_dh, both Sh(kappa) laws, and kappa-group medians."""
    from scipy import stats as sps

    from reactor.paper import closures, dimensionless

    pts = dimensionless.sh_local_table(case_table, resolution_name)
    wall = closures.load_sh_fit(resolution_name, "wall")
    flux = closures.load_sh_fit(resolution_name)

    med = (pts.groupby("kappa")
              .agg(Sh_med=("Sh", "median"),
                   q25=("Sh", lambda s: s.quantile(0.25)),
                   q75=("Sh", lambda s: s.quantile(0.75)),
                   Re_med=("Re_dh", "median"),
                   n=("Sh", "size"))
              .reset_index())
    flux_pooled = pd.DataFrame(flux["pooled"])
    r_sp = sps.spearmanr(pts["Re_dh"], pts["Sh"]).statistic

    wall_lbl = rf"$Sh_{{wall}}={wall['coeff']:.2f}\,\kappa^{{{wall['exp']:.2f}}}$"
    flux_lbl = rf"$Sh_{{flux}}={flux['coeff']:.2f}\,\kappa^{{{flux['exp']:.2f}}}$"

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.6))
    fig.subplots_adjust(wspace=0.20, left=0.05, right=0.99, top=0.82, bottom=0.13)

    # (a) local Sh vs Re_dh — flow-independence
    ax = axes[0]
    re_arr = np.logspace(np.log10(max(pts["Re_dh"].min() * 0.5, 1e-1)),
                         np.log10(pts["Re_dh"].max() * 2.0), 200)
    _sh_re_reference_lines(ax, re_arr)
    for fam, sub in pts.groupby("family"):
        ax.scatter(sub["Re_dh"], sub["Sh"], s=7, alpha=0.35,
                   color=FAMILY_COLORS.get(fam, "k"), zorder=3)
    ax.scatter(med["Re_med"], med["Sh_med"], marker="D", color="k", s=90,
               edgecolors="#1a1a1a", zorder=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$Re_{d_h}$  [–]", fontsize=16)
    ax.set_ylabel(r"$Sh_{\mathrm{NH_3}}$  [–]", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_title(rf"(a) $Sh$ vs $Re_{{d_h}}$"
                 f"\nSpearman $r={r_sp:.3f}$: Sh independent of Re",
                 fontsize=13.5, pad=5)
    handles = [mlines.Line2D([], [], marker=FAMILY_MARKERS[f], color="w",
                             markerfacecolor=FAMILY_COLORS[f],
                             markeredgecolor="#1a1a1a", ms=7, label=f)
               for f in sorted(FAMILY_COLORS)]
    handles += [
        mlines.Line2D([], [], marker="D", color="w", markerfacecolor="k", ms=8,
                      label="group median"),
        mlines.Line2D([], [], ls="--", color="k", lw=1.5, label="Poto et al. (2023)"),
        mlines.Line2D([], [], ls=":", color="#7f8c8d", lw=1.5,
                      label="Wakao & Funazkri (1978)"),
    ]
    ax.legend(handles=handles, fontsize=9.5, ncol=2, **LEG_KWARGS)
    ax.grid(True, which="both", ls=":", alpha=0.22)

    # (b) local points + both correlations vs kappa
    ax = axes[1]
    rng = np.random.default_rng(42)
    for fam, sub in pts.groupby("family"):
        jitter = sub["kappa"] * (1 + rng.normal(0, 0.006, len(sub)))
        ax.scatter(jitter, sub["Sh"], s=6, alpha=0.20,
                   color=FAMILY_COLORS.get(fam, "k"), zorder=2)
    kk = np.linspace(med["kappa"].min() * 0.92, med["kappa"].max() * 1.08, 200)
    ax.plot(kk, wall["coeff"] * kk**wall["exp"], "k-", lw=2.2,
            label=wall_lbl + rf"  ($R^2={wall['r_squared_pooled']:.3f}$)")
    ax.plot(kk, flux["coeff"] * kk**flux["exp"], "r-", lw=2.2,
            label=flux_lbl + rf"  ($R^2={flux['r_squared_pooled']:.3f}$)")
    ax.errorbar(med["kappa"], med["Sh_med"],
                yerr=[med["Sh_med"] - med["q25"], med["q75"] - med["Sh_med"]],
                fmt="ks", ms=8, capsize=3, lw=1.2, zorder=6,
                label=r"$Sh_{wall}$ median ± IQR")
    ax.plot(flux_pooled["kappa"], flux_pooled["Sh_group"], "r^", ms=9, mfc="none",
            mew=1.8, zorder=6, label=r"$Sh_{flux}$ group mean")
    ax.set_xlabel(r"$\kappa = r_\mathrm{mem}/r_\mathrm{max}$  [–]", fontsize=16)
    ax.set_ylabel(r"$Sh_{\mathrm{NH_3}}$  [–]", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_title("(b) Both Sh laws vs geometry", fontsize=13.5, pad=5)
    ax.legend(fontsize=9.5, loc="upper right", **LEG_KWARGS)
    ax.grid(True, ls=":", alpha=0.22)

    # (c) medians ± IQR only, with n counts
    ax = axes[2]
    ax.plot(kk, wall["coeff"] * kk**wall["exp"], "k-", lw=2.2, label=wall_lbl)
    ax.plot(kk, flux["coeff"] * kk**flux["exp"], "r-", lw=2.2, label=flux_lbl)
    ax.errorbar(med["kappa"], med["Sh_med"],
                yerr=[med["Sh_med"] - med["q25"], med["q75"] - med["Sh_med"]],
                fmt="ks", ms=9, capsize=3, lw=1.2, zorder=6,
                label=r"$Sh_{wall}$ median ± IQR")
    ax.plot(flux_pooled["kappa"], flux_pooled["Sh_group"], "r^", ms=10, mfc="none",
            mew=2.0, zorder=6, label=r"$Sh_{flux}$ group mean")
    for _, row in med.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["kappa"], row["q75"]),
                    textcoords="offset points", xytext=(5, 7), fontsize=9.5)
    ax.set_xlabel(r"$\kappa = r_\mathrm{mem}/r_\mathrm{max}$  [–]", fontsize=16)
    ax.set_ylabel(r"$Sh_{\mathrm{NH_3,median}}$  [–]", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_title(rf"(c) $\kappa$-group medians ± IQR", fontsize=13.5, pad=5)
    ax.legend(fontsize=9.5, loc="upper right", **LEG_KWARGS)
    ax.grid(True, ls=":", alpha=0.22)

    fig.suptitle(
        rf"Sh correlations from certified 2D data:  wall/film {wall_lbl}"
        rf"  vs  flux-matched {flux_lbl} (used by the corrected 1D model)",
        fontsize=13.5)
    return fig


def _ghsv_of(case_table: pd.DataFrame, case_id: str) -> float:
    """GHSV of a case [1/h], per catalyst volume (the case-table value)."""
    row = case_table[case_table["Case_ID"].astype(str) == case_id].iloc[0]
    return float(row["GHSV_h"])


def _sh_family_panel(ax, fig, resolution_name: str, case_table: pd.DataFrame,
                     family: str, *, x_key: str):
    """One Figure 13/14 panel: wall Sh profiles of a family, GHSV-coloured."""
    from matplotlib.colors import LogNorm

    from reactor.paper import dimensionless

    sub = case_table[case_table["Case_ID"].astype(str).str.startswith(family + " ")]
    ids = [str(c) for c in sub.sort_values("GHSV_h")["Case_ID"]]
    ids = [c for c in ids if c not in _unusable(resolution_name, [c])]
    ghsv = {c: _ghsv_of(case_table, c) for c in ids}
    norm = LogNorm(vmin=min(ghsv.values()), vmax=max(ghsv.values()))
    cmap = plt.cm.plasma

    ax.fill_between([0, 1] if x_key == "zL" else [1e-4, 3], *SH_CLASSICAL_RANGE,
                    color="#f5a623", alpha=0.35, zorder=0)
    re_lo, re_hi = np.inf, -np.inf
    kappa = None
    for case_id in ids:
        d = dimensionless.case_dimless(resolution_name, case_id)
        kappa = float(d["kappa"])
        sh = d["Sh_wall"][:, 2]
        ok = (d["z"] > Z_MIN_PLOT) & np.isfinite(sh) & (sh > 0.1) & (sh < 500)
        sh_s = _log_smooth(sh, ok)
        x = d["z"] / d["L"] if x_key == "zL" else d["z_star"]
        ax.plot(x[ok], sh_s[ok], "-", lw=1.9, alpha=0.9,
                color=cmap(norm(ghsv[case_id])), zorder=3)
        re_lo = min(re_lo, float(np.nanmin(d["Re_dh"][ok])))
        re_hi = max(re_hi, float(np.nanmax(d["Re_dh"][ok])))
    ax.set_yscale("log")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(r"GHSV  [h$^{-1}$]", fontsize=12.5)
    r_m = float(sub.iloc[0]["r_max_m"]) if "r_max_m" in sub.columns else float("nan")
    ax.set_title(rf"$\kappa={kappa:.3f}$  ($r={r_m:.2f}$ m, {family} group)"
                 f"\n$Re_{{d_h}}$ range: {re_lo:.1f}–{re_hi:.0f}"
                 f"  ({re_hi/max(re_lo, 1e-12):.0f}× span)", fontsize=12.5)
    ax.set_ylabel(r"$Sh_{\mathrm{NH_3}}$  [–]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.22)
    return kappa


def _log_smooth(values: np.ndarray, ok: np.ndarray, size: int = 7) -> np.ndarray:
    """Smooth values with a uniform filter in log space (NaN where not ``ok``)."""
    from scipy.ndimage import uniform_filter1d

    v = np.where(ok, np.log(np.maximum(values, 0.1)), 0.0)
    sm = np.exp(uniform_filter1d(v, size))
    return np.where(ok, sm, np.nan)


def figure_13_sh_whsv(resolution_name: str, case_table: pd.DataFrame,
                      families: Sequence[str] = ("G1", "G2")):
    """Manuscript Figure 13: Sh(z) collapse across GHSV at fixed kappa."""
    fig, axes = plt.subplots(1, len(families), figsize=(6.6 * len(families), 5.2))
    axes = np.atleast_1d(axes)
    for ax, family in zip(axes, families):
        _sh_family_panel(ax, fig, resolution_name, case_table, family, x_key="zL")
        ax.set_xlabel(r"$z/L$  [–]", fontsize=12)
        ax.set_xlim(0, 1)
        ax.text(0.5, np.sqrt(SH_CLASSICAL_RANGE[0] * SH_CLASSICAL_RANGE[1]),
                "Classical annular Sh", ha="center", va="center",
                fontsize=12, color="#b45309", transform=ax.get_yaxis_transform())
        handles = [
            mlines.Line2D([], [], color="#888", lw=2,
                          label=r"$Sh_{wall}(z)$ from 2D (colour = GHSV)"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#f5a623", alpha=0.35,
                          label=f"Classical annular Sh "
                                f"({SH_CLASSICAL_RANGE[0]:.1f}–{SH_CLASSICAL_RANGE[1]:.1f})"),
        ]
        ax.legend(handles=handles, fontsize=10.5, loc="upper right", **LEG_KWARGS)
    fig.suptitle(r"Re-independence of $Sh_{\mathrm{NH_3}}$ at fixed $\kappa$",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_14_sh_graetz(resolution_name: str, case_table: pd.DataFrame,
                        families: Sequence[str] = ("G1", "G2")):
    """Manuscript Figure 14: Sh vs the Graetz coordinate z* at fixed kappa."""
    fig, axes = plt.subplots(1, len(families), figsize=(6.6 * len(families), 5.4))
    axes = np.atleast_1d(axes)
    zs = np.logspace(-4, 0.5, 100)
    for ax, family in zip(axes, families):
        _sh_family_panel(ax, fig, resolution_name, case_table, family, x_key="z_star")
        ax.plot(zs, 0.538 * zs**(-1 / 3), "k-", lw=1.8, zorder=2)
        ax.plot(zs, 1.077 * zs**(-1 / 3), "k--", lw=1.8, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 3)
        ax.set_xlabel(r"$z^{*} = D_m\,z\,/\,(u_z\,l_c^{2})$  [–]", fontsize=14)
        ax.text(2e-3, np.sqrt(SH_CLASSICAL_RANGE[0] * SH_CLASSICAL_RANGE[1]),
                "Classical annular Sh", ha="center", va="center",
                fontsize=12, color="#b45309")
        handles = [
            mlines.Line2D([], [], color="#888", lw=2,
                          label=r"$Sh_{wall}(z)$ from 2D (colour = GHSV)"),
            mlines.Line2D([], [], color="k", lw=1.8,
                          label=r"Lévêque: $0.538\,(z^{*})^{-1/3}$"),
            mlines.Line2D([], [], color="k", lw=1.8, ls="--",
                          label=r"Graetz: $1.077\,(z^{*})^{-1/3}$"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#f5a623", alpha=0.35,
                          label=f"Classical annular Sh "
                                f"({SH_CLASSICAL_RANGE[0]:.1f}–{SH_CLASSICAL_RANGE[1]:.1f})"),
        ]
        ax.legend(handles=handles, fontsize=10.5, loc="upper right", **LEG_KWARGS)
    fig.suptitle(r"Dependence of $Sh_{\mathrm{NH_3}}$ on Graetz coordinate"
                 r" $z^{*}$ at fixed $\kappa$", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ── Figure 15: corrected CP per kappa group ──────────────────────────
# Manuscript layout: one representative case per kappa (CP_avg closest to
# the group median), four series per panel — plain 1D (CP = 1), the 2D
# reference, and the TWO corrected-1D variants (Sh(kappa) power law and
# the mechanistic screened closure), each reconstructed from its own
# cached 1D solution, with per-series MAE against the 2D reference.

KAPPA_PANEL_COLORS = ("#2980b9", "#e74c3c", "#27ae60",
                      "#f39c12", "#8e44ad", "#16a085")


def _cp_from_1d_case(resolution_name: str, model: str, case_id: str):
    """CP_NH3(z) implied by a cached corrected-1D solution.

    CP = c_wall/c_bulk with c_wall = c_bulk - J/k_cp. Bulk states on both
    sides come from profiles_axial.csv (ideal gas); k_cp is recomputed
    exactly as the model computes it (its class method, on the cached
    config's pinned closure coefficients); and J is the model's own law,
    J = beta * Perm*Rg*(T_ret c_ret - T_perm c_perm) with
    beta = k_cp/(k_cp + Perm*Rg*T_ret) — fully self-consistent with the
    cached 1D solution, no flux profile needed (flows.npz stores totals).
    """
    import importlib

    from reactor import ReactorConfig
    from reactor.config import get_membrane_permeances

    case_dir = settings.case_cache_dir(resolution_name, model, case_id)
    prof = pd.read_csv(Path(case_dir) / "profiles_axial.csv")
    config = cache.read_json(Path(case_dir) / "config.json")
    cfg = ReactorConfig.from_dict(config)

    mod = importlib.import_module(f"reactor.{settings.ONE_D_CORRECTED_VARIANT}")
    reactor = mod.MembraneReactor1D(config=cfg)

    def conc(flow_cols: list[str], T_col: str, p_col: str):
        """Concentrations, temperature and pressure from flow-profile columns (ideal gas)."""
        F = prof[flow_cols].to_numpy()
        y = F / np.maximum(F.sum(axis=1, keepdims=True), 1e-30)
        T = prof[T_col].to_numpy()
        p = prof[p_col].to_numpy() * 1e5
        return y * (p / (cfg.Rg * T))[:, None], T, p

    c_ret, T_ret, p_ret = conc(
        ["F_H2_ret_mol_s", "F_N2_ret_mol_s", "F_NH3_ret_mol_s"],
        "T_ret_K", "p_ret_bar")
    c_perm, T_perm, _ = conc(
        ["F_H2_perm_mol_s", "F_N2_perm_mol_s", "F_NH3_perm_mol_s"],
        "T_perm_K", "p_perm_bar")

    # k_cp via the model's own routine (needs its cpT pressure row set).
    reactor.cpT[:, 1, :-2] = c_ret
    reactor.cpT[:, 1, -2] = p_ret
    reactor.cpT[:, 1, -1] = T_ret
    kcp = reactor.compute_kcp_sh(c_ret, T_ret)

    z = prof["z_m"].to_numpy()
    P0, EA = get_membrane_permeances(list(cfg.species), cfg, z, cfg.Lsealing)
    perm = P0 * np.exp(-EA / (cfg.Rg * T_ret[:, None]))
    beta = kcp / np.maximum(kcp + perm * cfg.Rg * T_ret[:, None], 1e-300)
    J = beta * perm * cfg.Rg * (T_ret[:, None] * c_ret - T_perm[:, None] * c_perm)

    # Translate the variant's own flux into a *wall* CP with the physical
    # film coefficient (the wall-based Sh(kappa) refit — the manuscript's
    # CP definition). The variant's k_cp is a flux closure; for the
    # flux-matched law it folds permeate-side and thermal effects in and
    # is NOT the wall film coefficient.
    from reactor.paper import closures

    wall = closures.load_sh_fit(resolution_name, "wall")
    kappa = float(cfg.r_min) / float(cfg.r_max)
    l_c = float(cfg.r_max) - float(cfg.r_min)
    nz, nc = c_ret.shape
    y_ret = c_ret / np.maximum(c_ret.sum(axis=1, keepdims=True), 1e-30)
    D = np.asarray(reactor.correlation.diffusion(
        y_ret.reshape(nz, 1, nc), T_ret.reshape(nz, 1), p_ret.reshape(nz, 1)))
    while D.ndim > 2:
        D = D.squeeze(axis=1)
    k_wall = wall["coeff"] * kappa ** wall["exp"] * np.maximum(D, 1e-30) / l_c

    cp = 1.0 - J[:, INH3_IDX] / np.maximum(k_wall[:, INH3_IDX], 1e-30) \
        / np.maximum(c_ret[:, INH3_IDX], 1e-30)
    return z, cp


INH3_IDX = 2


def figure_15_cp_kgroup(resolution_name: str, case_table: pd.DataFrame,
                        summary_2d: pd.DataFrame):
    """Figure 15 — corrected-1D CP profiles, one representative case per kappa group."""
    from reactor.paper import closures

    try:
        wall_fit = closures.load_sh_fit(resolution_name, "wall")
    except FileNotFoundError:
        wall_fit = closures.fit_sh_kappa(case_table, resolution_name, kind="wall")
    per = pd.DataFrame(wall_fit["per_case"])
    cp_mean = summary_2d.set_index("Case_ID")["CP_mean"]
    per["CP_mean"] = per["Case_ID"].map(cp_mean)

    reps: list[tuple[float, str]] = []
    for kappa, grp in per.groupby("kappa"):
        grp = grp.dropna(subset=["CP_mean"])
        if grp.empty:
            continue
        med = grp["CP_mean"].median()
        reps.append((float(kappa),
                     grp.iloc[(grp["CP_mean"] - med).abs().argsort().iloc[0]]["Case_ID"]))

    ncol, nrow = 3, int(np.ceil(len(reps) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.2 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for j, (ax, (kappa, case_id)) in enumerate(zip(axes, reps)):
        color = KAPPA_PANEL_COLORS[j % len(KAPPA_PANEL_COLORS)]
        case_dir = settings.case_cache_dir(resolution_name, settings.MODEL_2D, case_id)
        fields = cache.load_fields(case_dir)
        cfg2 = cache.read_json(Path(case_dir) / "config.json")
        z2 = fields["z_c"]
        cp2 = kpi_mod.cp_profile(fields["y_ret"], fields["r_f_ret"])
        m2 = z2 > Z_MIN_PLOT
        zL2 = z2 / z2.max()

        def mae(z, cp):
            """Mean absolute error of a CP profile against the 2D reference."""
            ref = np.interp(z, z2[m2], cp2[m2])
            ok = (z > Z_MIN_PLOT) & np.isfinite(cp)
            return float(np.mean(np.abs(cp[ok] - ref[ok])))

        mae_1d = float(np.mean(np.abs(1.0 - cp2[m2])))
        ax.axhline(1.0, color="#999", ls=":", lw=1.4,
                   label=f"1D uncorrected   MAE={mae_1d:.3f}")
        ax.plot(zL2[m2], cp2[m2], "-", lw=2.4, color=color, label="2D reference")

        for model, ls, tag in ((settings.MODEL_1D_CORRECTED, "--", r"$Sh(\kappa)$ fit"),
                               (settings.MODEL_1D_SCREENED, "-.", "mechanistic")):
            try:
                z1, cp1 = _cp_from_1d_case(resolution_name, model, case_id)
            except (FileNotFoundError, OSError):
                continue
            ok = z1 > Z_MIN_PLOT
            ax.plot(z1[ok] / z1.max(), cp1[ok], ls, lw=1.9, color=color,
                    label=f"{tag}   MAE={mae(z1, cp1):.3f}")

        row = case_table[case_table["Case_ID"].astype(str) == case_id].iloc[0]
        fam = case_id.split(" ")[0]
        ax.set_title(
            rf"$\kappa={kappa:.3f}$  ($r={float(cfg2['r_max']):.3f}$ m)  |  {fam}"
            f"\nGHSV={_ghsv_of(case_table, case_id):,.0f}"
            rf" h$^{{-1}}$  |  T={float(row['T_ret_K']):.0f} K"
            rf"  P={float(row['p_ret_bar']):.0f} bar",
            fontsize=11)
        active = cp2[m2]
        ax.text(0.97, 0.05,
                rf"CP$_{{avg}}$={active.mean():.3f}  CP$_{{min}}$={active.min():.3f}",
                transform=ax.transAxes, ha="right", fontsize=9.5,
                bbox=dict(facecolor="white", edgecolor="#bbb"))
        ax.set_ylim(0.2, 1.75)
        ax.tick_params(labelsize=11)
        ax.grid(True, ls=":", alpha=0.25)
        ax.legend(fontsize=9, loc="upper left", **LEG_KWARGS)
    for ax in axes[len(reps):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(reps) - ncol):len(reps)]:
        ax.set_xlabel(r"$z/L$  [–]", fontsize=13.5)
    for j in range(0, len(reps), ncol):
        axes[j].set_ylabel(r"$CP_{\mathrm{NH_3}}(z)$  [–]", fontsize=13.5)
    fig.suptitle("Sherwood-corrected CP: one representative case per "
                 r"$\kappa$ — plain 1D, 2D reference, and both corrected-1D"
                 " closures", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ── Figure 5: membrane permeance validation ──────────────────────────
# Symbols are the measured
# single-gas permeances (currently digitized from the manuscript figure —
# provenance in data/inputs/permeation_exp_digitized_fig5.csv); the solid
# line is the Arrhenius membrane sub-model the 2D reactor uses, fitted to
# those points; the crosses are the full module simulation read back out
# through the experimentalist's log-mean driving-force formula.

FIG5_COLORS = {"NH3": "#1f77b4", "N2": "#ff7f0e", "H2": "#2ca02c"}


def figure_5_permeance(exp: pd.DataFrame, fit: dict,
                       module: pd.DataFrame | None = None):
    """Figure 5 — single-gas permeance vs 1/T: experiment and Arrhenius fit.

    Pass ``module`` to overlay the permeation-module LMDF read-out as a
    diagnostic; the paper figure omits it (not described in the caption,
    and the end-point LMDF averaging reads ~10 % low for NH3/H2 by
    construction).
    """
    from reactor import ReactorConfig

    Rg = ReactorConfig.from_defaults().Rg
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
    T_line = np.linspace(410, 640, 200)
    for ax, sp, pretty in zip(axes, ("NH3", "N2", "H2"),
                              ("NH$_3$", "N$_2$", "H$_2$")):
        color = FIG5_COLORS[sp]
        sub = exp[exp["species"] == sp]
        ax.plot(1e3 / sub["T_K"], sub["permeance_mol_m-2_s-1_Pa-1"] * 1e7,
                "o", ms=7, color=color, label="Experiment")
        law = fit[sp]["P0"] * np.exp(-fit[sp]["EA"] / (Rg * T_line))
        ax.plot(1e3 / T_line, law * 1e7, "-", lw=1.8, color=color,
                label="2D Model (Arrhenius fit)")
        if module is not None:
            msub = module[module["Component"] == sp]
            ax.plot(1e3 / msub["T_K"], msub["permeance_apparent"] * 1e7,
                    "x", ms=7, mew=1.6, color="#555",
                    label="module simulation (LMDF read-out)")
        ax.set_title(pretty, fontsize=14)
        ax.set_xlabel(r"$1/T$  [K$^{-1}$] $\times 10^{-3}$", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_ylabel(r"Permeance  [mol m$^{-2}$ s$^{-1}$ Pa$^{-1}$]"
                      r" $\times 10^{-7}$", fontsize=14)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls=":", alpha=0.2)
        ax.legend(fontsize=11, loc="lower right", **LEG_KWARGS)
    fig.tight_layout()
    return fig


def figure_rossetti_relative_error(rossetti: pd.DataFrame):
    """Relative error (Exp/Model - 1) vs GHSV, all Rossetti tests.

    Model-bias diagnostic (not a manuscript figure), computed from the
    cached validation table.
    """
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    rel = (rossetti["NH3_exp_volpct"] / rossetti["NH3_model_volpct"] - 1.0) * 100.0
    data = rossetti.assign(rel_err_pct=rel)
    for test, sub in data.groupby("Test"):
        sub = sub.sort_values("GHSV_h")
        ax.plot(sub["GHSV_h"] / 1e3, sub["rel_err_pct"], "o-", ms=5, lw=1.4,
                alpha=0.85, color=_ROSSETTI_TEST_COLORS.get(int(test), "k"),
                label=_rossetti_title(sub.iloc[0]))
    x_lo = data["GHSV_h"].min() / 1e3
    x_hi = data["GHSV_h"].max() / 1e3
    ax.axhline(0, color="black", lw=1.6, zorder=10)
    ax.axhline(10, color="gray", lw=1.0, ls="--", alpha=0.6)
    ax.axhline(-10, color="gray", lw=1.0, ls="--", alpha=0.6)
    ax.fill_between([x_lo, x_hi], -10, 10, color="gray", alpha=0.08, zorder=1)
    within = float((rel.abs() <= 10).mean() * 100)
    ax.set_xlabel(r"GHSV  [$\times 10^{3}$ h$^{-1}$]", fontsize=11)
    ax.set_ylabel(r"Relative error  (Exp/Model $-$ 1) $\times$ 100  [%]",
                  fontsize=11)
    label = _rossetti_model_label(rossetti).replace("Model", "model")
    ax.set_title(f"Relative error vs. GHSV — all tests, {label}"
                 f"  ({within:.0f}% of points within ±10%)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              **LEG_KWARGS)
    fig.tight_layout()
    return fig


def figure_16_mechanistic(resolution_name: str, case_table: pd.DataFrame):
    """Figure 16 — mechanistic validation of the CP closure.

    (a) Exact flux-matched Sherwood numbers of the reported cases against
    kappa, the Sh_opt power law, and the screened closure evaluated per
    case with no geometry fit. (b) k_cp,NH3(z) for a representative case:
    exact, power law, mechanistic.
    """
    from reactor.paper import closures, mechanistic

    plt.rcParams["font.family"] = FONT_FAMILY
    sh_fit = closures.load_sh_fit(resolution_name)
    per = pd.DataFrame(sh_fit["per_case"])
    per = per[~per["Case_ID"].isin(settings.CASES_598K)].reset_index(drop=True)
    ghsv = case_table.set_index("Case_ID")["GHSV_h"]
    per["GHSV"] = per["Case_ID"].map(ghsv)

    mech = []
    for cid in per["Case_ID"]:
        prof = mechanistic.case_profile(resolution_name, cid)
        i = prof["i_nh3"]
        sh_prof = (prof["kcp_screen"][:, i]
                   * (prof["r_max"] - prof["r_mem"]) / prof["D"][:, i])
        wgt = np.abs(prof["J"][:, i])
        ok = wgt > 0
        mech.append(float(np.sum(sh_prof[ok] * wgt[ok]) / np.sum(wgt[ok])))
    per["Sh_mech"] = mech

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    pts = ax.scatter(per["kappa"], per["Sh"], c=np.log10(per["GHSV"]),
                     cmap="viridis", s=36, zorder=4)
    kk = np.linspace(per["kappa"].min() * 0.92, per["kappa"].max() * 1.08, 100)
    ax.plot(kk, sh_fit["coeff"] * kk ** sh_fit["exp"], "k-", lw=2,
            label=rf"$Sh_{{opt}} = {sh_fit['coeff']:.2f}\,\kappa^{{{sh_fit['exp']:.2f}}}$")
    ax.plot(per["kappa"] * 1.045, per["Sh_mech"], "^", ms=8, mfc="none", mew=1.7,
            color="#d62728", zorder=5, label="mechanistic closure (no geometry fit)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa = r_\mathrm{mem}/r_\mathrm{max}$  [–]", fontsize=15)
    ax.set_ylabel(r"$Sh = k_\mathrm{cp}\,d_h/D_{\mathrm{NH_3}}$  [–]", fontsize=15)
    ax.tick_params(labelsize=13)
    cb = fig.colorbar(pts, ax=ax, pad=0.02)
    cb.set_label(r"log$_{10}$ GHSV [h$^{-1}$]", fontsize=13)
    ax.legend(fontsize=11.5, loc="lower left", **LEG_KWARGS)
    ax.set_title("(a)", loc="left", fontsize=13)

    ax = axes[1]
    prof = mechanistic.case_profile(resolution_name, FIG16_PROFILE_CASE)
    i = prof["i_nh3"]
    z = prof["z_c"]
    k_shk = (sh_fit["coeff"] * (prof["r_mem"] / prof["r_max"]) ** sh_fit["exp"]
             * prof["D"][:, i] / (prof["r_max"] - prof["r_mem"]))
    ax.plot(z, prof["kcp_exact"][:, i] * 1e3, "o", ms=4, color="#555555",
            label="exact (flux-matched from 2D)")
    ax.plot(z, k_shk * 1e3, "-", lw=2, color="#1f77b4",
            label=r"$Sh_{opt}(\kappa)$ power law")
    ax.plot(z, prof["kcp_screen"][:, i] * 1e3, "-", lw=2, color="#d62728",
            label="mechanistic (local state)")
    ax.set_xlabel(r"$z$  [m]", fontsize=15)
    ax.set_ylabel(r"$k_\mathrm{cp,NH_3}$  [mm s$^{-1}$]", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=11.5, loc="upper right", **LEG_KWARGS)
    ax.set_title("(b)", loc="left", fontsize=13)
    fig.tight_layout()
    return fig


#: Representative case of Figure 16(b) — mid-GHSV, compact annulus.
FIG16_PROFILE_CASE = "G2 — GHSV sweep_3000"
