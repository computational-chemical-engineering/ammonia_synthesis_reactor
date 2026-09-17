"""Dataset export and SHA-256 manifest — the archived-dataset payload.

``export()`` copies everything a reader needs to audit or re-plot the paper
out of the cache into ``dataset/`` (gitignored) and writes ``manifest.json``
with a SHA-256 checksum per file, package versions, the git commit, wall
times, and an explicit list of anything missing. ``verify()`` re-hashes a
dataset tree against its manifest — the acceptance check after a fresh
publication run.

The final deposit run is a deliberate, user-triggered act: it must happen
only after decisions D1/D2/D3 and the WHSV convention are settled and on a
clean commit. Until then use ``dry_run=True`` (report only) or draft-tier
exports to exercise the machinery.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

import reactor
from reactor import archive
from reactor.paper import cache, cases, provenance, settings

#: Files shipped per cached case, in cache order. fields.npz exists for the
#: 2D model only.
_CASE_FILES = ("config.json", "kpis.json", "solve_status.json", "meta.json",
               "profiles_axial.csv", "fields.npz", "flows.npz")

#: Closure fits and mechanistic-analysis artifacts shipped next to the KPI
#: summaries (the last two back Section 3.3.3's exact-tracing and
#: field-measurement numbers; reactor.paper.mechanistic regenerates them).
_CLOSURE_FILES = ("sh_cp_fit.json", "sh_wall_fit.json", "screened_cp_fit.json",
                  "exact_trace_kpis.csv", "mech_field_scan.json")

MANIFEST_NAME = "manifest.json"

#: Data descriptor written beside the manifest — generated from it, so
#: the counts and identifiers it quotes cannot go stale.
README_NAME = "README.md"


def _models(resolution_name: str) -> tuple[str, ...]:
    """Model subdirectories shipped in the dataset for this resolution."""
    return (settings.MODEL_2D, settings.MODEL_1D,
            settings.MODEL_1D_CORRECTED, settings.MODEL_1D_SCREENED)


def _sha256(path: Path) -> str:
    """SHA-256 hex digest of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _selected_case_ids(include_598k: str) -> list[str]:
    """Case IDs to ship, after applying the 598 K reporting switch."""
    table = cases.load_case_table()
    selected = cases.select_cases(table, include_598k=include_598k)
    return [str(c) for c in selected["Case_ID"]]


def _collect(resolution_name: str, include_598k: str) -> tuple[list[tuple[Path, Path]], list[str]]:
    """(source, dataset-relative destination) pairs, plus what is missing."""
    root = settings.cache_root(resolution_name)
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []

    def want(src: Path, rel: Path, *, optional: bool = False) -> None:
        """Queue ``src`` for export at ``rel``, or record it as missing (unless optional)."""
        if src.exists():
            pairs.append((src, rel))
        elif not optional:
            missing.append(str(rel))

    # The published dataset is always complete: every solved case ships,
    # 598 K pair included, regardless of the figure-reporting switch (which
    # the manifest still records).
    case_ids = _selected_case_ids("include")
    for model in _models(resolution_name):
        want(settings.summary_csv(resolution_name, model),
             Path("kpis") / settings.summary_csv(resolution_name, model).name)
        model_missing = 0
        for case_id in case_ids:
            case_dir = settings.case_cache_dir(resolution_name, model, case_id)
            if not case_dir.exists():
                model_missing += 1
                continue
            for name in _CASE_FILES:
                src = case_dir / name
                if src.exists():
                    pairs.append((src, Path("fields") / model / case_id / name))
        if model_missing:
            missing.append(f"fields/{model}: {model_missing}/{len(case_ids)} case dirs")

    for name in _CLOSURE_FILES:
        want(root / name, Path("closures") / name)

    # The published Rossetti table is the 2D one (manuscript caption); the
    # 1D table ships alongside as the documented <=0.3% cross-check.
    want(settings.validation_dir() / "rossetti_2d.csv",
         Path("validation") / "rossetti_2d.csv")
    want(settings.validation_dir() / "rossetti_1d.csv",
         Path("validation") / "rossetti_1d.csv", optional=True)
    want(settings.validation_dir() / "permeation_1d.csv",
         Path("validation") / "permeation_1d.csv")
    want(settings.validation_dir() / "permeation_arrhenius_fit.json",
         Path("validation") / "permeation_arrhenius_fit.json")
    want(root / "weisz_prater_scan.json",
         Path("validation") / "weisz_prater_scan.json")

    fig_dir = settings.figures_dir(resolution_name)
    if fig_dir.exists():
        for src in sorted(fig_dir.rglob("*")):
            if src.suffix in (".png", ".pdf") or src.name == "si_field_map.csv":
                pairs.append((src, Path("figures") / src.relative_to(fig_dir)))
    else:
        missing.append("figures/")

    # Derived dimensionless tables (SI Figures S.2-S.5, Figure 12 points).
    dimless = root / "dimless"
    for name in ("dimless_scalars.csv", "sh_local_points.csv"):
        want(dimless / name, Path("dimensionless") / name, optional=True)

    # Inputs, so the archive stands on its own: every case's config.json
    # names the property database, and the validation figures need the
    # experimental tables. Without these the dataset documents results
    # that cannot be recomputed from it.
    # config.json names this as "data/properties_database.json"; the
    # package ships it beside the code and resolves it there.
    want(Path(reactor.__file__).parent / "data" / "properties_database.json",
         Path("inputs") / "properties_database.json")
    inputs = settings.project_root() / "data" / "inputs"
    if inputs.exists():
        for src in sorted(inputs.iterdir()):
            if src.is_file():
                pairs.append((src, Path("inputs") / src.name))
    else:
        missing.append("inputs/")

    want(settings.project_root() / "LICENSE", Path("LICENSE"))

    return pairs, missing


def _compute_provenance(out_root: Path) -> dict[str, Any]:
    """Git state of the runs that produced the fields, from the case meta.

    Distinct from the manifest's own ``provenance``, which records the
    *export*. The two differ whenever the cache predates the current
    commit — normal here, because the certified caches are never re-run
    for cosmetic changes. Recording both keeps that visible instead of
    implying one clean commit reproduces every field.
    """
    commits: dict[str, dict[str, Any]] = {}
    for meta_path in sorted((out_root / "fields").rglob("meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        prov = meta.get("provenance") or meta
        commit = prov.get("git_commit")
        if not commit:
            continue
        entry = commits.setdefault(commit, {
            "git_branch": prov.get("git_branch"),
            "git_dirty": prov.get("git_dirty"),
            "n_cases": 0,
        })
        entry["n_cases"] += 1
    return {
        "note": ("Git state of the solver runs that produced fields/, which "
                 "predates the export commit: the certified caches are never "
                 "re-run for changes that do not alter the physics."),
        "commits": dict(sorted(commits.items(),
                               key=lambda kv: -kv[1]["n_cases"])),
    }


def _readme_text(manifest: dict[str, Any]) -> str:
    """The data descriptor written to ``dataset/README.md``.

    Generated rather than hand-kept so the counts and identifiers in it
    cannot drift from the manifest they describe.
    """
    ids = manifest["identifiers"]
    n_cases = manifest.get("n_cases_per_model", "?")
    dataset_doi = ids["dataset_doi"] or "(pending — assigned on deposit)"
    code_doi = ids["code_doi"] or "(pending)"
    commits = manifest.get("compute_provenance", {}).get("commits", {})
    commit_lines = "\n".join(
        f"| `{c[:8]}` | {v.get('git_branch')} | {v.get('n_cases')} | "
        f"{'yes' if v.get('git_dirty') else 'no'} |"
        for c, v in commits.items()) or "| — | — | — | — |"

    return f"""\
# Ammonia synthesis packed bed membrane reactor — simulation dataset

Two-dimensional axisymmetric, non-isothermal simulations of a packed bed
membrane reactor for ammonia synthesis over a Ru/C catalyst, together
with the one-dimensional models compared against them and every figure
of the accompanying paper.

- Dataset DOI: {dataset_doi}
- Source code: {ids['code_repository']}
- Code archive DOI: {code_doi}
- Paper DOI: {ids['paper_doi'] or '(pending — in review)'}
- Exported: {manifest['created_utc']} from commit `{manifest['provenance'].get('git_commit', '?')[:8]}`
- Contents: {manifest['n_files']} files, resolution tier `{manifest['resolution']}`
- Licence: MIT (see `LICENSE`)

## What is here

```
kpis/            one row per case per model — start here
fields/          per-case solver output, one directory per case
  2d/            the 2D axisymmetric model (the reference)
  1d/            plain 1D plug-flow model
  1d_corrected/  1D with the fitted Sherwood correction Sh = a·kappa^b
  1d_screened/   1D with the mechanistic reaction-screened closure
closures/        fitted closures + the mechanistic analysis artifacts
validation/      kinetics and permeation validation against experiment
dimensionless/   derived dimensionless tables (Peclet, Damkohler, Sherwood)
figures/         every figure of the paper (PNG at 200 dpi, PDF vector)
inputs/          property database and experimental input tables
manifest.json    SHA-256 of every file, plus the provenance below
```

Each of the four model directories holds **{n_cases} cases**: the 50
operating conditions reported in the paper plus the two 598 K cases,
which are shipped in full but excluded from the paper's figures (they sit
in a solution-multiplicity window and are discussed separately). The
manifest field `figures_include_598k` records that figure convention —
it does **not** mean data is missing.

## Case directories

`fields/<model>/<Case_ID>/` — the `Case_ID` is `<family> — <sweep>_<value>`,
e.g. `G1 — GHSV sweep_50`. Families G1–G8 vary one design parameter at a
time; the value suffix is that parameter's setting (GHSV in h⁻¹,
pressure in bar, temperature in K, radius in units given by the sweep).
Each directory contains:

| file | contents |
|---|---|
| `config.json` | every reactor/solver parameter of the run |
| `kpis.json` | the key performance indicators and solver certificates |
| `profiles_axial.csv` | axial profiles, units in the column names |
| `fields.npz` | the 2D field arrays (below) |
| `flows.npz` | molar flows, mol s⁻¹ |
| `solve_status.json` | convergence outcome and residual norms |
| `meta.json` | runtime and the git state of *that solver run* |

## Units and array layout (`fields.npz`)

Grids are cell-centred (`_c`, length n) and face-centred (`_f`, n+1);
`ret` is the retentate (reaction) side, `perm` the permeate side. The
species axis has length 3 and is always ordered **`["H2", "N2", "NH3"]`**
(also recorded in each `config.json`).

| array | shape | units |
|---|---|---|
| `z_c`, `z_f` | (nz), (nz+1) | m — axial coordinate |
| `r_c_ret`, `r_f_ret`, `r_c_perm`, `r_f_perm` | (nr), (nr+1) | m — radial coordinate |
| `T_ret`, `T_perm` | (nz, nr) | K |
| `p_ret_bar`, `p_perm_bar` | (nz, nr) | bar |
| `c_ret`, `c_perm` | (nz, nr, 3) | mol m⁻³ |
| `y_ret`, `y_perm` | (nz, nr, 3) | mole fraction, dimensionless |
| `u_*_ax`, `u_*_rad` | (nz+1, nr), (nz, nr+1) | m s⁻¹ — superficial velocity |
| `flux_*_ax`, `flux_*_rad` | (nz+1, nr, 3), (nz, nr+1, 3) | mol m⁻² s⁻¹ — total (diffusive + convective) |
| `reaction_source_ret` | (nz, nr, 3) | mol m⁻³ s⁻¹ |

Radial fluxes are signed along +r, so the ammonia flux *into* the
membrane at the inner wall is `-flux_ret_rad[:, 0, 2]`.

## Provenance

`manifest.json` carries two distinct provenance records, because they
genuinely differ:

- `provenance` — the commit that produced this **export** (clean tree).
- `compute_provenance` — the commits that produced the **fields**. The
  solver caches are certified and deliberately never re-run for changes
  that do not alter the physics, so they predate the export commit:

| commit | branch | cases | dirty tree |
|---|---|---|---|
{commit_lines}

"dirty tree" means uncommitted edits were present during that run; the
edits concerned analysis and plotting code, not the solver. The
authoritative check on the numbers is not the commit but the per-case
certificates in `kpis.json` (`convergence_certificate`,
`eigenvalue_certificate`, `element_balance_ok`) and the SHA-256 manifest.

## Reproducing

Everything here regenerates from the source repository:

```bash
git clone {ids['code_repository']}
cd ammonia_synthesis_reactor
pip install -e .                      # pulls pymrm from PyPI
python -c "from reactor.paper import dataset; print(dataset.verify())"
```

`dataset.verify()` re-hashes this tree against `manifest.json`;
`dataset.restore_cache()` loads it into the solver cache so the figure
notebooks (`notebooks/paper_figures.ipynb`, `notebooks/si_figures.ipynb`)
re-render every figure without re-solving. Re-running the sweeps from
scratch instead costs about {(manifest.get('sweep_runtime_2d_s') or 0) / 3600:.0f} CPU-hours for the 2D model.

## Third-party inputs

`inputs/ammonia_synthesis_data_rossetti_et_al.csv` holds experimental
values we transcribed from

> I. Rossetti, N. Pernicone, F. Ferrero, L. Forni, "Kinetic study of
> ammonia synthesis on a promoted Ru/C catalyst", Ind. Eng. Chem. Res.
> 45 (2006) 4150-4155. https://doi.org/10.1021/ie051398g

and `inputs/s1_diffusivity_chapman.csv` holds literature diffusivities we
digitized from the source cited in the paper's supplementary material.
Both are included solely so the validation figures are reproducible:
please cite those original publications for the measurements, not this
dataset. Everything else here — all simulation output, the property
database and the permeation measurements — is our own, under the MIT
licence in `LICENSE`.
"""


def export(
    resolution_name: str,
    *,
    include_598k: str | None = None,
    dry_run: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Assemble the dataset tree and manifest; report what is missing.

    With ``dry_run=True`` nothing is written — the returned report lists
    the files that would ship and the gaps. The dataset is always
    complete: the full KPI summaries and all case directories ship, the
    598 K pair included. ``settings.INCLUDE_598K`` controls only how the
    *figures* report those cases; the manifest records its value.
    """
    include_598k = settings.INCLUDE_598K if include_598k is None else include_598k
    if include_598k not in settings.INCLUDE_598K_CHOICES:
        raise ValueError(f"include_598k must be one of {settings.INCLUDE_598K_CHOICES}")
    out_root = settings.dataset_dir() if out_dir is None else Path(out_dir)

    pairs, missing = _collect(resolution_name, include_598k)
    report: dict[str, Any] = {
        "resolution": resolution_name,
        "include_598k": include_598k,
        "n_files": len(pairs),
        "total_bytes": int(sum(src.stat().st_size for src, _ in pairs)),
        "missing": missing,
        "dry_run": dry_run,
        "out_dir": str(out_root),
    }
    if dry_run:
        return report

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    files: dict[str, dict[str, Any]] = {}
    for src, rel in pairs:
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files[str(rel)] = {"sha256": _sha256(dst), "bytes": dst.stat().st_size}

    # Wall-time accounting from the 2D summary (the expensive half).
    runtime_s = None
    summary_path = settings.summary_csv(resolution_name, settings.MODEL_2D)
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if "runtime_s" in df.columns:
            runtime_s = float(df["runtime_s"].sum())

    manifest = {
        "dataset": "ammonia_synthesis_membrane_reactor",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolution": resolution_name,
        # Named for what it actually controls: every solved case ships,
        # this only records how the *figures* report the 598 K pair.
        # (The old key name read as "598 K excluded from the data".)
        "figures_include_598k": include_598k,
        "include_598k": include_598k,
        "n_cases_per_model": len(_selected_case_ids("include")),
        "models": list(_models(resolution_name)),
        "species_order": ["H2", "N2", "NH3"],
        "cache_schema_version": settings.CACHE_SCHEMA_VERSION,
        "whsv_convention": settings.WHSV_CONVENTION,
        "one_d_variant": settings.ONE_D_VARIANT,
        "one_d_corrected_variant": settings.ONE_D_CORRECTED_VARIANT,
        "sweep_runtime_2d_s": runtime_s,
        "identifiers": archive.identifiers(),
        "provenance": provenance.stamp(),
        "missing": missing,
        "n_files": len(files),
        "files": files,
    }
    manifest["compute_provenance"] = _compute_provenance(out_root)
    (out_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    # The descriptor a stranger reads first; not in the manifest, because
    # it is written from it.
    (out_root / README_NAME).write_text(_readme_text(manifest))
    report["manifest"] = str(out_root / MANIFEST_NAME)
    report["readme"] = str(out_root / README_NAME)
    return report


def restore_cache(dataset_root: Path | None = None) -> dict[str, Any]:
    """Populate the local cache from an exported dataset tree (export's inverse).

    Meant for a fresh environment (e.g. Google Colab) where the archived
    dataset was downloaded instead of solved: every dataset file is copied
    back to the cache location ``export()`` took it from, after which the
    notebooks re-render every figure from cache exactly as on a machine
    that ran the sweeps. Figures are not restored — the notebooks
    regenerate them. Existing cache files are NEVER overwritten (a local
    certified cache always wins); they are counted as skipped.
    """
    root = settings.dataset_dir() if dataset_root is None else Path(dataset_root)
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    resolution = manifest["resolution"]
    cache_root_dir = settings.cache_root(resolution)

    restored: list[str] = []
    skipped = 0
    for rel in manifest["files"]:
        parts = Path(rel).parts
        if parts[0] == "figures":
            continue
        if parts[0] in ("kpis", "closures"):
            dst = cache_root_dir / Path(*parts[1:])
        elif parts[0] == "fields":
            dst = cache_root_dir / Path(*parts[1:])
        elif rel == "validation/weisz_prater_scan.json":
            dst = cache_root_dir / "weisz_prater_scan.json"
        elif parts[0] == "validation":
            dst = settings.validation_dir() / Path(*parts[1:])
        elif parts[0] == "dimensionless":
            dst = cache_root_dir / "dimless" / Path(*parts[1:])
        else:
            continue
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dst)
        restored.append(rel)

    return {
        "resolution": resolution,
        "n_restored": len(restored),
        "n_skipped_existing": skipped,
        "cache_root": str(cache_root_dir),
    }


def verify(dataset_root: Path | None = None) -> dict[str, Any]:
    """Re-hash a dataset tree against its manifest (sha256sum -c equivalent)."""
    root = settings.dataset_dir() if dataset_root is None else Path(dataset_root)
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    bad: list[str] = []
    absent: list[str] = []
    for rel, entry in manifest["files"].items():
        path = root / rel
        if not path.exists():
            absent.append(rel)
        elif _sha256(path) != entry["sha256"]:
            bad.append(rel)
    extra = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME
        and str(p.relative_to(root)) not in manifest["files"]
    ]
    return {
        "ok": not bad and not absent,
        "n_checked": len(manifest["files"]),
        "mismatched": bad,
        "absent": absent,
        "untracked": extra,
    }
