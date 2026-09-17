# Ammonia Synthesis Membrane Reactor

A two-dimensional, axisymmetric, non-isothermal model of a packed-bed
membrane reactor for ammonia synthesis

&nbsp;&nbsp;&nbsp;&nbsp; N₂ + 3 H₂ ⇌ 2 NH₃ &nbsp;&nbsp; (Ru/C catalyst, NH₃-selective membrane)

together with its 1D counterparts and the pipeline that generates **every
simulation figure and the published dataset of the accompanying paper**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/computational-chemical-engineering/ammonia_synthesis_reactor/blob/main/notebooks/paper_figures.ipynb)
paper figures &nbsp;·&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/computational-chemical-engineering/ammonia_synthesis_reactor/blob/main/notebooks/si_figures.ipynb)
SI figures

The finite-volume discretisation is built on
[pymrm](https://github.com/computational-chemical-engineering/pymrm), the
Python Package for Multiphase Reactor Modeling
([PyPI](https://pypi.org/project/pymrm/) ·
[documentation](https://computational-chemical-engineering.github.io/pymrm-book)). The transport model couples species, energy and momentum
(Ergun/Poiseuille) balances on retentate and permeate domains, Temkin-type ammonia synthesis kinetics, and Arrhenius membrane permeation, solved by a
fully implicit pseudo-transient Newton method with an error-weighted convergence norm and per-case convergence certificates.

## Installation

```bash
git clone https://github.com/computational-chemical-engineering/ammonia_synthesis_reactor
cd ammonia_synthesis_reactor
pip install -e ".[test,notebooks]"
```

Installing the package resolves every dependency from PyPI, `pymrm`
included (>= 2.3.1; it is a regular PyPI package — `pip install pymrm` —
so nothing has to be available locally beforehand). The extras add
`pytest` (`test`) and Jupyter (`notebooks`); a plain `pip install -e .`
installs just the library, which cannot run the test suite or the
notebooks.

The notebooks do need the full clone, not just the `.ipynb` files: they
read the case tables from `data/inputs/` and cache results under
`results/`. On Google Colab the badge link takes care of the clone and the
install automatically; the dataset download it also needs becomes
available when the dataset deposit lands (see below).

Python ≥ 3.11. Use `from reactor import ...` (never `from src.reactor ...`).

Quick smoke test:

```bash
python -c "from reactor import MembraneReactor; r = MembraneReactor(L=1.0, r_max=0.0165, p_ret_out=29.83e5, T_ret_in=643, F_ret_in=0.1); r.solve(num_timesteps=5, verbose=0); print('ok')"
python -m pytest tests/          # full test suite
```

## Reproducing the paper

The publication vehicle is **`notebooks/paper_figures.ipynb`** (main-text
Figures 2–16) and **`notebooks/si_figures.ipynb`** (Supplementary Figures
S.1–S.57). Both are thin drivers over `reactor.paper`, a cached pipeline:
every solved case is cached under
`results/paper/<resolution>/<model>/<case_id>/`,
figure cells read only from that cache, and a complete re-render takes
about two minutes.

### 1. Solve the cases (once, as background scripts)

```bash
python -m scripts.run_paper_sweep --resolution draft                       # 2D, 24x60 grid, ~20 min
python -m scripts.run_paper_sweep --resolution draft --model 1d            # plain 1D, minutes
python -m scripts.run_paper_sweep --resolution draft --model 1d_corrected
python -m scripts.run_paper_sweep --resolution draft --model 1d_screened

# publication tier (40x100): ~80 min for the 2D sweep on a desktop CPU
nohup python -u -m scripts.run_paper_sweep --resolution publication > sweep.log 2>&1 &
```

All 52 cases solve from cold starts. Convergence is judged by an
error-weighted RMS norm (`wrms <= 1`) combined with a KPI-stagnation gate;
outcomes are classified `converged` / `floored` / `oscillatory`, and every
cached case carries a machine-checkable convergence certificate (plus an
eigenvalue certificate where the temperature-continuation ladder was used).
The two 598 K cases are dynamically delicate and always solved through that
ladder (~15–30 min each); the `INCLUDE_598K` switch controls only whether
they are *reported*.

### 2. Run the notebooks

Run top-to-bottom. `paper_figures.ipynb` defaults to the draft tier and
`si_figures.ipynb` to publication; override either through the
`REACTOR_RESOLUTION` environment variable (`draft` | `publication`) set
before starting Jupyter, or by editing the `RESOLUTION` line in the
settings cell (the one after the Colab bootstrap). The chosen tier must
match the sweeps you ran in step 1. Every figure lands in
`results/paper/<resolution>/figures/` (SI figures in a `si/` subfolder) as
PNG and PDF, sized to drop into the manuscript.

### 3. Dataset export

The final notebook section assembles the archived-dataset payload
(`dataset/`, never committed) and writes `manifest.json` with per-file
SHA-256 checksums, package versions, the git commit and wall times;
`reactor.paper.dataset.verify` re-hashes a tree against its manifest so a
re-run can be compared with the published dataset file by file.
`reactor.paper.dataset.restore_cache` is the inverse: it populates the
local cache from a downloaded dataset tree (never overwriting existing
cache files), so the notebooks re-render every figure without solving
anything.

### Running on Google Colab

Both notebooks carry an "Open in Colab" badge and a bootstrap cell that is
a no-op on a local checkout. On Colab it installs this package (pulling
`pymrm` from PyPI), downloads the archived dataset, verifies it against its
SHA-256 manifest and restores the cache — no sweeps needed. The dataset
DOI and its download URL live in one place, `src/reactor/archive.py`;
until the deposit lands the bootstrap cell stops with a clear message
naming exactly what to set.

## Repository layout

| path | contents |
|---|---|
| `src/reactor/` | the models: `membrane_reactor.py` (2D), `membrane_reactor_1d.py` (plain 1D, CP = 1), `membrane_reactor_1d_corrected.py` (1D with a concentration-polarization closure — fitted Sherwood law or mechanistic reaction-screening via `cp_closure.py`), plus kinetics, mixture properties, convergence and stability machinery |
| `src/reactor/paper/` | the paper pipeline: case tables, cached runner, KPI definitions, closure fits, dimensionless analysis, validation runs, all figure functions, dataset export |
| `scripts/run_paper_sweep.py` | background sweep driver (also installed as `reactor-paper-sweep`) |
| `notebooks/` | `paper_figures.ipynb`, `si_figures.ipynb` |
| `data/inputs/` | everything needed to reproduce the study (see below) |
| `tests/` | pytest suite covering the case tables, cache discipline, convergence norm and classifier, 1D conservation identities, validation runs, archive identifiers and dataset manifest |

## Data

All inputs are versioned in `data/inputs/`:

- `cases_to_run.xlsx` — the G1–G8 case table (52 operating points: GHSV,
  pressure, temperature and radius sweeps).
- `ammonia_synthesis_data_rossetti_et_al.csv` — the experimental kinetics
  dataset of Rossetti et al. used for the Figure 3/4/S.6 validation
  (19 test conditions, 125 points).
- `permeation_cases_1d.csv` — the single-gas membrane permeation test
  matrix behind Figure 5.
- `permeation_exp_measured_fig5.csv` — the measured single-gas
  CMS-membrane permeances behind Figure 5 and the Arrhenius fit (this is
  the file the pipeline reads).
- `permeation_exp_digitized_fig5.csv` — the earlier placeholder, read off
  the manuscript figure; kept only as a cross-check (the two agree to
  ≤ 0.5 % on every point).
- `s1_diffusivity_chapman.csv` — digitized literature NH₃–H₂ diffusivities
  behind SI Figure S.1.

Two of these files hold **third-party experimental data** that we
transcribed or digitized so the validation is reproducible
(`ammonia_synthesis_data_rossetti_et_al.csv`,
`s1_diffusivity_chapman.csv`). Each carries its source in its header;
cite the original publications for those measurements. The MIT licence
covers our code and our own outputs, not the underlying third-party
measurements.

Generated results (`results/`) and the dataset payload (`dataset/`) are not
in git; the archived dataset is published separately with a DOI and can be
regenerated from this repository alone (case tables in, manifest-verified
tree out).

Case identifiers contain em-dashes (e.g. `G1 — GHSV sweep_50`): quote them
in shells and prefer `pathlib` when scripting.

## Citing

See `CITATION.cff`, or run `python -c "from reactor import archive;
print(archive.citation())"`. Please cite the accompanying paper for the
science and this repository (or the archived dataset DOI) for the
implementation and data.

- **Code archive** (Zenodo): <https://doi.org/10.5281/zenodo.22811033> —
  the concept DOI, which always resolves to the latest release. Release
  v1.1.0 is <https://doi.org/10.5281/zenodo.22811034>.
- **Dataset** (4TU.ResearchData):
  <https://doi.org/10.4121/e03a6e99-6ddc-4c10-8d92-fb36335cdb43>, CC BY 4.0.
- **Paper** — DOI pending; it will be added here on publication.

## License

MIT — see `LICENSE`.
