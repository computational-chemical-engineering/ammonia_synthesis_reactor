"""Run the paper sweep into the cache — the long-running half of the notebook.

Sweeps belong here, as a plain background script, not in a live notebook
cell: a publication-resolution 2D sweep is 1.5-2.5 h. The notebook then
calls ``reactor.paper.runner.run_sweep`` and finds everything cached.

Usage::

    python -m scripts.run_paper_sweep --resolution draft
    nohup python -u -m scripts.run_paper_sweep --resolution publication \
        > sweep_publication.log 2>&1 &

Both 598 K cases are always solved and cached regardless of deferred
the ``INCLUDE_598K`` switch — it controls figures and dataset, never compute.
Expect ~15-30 min each: they reach their steady state through the
temperature-continuation ladder.
"""
from __future__ import annotations

import argparse
import sys
import time

from reactor.paper import cases, runner, settings


def parse_args() -> argparse.Namespace:
    """Parse the sweep's command line: resolution, model, force, case filter, verbosity."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--resolution", default="draft", choices=sorted(settings.RESOLUTIONS))
    p.add_argument("--model", default=settings.MODEL_2D,
                   choices=[settings.MODEL_2D, settings.MODEL_1D,
                            settings.MODEL_1D_CORRECTED,
                            settings.MODEL_1D_SCREENED])
    p.add_argument("--force", action="store_true", help="ignore the cache")
    p.add_argument("--case-id", action="append", default=None,
                   help="run only this Case_ID (repeatable)")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main() -> int:
    """Run the sweep for the chosen model/resolution; exit 1 if any case failed."""
    args = parse_args()
    res = settings.resolution(args.resolution)
    table = cases.load_case_table()

    print(f"resolution {res.name}: {res.num_r}x{res.num_z}, "
          f"steady_state_atol={res.steady_state_atol:g}", flush=True)
    print(f"pressure row: {settings.PRESSURE_EQUATION} | cold starts | "
          f"{len(table)} cases (INCLUDE_598K={settings.INCLUDE_598K!r}; both 598 K cases always solved)",
          flush=True)

    start = time.perf_counter()
    summary = runner.run_sweep(
        table, res.name, model=args.model,
        force=args.force, verbose=args.verbose, only=args.case_id,
    )
    elapsed = time.perf_counter() - start

    failed = summary[summary["status"] == "failed"]["Case_ID"].tolist() if not summary.empty else []
    print(f"\nwall {elapsed / 60:.1f} min", flush=True)
    if failed:
        print(f"FAILED ({len(failed)}): " + "; ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
