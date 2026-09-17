"""Build the dataset archive to upload to 4TU.ResearchData.

Exports the publication-tier dataset, verifies it against its own
manifest, and writes a zip whose single top-level directory is
``dataset/`` — the layout the Colab bootstrap unpacks and the
convention archives expect.

    python -m scripts.make_deposit                  # build from HEAD
    python -m scripts.make_deposit --allow-dirty    # rehearsal only

The deposit is a deliberate act, so this refuses to run on a dirty tree
unless told otherwise: the manifest stamps the commit, and a stamp that
does not identify the code is worse than none. It also reports whether
the dataset DOI is set (see ``reactor.archive``) — deposit reads best
when the DOI is reserved first, so the archive can name itself.
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

from reactor import archive
from reactor.paper import dataset, provenance, settings


def build_zip(dataset_dir: Path, out_path: Path) -> tuple[int, int]:
    """Zip ``dataset_dir`` under a single top-level ``dataset/`` prefix."""
    n_files = 0
    total = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src in sorted(dataset_dir.rglob("*")):
            if not src.is_file():
                continue
            zf.write(src, Path("dataset") / src.relative_to(dataset_dir))
            n_files += 1
            total += src.stat().st_size
    return n_files, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="publication")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where to write the zip (default: project root)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="rehearsal on a dirty tree; never for the real deposit")
    args = parser.parse_args(argv)

    stamp = provenance.stamp()
    commit = (stamp.get("git_commit") or "unknown")[:7]
    if stamp.get("git_dirty") and not args.allow_dirty:
        print("refusing to build: the working tree is dirty, so the manifest "
              "commit would not identify this code.\n"
              "Commit first, or pass --allow-dirty for a rehearsal.")
        return 1

    if archive.DATASET_DOI is None:
        print("note: reactor.archive.DATASET_DOI is not set. Reserving the DOI "
              "at 4TU first and setting it here lets the archive name itself; "
              "otherwise the descriptor will say 'pending'.\n")

    report = dataset.export(args.resolution, dry_run=False)
    if report["missing"]:
        print("refusing to build: the export is incomplete:")
        for item in report["missing"]:
            print("  -", item)
        return 1

    check = dataset.verify()
    if not check["ok"]:
        print("refusing to build: the exported tree does not match its manifest:")
        print(" ", check)
        return 1

    out_dir = args.out_dir or settings.project_root()
    date = time.strftime("%Y-%m-%d", time.gmtime())
    out_path = out_dir / f"dataset_4tu_{date}_{commit}.zip"
    n_files, total = build_zip(Path(report["out_dir"]), out_path)

    print(f"exported  {report['n_files']} files, verified {check['n_checked']}")
    print(f"wrote     {out_path.name}")
    print(f"          {n_files} files, {total / 1e6:.1f} MB uncompressed, "
          f"{out_path.stat().st_size / 1e6:.1f} MB compressed")
    print(f"commit    {commit} (clean tree)" if not stamp.get("git_dirty")
          else f"commit    {commit} (DIRTY — rehearsal only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
