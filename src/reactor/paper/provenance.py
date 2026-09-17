"""Version, commit and environment stamping for the paper pipeline."""
from __future__ import annotations

import platform
import subprocess
import sys
from functools import lru_cache
from typing import Any

from reactor.paths import project_root

_PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "pymrm", "h5py", "openpyxl")


def _git(*args: str) -> str | None:
    """Run a git command in the project root; return stripped stdout or None on failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root()), *args],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def package_versions() -> dict[str, str]:
    """Installed versions of the packages relevant to the pipeline."""
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


@lru_cache(maxsize=1)
def stamp() -> dict[str, Any]:
    """Provenance record embedded in every cached case and the manifest."""
    dirty = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def describe() -> str:
    """One-line human-readable stamp for the notebook's setup cell."""
    s = stamp()
    commit = (s["git_commit"] or "unknown")[:10]
    dirty = " (dirty)" if s["git_dirty"] else ""
    pkgs = ", ".join(f"{k} {v}" for k, v in sorted(s["packages"].items()))
    return f"commit {commit}{dirty} on {s['git_branch']} | python {s['python']} | {pkgs}"
