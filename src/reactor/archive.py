"""Persistent identifiers of the archived code, dataset and paper.

One place to edit when an archive is deposited. The notebooks, the README
generator, the dataset manifest and anything else that needs a DOI read it
from here, so a deposit never has to be chased through the tree.

``None`` means "not deposited yet". Code that needs an identifier should
call the accessor (``dataset_zip_url()``, ``dataset_landing_page()``)
rather than the constant, so an unset DOI fails with an instruction
instead of a ``TypeError`` three frames later.

Filling these in, in deposit order:

1. Reserve the dataset DOI at 4TU.ResearchData (Figshare reserves a DOI
   before publication), set ``DATASET_DOI`` and ``DATASET_ZIP_URL``,
   commit, and re-run the dataset export so the manifest carries the
   identifier. Then upload that zip and publish (embargoed).
2. After the GitHub release is archived by Zenodo, set ``CODE_DOI`` (the
   concept DOI, which always resolves to the newest version) and
   ``CODE_VERSION_DOI`` (this release).
3. On acceptance, set ``PAPER_DOI``.

``CITATION.cff`` carries the same identifiers for citation managers;
``tests/test_archive.py`` pins the two together so they cannot drift.
"""
from __future__ import annotations

from typing import Any

#: Authors of the code and the dataset, in citation order. Mirrored in
#: ``CITATION.cff`` and ``.zenodo.json``; the dataset descriptor reads
#: them from here so a deposited archive names its authors.
AUTHORS = (
    {"name": "Gargiulo, Iolanda",
     "orcid": "0009-0003-2508-6373",
     "affiliation": "Eindhoven University of Technology"},
    {"name": "Peters, E. A. J. F.",
     "orcid": "0000-0001-6099-3583",
     "affiliation": "Eindhoven University of Technology"},
)

#: Public source repository.
CODE_REPOSITORY = "https://github.com/computational-chemical-engineering/ammonia_synthesis_reactor"

#: Zenodo *concept* DOI — resolves to the latest archived release.
CODE_DOI: str | None = None

#: Zenodo DOI of this specific release.
CODE_VERSION_DOI: str | None = None

#: 4TU.ResearchData DOI of the archived dataset.
DATASET_DOI: str | None = None

#: Direct download URL of the dataset archive (a zip holding one
#: top-level ``dataset/``). 4TU serves these from its file API; the
#: landing page derived from :data:`DATASET_DOI` always works as a
#: fallback for a human reader.
DATASET_ZIP_URL: str | None = None

#: Title under which the dataset is deposited.
DATASET_TITLE = ("Two-dimensional model of an ammonia synthesis packed bed "
                 "membrane reactor: simulation dataset")

#: Archive hosting the dataset.
DATASET_REPOSITORY = "4TU.ResearchData"

#: DOI of the accompanying paper, once published.
PAPER_DOI: str | None = None


def doi_url(doi: str) -> str:
    """Resolver URL for a bare DOI (idempotent if already a URL)."""
    if doi.startswith(("http://", "https://")):
        return doi
    return f"https://doi.org/{doi.removeprefix('doi:')}"


def dataset_landing_page() -> str:
    """Landing page of the archived dataset."""
    if DATASET_DOI is None:
        raise RuntimeError(
            "the dataset DOI is not set yet — deposit the dataset at "
            f"{DATASET_REPOSITORY} (or reserve its DOI) and set "
            "reactor.archive.DATASET_DOI")
    return doi_url(DATASET_DOI)


def dataset_zip_url() -> str:
    """Direct download URL of the dataset archive.

    Raises with the fix rather than returning ``None``: this is what the
    Colab bootstrap calls, where a silent ``None`` would surface as an
    unreadable ``wget`` error.
    """
    if DATASET_ZIP_URL is None:
        raise RuntimeError(
            "the dataset download URL is not set yet — after the "
            f"{DATASET_REPOSITORY} deposit, set reactor.archive."
            "DATASET_ZIP_URL to the direct link of the dataset zip "
            "(the landing page is reachable from DATASET_DOI)")
    return DATASET_ZIP_URL


def identifiers() -> dict[str, Any]:
    """The identifier block stamped into the dataset manifest."""
    return {
        "code_repository": CODE_REPOSITORY,
        "code_doi": CODE_DOI,
        "code_version_doi": CODE_VERSION_DOI,
        "dataset_doi": DATASET_DOI,
        "dataset_repository": DATASET_REPOSITORY,
        "paper_doi": PAPER_DOI,
    }


def citation() -> str:
    """Human-readable "how to cite this work" block.

    Names every identifier that exists and says plainly which ones do
    not yet, so a reader is never left guessing whether a missing DOI is
    an omission or a pending deposit.
    """
    lines = ["Ammonia synthesis membrane reactor — how to cite:", ""]
    lines.append(f"  code:    {CODE_REPOSITORY}")
    if CODE_DOI:
        lines.append(f"           {doi_url(CODE_DOI)}  (all versions)")
    if CODE_VERSION_DOI:
        lines.append(f"           {doi_url(CODE_VERSION_DOI)}  (this release)")
    if not (CODE_DOI or CODE_VERSION_DOI):
        lines.append("           archive DOI pending")
    lines.append(f"  dataset: {DATASET_TITLE} ({DATASET_REPOSITORY})")
    lines.append(f"           {doi_url(DATASET_DOI) if DATASET_DOI else 'DOI pending'}")
    lines.append(f"  paper:   {doi_url(PAPER_DOI) if PAPER_DOI else 'DOI pending (in review)'}")
    return "\n".join(lines)
