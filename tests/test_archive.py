"""Tests for reactor.archive — the DOIs of the deposited artifacts.

The point of these is drift: the identifiers live in three places a
reader may look (the module, CITATION.cff, the dataset manifest), and a
deposit that updates only one of them is worse than no DOI at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reactor import archive

CITATION = Path(__file__).resolve().parents[1] / "CITATION.cff"


def _citation() -> dict:
    return yaml.safe_load(CITATION.read_text())


def test_doi_url_accepts_bare_and_prefixed_dois():
    assert archive.doi_url("10.4121/abc") == "https://doi.org/10.4121/abc"
    assert archive.doi_url("doi:10.4121/abc") == "https://doi.org/10.4121/abc"
    # already a URL: unchanged, so round-tripping a stored link is safe
    url = "https://doi.org/10.4121/abc"
    assert archive.doi_url(url) == url


@pytest.mark.parametrize("accessor, attr", [
    (archive.dataset_zip_url, "DATASET_ZIP_URL"),
    (archive.dataset_landing_page, "DATASET_DOI"),
])
def test_unset_identifier_raises_with_instructions(accessor, attr, monkeypatch):
    monkeypatch.setattr(archive, attr, None)
    with pytest.raises(RuntimeError, match="not set yet"):
        accessor()


def test_set_identifier_is_returned(monkeypatch):
    monkeypatch.setattr(archive, "DATASET_DOI", "10.4121/xyz")
    monkeypatch.setattr(archive, "DATASET_ZIP_URL", "https://data.4tu.nl/file.zip")
    assert archive.dataset_landing_page() == "https://doi.org/10.4121/xyz"
    assert archive.dataset_zip_url() == "https://data.4tu.nl/file.zip"


def test_identifiers_block_has_every_key_the_manifest_promises():
    keys = set(archive.identifiers())
    assert keys == {"code_repository", "code_doi", "code_version_doi",
                    "dataset_doi", "dataset_repository", "paper_doi"}


def test_citation_block_names_pending_archives():
    text = archive.citation()
    assert archive.CODE_REPOSITORY in text
    for doi, label in ((archive.DATASET_DOI, "dataset"), (archive.PAPER_DOI, "paper")):
        if doi is None:
            assert "pending" in text
        else:
            assert archive.doi_url(doi) in text


def test_every_author_has_an_orcid():
    for author in archive.AUTHORS:
        assert author["orcid"], author["name"]


def test_authors_and_orcids_match_citation_cff_and_zenodo():
    """One author list, three files that must agree on it."""
    cff_authors = _citation()["authors"]
    zenodo = json.loads((CITATION.parent / ".zenodo.json").read_text())

    assert len(cff_authors) == len(archive.AUTHORS) == len(zenodo["creators"])
    for author, cff, zen in zip(archive.AUTHORS, cff_authors, zenodo["creators"]):
        family, given = author["name"].split(", ")
        assert cff["family-names"] == family
        assert cff["given-names"] == given
        assert zen["name"] == author["name"]
        assert cff["orcid"].endswith(author["orcid"])
        assert zen["orcid"] == author["orcid"]


def test_citation_cff_agrees_with_archive_module():
    """CITATION.cff must carry the same identifiers, once they exist."""
    cff = _citation()
    assert cff["repository-code"].rstrip("/") == archive.CODE_REPOSITORY.rstrip("/")

    dois = {i.get("value") for i in cff.get("identifiers", [])
            if i.get("type") == "doi"}
    if archive.CODE_DOI is not None:
        assert archive.CODE_DOI in dois or cff.get("doi") == archive.CODE_DOI
    if archive.DATASET_DOI is not None:
        related = {r.get("value") for r in cff.get("references", [])
                   if isinstance(r, dict)}
        assert archive.DATASET_DOI in dois | related
