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


def test_code_and_dataset_licences_are_distinct_and_consistent():
    """The code is MIT, the archived data is CC BY — keep them straight.

    The repository LICENSE and .zenodo.json must state the *code* licence;
    the dataset's own LICENSE (generated at export) states the data one.
    Conflating them is the failure mode this guards.
    """
    from reactor.paper import dataset

    assert archive.CODE_LICENSE == "MIT"
    assert archive.DATASET_LICENSE.startswith("CC BY")
    # 4TU's table: 1 = CC BY 4.0, 2 = CC0, 3 = MIT. Off-by-one here would
    # deposit under the wrong terms silently.
    assert archive.DATASET_LICENSE_4TU_ID == 1

    assert json.loads((CITATION.parent / ".zenodo.json").read_text())["license"] \
        == archive.CODE_LICENSE
    assert _citation()["license"] == archive.CODE_LICENSE
    assert (CITATION.parent / "LICENSE").read_text().lstrip().startswith("MIT")

    data_licence = dataset._license_text()
    assert archive.DATASET_LICENSE in data_licence
    assert archive.DATASET_LICENSE_URL in data_licence
    # the third-party carve-out must survive any rewording
    assert "rossetti" in data_licence.lower()
    assert "chapman" in data_licence.lower()


def test_related_identifiers_grow_with_the_dois(monkeypatch):
    assert archive.related_identifiers() == [] or archive.DATASET_DOI or archive.PAPER_DOI

    monkeypatch.setattr(archive, "DATASET_DOI", "10.4121/xyz")
    monkeypatch.setattr(archive, "PAPER_DOI", "10.1016/j.example.2026.01.001")
    rel = {r["identifier"]: r["relation"] for r in archive.related_identifiers()}
    assert rel == {"10.4121/xyz": "isSourceOf",
                   "10.1016/j.example.2026.01.001": "isSupplementTo"}


def test_zenodo_json_carries_the_cross_links_once_minted():
    """The archives must point at each other, not just exist.

    Skipped while every sibling DOI is pending; the moment one is set in
    archive.py this fails until .zenodo.json names it, which is what keeps
    the dataset/code/paper links from being half-wired.
    """
    expected = archive.related_identifiers()
    if not expected:
        pytest.skip("no sibling DOI minted yet")
    zenodo = json.loads((CITATION.parent / ".zenodo.json").read_text())
    have = {r["identifier"]: r["relation"]
            for r in zenodo.get("related_identifiers", [])}
    for entry in expected:
        assert have.get(entry["identifier"]) == entry["relation"], entry


def test_citation_cff_agrees_with_archive_module():
    """CITATION.cff must carry the same identifiers, once they exist."""
    cff = _citation()
    assert cff["repository-code"].rstrip("/") == archive.CODE_REPOSITORY.rstrip("/")

    dois = {i.get("value") for i in cff.get("identifiers", [])
            if i.get("type") == "doi"}
    if archive.CODE_DOI is not None:
        assert archive.CODE_DOI in dois or cff.get("doi") == archive.CODE_DOI
    if archive.DATASET_DOI is not None:
        # CFF 1.2.0 spells a reference's DOI as `doi:`, not `value:` —
        # `value:` belongs to `identifiers:` entries.
        related = {r.get("doi") for r in cff.get("references", [])
                   if isinstance(r, dict)}
        assert archive.DATASET_DOI in dois | related
