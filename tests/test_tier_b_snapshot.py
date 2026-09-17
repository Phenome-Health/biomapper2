"""Loader tests for the pinned Tier B freeze (core/tier_b_snapshot.py).

Fixture-only: the loader reads a committed freeze TSV under tests/fixtures/. No network, no real
data file. ``BIOMAPPER2_TIER_B_SNAPSHOT_PATH`` is pointed at the fixture per test and the module
cache is reset around each so the process-global singleton never leaks a freeze into the rest of the
suite. Mirrors tests/test_refmet_snapshot.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biomapper2.core import tier_b_snapshot

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "tier_b_freeze_fixture.tsv"


@pytest.fixture(autouse=True)
def _reset_snapshot_cache():
    tier_b_snapshot.reset()
    yield
    tier_b_snapshot.reset()


@pytest.fixture
def loaded(monkeypatch):
    monkeypatch.setenv("BIOMAPPER2_TIER_B_SNAPSHOT_PATH", str(FIXTURE))
    tier_b_snapshot.reset()
    return tier_b_snapshot


def test_is_present_false_when_unconfigured(monkeypatch):
    monkeypatch.delenv("BIOMAPPER2_TIER_B_SNAPSHOT_PATH", raising=False)
    tier_b_snapshot.reset()
    assert tier_b_snapshot.is_present() is False
    assert tier_b_snapshot.version() is None
    assert tier_b_snapshot.lookup("glucose") is None


def test_is_present_true_and_version_from_sidecar(loaded):
    assert loaded.is_present() is True
    assert loaded.version() == "tier-b-fixture-v1"


def test_resolved_hit_carries_inchikey_source_and_version(loaded):
    hit = loaded.lookup("glucose")
    assert hit is not None
    assert hit.inchikey == "WQZGKKKJIJFFOK-GASJEMHNSA-N"
    assert hit.source == "pubchem"
    assert hit.version == "tier-b-fixture-v1"


def test_canonical_mw_source_and_first_block_form_are_accepted(loaded):
    # metabolomics-workbench is a canonical source; a 14-letter first-block-only key is a valid form.
    mw = loaded.lookup("citrate")
    assert mw is not None
    assert mw.source == "metabolomics-workbench"
    first_block = loaded.lookup("alanine")
    assert first_block is not None
    assert first_block.inchikey == "QNAYBMKLOCPYGJ"


def test_empty_inchikey_is_a_positively_frozen_no_structure(loaded):
    # A row present in the corpus with no InChIKey positively records "no structure for this name":
    # it returns a hit (so the tier does NOT go to the network) whose inchikey is None.
    hit = loaded.lookup("X-99999")
    assert hit is not None
    assert hit.inchikey is None


def test_miss_returns_none(loaded):
    assert loaded.lookup("a name that is not in the freeze at all") is None


def test_normalized_lookup_ignores_case_and_surrounding_whitespace(loaded):
    hit = loaded.lookup("  GLUCOSE ")
    assert hit is not None
    assert hit.inchikey == "WQZGKKKJIJFFOK-GASJEMHNSA-N"


def test_malformed_header_is_rejected_not_trusted(monkeypatch, tmp_path):
    # A freeze missing the required 'inchikey' column is corrupt. It must be REJECTED (is_present
    # False -> live fallback), never silently served as an authoritative empty corpus that turns
    # every name into a miss.
    freeze = tmp_path / "bad_freeze.tsv"
    freeze.write_text("name\twrong_column\nglucose\tX\n", encoding="utf-8")
    monkeypatch.setenv("BIOMAPPER2_TIER_B_SNAPSHOT_PATH", str(freeze))
    tier_b_snapshot.reset()
    assert tier_b_snapshot.is_present() is False
    assert tier_b_snapshot.lookup("glucose") is None
