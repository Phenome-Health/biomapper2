"""Loader validation for the Tier B freeze (Greptile #86 findings 2 and 3).

A RESOLVED freeze row must carry a canonical source and a well-formed InChIKey. A row that fails
either check is SKIPPED at load (the name falls back to the live path) rather than becoming trusted
evidence -- never assign pubchem provenance to missing data, and never return a truncated key as
RESOLVED. Fixture-only: each case writes a tiny freeze under tmp_path and points the env at it.
"""

from __future__ import annotations

import pytest

from biomapper2.core import tier_b_snapshot

pytestmark = pytest.mark.unit

GOOD_KEY = "WQZGKKKJIJFFOK-GASJEMHNSA-N"


@pytest.fixture(autouse=True)
def _reset_snapshot_cache():
    tier_b_snapshot.reset()
    yield
    tier_b_snapshot.reset()


def _freeze(monkeypatch, tmp_path, body: str):
    path = tmp_path / "freeze.tsv"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("BIOMAPPER2_TIER_B_SNAPSHOT_PATH", str(path))
    tier_b_snapshot.reset()


def test_resolved_row_with_empty_source_is_skipped(monkeypatch, tmp_path):
    _freeze(monkeypatch, tmp_path, f"name\tinchikey\tsource\nglucose\t{GOOD_KEY}\t\n")
    assert tier_b_snapshot.lookup("glucose") is None  # skipped -> falls back to live


def test_resolved_row_with_non_canonical_source_is_skipped(monkeypatch, tmp_path):
    _freeze(monkeypatch, tmp_path, f"name\tinchikey\tsource\nglucose\t{GOOD_KEY}\tsome_random_db\n")
    assert tier_b_snapshot.lookup("glucose") is None


def test_malformed_inchikey_is_skipped(monkeypatch, tmp_path):
    # A truncated / typo'd key must never be returned as RESOLVED evidence.
    _freeze(monkeypatch, tmp_path, "name\tinchikey\tsource\nglucose\tWQZGKKK-GASJEMHNSA-N\tpubchem\n")
    assert tier_b_snapshot.lookup("glucose") is None


def test_source_is_normalized_strip_and_casefold(monkeypatch, tmp_path):
    # " PubChem " normalizes to the canonical "pubchem" and is accepted.
    _freeze(monkeypatch, tmp_path, f"name\tinchikey\tsource\nglucose\t{GOOD_KEY}\t  PubChem  \n")
    hit = tier_b_snapshot.lookup("glucose")
    assert hit is not None
    assert hit.source == "pubchem"


def test_a_bad_row_does_not_reject_the_whole_freeze(monkeypatch, tmp_path):
    # Per-row skip, not whole-freeze rejection: a good row alongside a bad one still loads.
    body = f"name\tinchikey\tsource\nglucose\t{GOOD_KEY}\tpubchem\nbad\tNOTAKEY\tpubchem\n"
    _freeze(monkeypatch, tmp_path, body)
    assert tier_b_snapshot.lookup("glucose") is not None
    assert tier_b_snapshot.lookup("bad") is None


def test_all_rejected_corpus_is_not_present_so_default_stays_inert(monkeypatch, tmp_path):
    """Finding 1: a header-valid corpus whose EVERY resolved row is rejected reduces to an empty
    lookup table. It must read as NOT present so the default posture falls to INERT (no live calls),
    while an explicit truthy override still goes live-behind-breaker."""
    from biomapper2 import config

    # Both rows rejected: one non-canonical source, one malformed inchikey. No frozen-unresolvable row.
    body = f"name\tinchikey\tsource\na\t{GOOD_KEY}\tsome_random_db\nb\tNOTAKEY\tpubchem\n"
    _freeze(monkeypatch, tmp_path, body)
    assert tier_b_snapshot.is_present() is False

    monkeypatch.delenv("BIOMAPPER2_TIER_B_ENABLED", raising=False)
    assert config.resolve_tier_b_state(tier_b_snapshot.is_present()) == config.TIER_B_STATE_INERT
    monkeypatch.setenv("BIOMAPPER2_TIER_B_ENABLED", "true")
    assert config.resolve_tier_b_state(tier_b_snapshot.is_present()) == config.TIER_B_STATE_ENABLED_LIVE


def test_all_unresolvable_corpus_is_still_present(monkeypatch, tmp_path):
    """A corpus of only frozen-unresolvable rows (empty inchikeys) is a valid, usable freeze: those
    are deterministic-unresolvable entries, so it stays present rather than falling to live."""
    _freeze(monkeypatch, tmp_path, "name\tinchikey\tsource\nX-1\t\t\nX-2\t\t\n")
    assert tier_b_snapshot.is_present() is True
    assert tier_b_snapshot.lookup("X-1") is not None
