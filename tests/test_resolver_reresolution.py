"""Unit 6: structure-guided re-resolution over the one-to-many candidate set.

On a detected contradiction the resolver uses the QUERY's independent structure (the anchor) to pick
the correct DISTINCT candidate from the conflated vote's losers. It NEVER reads the committed node's
own InChIKey as the anchor (KTD5). Across-node conflation switches; within-node / no-match refuses;
an ambiguous multi-match refuses; and the flag defaults off. Every test injects a fake structure
resolver — no network.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import biomapper2.config as config
from biomapper2.core.resolver import Resolver

QUERY_ANCHOR = "KILNVBDSWZSGLL-KXQOOQHDSA-N"  # the query's independent structure


class _FakeStructureResolver:
    """Maps a node id -> its FULL structural InChIKey(s). A value may be a single key or a list of
    graph-asserted keys (StructureResolver.structural_inchikeys returns all of them)."""

    def __init__(self, keys: dict[str, str | list[str] | None]) -> None:
        self._keys = keys
        self.asked: list[str] = []

    def structural_inchikey(self, node_id, node_name=None, records=None):
        val = self._keys.get(node_id)
        return val[0] if isinstance(val, list) else val

    def structural_inchikeys(self, node_id, node_name=None, records=None):
        self.asked.append(node_id)
        val = self._keys.get(node_id)
        if val is None:
            return []
        return list(val) if isinstance(val, list) else [val]


def _resolver(struct_keys: dict[str, str | list[str] | None], names: dict[str, str] | None = None) -> Resolver:
    linker = MagicMock()
    linker.get_node_records.return_value = {nid: {"name": (names or {}).get(nid)} for nid in struct_keys}
    r = Resolver(linker=linker, biolink_client=MagicMock())
    r.structure_resolver = cast(Any, _FakeStructureResolver(struct_keys))
    return r


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setattr(config, "RERESOLUTION_ENABLED", True)


def test_across_node_conflation_switches_to_the_matching_candidate():
    # committed node's structure differs from the anchor; a distinct candidate matches it.
    r = _resolver(
        {"CHEBI:committed": "WRONGWRONGWRNG-XXXXXXXXXX-N", "CHEBI:correct": QUERY_ANCHOR},
        names={"CHEBI:correct": "PC 16:0/18:1"},
    )
    new_id, reason = r.reresolve_on_contradiction(
        candidates=["CHEBI:committed", "CHEBI:correct"],
        query_independent_inchikey=QUERY_ANCHOR,
        committed_kg_id="CHEBI:committed",
    )
    assert new_id == "CHEBI:correct"
    assert reason == "reresolved"


def test_the_committed_node_is_never_the_comparison_anchor():
    # KTD5: if the committed node's own key were (wrongly) used as the anchor, the ONLY candidate
    # that matches it is itself, and re-resolution would spuriously "confirm" the committed node.
    # The anchor is the QUERY key, which matches a different candidate — so we must switch, and the
    # committed node's structure must not be consulted as the anchor.
    struct = _FakeStructureResolver({"CHEBI:committed": "CCCCCCCCCCCCCC-YYYYYYYYYY-N", "CHEBI:other": QUERY_ANCHOR})
    r = _resolver({"CHEBI:other": QUERY_ANCHOR})
    r.structure_resolver = cast(Any, struct)
    new_id, reason = r.reresolve_on_contradiction(
        candidates=["CHEBI:committed", "CHEBI:other"],
        query_independent_inchikey=QUERY_ANCHOR,
        committed_kg_id="CHEBI:committed",
    )
    assert new_id == "CHEBI:other"
    assert reason == "reresolved"
    # The committed node's structural key was never fetched as the anchor.
    assert "CHEBI:committed" not in struct.asked


def test_within_node_no_distinct_match_refuses():
    r = _resolver({"CHEBI:committed": "AAAAAAAAAAAAAA-XX-N", "CHEBI:b": "BBBBBBBBBBBBBB-XX-N"})
    new_id, reason = r.reresolve_on_contradiction(
        candidates=["CHEBI:committed", "CHEBI:b"],
        query_independent_inchikey=QUERY_ANCHOR,
        committed_kg_id="CHEBI:committed",
    )
    assert new_id == "CHEBI:committed"
    assert reason == "reresolution_refused_no_match"


def test_ambiguous_multi_match_refuses():
    r = _resolver({"CHEBI:committed": "ZZ-XX-N", "CHEBI:a": QUERY_ANCHOR, "CHEBI:b": QUERY_ANCHOR})
    new_id, reason = r.reresolve_on_contradiction(
        candidates=["CHEBI:committed", "CHEBI:a", "CHEBI:b"],
        query_independent_inchikey=QUERY_ANCHOR,
        committed_kg_id="CHEBI:committed",
    )
    assert new_id == "CHEBI:committed"
    assert reason == "reresolution_ambiguous"


def test_disabled_is_a_no_op(monkeypatch):
    monkeypatch.setattr(config, "RERESOLUTION_ENABLED", False)
    r = _resolver({"CHEBI:committed": "ZZ-XX-N", "CHEBI:correct": QUERY_ANCHOR})
    new_id, reason = r.reresolve_on_contradiction(
        candidates=["CHEBI:committed", "CHEBI:correct"],
        query_independent_inchikey=QUERY_ANCHOR,
        committed_kg_id="CHEBI:committed",
    )
    assert new_id == "CHEBI:committed"
    assert reason == "reresolution_disabled"


def test_no_anchor_refuses():
    r = _resolver({"CHEBI:committed": "ZZ-XX-N", "CHEBI:correct": QUERY_ANCHOR})
    new_id, reason = r.reresolve_on_contradiction(
        candidates=["CHEBI:committed", "CHEBI:correct"],
        query_independent_inchikey=None,
        committed_kg_id="CHEBI:committed",
    )
    assert new_id == "CHEBI:committed"
    assert reason == "reresolution_refused_no_match"


def test_positive_control_a_wrong_candidate_key_does_not_match():
    # If the correct candidate's key is corrupted, no candidate matches and re-resolution refuses —
    # proving the match is real, not always-true.
    r = _resolver({"CHEBI:committed": "ZZ-XX-N", "CHEBI:correct": "TOTALLYWRONGXX-XXXXXXXXXX-N"})
    new_id, reason = r.reresolve_on_contradiction(
        candidates=["CHEBI:committed", "CHEBI:correct"],
        query_independent_inchikey=QUERY_ANCHOR,
        committed_kg_id="CHEBI:committed",
    )
    assert reason == "reresolution_refused_no_match"


def test_candidate_matched_on_a_non_first_inchikey():
    # A candidate carries several graph-asserted InChIKeys and only the SECOND matches the anchor.
    # Matching on the first key alone would miss it; structural_inchikeys must accept ANY key.
    r = _resolver(
        {
            "CHEBI:committed": "WRONGWRONGWRNG-XXXXXXXXXX-N",
            "CHEBI:correct": ["DECOYDECOYDEC-YYYYYYYYYY-N", QUERY_ANCHOR],
        }
    )
    new_id, reason = r.reresolve_on_contradiction(
        candidates=["CHEBI:committed", "CHEBI:correct"],
        query_independent_inchikey=QUERY_ANCHOR,
        committed_kg_id="CHEBI:committed",
    )
    assert new_id == "CHEBI:correct"
    assert reason == "reresolved"
