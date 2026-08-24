"""The mapper must not spend a Tier B lookup on a row the certificate cannot be about.

Every term of that predicate was previously unpinned: dropping `equivalent_ids_lookup_ok`, dropping
the `chosen_kg_id is not None` term, or dropping the whole guard all left the full suite green. The
cost of getting it wrong is not a wrong verdict -- `issue()` discards the result for out-of-scope
rows -- it is a throttled round trip against two external services for every gene symbol and every
uncommitted row, and those names entering the resolution-rate denominator that must accompany every
operating point on Figure 5.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from biomapper2.core.tier_b import TierBOutcome, TierBResult


class _RecordingTierB:
    def __init__(self):
        self.looked_up: list[str | None] = []

    def lookup(self, name):
        self.looked_up.append(name)
        return TierBResult(source="pubchem", inchikey_block="AAAAAAAAAAAAAA", outcome=TierBOutcome.RESOLVED)


class _Mapper:
    """The narrowest object exposing `_issue_certificate`, bound to a stub resolver and Tier B."""

    def __init__(self, tier_b, is_small_molecule=True):
        from biomapper2.mapper import Mapper

        self.tier_b = tier_b
        self.resolver = SimpleNamespace(is_small_molecule=lambda category: is_small_molecule)
        self._issue_certificate = Mapper._issue_certificate.__get__(self, _Mapper)


@pytest.mark.parametrize(
    "in_scope,chosen_kg_id,lookup_ok,is_small_molecule",
    [
        (True, "CHEBI:1", True, True),  # the only shape a certificate can be about
        (False, None, True, True),  # nothing committed
        (False, "CHEBI:1", False, True),  # /get-nodes outage: structure unknown, not absent
        (False, "HGNC:1", True, False),  # out of scope entirely
    ],
)
def test_tier_b_is_consulted_only_for_rows_the_certificate_covers(in_scope, chosen_kg_id, lookup_ok, is_small_molecule):
    tier_b = _RecordingTierB()
    mapper = _Mapper(tier_b, is_small_molecule=is_small_molecule)
    mapper._issue_certificate(
        query_name="acetate",
        category="biolink:SmallMolecule" if is_small_molecule else "biolink:Gene",
        chosen_kg_id=chosen_kg_id,
        kg_equivalent_ids={"INCHIKEY": ["AAAAAAAAAAAAAA-UHFFFAOYSA-N"]},
        equivalent_ids_lookup_ok=lookup_ok,
        selection_conflict=None,
        kg_ids_assigned={},
    )
    assert tier_b.looked_up == (["acetate"] if in_scope else [])


def test_no_lookup_at_all_when_tier_b_is_disabled():
    """The default posture: Tier B is opt-in, so the ordinary path must make no external call."""
    mapper = _Mapper(None)
    certificate = mapper._issue_certificate(
        query_name="acetate",
        category="biolink:SmallMolecule",
        chosen_kg_id="CHEBI:1",
        kg_equivalent_ids={"INCHIKEY": ["AAAAAAAAAAAAAA-UHFFFAOYSA-N"]},
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
        kg_ids_assigned={},
    )
    assert certificate.tier_b_outcome is TierBOutcome.OFF
    assert certificate.independent_source is None
