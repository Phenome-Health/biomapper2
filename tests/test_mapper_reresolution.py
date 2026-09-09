"""Unit 7: Mapper Step 6.5 — structure-guided re-resolution wiring.

When a certificate is CONTRADICTED and the flag is on, the Mapper asks the resolver to swap the
conflated node for the correct distinct candidate, then RE-RUNS enrichment + certificate on the
swapped node under a single-attempt loop guard (a still-contradicted swap is a logged REFUSE, never
recursion). Flag-off is today's behavior byte-for-byte. Uses the narrow bound-method stub pattern of
test_mapper_tier_b_scoping — no full Mapper, no network.
"""

from __future__ import annotations

import pytest

import biomapper2.config as config
from biomapper2.core.certificate import CertificateState, TierBOutcome, TierBResult

COMMITTED_KEY = "AAAAAAAAAAAAAA-BBBBBBBBBB-N"  # committed node's KG structure
ANCHOR_KEY = "KILNVBDSWZSGLL-KXQOOQHDSA-N"  # the query's independent structure (Tier B)


class _TierB:
    def __init__(self, block=ANCHOR_KEY):
        self._block = block
        self.calls = 0

    def lookup(self, name):
        self.calls += 1
        return TierBResult(source="lipidmaps", inchikey_block=self._block, outcome=TierBOutcome.RESOLVED)


class _Resolver:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def is_small_molecule(self, category):
        return True

    def reresolve_on_contradiction(self, *, candidates, query_independent_inchikey, committed_kg_id):
        self.calls += 1
        assert query_independent_inchikey == ANCHOR_KEY, "anchor must be the query independent structure"
        return self._result


class _Linker:
    """Returns scripted equivalent-ids per node id for the swap re-enrichment."""

    def __init__(self, per_node):
        self._per_node = per_node

    def get_equivalent_ids_checked(self, ids):
        kid = ids[0]
        return {kid: self._per_node.get(kid, {})}, True


class _Mapper:
    def __init__(self, tier_b, resolver, linker):
        from biomapper2.mapper import Mapper

        self.tier_b = tier_b
        self.resolver = resolver
        self.linker = linker
        self._issue_certificate = Mapper._issue_certificate.__get__(self, _Mapper)
        self._enrich_equivalent_ids = Mapper._enrich_equivalent_ids.__get__(self, _Mapper)
        self._certify_and_reresolve = Mapper._certify_and_reresolve.__get__(self, _Mapper)


def _call(mapper, chosen="CHEBI:committed", kg_ids=None):
    return mapper._certify_and_reresolve(
        query_name="PC 16:0/18:1",
        category="biolink:SmallMolecule",
        chosen_kg_id=chosen,
        kg_equivalent_ids={"INCHIKEY": [COMMITTED_KEY]},
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
        kg_ids=kg_ids or {"CHEBI:committed": ["a"], "CHEBI:correct": ["b"]},
        kg_ids_assigned={},
    )


@pytest.fixture
def _on(monkeypatch):
    monkeypatch.setattr(config, "RERESOLUTION_ENABLED", True)


def test_contradiction_swaps_to_matching_candidate_with_fresh_certificate(_on):
    resolver = _Resolver(("CHEBI:correct", "reresolved"))
    linker = _Linker({"CHEBI:correct": {"INCHIKEY": [ANCHOR_KEY]}})  # correct node structurally matches anchor
    mapper = _Mapper(_TierB(), resolver, linker)

    cert, new_id, new_equiv = _call(mapper)

    assert new_id == "CHEBI:correct"
    assert cert.state is CertificateState.CORROBORATED
    assert new_equiv == {"INCHIKEY": [ANCHOR_KEY]}
    assert resolver.calls == 1


def test_still_contradicted_swap_is_a_single_attempt_refuse(_on):
    # The resolver hands back a candidate, but the swapped node ALSO contradicts the anchor.
    resolver = _Resolver(("CHEBI:alsobad", "reresolved"))
    linker = _Linker({"CHEBI:alsobad": {"INCHIKEY": ["ZZZZZZZZZZZZZZ-XXXXXXXXXX-N"]}})
    mapper = _Mapper(_TierB(), resolver, linker)

    cert, new_id, _ = _call(mapper)

    assert new_id == "CHEBI:committed", "a still-contradicted swap must NOT be committed"
    assert cert.state is CertificateState.CONTRADICTED
    assert cert.refusal_reason == "reresolution_still_contradicted"
    assert resolver.calls == 1, "re-resolution runs at most once (no recursion)"


def test_refused_no_match_keeps_committed_and_records_reason(_on):
    resolver = _Resolver(("CHEBI:committed", "reresolution_refused_no_match"))
    mapper = _Mapper(_TierB(), resolver, _Linker({}))

    cert, new_id, _ = _call(mapper)

    assert new_id == "CHEBI:committed"
    assert cert.state is CertificateState.CONTRADICTED
    assert cert.refusal_reason == "reresolution_refused_no_match"


def test_flag_off_is_todays_behavior(monkeypatch):
    monkeypatch.setattr(config, "RERESOLUTION_ENABLED", False)
    resolver = _Resolver(("CHEBI:correct", "reresolved"))
    mapper = _Mapper(_TierB(), resolver, _Linker({}))

    cert, new_id, new_equiv = _call(mapper)

    assert new_id == "CHEBI:committed"
    assert cert.state is CertificateState.CONTRADICTED  # unchanged, no swap
    assert cert.refusal_reason is None
    assert resolver.calls == 0, "re-resolution must not even be consulted when the flag is off"
    assert new_equiv == {"INCHIKEY": [COMMITTED_KEY]}


def test_reresolution_default_is_off_in_config():
    """Default-off is the contract: re-resolution is a gated production change, not a default the
    pipeline drifts into. Only a supervised, benchmark-cleared run enables it."""
    import os

    from biomapper2 import config as _config

    if os.getenv("BIOMAPPER2_RERESOLUTION_ENABLED", "").strip().lower() in {"1", "true", "yes"}:
        pytest.skip("env explicitly enables re-resolution")
    assert _config.RERESOLUTION_ENABLED is False


def test_reresolution_without_tier_b_is_a_loud_configuration_error(monkeypatch):
    """RERESOLUTION_ENABLED requires TIER_B_ENABLED (a CONTRADICTED certificate only comes from Tier
    B). The dependency is surfaced at build time, not as a silent no-op."""
    import biomapper2.mapper as mapper_mod

    monkeypatch.setattr(mapper_mod, "TIER_B_ENABLED", False)
    monkeypatch.setattr(mapper_mod, "RERESOLUTION_ENABLED", True)
    with pytest.raises(ValueError, match="requires TIER_B_ENABLED"):
        mapper_mod.Mapper._build_tier_b(None)
