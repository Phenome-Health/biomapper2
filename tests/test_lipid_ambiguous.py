"""Unit 1: a lipid name that LIPID MAPS maps to several DISTINCT connectivities is AMBIGUOUS, not a
false unresolvable. Variants of one skeleton (same first block, different stereo / double-bond
position) still collapse to a single RESOLVED structure. All fakes; no pygoslin, no network.
"""

from __future__ import annotations

from biomapper2.core.certificate import TierBOutcome
from biomapper2.core.lipid_structure_resolver import LipidStructureResolver


class _FakeParse:
    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name


class _FakeGrammar:
    def __init__(self, canonical: str) -> None:
        self._canonical = canonical

    def parse(self, name):
        return _FakeParse(self._canonical) if name else None


class _FakeEnricher:
    def __init__(self, candidates, ok: bool = True) -> None:
        self._candidates = candidates
        self._ok = ok

    def enrich(self, canonical_name):
        return {}

    def enrich_checked(self, canonical_name):
        return {}, self._ok

    def candidates_checked(self, canonical_name):
        return [dict(c) for c in self._candidates], self._ok


def _resolver(candidates, canonical="PC 34:1", ok=True) -> LipidStructureResolver:
    return LipidStructureResolver(grammar=_FakeGrammar(canonical), enricher=_FakeEnricher(candidates, ok))


def test_distinct_connectivities_are_ambiguous_with_the_candidate_set() -> None:
    result = _resolver(
        [
            {"inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N", "lm_id": "LM2"},
            {"inchi_key": "CCCCCCCCCCCCCC-DDDDDDDDDD-N", "lm_id": "LM1"},
        ]
    ).resolve("PC 34:1")
    assert result.outcome is TierBOutcome.AMBIGUOUS
    assert result.inchikey_block is None
    # Sorted, so the set is order-independent (LIPID MAPS row order is not stable).
    assert result.candidate_inchikeys == ("AAAAAAAAAAAAAA-BBBBBBBBBB-N", "CCCCCCCCCCCCCC-DDDDDDDDDD-N")


def test_one_connectivity_many_stereo_resolves_connectivity_only() -> None:
    # Same first block (one connectivity), different stereo layers: stereo is NOT pinned, so assert
    # first block only. A full key would pick an arbitrary stereo and could false-contradict a node
    # carrying a different valid variant.
    result = _resolver(
        [
            {"inchi_key": "AAAAAAAAAAAAAA-XXXXXXXXXX-N"},
            {"inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N"},
        ]
    ).resolve("PC 16:0/18:1")
    assert result.outcome is TierBOutcome.RESOLVED
    assert result.inchikey_block == "AAAAAAAAAAAAAA"  # connectivity only, no arbitrary stereo
    assert result.candidate_inchikeys == ()


def test_single_full_key_pins_stereo() -> None:
    # Exactly one candidate: connectivity AND stereo are pinned, so the full key is asserted.
    result = _resolver([{"inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N"}]).resolve("PC 16:0/18:1")
    assert result.outcome is TierBOutcome.RESOLVED
    assert result.inchikey_block == "AAAAAAAAAAAAAA-BBBBBBBBBB-N"
