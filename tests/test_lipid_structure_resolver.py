"""Unit 2: LipidStructureResolver — Goslin parse -> LIPID MAPS REST -> FULL InChIKey + source tag.

Goslin alone emits no InChIKey; binding the canonical shorthand to a structure is a database lookup.
This resolver is the lipid hop of the ``independent-structure-iface``: it produces a
``name -> independent InChIKey`` verdict for BOTH the query side (Tier B) and the candidate side
(StructureResolver). Every test injects a fake grammar and a fake enricher — no pygoslin, no network.
"""

from __future__ import annotations

from biomapper2.core.certificate import TierBOutcome
from biomapper2.core.lipid_structure_resolver import LipidStructureResolver

FULL_KEY = "KILNVBDSWZSGLL-KXQOOQHDSA-N"


class _FakeParse:
    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name


class _FakeGrammar:
    """Parses only names starting with a known lipid head group; everything else is a non-lipid."""

    def __init__(self, canonical: str | None = "PC 16:0/18:1") -> None:
        self._canonical = canonical

    def parse(self, name):
        if name and str(name).startswith(("PC", "PE", "TG", "SM")):
            return _FakeParse(self._canonical or str(name))
        return None


class _FakeEnricher:
    """Returns a scripted ``(mapping, ok)`` for enrich_checked; records the canonical names asked."""

    def __init__(self, mapping=None, ok=True) -> None:
        self._mapping = mapping if mapping is not None else {}
        self._ok = ok
        self.calls: list[str] = []

    def enrich(self, canonical_name):
        return self.enrich_checked(canonical_name)[0]

    def enrich_checked(self, canonical_name):
        self.calls.append(canonical_name)
        return dict(self._mapping), self._ok


def _resolver(mapping=None, ok=True, canonical="PC 16:0/18:1"):
    return LipidStructureResolver(grammar=_FakeGrammar(canonical), enricher=_FakeEnricher(mapping, ok))


def test_non_lipid_name_is_unresolvable_and_never_calls_the_enricher():
    enricher = _FakeEnricher({"INCHIKEY": FULL_KEY})
    r = LipidStructureResolver(grammar=_FakeGrammar(), enricher=enricher)
    result = r.resolve("glucose")
    assert result.outcome is TierBOutcome.UNRESOLVABLE
    assert result.inchikey_block is None
    assert enricher.calls == [], "a non-lipid parse-miss must not spend a LIPID MAPS lookup"


def test_lipid_hit_resolves_to_a_full_key_with_the_lipidmaps_source_tag():
    result = _resolver({"INCHIKEY": FULL_KEY}).resolve("PC 16:0/18:1")
    assert result.outcome is TierBOutcome.RESOLVED
    assert result.inchikey_block == FULL_KEY  # FULL key, block2 present
    assert "-" in result.inchikey_block and len(result.inchikey_block.split("-")) == 3
    assert result.source == "lipidmaps"


def test_parsed_lipid_the_enricher_cannot_bind_is_unresolvable():
    # A clean answer with no InChIKey means "no structure for this lipid", not a failure.
    result = _resolver({"LIPIDMAPS": "LMGP01010001"}, ok=True).resolve("PC 16:0/18:1")
    assert result.outcome is TierBOutcome.UNRESOLVABLE
    assert result.inchikey_block is None


def test_enricher_failure_is_lookup_failed_and_not_cached():
    r = _resolver({}, ok=False)
    result = r.resolve("PC 16:0/18:1")
    assert result.outcome is TierBOutcome.LOOKUP_FAILED
    # Not cached: a later successful attempt for the same name must be able to resolve.
    assert "PC 16:0/18:1" not in r._memo


def test_resolved_results_are_memoized():
    r = _resolver({"INCHIKEY": FULL_KEY})
    r.resolve("PC 16:0/18:1")
    r.resolve("PC 16:0/18:1")
    # The underlying enricher was consulted once; the second call is served from the memo.
    assert r._enricher.calls == ["PC 16:0/18:1"]


def test_the_canonical_name_is_what_gets_enriched_not_the_raw_input():
    # Goslin normalizes shorthand; the enricher must be handed the canonical form.
    r = _resolver({"INCHIKEY": FULL_KEY}, canonical="PC 16:0/18:1")
    r.resolve("PC(16:0/18:1)")
    assert r._enricher.calls == ["PC 16:0/18:1"]


def test_positive_control_a_wrong_expected_key_fails():
    result = _resolver({"INCHIKEY": FULL_KEY}).resolve("PC 16:0/18:1")
    assert result.inchikey_block != "WRONGWRONGWRONG-XXXXXXXXXX-N"
