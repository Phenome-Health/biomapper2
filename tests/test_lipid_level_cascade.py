"""Unit 3: lipid lookups cascade from the input's EFFECTIVE query level down to species across the
sources, recording the level each hit was found at. The sn-position trust policy (D1) downgrades a
"/" input to molecular-species for querying by default; Kestrel is queried only for levels no other
source hit (D3). Real (offline) pygoslin parses; all binding sources are injected fakes, no network.
"""

from __future__ import annotations

import pytest

from biomapper2.core.annotators.goslin_lipid import GoslinLipidAnnotator

_SM = "biolink:SmallMolecule"


class _FakeSource:
    """A name-binding source (RefMet /match or Kestrel). Records the names it was queried with and
    returns a vote block only for the names it was told to know."""

    def __init__(self, slug: str, known: dict | None = None) -> None:
        self.slug = slug
        self.seen_names: list[str] = []
        self._known = known or {}

    def get_annotations(self, entity, name_field, category, prefixes=None, **kwargs):
        name = entity.get(name_field)
        self.seen_names.append(name)
        vocab_map = self._known.get(name, {})
        return {self.slug: {vocab: {id_: {} for id_ in ids} for vocab, ids in vocab_map.items()}}


class _FakeEnricher:
    """A LIPID MAPS enricher whose candidate set depends on the level name (via candidates_checked)."""

    def __init__(self, by_level: dict[str, list[dict]]) -> None:
        self.seen: list[str] = []
        self._by_level = by_level

    def enrich(self, canonical_name):
        return {}

    def enrich_checked(self, canonical_name):
        return {}, True

    def candidates_checked(self, canonical_name):
        self.seen.append(canonical_name)
        return [dict(c) for c in self._by_level.get(canonical_name, [])], True


def _binder(known):
    return _FakeSource("metabolomics-workbench", known)


def _kestrel(known):
    return _FakeSource("kestrel-hybrid-search", known)


def _vote_meta(out, vocab, id_):
    return out["goslin-lipid"][vocab][id_]


def test_falls_back_to_species_when_only_species_hits():
    # Only the species name is known, so the cascade walks past molecular-species to species.
    binder = _binder({"PC 34:1": {"refmet_id": ["RM0001"]}})
    ann = GoslinLipidAnnotator(binder=binder)  # pyright: ignore[reportArgumentType]
    out = ann.get_annotations({"name": "PC 16:0/18:1"}, name_field="name", category=_SM)
    meta = _vote_meta(out, "refmet_id", "RM0001")
    assert meta["matched_level"] == "species"
    assert meta["query_lipid_level_asserted"] == "sn_position"
    assert meta["query_lipid_level_effective"] == "molecular_species"


def test_matched_level_records_the_molecular_species_hit():
    # The molecular-species name is known, so the cascade stops there (more specific than species).
    binder = _binder({"PC 16:0_18:1": {"refmet_id": ["RM0090134"]}})
    ann = GoslinLipidAnnotator(binder=binder)  # pyright: ignore[reportArgumentType]
    out = ann.get_annotations({"name": "PC 16:0/18:1"}, name_field="name", category=_SM)
    assert _vote_meta(out, "refmet_id", "RM0090134")["matched_level"] == "molecular_species"


def test_trust_off_downgrades_and_never_queries_the_slash_name():
    # Default trust OFF: the sn-position "/" name is downgraded out of the query set entirely.
    binder = _binder({"PC 16:0_18:1": {"refmet_id": ["RM0090134"]}})
    ann = GoslinLipidAnnotator(binder=binder)  # pyright: ignore[reportArgumentType]
    out = ann.get_annotations({"name": "PC 16:0/18:1"}, name_field="name", category=_SM)
    assert "PC 16:0/18:1" not in binder.seen_names
    assert "PC 16:0_18:1" in binder.seen_names
    meta = _vote_meta(out, "refmet_id", "RM0090134")
    assert meta["query_lipid_level_asserted"] == "sn_position"
    assert meta["query_lipid_level_effective"] == "molecular_species"


def test_sn_level_hit_recorded_when_trust_enabled():
    # Trust ON: the effective level equals the asserted sn-position, so the slash name is queried first.
    ann = GoslinLipidAnnotator(
        binder=_binder({"PC 16:0/18:1": {"refmet_id": ["RM0015001"]}}),  # pyright: ignore[reportArgumentType]
        trust_sn_position=True,
    )
    out = ann.get_annotations({"name": "PC 16:0/18:1"}, name_field="name", category=_SM)
    meta = _vote_meta(out, "refmet_id", "RM0015001")
    assert meta["matched_level"] == "sn_position"
    assert meta["query_lipid_level_effective"] == "sn_position"


def test_kestrel_skips_levels_already_hit_by_another_source():
    # The binder hits at molecular-species, so Kestrel must not re-query that level (D3); it only
    # queries species, and its hit there is recorded at the species level.
    binder = _binder({"PC 16:0_18:1": {"refmet_id": ["RM0090134"]}})
    kestrel = _kestrel({"PC 16:0_18:1": {"CHEBI": ["1"]}, "PC 34:1": {"CHEBI": ["2"]}})
    ann = GoslinLipidAnnotator(binder=binder, kestrel=kestrel)  # pyright: ignore[reportArgumentType]
    out = ann.get_annotations({"name": "PC 16:0/18:1"}, name_field="name", category=_SM)
    assert "PC 16:0_18:1" not in kestrel.seen_names
    assert "PC 34:1" in kestrel.seen_names
    assert _vote_meta(out, "CHEBI", "2")["matched_level"] == "species"


def test_enrichment_cascade_records_matched_level_and_flags_fired():
    # LIPID MAPS enrichment (multi-row aware) hits at molecular-species and stamps that level.
    enr = _FakeEnricher({"PC 16:0_18:1": [{"lm_id": "LMGP01010001", "inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N"}]})
    ann = GoslinLipidAnnotator(binder=_binder({}), enrichment=enr)  # pyright: ignore[reportArgumentType]
    out = ann.get_annotations({"name": "PC 16:0/18:1"}, name_field="name", category=_SM)
    meta = _vote_meta(out, "LIPIDMAPS", "LMGP01010001")
    assert meta["matched_level"] == "molecular_species"
    assert meta["lipidmaps_rest_enrichment_fired"] is True


@pytest.mark.integration
@pytest.mark.third_party
@pytest.mark.requires_api
def test_end_to_end_map_entity_to_kg_exercises_the_cascade():
    # Smoke check only (live services drift): the sn-position lipid runs the whole pipeline and a
    # certificate comes back. All correctness assertions live in the offline tests above.
    from biomapper2.mapper import Mapper

    result = Mapper().map_entity_to_kg(
        item={"name": "PC 16:0/18:1"},
        name_field="name",
        provided_id_fields=[],
        entity_type="metabolite",
        annotation_mode="all",
        candidate_limit=1,
    )
    assert isinstance(result, dict)
    assert "resolution_certificate" in result
