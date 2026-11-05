"""Lipid resolution coarsening fix (2026-11-05): D1 narrowed, relation vs asserted level, RefMet
precedence over a generalized Goslin vote, and the lipid class guard.

Evidence: the 2026-11-05 cross-cohort rerun committed ``12,13-dihome`` / ``9,10-dihome`` to ``FA 18:1;O2``,
``5-hete`` / ``12-hete`` to ``FA 20:4;O``, ``5-HEPE`` to ``FA 20:5;O`` (all labelled exact) and the
plasmalogen ``...-gpe (p-16:0/18:1)*`` to the diacylglycerol ``DG 16:0/18:1(9Z)/0:0``.

No live services. The RefMet ``/match`` answers, Kestrel votes and KG node names/InChIKeys below were
RECORDED from live calls on 2026-11-05 (RefMet ``/match``; the public Kestrel ``/get-nodes``; a local
engine trace of the two evidence rows, which reproduced the production rows' ``assigned_ids`` and
chosen node exactly). Responses not recorded are not invented: a name absent from ``_REFMET`` is a
recorded no-match only where noted, and the post-fix Kestrel cascade votes (which the fix changes and
which were not recorded) are modelled as a single species-level vote, stated per test.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest

from biomapper2.core.annotators.goslin_grammar import LipidGrammar
from biomapper2.core.annotators.goslin_lipid import GoslinLipidAnnotator
from biomapper2.core.resolver import (
    GOSLIN_LIPID_ANNOTATOR,
    LIPID_GENERALIZED_HINT,
    REFMET_ANNOTATOR,
    Resolver,
    build_lipid_resolution,
    lipid_mapping_relation,
)

_SM = "biolink:SmallMolecule"
_KESTREL = "kestrel-hybrid-search"

# RefMet /match, recorded live 2026-11-05 (query -> refmet_id). Every Goslin structure-level rendering
# of the oxylipins below (e.g. "FA 18:1(9Z);12OH,13OH", "FA 20:4;OH") was also queried and returned
# the "-" no-match record, so leaving them out of this map IS the recorded answer.
_REFMET: dict[str, str] = {
    "12,13-DiHOME": "RM0138124",
    "9,10-DiHOME": "RM0074128",
    "FA 18:1;O2": "RM0224117",
    "5-HETE": "RM0158210",
    "12-HETE": "RM0152612",
    "15-HETE": "RM0153441",
    "FA 20:4;O": "RM0224126",
    "5-HEPE": "RM0155791",
    "FA 20:5;O": "RM0224127",
    "PE P-16:0/18:1": "RM0134760",
    "1-(1-enyl-palmitoyl)-2-oleoyl-GPE (P-16:0/18:1)*": "RM0134760",
    "PC 16:0/18:1": "RM0015001",
    "PC 16:0_18:1": "RM0090134",
}

# KG node names (Kestrel /get-nodes, recorded 2026-11-05). INCHIKEY presence mirrors the record.
_NODE_NAMES: dict[str, str] = {
    "RM:0138124": "12,13-DiHOME",
    "RM:0074128": "9,10-DiHOME",
    "RM:0224117": "FA 18:1;O2",
    "RM:0139037": "9-HpOME",
    "RM:0158210": "5-HETE",
    "RM:0152612": "12-HETE",
    "RM:0153066": "9-HETE",
    "RM:0224126": "FA 20:4;O",
    "RM:0155791": "5-HEPE",
    "RM:0153119": "11-HEPE",
    "RM:0224127": "FA 20:5;O",
    "RM:0134760": "PE P-16:0/18:1",
    "RM:0133287": "DG 16:0/18:1(9Z)/0:0",
    "RM:0015001": "PC 16:0/18:1",
    "RM:0090134": "PC 16:0_18:1",
    "RM:0015038": "PE 18:0/18:1",
    "RM:0161692": "DG 18:0/18:1(9Z)/0:0",
    "RM:0135445": "sphinganine",
    # Non-lipid names (not recorded; any non-lipid name serves) for the no-op regression.
    "CHEBI:100": "glucose",
    "CHEBI:200": "alpha-D-glucose",
}


# --------------------------------------------------------------------------------------------- fakes


class _RecordedRefMet:
    """RefMet /match binder replaying the recorded answers; records every query it receives."""

    slug = REFMET_ANNOTATOR

    def __init__(self) -> None:
        self.queries: list[str] = []

    def get_annotations(self, entity, name_field, category, prefixes=None, **kwargs):
        name = entity.get(name_field)
        self.queries.append(name)
        rid = _REFMET.get(name)
        return {self.slug: {"refmet_id": {rid: {}}} if rid else {}}


class _NodeLinker:
    """``get_node_records`` over the recorded node names; counts calls."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def get_node_records(self, ids):
        self.calls.append(list(ids))
        return {i: {"name": _NODE_NAMES[i], "equivalent_ids": {}} for i in ids if i in _NODE_NAMES}


class _Biolink:
    def get_descendants(self, category):
        return {_SM} if category == _SM else set()


def _resolver(connectivity: bool | None = None) -> tuple[Resolver, _NodeLinker, list]:
    linker = _NodeLinker()
    r = Resolver(linker=linker, biolink_client=_Biolink())
    seen: list = []

    def _conn(a, b):
        seen.append((a, b))
        return connectivity

    r._connectivity_match = _conn  # type: ignore[method-assign]
    return r, linker, seen


def _resolve(r: Resolver, row: pd.Series) -> dict[str, Any]:
    out = r.resolve(row, category=_SM)
    assert isinstance(out, pd.Series)
    return out.to_dict()


def _goslin_meta(asserted: str, effective: str, matched: str, level_names: dict[str, str]) -> dict[str, Any]:
    return {
        "goslin_canonical": level_names.get("SPECIES"),
        "goslin_level_names": level_names,
        "goslin_dialect": "Goslin",
        "query_lipid_level_asserted": asserted,
        "query_lipid_level_effective": effective,
        "matched_level": matched,
    }


def _entity(
    *,
    refmet: str | None,
    kestrel_raw: str | None,
    goslin: dict[str, dict[str, Any]] | None = None,
) -> pd.Series:
    """A resolver input row: curie-keyed votes per annotator, mirroring the recorded trace shape.

    ``goslin`` maps a KG curie to its goslin vote metadata (keyed under the RAW id the annotator stamps).
    """
    assigned: dict[str, Any] = {}
    kg_assigned: dict[str, dict[str, list[str]]] = {}
    kg_ids: dict[str, list[str]] = {}
    if refmet:
        assigned[REFMET_ANNOTATOR] = {"refmet_id": {refmet.replace(":", ""): {}}}
        kg_assigned[REFMET_ANNOTATOR] = {refmet: [refmet]}
        kg_ids.setdefault(refmet, [refmet])
    if kestrel_raw:
        assigned[_KESTREL] = {"RM": {kestrel_raw.split(":", 1)[1]: {"score": 1.0}}}
        kg_assigned[_KESTREL] = {kestrel_raw: [kestrel_raw]}
        kg_ids.setdefault(kestrel_raw, [kestrel_raw])
    if goslin:
        assigned[GOSLIN_LIPID_ANNOTATOR] = {
            "refmet_id": {curie.replace(":", ""): meta for curie, meta in goslin.items()}
        }
        kg_assigned[GOSLIN_LIPID_ANNOTATOR] = {curie: [curie] for curie in goslin}
        for curie in goslin:
            kg_ids.setdefault(curie, [curie])
    return pd.Series(
        {"kg_ids": kg_ids, "kg_ids_provided": {}, "kg_ids_assigned": kg_assigned, "assigned_ids": assigned}
    )


# ------------------------------------------------------------------- (a) D1 narrowed in the annotator

# (input spelling, asserted level, finest rendering the cascade must query FIRST, matched RefMet id)
_SINGLE_CHAIN = [
    ("12,13-dihome", "FULL_STRUCTURE", "FA 18:1(9Z);12OH,13OH", "RM0224117"),
    ("12,13-DiHOME", "FULL_STRUCTURE", "FA 18:1(9Z);12OH,13OH", "RM0224117"),
    ("9,10-dihome", "FULL_STRUCTURE", "FA 18:1(12Z);9OH,10OH", "RM0224117"),
    ("5-hete", "STRUCTURE_DEFINED", "FA 20:4;OH", "RM0224126"),
    ("12-HETE", "FULL_STRUCTURE", "FA 20:4(5Z,8Z,10E,14Z);12OH", "RM0224126"),
    ("15-HETE", "FULL_STRUCTURE", "FA 20:4(5Z,8Z,11Z,13Z);15OH", "RM0224126"),
    ("5-HEPE", "STRUCTURE_DEFINED", "FA 20:5;OH", "RM0224127"),
]


@pytest.fixture(scope="module")
def grammar() -> LipidGrammar:
    return LipidGrammar()


@pytest.mark.parametrize("name,asserted,first_query,goslin_id", _SINGLE_CHAIN)
def test_single_chain_oxylipin_keeps_asserted_level(grammar, name, asserted, first_query, goslin_id) -> None:
    binder = _RecordedRefMet()
    out = GoslinLipidAnnotator(grammar=grammar, binder=cast(Any, binder), trust_sn_position=False).get_annotations(
        {"name": name}, "name", _SM
    )
    meta = out["goslin-lipid"]["refmet_id"][goslin_id]
    # D1 no longer caps a single-chain parse: effective == asserted (it was molecular_species before).
    assert meta["query_lipid_level_asserted"] == asserted.lower()
    assert meta["query_lipid_level_effective"] == asserted.lower()
    assert binder.queries[0] == first_query
    # Recorded: RefMet answers none of Goslin's structure-level renderings, so the cascade still binds
    # the species-string class record at the SN_POSITION rendering. Correct ON ITS OWN only with (c).
    assert meta["matched_level"] == "sn_position"


@pytest.mark.parametrize(
    "name,expected_first_query",
    [
        ("PC 16:0/18:1", "PC 16:0_18:1"),
        ("PE P-16:0/18:1", "PE P-16:0_18:1"),
        ("TG 16:0/18:1/18:2", "TG 16:0_18:1_18:2"),
        ("Cer d18:1/16:0", "Cer 18:1;O2/16:0"),
        ("LPC 18:3/0:0", "LPC 18:3"),  # lyso with an sn placeholder: a regio claim, still capped
    ],
)
def test_multi_slot_slash_names_still_capped_to_molecular_species(grammar, name, expected_first_query) -> None:
    binder = _RecordedRefMet()
    annotator = GoslinLipidAnnotator(grammar=grammar, binder=cast(Any, binder), trust_sn_position=False)
    parsed = grammar.parse(name)
    assert parsed is not None
    asserted = annotator._asserted_level(parsed)
    assert annotator._effective_level(asserted, parsed) == "MOLECULAR_SPECIES"
    annotator.get_annotations({"name": name}, "name", _SM)
    assert binder.queries[0] == expected_first_query


@pytest.mark.parametrize("name", ["CAR 2:0", "LPC 16:0", "LPE P-16:0", "PC 34:1"])
def test_single_slot_or_coarse_names_are_never_capped(grammar, name) -> None:
    annotator = GoslinLipidAnnotator(grammar=grammar, binder=cast(Any, _RecordedRefMet()), trust_sn_position=False)
    parsed = grammar.parse(name)
    assert parsed is not None
    asserted = annotator._asserted_level(parsed)
    assert annotator._effective_level(asserted, parsed) == asserted


def test_trust_on_is_unchanged(grammar) -> None:
    annotator = GoslinLipidAnnotator(grammar=grammar, binder=cast(Any, _RecordedRefMet()), trust_sn_position=True)
    parsed = grammar.parse("PC 16:0/18:1")
    assert parsed is not None
    assert annotator._effective_level("SN_POSITION", parsed) == "SN_POSITION"


# ------------------------------------------------- (b) relation is measured against the ASSERTED level


def test_relation_against_asserted_level() -> None:
    assert lipid_mapping_relation("molecular_species", "sn_position") == ("broad", "skos:broadMatch")
    assert lipid_mapping_relation("molecular_species", "full_structure") == ("broad", "skos:broadMatch")
    assert lipid_mapping_relation("full_structure", "full_structure") == ("exact", "skos:exactMatch")
    assert lipid_mapping_relation(None, "full_structure") == ("unknown", None)


def test_capped_goslin_commit_reads_broad_and_hints_generalized() -> None:
    # The production 12,13-dihome row's goslin metadata (asserted full_structure, capped to molecular
    # species) committed to the class record: it reads broad now, not exact.
    names = {"SPECIES": "FA 18:1;O2", "CLASS": "FA"}
    row = _entity(
        refmet=None,
        kestrel_raw=None,
        goslin={"RM:0224117": _goslin_meta("full_structure", "molecular_species", "molecular_species", names)},
    )
    r, _, _ = _resolver()
    out = _resolve(r, row)
    assert out["chosen_kg_id"] == "RM:0224117"
    assert out["chosen_kg_id_lipid_hint"] == LIPID_GENERALIZED_HINT
    row["chosen_kg_id"] = out["chosen_kg_id"]
    lipid = build_lipid_resolution(row)
    assert lipid is not None
    assert (lipid["mapping_relation"], lipid["mapping_predicate"]) == ("broad", "skos:broadMatch")


# --------------------------------------------- (c) RefMet's finer raw-name record beats a generalized vote


@pytest.mark.parametrize(
    "refmet,goslin_refmet,goslin_kestrel,asserted,level_names",
    [
        # Production votes, recorded (pre-fix: goslin capped to molecular_species, its Kestrel vote at
        # species landed on a DIFFERENT specific isomer). Kestrel's raw-name search agreed with RefMet.
        ("RM:0138124", "RM:0224117", "RM:0139037", "full_structure", {"SPECIES": "FA 18:1;O2", "CLASS": "FA"}),
        ("RM:0074128", "RM:0224117", "RM:0139037", "full_structure", {"SPECIES": "FA 18:1;O2", "CLASS": "FA"}),
        ("RM:0158210", "RM:0224126", "RM:0153066", "structure_defined", {"SPECIES": "FA 20:4;O", "CLASS": "FA"}),
        ("RM:0152612", "RM:0224126", "RM:0153066", "full_structure", {"SPECIES": "FA 20:4;O", "CLASS": "FA"}),
        ("RM:0155791", "RM:0224127", "RM:0153119", "structure_defined", {"SPECIES": "FA 20:5;O", "CLASS": "FA"}),
    ],
)
@pytest.mark.parametrize("effective_is_capped", [True, False])
def test_oxylipins_keep_their_specific_refmet_entry(
    refmet, goslin_refmet, goslin_kestrel, asserted, level_names, effective_is_capped
) -> None:
    effective = "molecular_species" if effective_is_capped else asserted
    # Post-fix the goslin RefMet vote matches at sn_position (recorded RefMet behaviour); pre-fix it
    # matched at molecular_species. Either is coarser than the asserted level.
    goslin_level = "molecular_species" if effective_is_capped else "sn_position"
    row = _entity(
        refmet=refmet,
        kestrel_raw=refmet,
        goslin={
            goslin_refmet: _goslin_meta(asserted, effective, goslin_level, level_names),
            goslin_kestrel: _goslin_meta(asserted, effective, "species", level_names),
        },
    )
    r, _, connectivity_calls = _resolver(connectivity=None)
    out = _resolve(r, row)
    assert out["chosen_kg_id"] == refmet
    assert out["chosen_kg_id_review"] is None
    assert connectivity_calls == []  # decided by rule (c), never by the InChIKey test or a numeric coin
    row["chosen_kg_id"] = out["chosen_kg_id"]
    lipid = build_lipid_resolution(row)
    assert lipid is not None and lipid["mapping_relation"] == "unknown"  # no goslin vote backs RefMet's node


def test_hete_isomers_stay_distinct() -> None:
    chosen = {}
    for name, refmet in (("5-HETE", "RM:0158210"), ("12-HETE", "RM:0152612")):
        names = {"SPECIES": "FA 20:4;O", "CLASS": "FA"}
        row = _entity(
            refmet=refmet,
            kestrel_raw=refmet,
            goslin={"RM:0224126": _goslin_meta("full_structure", "full_structure", "sn_position", names)},
        )
        r, _, _ = _resolver()
        chosen[name] = _resolve(r, row)["chosen_kg_id"]
    assert chosen == {"5-HETE": "RM:0158210", "12-HETE": "RM:0152612"}


def test_annotator_plus_resolver_end_to_end_for_necs_spelling(grammar) -> None:
    # The real annotator (recorded RefMet answers) produces the goslin vote; the resolver then sees the
    # MW raw-name vote for 12,13-DiHOME, the goslin class-record vote, and a tie.
    goslin = GoslinLipidAnnotator(grammar=grammar, binder=cast(Any, _RecordedRefMet()), trust_sn_position=False)
    votes = goslin.get_annotations({"name": "12,13-dihome"}, "name", _SM)["goslin-lipid"]["refmet_id"]
    assert list(votes) == ["RM0224117"]
    row = _entity(refmet="RM:0138124", kestrel_raw=None, goslin={"RM:0224117": votes["RM0224117"]})
    r, _, _ = _resolver()
    out = _resolve(r, row)
    assert out["chosen_kg_id"] == "RM:0138124"
    assert out["chosen_kg_id_review"] is None
    assert out["chosen_kg_id_lipid_hint"] is None


def test_multi_chain_refmet_sn_record_beats_capped_molecular_species_vote() -> None:
    # D1 still caps the GOSLIN QUERY of "PC 16:0/18:1" to molecular species ("PC 16:0_18:1" ->
    # RM:0090134), but RefMet's own raw-name record "PC 16:0/18:1" is finer and must not be overridden.
    names = {"SPECIES": "PC 34:1", "CLASS": "PC"}
    row = _entity(
        refmet="RM:0015001",
        kestrel_raw=None,
        goslin={"RM:0090134": _goslin_meta("sn_position", "molecular_species", "molecular_species", names)},
    )
    r, _, _ = _resolver()
    out = _resolve(r, row)
    assert out["chosen_kg_id"] == "RM:0015001"


def test_goslin_vote_at_asserted_level_is_not_overridden() -> None:
    # Not a generalization (matched == asserted): rule (c) must stay out; existing logic decides.
    names = {"SPECIES": "PC 34:1", "CLASS": "PC"}
    row = _entity(
        refmet="RM:0015001",
        kestrel_raw=None,
        goslin={"RM:0090134": _goslin_meta("molecular_species", "molecular_species", "molecular_species", names)},
    )
    r, _, connectivity_calls = _resolver(connectivity=None)
    out = _resolve(r, row)
    assert out["chosen_kg_id"] == "RM:0090134"  # numeric tie-break among the two, as before
    assert out["chosen_kg_id_review"] == "conflict_no_structure"
    assert connectivity_calls == [("RM:0015001", "RM:0090134")]


# --------------------------------------------------------------------------------- (d) lipid class guard


def test_plasmalogen_pe_is_not_committed_to_a_dg_node() -> None:
    # The recorded production votes for 1-(1-enyl-palmitoyl)-2-oleoyl-GPE (P-16:0/18:1)*: RefMet's raw
    # name -> PE P-16:0/18:1 (RM:0134760), Kestrel hybrid search -> DG 16:0/18:1(9Z)/0:0 (RM:0133287). The
    # name does not parse in Goslin, a 1-1 tie, and the numeric tie-break picked the DG (its RM
    # local id is the lower one); neither node has a usable InChIKey pair, so the old path committed the DG.
    row = _entity(refmet="RM:0134760", kestrel_raw="RM:0133287")
    r, linker, connectivity_calls = _resolver(connectivity=None)
    out = _resolve(r, row)
    assert out["chosen_kg_id"] == "RM:0134760"
    assert out["chosen_kg_id_review"] == "divergent_refmet"
    assert connectivity_calls == []
    assert len(linker.calls) == 1


def test_class_guard_uses_query_class_when_goslin_parsed_and_refmet_absent() -> None:
    names = {"SPECIES": "PE O-34:2", "CLASS": "PE"}
    row = _entity(
        refmet=None,
        kestrel_raw="RM:0133287",
        # species-level so the Unit 4 tie-break does not already prefer it: the numeric fallback picks
        # the DG (lower RM local id) and only the class guard moves the commit back to the PE.
        goslin={"RM:0134760": _goslin_meta("sn_position", "molecular_species", "species", names)},
    )
    r, _, _ = _resolver()
    out = _resolve(r, row)
    assert out["chosen_kg_id"] == "RM:0134760"


def test_class_guard_refuses_when_every_candidate_is_the_wrong_class() -> None:
    names = {"SPECIES": "PE O-34:2", "CLASS": "PE"}
    row = _entity(refmet=None, kestrel_raw="RM:0133287", goslin=None)
    row["assigned_ids"][GOSLIN_LIPID_ANNOTATOR] = {
        "refmet_id": {"RM0161692": _goslin_meta("sn_position", "molecular_species", "species", names)}
    }
    row["kg_ids_assigned"][GOSLIN_LIPID_ANNOTATOR] = {"RM:0161692": ["RM:0161692"]}
    row["kg_ids"]["RM:0161692"] = ["RM:0161692"]
    r, _, _ = _resolver()
    out = _resolve(r, row)
    assert out["chosen_kg_id"] is None  # two DG nodes for a PE query: never commit a known class error
    assert out["chosen_kg_id_review"] is None


@pytest.mark.parametrize(
    "refmet,kestrel",
    [
        ("RM:0015001", "RM:0133287"),  # POPC: 1-palmitoyl-2-oleoyl-GPC (16:0/18:1) -> PC 16:0/18:1
        ("RM:0015038", "RM:0161692"),  # SOPE: 1-stearoyl-2-oleoyl-GPE (18:0/18:1) -> PE 18:0/18:1
    ],
)
def test_case_fix_glycerophospholipids_still_resolve_to_refmet_unflagged(refmet, kestrel) -> None:
    # #98's POPC/SOPE improvements (recorded votes): RefMet already wins these ties, so the guard never
    # runs (no node-name lookup) and the outcome is byte-identical.
    row = _entity(refmet=refmet, kestrel_raw=kestrel)
    r, linker, _ = _resolver()
    out = _resolve(r, row)
    assert (out["chosen_kg_id"], out["chosen_kg_id_review"]) == (refmet, None)
    assert linker.calls == []


def test_non_lipid_conflict_is_unchanged() -> None:
    # Names that do not parse as lipids carry no class: the guard is a no-op and the old
    # majority + conflict_no_structure outcome stands.
    row = pd.Series(
        {
            "kg_ids": {"CHEBI:100": ["x"], "CHEBI:200": ["y"]},
            "kg_ids_provided": {},
            "kg_ids_assigned": {REFMET_ANNOTATOR: {"CHEBI:200": ["y"]}, _KESTREL: {"CHEBI:100": ["x"]}},
            "assigned_ids": {},
        }
    )
    r, _, connectivity_calls = _resolver(connectivity=None)
    out = _resolve(r, row)
    assert (out["chosen_kg_id"], out["chosen_kg_id_review"]) == ("CHEBI:100", "conflict_no_structure")
    assert connectivity_calls == [("CHEBI:200", "CHEBI:100")]


def test_guard_degrades_to_noop_when_node_lookup_fails() -> None:
    class _Boom:
        def get_node_records(self, ids):
            raise RuntimeError("kestrel down")

    r = Resolver(linker=_Boom(), biolink_client=_Biolink())
    r._connectivity_match = lambda a, b: None  # type: ignore[method-assign]
    out = _resolve(r, _entity(refmet="RM:0134760", kestrel_raw="RM:0133287"))
    assert (out["chosen_kg_id"], out["chosen_kg_id_review"]) == ("RM:0133287", "conflict_no_structure")


# ------------------------------------------------- Kestrel ceiling + committed-node level for relations


class _RecordingKestrel:
    slug = _KESTREL

    def __init__(self) -> None:
        self.queries: list[str] = []

    def get_annotations(self, entity, name_field, category, prefixes=None, **kwargs):
        self.queries.append(entity.get(name_field))
        return {self.slug: {}}


@pytest.mark.parametrize("name", ["12,13-dihome", "12-HETE", "5-HEPE"])
def test_kestrel_never_queried_at_structure_level_renderings(grammar, name) -> None:
    # Recorded 2026-11-05: Kestrel hybrid search answered the FULL_STRUCTURE renderings of 12,13-DiHOME,
    # 9,10-DiHOME and 12-HETE with one and the same unrelated HMDB node. Kestrel keeps the level-only cap.
    kestrel = _RecordingKestrel()
    GoslinLipidAnnotator(
        grammar=grammar, binder=cast(Any, _RecordedRefMet()), kestrel=cast(Any, kestrel), trust_sn_position=False
    ).get_annotations({"name": name}, "name", _SM)
    parsed = grammar.parse(name) or grammar.parse(name.upper().replace("DIHOME", "DiHOME"))
    assert parsed is not None
    finer = {parsed.level_names.get(lvl) for lvl in ("STRUCTURE_DEFINED", "FULL_STRUCTURE", "COMPLETE_STRUCTURE")}
    assert kestrel.queries, "Kestrel cascade should still run at molecular-species/species"
    assert not (set(kestrel.queries) & (finer - {None}))


def test_relation_reads_the_committed_nodes_own_level() -> None:
    # sphinganine: RefMet answers the species rendering "SPB 18:0;O2" with its full-structure
    # "sphinganine" record (recorded), so the goslin matched level (sn_position) under-states the node.
    names = {"SPECIES": "SPB 18:0;O2", "CLASS": "SPB"}
    row = _entity(
        refmet="RM:0135445",
        kestrel_raw="RM:0135445",
        goslin={"RM:0135445": _goslin_meta("full_structure", "full_structure", "sn_position", names)},
    )
    r, _, _ = _resolver()
    out = _resolve(r, row)
    assert out["chosen_kg_id"] == "RM:0135445"
    assert out["chosen_kg_id_lipid_hint"] is None
    row["chosen_kg_id"] = out["chosen_kg_id"]
    level = r.node_lipid_readings(["RM:0135445"])["RM:0135445"][1]
    assert level == "full_structure"
    lipid = build_lipid_resolution(row, node_level=level)
    assert lipid is not None and lipid["mapping_relation"] == "exact"
    # Without the node's own level the matched level alone would read broad.
    lipid_no_node = build_lipid_resolution(row)
    assert lipid_no_node is not None and lipid_no_node["mapping_relation"] == "broad"


def test_refmet_committed_oxylipin_reads_exact_via_its_node_name() -> None:
    names = {"SPECIES": "FA 18:1;O2", "CLASS": "FA"}
    row = _entity(
        refmet="RM:0138124",
        kestrel_raw=None,
        goslin={"RM:0224117": _goslin_meta("full_structure", "full_structure", "sn_position", names)},
    )
    r, _, _ = _resolver()
    out = _resolve(r, row)
    assert out["chosen_kg_id"] == "RM:0138124"
    row["chosen_kg_id"] = out["chosen_kg_id"]
    level = r.node_lipid_readings(["RM:0138124"])["RM:0138124"][1]
    lipid = build_lipid_resolution(row, node_level=level)
    assert lipid is not None
    assert (lipid["mapping_relation"], lipid["mapping_predicate"]) == ("exact", "skos:exactMatch")


def test_committed_level_is_the_more_specific_of_match_and_node_name() -> None:
    from biomapper2.core.resolver import committed_lipid_level

    assert committed_lipid_level("sn_position", "full_structure") == "full_structure"  # sphinganine
    assert committed_lipid_level("full_structure", "structure_defined") == "full_structure"  # 5-KETE
    assert committed_lipid_level("sn_position", None) == "sn_position"  # FA 18:1;O2 (name does not parse)
    assert committed_lipid_level(None, None) is None
