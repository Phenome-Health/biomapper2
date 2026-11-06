"""Metabolon-style lipid names take the Goslin lipid path from their chain parenthetical.

Offline. RefMet answers are the live ``/match`` responses recorded 2026-11-06 for exactly the queries this
route builds from the NECS / Xu / Arivale panels (tests/fixtures/refmet_match_metabolon_lipid_route.json).

Before this route, a name such as ``sphingomyelin (d18:1/17:0, d17:1/18:0, d19:1/16:0)`` never parsed with
Goslin as a whole, RefMet ``/match`` did not answer the vendor spelling, and Kestrel's text search put 8
distinct NECS sphingomyelins on one generic node (HMDB:HMDB0013468) and 5 distinct ceramides on one
``Cer 18:1;O2/18:0`` node (RM:0135486), with no review flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from biomapper2.core.annotators.goslin_grammar import LipidGrammar
from biomapper2.core.annotators.goslin_lipid import GoslinLipidAnnotator
from biomapper2.core.annotators.metabolon_lipid_name import (
    ParentheticalLipidQuery,
    name_is_mixture,
    parenthetical_lipid_query,
)
from biomapper2.core.resolver import Resolver, build_lipid_resolution, lipid_flat_columns

_FIXTURE = Path(__file__).parent / "fixtures" / "refmet_match_metabolon_lipid_route.json"
_REFMET = {q: a["refmet_id"] for q, a in json.loads(_FIXTURE.read_text())["answers"].items()}

# The 8 NECS sphingomyelins and 5 ceramides that collapsed on engine v1.2.1, with the species each one is.
NECS_COLLAPSED_SM = {
    "sphingomyelin (d18:1/20:2, d18:2/20:1, d16:1/22:2)*": "SM 38:3;O2",
    "sphingomyelin (d18:1/22:1, d18:2/22:0, d16:1/24:1)*": "SM 40:2;O2",
    "sphingomyelin (d18:1/22:2, d18:2/22:1, d16:1/24:2)*": "SM 40:3;O2",
    "sphingomyelin (d17:1/16:0, d18:1/15:0, d16:1/17:0)*": "SM 33:1;O2",
    "sphingomyelin (d18:1/17:0, d17:1/18:0, d19:1/16:0)": "SM 35:1;O2",
    "sphingomyelin (d18:2/23:0, d18:1/23:1, d17:1/24:1)*": "SM 41:2;O2",
    "sphingomyelin (d18:1/21:0, d17:1/22:0, d16:1/23:0)*": "SM 39:1;O2",
    "sphingomyelin (d18:1/25:0, d19:0/24:1, d20:1/23:0, d19:1/24:0)*": "SM 43:1;O2",
}
NECS_COLLAPSED_CER = {
    "ceramide (d16:1/24:1, d18:1/22:1)*": "Cer 40:2;O2",
    "ceramide (d18:1/14:0, d16:1/16:0)*": "Cer 32:1;O2",
    "ceramide (d18:1/17:0, d17:1/18:0)*": "Cer 35:1;O2",
    "ceramide (d18:2/24:1, d18:1/24:2)*": "Cer 42:3;O2",
    "ceramide (d18:1/20:0, d16:1/22:0, d20:1/18:0)*": "Cer 38:1;O2",
}


@pytest.fixture(scope="module")
def grammar() -> LipidGrammar:
    return LipidGrammar()


class _RecordedRefMet:
    """RefMet /match binder replaying the recorded answers. Records every name it was asked."""

    slug = "metabolomics-workbench"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def get_annotations(self, entity, name_field, category, prefixes=None, **kwargs):
        name = entity.get(name_field)
        self.seen.append(name)
        refmet_id = _REFMET.get(name)
        return {self.slug: {"refmet_id": {refmet_id: {}}}} if refmet_id else {}


def _annotate(name: str, grammar: LipidGrammar) -> tuple[dict, _RecordedRefMet]:
    binder = _RecordedRefMet()
    ann = GoslinLipidAnnotator(grammar=grammar, binder=binder)  # pyright: ignore[reportArgumentType]
    return ann.get_annotations({"name": name}, name_field="name", category="biolink:SmallMolecule"), binder


# ------------------------------------------------ helper rules ------------------------------------------------ #


@pytest.mark.parametrize("name,species", sorted({**NECS_COLLAPSED_SM, **NECS_COLLAPSED_CER}.items()))
def test_isomer_lists_resolve_at_their_shared_species(grammar, name, species):
    route = parenthetical_lipid_query(name, grammar)
    assert route is not None
    assert route.species_level is True
    assert route.parse.level_names["SPECIES"] == species
    assert route.parse.level == "LipidLevel.SPECIES"
    assert route.parse.chains == ()  # never one listed isomer
    assert set(route.parse.level_names) <= {"CATEGORY", "CLASS", "SPECIES"}


def _route(name: str, grammar: LipidGrammar) -> ParentheticalLipidQuery:
    route = parenthetical_lipid_query(name, grammar)
    assert route is not None, name
    return route


def test_the_collapsed_names_now_have_distinct_species(grammar):
    species = [_route(n, grammar).parse.level_names["SPECIES"] for n in NECS_COLLAPSED_SM]
    assert len(set(species)) == len(NECS_COLLAPSED_SM)
    species = [_route(n, grammar).parse.level_names["SPECIES"] for n in NECS_COLLAPSED_CER]
    assert len(set(species)) == len(NECS_COLLAPSED_CER)


@pytest.mark.parametrize(
    "name,expected,species_level",
    [
        ("dihomo-linolenate (20:3n3 or n6)", "FA 20:3", False),
        ("dihomolinolenate (20:3n3 or 3n6)", "FA 20:3", False),
        ("linolenate [alpha or gamma; (18:3n3 or 6)]", "FA 18:3", False),
        ("butyrate/isobutyrate (4:0)", "FA 4:0", False),
        ("oleate/vaccenate (18:1)", "FA 18:1", False),
        ("dihomo-linolenoylcarnitine (C20:3n3 or 6)*", "CAR 20:3", False),
        ("1-dihomo-linolenoyl-GPC (20:3n3 or 6)*", "LPC 20:3", False),
        ("1-stearoyl-2-dihomo-linolenoyl-GPI (18:0/20:3n3 or 6)*", "PI 38:3", False),
        ("diacylglycerol (14:0/18:1, 16:0/16:1) [1]*", "DG 32:1", True),
        ("diacylglycerol (16:1/18:2 [2], 16:0/18:3 [1])*", "DG 34:3", True),
        ("phosphatidylcholine (16:0/22:5n3, 18:1/20:4)*", "PC 38:5", True),
        ("glycosyl ceramide (d18:2/24:1, d18:1/24:2)*", "HexCer 42:3;O2", True),
    ],
)
def test_mixture_labels_resolve_at_the_level_their_alternatives_share(grammar, name, expected, species_level):
    route = parenthetical_lipid_query(name, grammar)
    assert route is not None
    assert route.parse.level_names["SPECIES"] == expected
    assert route.species_level is species_level


def test_lyso_acyl_position_is_kept_as_an_empty_slot(grammar):
    # "1-" names the sn-1 acyl: kept as LPC 20:3/0:0 (RefMet's record for the vendor name), so the
    # sn-position trust policy and RefMet precedence see the same claim as for written-out shorthand.
    assert _route("1-dihomo-linolenoyl-GPC (20:3n3 or 6)*", grammar).shorthand == "LPC 20:3/0:0"
    assert _route("2-dihomo-linolenoyl-GPC (20:3n3 or 6)*", grammar).shorthand == "LPC 0:0/20:3"
    out, binder = _annotate("1-dihomo-linolenoyl-GPE (20:3n3 or 6)*", grammar)
    assert binder.seen[0] == "LPE 20:3"  # trust-off cap to molecular species, as for "LPE 20:3/0:0"
    meta = out["goslin-lipid"]["refmet_id"]["RM0012621"]
    assert meta["query_lipid_level_asserted"] == "sn_position"


@pytest.mark.parametrize(
    "name",
    [
        # R1: single-isomer prose names are not routed; RefMet answers them at their asserted level today.
        "palmitate (16:0)",
        "stearoyl sphingomyelin (d18:1/18:0)",
        "1-palmitoyl-gpi (16:0)",
        "1-palmitoyl-2-oleoyl-GPC (16:0/18:1)",
        # R4: branched (anteiso / iso) chains would merge into the straight-chain species.
        "(14 or 15)-methylpalmitate (a17:0 or i17:0)",
        # R2: a nested functional-group parenthetical, or text after the parenthetical, is refused.
        "sphingomyelin (d18:1/16:0(OH), d18:1/16:0)",
        "sphingomyelin (d18:1/14:0, d16:1/16:0) or something",
        # R3: an unsupported head.
        "branched-chain, straight-chain, or cyclopropyl 10:1 fatty acid (1)*",
        "mannitol/sorbitol",
        "13-HODE + 9-HODE",
        # R5: listed alternatives with DIFFERENT sum compositions share no species.
        "sphingomyelin (d18:1/16:0, d18:1/18:0)",
        # R4: a sphingolipid chain without its long-chain-base prefix.
        "ceramide (18:1/14:0, 16:1/16:0)",
    ],
)
def test_refused_names_are_not_routed(grammar, name):
    assert parenthetical_lipid_query(name, grammar) is None


@pytest.mark.parametrize(
    "name",
    [
        "dihomo-linolenate (20:3n3 or n6)",
        "butyrate/isobutyrate (4:0)",
        "13-HODE + 9-HODE",
        "aconitate [cis or trans]",
        "mannitol/sorbitol",
        "sphingomyelin (d18:1/14:0, d16:1/16:0)*",
        "dimethylarginine (sdma + adma)",
    ],
)
def test_mixture_detected(name):
    assert name_is_mixture(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "12,13-DiHOME",
        "PC 16:0/18:1",
        "PE 18:0/18:1",
        "SM 18:1;O2/18:0",
        "palmitate (16:0)",
        "stearoyl sphingomyelin (d18:1/18:0)",
        "1-palmitoyl-2-oleoyl-GPC (16:0/18:1)",
        "1-(1-enyl-palmitoyl)-2-oleoyl-GPC (P-16:0/18:1)*",
        "androstenediol (3beta,17beta) disulfate (1)",
        "(+)-alpha-tocopherol",
        "(+/-)-12-HETE",
        "glucose",
        "",
        None,
    ],
)
def test_not_a_mixture(name):
    assert name_is_mixture(name) is False


# ------------------------------------------------ annotator ------------------------------------------------ #


def test_collapsed_sphingomyelins_and_ceramides_bind_to_distinct_refmet_species(grammar):
    bound = {}
    for name, species in {**NECS_COLLAPSED_SM, **NECS_COLLAPSED_CER}.items():
        out, binder = _annotate(name, grammar)
        assert binder.seen == [species]  # one species query, never a listed isomer
        votes = out["goslin-lipid"]["refmet_id"]
        (refmet_id,) = votes
        meta = votes[refmet_id]
        assert meta["matched_level"] == "species"
        assert meta["query_lipid_level_asserted"] == "species"
        assert meta["name_is_mixture"] is True
        assert meta["query_species_from_parenthetical"] is True
        bound[name] = refmet_id
    # 13 names, 13 distinct RefMet records (v1.2.1 put them on 2 nodes).
    assert len(set(bound.values())) == len(bound) == 13
    assert "RM0135486" not in bound.values()


def test_both_spellings_of_dihomo_linolenate_query_fa_20_3(grammar):
    necs, necs_binder = _annotate("dihomo-linolenate (20:3n3 or n6)", grammar)
    xu, xu_binder = _annotate("dihomolinolenate (20:3n3 or 3n6)", grammar)
    assert necs_binder.seen[0] == xu_binder.seen[0] == "FA 20:3"
    assert set(necs["goslin-lipid"]["refmet_id"]) == set(xu["goslin-lipid"]["refmet_id"]) == {"RM0152308"}


def test_lower_case_necs_spelling_routes_like_the_metabolon_spelling(grammar):
    lower, _ = _annotate("dihomo-linolenoylcarnitine (c20:3n3 or 6)*", grammar)
    upper, _ = _annotate("dihomo-linolenoylcarnitine (C20:3n3 or 6)*", grammar)
    assert lower == upper
    assert set(lower["goslin-lipid"]["refmet_id"]) == {"RM0154160"}  # CAR 20:3


@pytest.mark.parametrize(
    "name,first_query",
    [
        ("PC 16:0/18:1", "PC 16:0_18:1"),  # POPC: trust-off cap to molecular species, as before
        ("PE 18:0/18:1", "PE 18:0_18:1"),  # SOPE
        ("12,13-dihome", "FA 18:1(9Z);12OH,13OH"),  # oxylipin via the case fix, single chain keeps its level
    ],
)
def test_names_goslin_parses_whole_never_take_the_route(grammar, name, first_query):
    out, binder = _annotate(name, grammar)
    assert binder.seen[0] == first_query
    for votes in out.get("goslin-lipid", {}).values():
        for meta in votes.values():
            assert meta["query_species_from_parenthetical"] is False
            assert meta["name_is_mixture"] is False


@pytest.mark.parametrize("name", ["palmitate (16:0)", "stearoyl sphingomyelin (d18:1/18:0)", "glucose"])
def test_unrouted_names_still_fall_through_without_a_query(grammar, name):
    out, binder = _annotate(name, grammar)
    assert out == {}
    assert binder.seen == []


@pytest.mark.parametrize("name", ["EDTA", "edta", " EDTA "])
def test_edta_is_not_read_as_an_epoxy_fatty_acid(grammar, name):
    assert grammar.parse(name) is None
    out, binder = _annotate(name, grammar)
    assert out == {} and binder.seen == []


# ---------------------------------------------- resolution + output ---------------------------------------------- #


def _entity_with_kestrel_generic_vote(name: str, grammar: LipidGrammar) -> dict:
    """A resolver input shaped like the pipeline output: Kestrel's generic sphingomyelin node vs the
    goslin species node, one supporting curie each (the v1.2.1 tie that Kestrel won by namespace order)."""
    out, _ = _annotate(name, grammar)
    (raw,) = out["goslin-lipid"]["refmet_id"]
    goslin_node = f"RM:{raw[2:]}"
    kestrel_node = "HMDB:HMDB0013468"
    return {
        "name": name,
        "assigned_ids": {**out, "kestrel-hybrid-search": {"HMDB": {"HMDB0013468": {}}}},
        "kg_ids": {kestrel_node: [kestrel_node], goslin_node: [goslin_node]},
        "kg_ids_provided": {},
        "kg_ids_assigned": {
            "kestrel-hybrid-search": {kestrel_node: [kestrel_node]},
            "goslin-lipid": {goslin_node: [goslin_node]},
        },
    }


def test_resolver_commits_the_species_node_not_the_generic_sphingomyelin(grammar):
    resolver = Resolver()
    chosen = {}
    for name in NECS_COLLAPSED_SM:
        entity = _entity_with_kestrel_generic_vote(name, grammar)
        row = resolver.resolve(entity)
        assert isinstance(row, pd.Series)
        chosen[name] = row["chosen_kg_id"]
        assert row["chosen_kg_id"] != "HMDB:HMDB0013468"
        assert row["chosen_kg_id_review"] is None  # the closed whitelist is untouched
    assert len(set(chosen.values())) == len(NECS_COLLAPSED_SM)


def test_lipid_resolution_exposes_the_review_signals(grammar):
    name = "sphingomyelin (d18:1/17:0, d17:1/18:0, d19:1/16:0)"
    entity = _entity_with_kestrel_generic_vote(name, grammar)
    entity["chosen_kg_id"] = "RM:0059990"
    lr = build_lipid_resolution(entity)
    assert lr is not None
    assert lr["name_is_mixture"] is True
    assert lr["query_species_from_parenthetical"] is True
    assert lr["query_lipid_level_asserted"] == "species"
    assert lr["matched_lipid_level"] == "species"
    assert lr["mapping_relation"] == "exact"
    flat = lipid_flat_columns(lr)
    assert flat["lipid_name_is_mixture"] is True
    assert flat["lipid_query_species_from_parenthetical"] is True
    # The existing columns are all still there.
    assert {"lipid_mapping_relation", "lipid_goslin_formula", "lipid_ambiguous"} <= set(flat)


def test_lipid_resolution_from_votes_without_the_new_keys_reads_false():
    entity = {
        "chosen_kg_id": "RM:0015001",
        "assigned_ids": {
            "goslin-lipid": {
                "refmet_id": {"RM0015001": {"query_lipid_level_asserted": "species", "matched_level": "species"}}
            }
        },
        "kg_ids_assigned": {"goslin-lipid": {"RM:0015001": ["RM:0015001"]}},
    }
    lr = build_lipid_resolution(entity)
    assert lr is not None
    assert lr["name_is_mixture"] is False
    assert lr["query_species_from_parenthetical"] is False
