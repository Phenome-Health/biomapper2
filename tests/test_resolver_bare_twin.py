"""Bare RM twin: the resolver must not commit a structure-less RefMet node over its annotated twin.

Shapes are the live votes captured 2026-11-11 (prod v1.4.0, KG 2.3.0) for four queries. No live APIs.
"""

from unittest.mock import MagicMock

from biomapper2.core.resolver import Resolver

SM = "biolink:SmallMolecule"
MW = "metabolomics-workbench"
KESTREL = "kestrel-hybrid-search"
GOSLIN = "goslin-lipid"


def _resolver(conn_result, structured: set[str]):
    """Resolver with a fixed connectivity answer; ``structured`` = nodes whose KG record has an InChIKey."""
    r = Resolver(linker=MagicMock(), biolink_client=MagicMock())
    r.structure_resolver = MagicMock()
    r.structure_resolver.connectivity_match.return_value = conn_result
    r.structure_resolver.kg_structured_nodes.side_effect = lambda nodes: {n for n in nodes if n in structured}
    r._is_small_molecule = lambda category: category == SM
    r._preferred_prefixes = lambda category: {"RM", "CHEBI", "HMDB"}
    return r


# --- same molecule, RefMet's node is bare (4-pyridoxic acid, IPA, trans-4-hydroxy-L-proline) ----------


def test_same_molecule_prefers_the_structured_twin_over_bare_refmet_node():
    kg_ids = {"RM:0005375": ["RM:0005375"], "RM:0030084": ["RM:0030084"]}
    assigned = {MW: {"RM:0030084": ["RM:0030084"]}, KESTREL: {"RM:0005375": ["RM:0005375"]}}
    r = _resolver(True, structured={"RM:0005375"})
    assert r._choose_best_kg_id(kg_ids, assigned, SM) == ("RM:0005375", None)


def test_same_molecule_still_prefers_refmet_when_refmet_node_is_structured():
    kg_ids = {"CHEBI:bmp": ["a", "b"], "CHEBI:refmet": ["RM:1"]}
    assigned = {MW: {"CHEBI:refmet": ["RM:1"]}}
    r = _resolver(True, structured={"CHEBI:bmp", "CHEBI:refmet"})
    assert r._choose_best_kg_id(kg_ids, assigned, SM) == ("CHEBI:refmet", None)


def test_same_molecule_keeps_refmet_when_neither_node_is_structured():
    kg_ids = {"RM:0005375": ["RM:0005375"], "RM:0030084": ["RM:0030084"]}
    assigned = {MW: {"RM:0030084": ["RM:0030084"]}, KESTREL: {"RM:0005375": ["RM:0005375"]}}
    r = _resolver(True, structured=set())
    assert r._choose_best_kg_id(kg_ids, assigned, SM) == ("RM:0030084", None)


def test_different_molecule_still_flips_to_refmet_with_flag_even_if_bare():
    kg_ids = {"RM:0005375": ["RM:0005375"], "RM:0030084": ["RM:0030084"]}
    assigned = {MW: {"RM:0030084": ["RM:0030084"]}, KESTREL: {"RM:0005375": ["RM:0005375"]}}
    r = _resolver(False, structured={"RM:0005375"})
    assert r._choose_best_kg_id(kg_ids, assigned, SM) == ("RM:0030084", "divergent_refmet")


# --- count tie broken by annotator agreement (cholesterol) -------------------------------------------


CHOL_KG_IDS = {"RM:0001746": ["RM:0001746"], "RM:0007253": ["RM:0007253"], "RM:0135639": ["RM:0135639"]}
CHOL_ASSIGNED = {
    MW: {"RM:0135639": ["RM:0135639"]},
    GOSLIN: {"RM:0007253": ["RM:0007253"], "RM:0001746": ["RM:0001746"]},
    KESTREL: {"RM:0135639": ["RM:0135639"]},
}


def test_count_tie_goes_to_the_node_two_annotators_agree_on():
    r = _resolver(None, structured={"RM:0135639"})
    r.node_lipid_readings = lambda ids: {i: (None, None) for i in ids}
    chosen, flag = r._choose_best_kg_id(
        CHOL_KG_IDS,
        CHOL_ASSIGNED,
        SM,
        lipid_levels={"RM:0007253": "species", "RM:0001746": "class"},
        effective_level="species",
    )
    assert (chosen, flag) == ("RM:0135639", None)


def test_agreement_tie_break_beats_lipid_level_and_numeric_order():
    r = _resolver(None, structured=set())
    picked = r._stable_majority(CHOL_KG_IDS, SM, {"RM:0007253": "species"}, "species", kg_ids_assigned=CHOL_ASSIGNED)
    assert picked == "RM:0135639"


def test_equal_agreement_falls_back_to_existing_tie_break():
    r = _resolver(None, structured=set())
    kg_ids = {"RM:2": ["RM:2"], "RM:1": ["RM:1"]}
    assigned = {MW: {"RM:2": ["RM:2"]}, KESTREL: {"RM:1": ["RM:1"]}}
    assert r._stable_majority(kg_ids, SM, kg_ids_assigned=assigned) == "RM:1"


def test_strict_count_winner_ignores_agreement():
    r = _resolver(None, structured=set())
    kg_ids = {"RM:1": ["a", "b"], "RM:2": ["c"]}
    assigned = {MW: {"RM:2": ["c"]}, KESTREL: {"RM:2": ["c"]}, GOSLIN: {"RM:1": ["a", "b"]}}
    assert r._stable_majority(kg_ids, SM, kg_ids_assigned=assigned) == "RM:1"


def test_structure_check_reuses_the_connectivity_read():
    """One node-record read, same order as connectivity_match, so the HTTP cache serves it."""
    from biomapper2.core.structure_resolver import StructureResolver

    linker = MagicMock()
    linker.get_node_records.return_value = {
        "RM:0005375": {"equivalent_ids": {"INCHIKEY": ["HXACOUQIXZGNBF-UHFFFAOYSA-N"]}},
        "RM:0030084": {"equivalent_ids": {"RM": ["RM:0030084"]}},
    }
    sr = StructureResolver.__new__(StructureResolver)
    sr.linker = linker
    assert sr.kg_structured_nodes(["RM:0030084", "RM:0005375"]) == {"RM:0005375"}
    linker.get_node_records.assert_called_once_with(["RM:0030084", "RM:0005375"])
