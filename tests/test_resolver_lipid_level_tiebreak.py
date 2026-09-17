"""Level-aware lipid tie-break tests for the resolver (Unit 4, no live services).

The resolver's deterministic majority tie-break was level-blind: on a genuine tie the lower-numeric
local id won, which is arbitrary with respect to lipid specificity and could commit a broader node
over an exact-level one with no flag. Unit 4 makes the tie-break prefer, among tied lipid candidates
with known levels, the node whose matched level equals the EFFECTIVE query level, then falls back to
today's namespace/numeric order. When the committed lipid node is broader than the effective query
level, a new additive ``lipid_generalized`` hint is set on its own field, never on the closed
``selection_conflict`` whitelist. Non-lipid ties are unchanged.

All level metadata is injected as fixtures shaped like the goslin-lipid annotator's real output
(``assigned_ids`` for the levels, ``kg_ids_assigned`` for the linked nodes joined on the local id).
"""

from unittest.mock import MagicMock

import pandas as pd

from biomapper2.core.resolver import LIPID_GENERALIZED_HINT, Resolver, _lipid_level_context

# Worked-example shape: an sn/molecular-species node and a species node, one supporting curie each
# (a genuine count tie). The exact-level node carries the HIGHER numeric local id, so a pick that
# lands on it proves the level preference overrode the numeric fallback (which prefers the lower one).
_EXACT_NODE = "RM:0015001"  # matched at molecular_species == effective query level
_SPECIES_NODE = "RM:0010728"  # matched at species (broader)
_EFFECTIVE = "molecular_species"

_KG_IDS_TIE = {_EXACT_NODE: [_EXACT_NODE], _SPECIES_NODE: [_SPECIES_NODE]}
_LEVELS_TIE = {_EXACT_NODE: "molecular_species", _SPECIES_NODE: "species"}


def _resolver() -> Resolver:
    """Resolver with mocked deps; the tie-break under test is pure over its dict/level args."""
    return Resolver(linker=MagicMock(), biolink_client=MagicMock())


def _goslin_assigned_ids(levels_by_local: dict[str, str], effective: str) -> dict:
    """Fixture shaped like GoslinLipidAnnotator output: {slug: {vocab: {local_id: metadata}}}."""
    votes = {
        local: {"matched_level": level, "query_lipid_level_effective": effective}
        for local, level in levels_by_local.items()
    }
    return {"goslin-lipid": {"REFMET": votes}}


def _goslin_kg_ids_assigned(nodes: list[str]) -> dict:
    """Fixture shaped like the linker output: {slug: {kg_id: [supporting curies]}}."""
    return {"goslin-lipid": {node: [node] for node in nodes}}


def test_exact_level_node_wins_lipid_tie_over_numeric_order() -> None:
    # The exact-level node has the higher numeric local id; the level preference must still pick it.
    chosen = _resolver()._stable_majority(_KG_IDS_TIE, None, _LEVELS_TIE, _EFFECTIVE)
    assert chosen == _EXACT_NODE


def test_no_exact_level_falls_back_to_numeric_unchanged() -> None:
    # Both tied nodes are broader than the effective level -> no exact match -> today's numeric pick.
    both_species = {_EXACT_NODE: "species", _SPECIES_NODE: "species"}
    chosen = _resolver()._stable_majority(_KG_IDS_TIE, None, both_species, "sn_position")
    assert chosen == _SPECIES_NODE  # lower numeric local id, the pre-Unit-4 behavior


def test_non_lipid_tie_byte_identical_without_and_with_empty_levels() -> None:
    # A non-lipid tie (no level context) resolves exactly as before, and passing empty lipid args is
    # the same code path as omitting them.
    r = _resolver()
    baseline = r._stable_majority(_KG_IDS_TIE, None)
    with_empty = r._stable_majority(_KG_IDS_TIE, None, {}, None)
    assert baseline == with_empty == _SPECIES_NODE


def test_resolve_exact_level_end_to_end_sets_no_hint() -> None:
    # Full resolve() over the worked-example fixture: the exact-level node is committed and, because it
    # matches the effective level (not broader), no generalization hint is set.
    r = _resolver()
    entity = pd.Series(
        {
            "kg_ids": _KG_IDS_TIE,
            "kg_ids_provided": {},
            "kg_ids_assigned": _goslin_kg_ids_assigned([_EXACT_NODE, _SPECIES_NODE]),
            # RAW refmet_id keys ("RM0015001"), the form goslin actually stamps -- NOT the numeric
            # local part. The curie is "RM:0015001", so a plain local-part join would miss these and
            # lose the level; the raw-id reconciliation is what makes this pass.
            "assigned_ids": _goslin_assigned_ids(
                {"RM0015001": "molecular_species", "RM0010728": "species"}, _EFFECTIVE
            ),
        }
    )
    out = r.resolve(entity)
    assert isinstance(out, pd.Series)
    assert out["chosen_kg_id"] == _EXACT_NODE
    assert out["chosen_kg_id_lipid_hint"] is None


def test_resolve_species_only_sets_generalized_hint() -> None:
    # A more-specific query but only species-level nodes are available: species is committed AND the
    # generalization hint fires because the committed node is broader than the effective query level.
    r = _resolver()
    entity = pd.Series(
        {
            "kg_ids": _KG_IDS_TIE,
            "kg_ids_provided": {},
            "kg_ids_assigned": _goslin_kg_ids_assigned([_EXACT_NODE, _SPECIES_NODE]),
            "assigned_ids": _goslin_assigned_ids({"RM0015001": "species", "RM0010728": "species"}, _EFFECTIVE),
        }
    )
    out = r.resolve(entity)
    assert isinstance(out, pd.Series)
    assert out["chosen_kg_id"] == _SPECIES_NODE  # numeric fallback among two broader nodes
    assert out["chosen_kg_id_lipid_hint"] == LIPID_GENERALIZED_HINT


def test_resolve_non_lipid_row_has_null_hint_and_unchanged_pick() -> None:
    # Regression: a non-lipid tie (no goslin votes) commits the lower-numeric node with no hint.
    r = _resolver()
    kg_ids = {"CHEBI:200": ["x"], "CHEBI:100": ["y"]}
    entity = pd.Series({"kg_ids": kg_ids, "kg_ids_provided": {}, "kg_ids_assigned": {}})
    out = r.resolve(entity)
    assert isinstance(out, pd.Series)
    assert out["chosen_kg_id"] == "CHEBI:100"
    assert out["chosen_kg_id_lipid_hint"] is None
    assert out["chosen_kg_id_review"] is None


def test_lipid_level_context_joins_and_takes_most_specific() -> None:
    # The helper joins goslin metadata (assigned_ids) to linked nodes (kg_ids_assigned) on the local
    # id, and a node backed by two goslin votes inherits the MORE specific level.
    entity = {
        "assigned_ids": _goslin_assigned_ids({"0015001": "molecular_species", "0010728": "species"}, _EFFECTIVE),
        "kg_ids_assigned": {"goslin-lipid": {"RM:node": ["RM:0015001", "RM:0010728"]}},
    }
    levels, effective = _lipid_level_context(entity)
    assert effective == _EFFECTIVE
    assert levels == {"RM:node": "molecular_species"}
