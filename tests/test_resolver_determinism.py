"""Determinism regression tests for Resolver._choose_best_kg_id majority tie-break.

The default majority vote (max() over kg_ids_dict) had no stable tie-break, so a count tie
resolved by dict-iteration order = Kestrel API-response order = batch/process history. These tests
pin the tie-break to a deterministic total order and assert a genuine tie is logged (the coin-flip
stays visible in run logs). No network: get_descendants is stubbed and _choose_best_kg_id is pure
over its dict/category args here.
"""

import logging
from unittest.mock import MagicMock

from biomapper2.core.resolver import REFMET_ANNOTATOR, Resolver

# Biolink descendant sets the resolver consults: SmallMolecule's subtree includes Drug so a descendant
# category inherits the canonical-namespace policy; Disease is disjoint.
_DESCENDANTS = {
    "biolink:SmallMolecule": {"biolink:SmallMolecule", "biolink:Drug"},
    "biolink:Disease": {"biolink:Disease"},
}


def _resolver(descendants: dict[str, set[str]] | None = None) -> Resolver:
    r = Resolver(linker=MagicMock(), biolink_client=MagicMock())
    mapping = descendants or {}
    r.biolink_client.get_descendants.side_effect = lambda c: mapping.get(c, set())
    return r


def test_count_tie_is_order_independent() -> None:
    # T1 (RED before fix): two candidates tie on supporting-curie count. Building the dict in
    # opposite insertion orders simulates two Kestrel response orders across a batch.
    r = _resolver()
    ab, _ = r._choose_best_kg_id({"CHEBI:200": ["x"], "CHEBI:100": ["y"]})
    ba, _ = r._choose_best_kg_id({"CHEBI:100": ["y"], "CHEBI:200": ["x"]})
    assert ab == ba


def test_clear_majority_unchanged_and_unflagged(caplog) -> None:
    # T2 (R3 no-op): a strict count winner is returned regardless of order, with no review flag and
    # no tie warning (a clear majority is not a coin-flip).
    r = _resolver()
    with caplog.at_level(logging.WARNING):
        for d in ({"CHEBI:1": ["a", "b"], "CHEBI:2": ["c"]}, {"CHEBI:2": ["c"], "CHEBI:1": ["a", "b"]}):
            chosen, flag = r._choose_best_kg_id(d)
            assert chosen == "CHEBI:1"
            assert flag is None
    assert "majority tie" not in caplog.text


def test_preferred_prefix_wins_cross_namespace_tie() -> None:
    # T3 (R2): on a tie the preferred-namespace candidate beats a non-preferred one. Numeric fallback
    # alone would also pick CHEBI:500 here, so the point is proven by T_drug below; this pins the
    # exact-category path.
    r = _resolver(_DESCENDANTS)
    chosen, flag = r._choose_best_kg_id(
        {"PUBCHEM.COMPOUND:999": ["x"], "CHEBI:500": ["y"]},
        category="biolink:SmallMolecule",
    )
    assert chosen == "CHEBI:500"
    assert flag is None  # determinism fix does not touch the certificate review-flag channel


def test_descendant_category_inherits_preferred_namespace() -> None:
    # T6 (Greptile #1): a descendant category (biolink:Drug) must inherit SmallMolecule's CHEBI policy.
    # Numeric fallback alone would prefer the lower-local-id PubChem sibling (PUBCHEM.COMPOUND:1); the
    # inherited CHEBI preference must instead pick CHEBI:500.
    r = _resolver(_DESCENDANTS)
    chosen, _ = r._choose_best_kg_id(
        {"PUBCHEM.COMPOUND:1": ["x"], "CHEBI:500": ["y"]},
        category="biolink:Drug",
    )
    assert chosen == "CHEBI:500"


def test_same_namespace_tie_stable_numeric_and_logged(caplog) -> None:
    # T4 (retinol-shaped, honesty): two CHEBI siblings tie -> numeric-lower local id, deterministic
    # across orders, and the coin-flip is logged. String sort would misorder CHEBI:12777 vs CHEBI:983;
    # the numeric key must not.
    r = _resolver(_DESCENDANTS)
    with caplog.at_level(logging.WARNING):
        for d in (
            {"CHEBI:132246": ["x"], "CHEBI:12777": ["y"]},
            {"CHEBI:12777": ["y"], "CHEBI:132246": ["x"]},
        ):
            chosen, flag = r._choose_best_kg_id(d, category="biolink:SmallMolecule")
            assert chosen == "CHEBI:12777"
            assert flag is None
    assert "majority tie" in caplog.text


def test_tie_with_refmet_vote_is_determinized() -> None:
    # T5 (R4, Greptile #2): descendants are configured so _is_small_molecule is True and the row
    # ENTERS the RefMet branch (the earlier version left get_descendants an unconfigured MagicMock and
    # returned before it). RefMet votes for one tied sibling; the stable majority must land on that node
    # identically across insertion orders, so the RefMet-branch outcome (chosen + flag) is deterministic.
    results = set()
    for d in (
        {"CHEBI:132246": ["x"], "CHEBI:12777": ["y"]},
        {"CHEBI:12777": ["y"], "CHEBI:132246": ["x"]},
    ):
        r = _resolver(_DESCENDANTS)
        assigned = {REFMET_ANNOTATOR: {"CHEBI:12777": ["y"]}}
        chosen, flag = r._choose_best_kg_id(d, kg_ids_assigned=assigned, category="biolink:SmallMolecule")
        results.add((chosen, flag))
    assert results == {("CHEBI:12777", None)}, f"non-deterministic tie x RefMet outcome: {results}"
