"""Axis 3 determinism: Kestrel candidate order must not decide an EXACT score tie.

Each Kestrel annotator commits a single node per entity via order-trusting selection (`term_results[0]`,
`max(..., key=score)`, first-on-category scan). `stable_result_order` normalizes the candidate list at
ingestion so an exact score tie resolves on the `id` CURIE, not Kestrel's run-varying response order.
The resolver's own tie-break (#67) cannot reach this, because it acts on the one-node-per-annotator SET
these selections already reduced to. Network is monkeypatched — no live Kestrel call.
"""

from __future__ import annotations

from biomapper2.core.annotators.base import stable_result_order
from biomapper2.core.annotators.kestrel_hybrid import KestrelHybridSearchAnnotator
from biomapper2.core.annotators.kestrel_text import KestrelTextSearchAnnotator

SM = ["biolink:SmallMolecule"]


def test_orders_by_score_descending() -> None:
    rows = [{"id": "CHEBI:1", "score": 0.2}, {"id": "CHEBI:2", "score": 0.9}, {"id": "CHEBI:3", "score": 0.5}]
    assert [r["id"] for r in stable_result_order(rows)] == ["CHEBI:2", "CHEBI:3", "CHEBI:1"]


def test_exact_tie_breaks_on_id_independent_of_input_order() -> None:
    a = {"id": "CHEBI:200", "score": 0.8}
    b = {"id": "CHEBI:100", "score": 0.8}
    # Same tied set, opposite arrival orders (two Kestrel responses) -> identical order + winner.
    assert stable_result_order([a, b]) == stable_result_order([b, a])
    assert stable_result_order([a, b])[0]["id"] == "CHEBI:100"


def test_missing_or_none_score_sorts_as_zero() -> None:
    rows = [{"id": "CHEBI:1"}, {"id": "CHEBI:2", "score": None}, {"id": "CHEBI:3", "score": 1.0}]
    # The scored row rises above the missing/None-score rows (which tie at 0.0 and break on id).
    assert [r["id"] for r in stable_result_order(rows)] == ["CHEBI:3", "CHEBI:1", "CHEBI:2"]


def test_non_numeric_score_is_tolerated() -> None:
    rows = [{"id": "CHEBI:9", "score": "oops"}, {"id": "CHEBI:8", "score": 0.4}]
    # A non-numeric score coerces to 0.0 rather than raising; the real 0.4 row wins.
    assert [r["id"] for r in stable_result_order(rows)] == ["CHEBI:8", "CHEBI:9"]


def test_none_and_empty_inputs_return_empty_list() -> None:
    assert stable_result_order(None) == []
    assert stable_result_order([]) == []


def test_nan_score_is_folded_to_zero_and_stays_deterministic() -> None:
    nan = float("nan")
    real = {"id": "CHEBI:5", "score": 0.3}
    a = {"id": "CHEBI:9", "score": nan}
    b = {"id": "CHEBI:1", "score": nan}
    # NaN is unorderable, so without folding it timsort would leave NaN rows in arrival order. Folded to
    # 0.0: the real 0.3 row leads and the two NaN rows order by id, identically across input orders.
    assert [r["id"] for r in stable_result_order([a, real, b])] == ["CHEBI:5", "CHEBI:1", "CHEBI:9"]
    assert stable_result_order([a, b]) == stable_result_order([b, a])


def test_text_annotator_commit_is_order_independent_on_a_tie(monkeypatch) -> None:
    # Two on-category rows tie on score; the committed node must be identical across response orders.
    a = {"id": "CHEBI:200", "score": 0.8, "categories": SM}
    b = {"id": "CHEBI:100", "score": 0.8, "categories": SM}

    def commit(rows: list[dict]) -> dict:
        ann = KestrelTextSearchAnnotator()
        monkeypatch.setattr(ann, "_kestrel_text_search", lambda t, c, p, limit=1: {t: rows})
        return ann.get_annotations(
            {"name": "q"}, name_field="name", category="biolink:SmallMolecule", accepted_categories=set(SM)
        )[ann.slug]

    assert commit([a, b]) == commit([b, a]) == {"CHEBI": {"100": {"score": 0.8}}}


def test_hybrid_annotator_commit_is_order_independent_on_a_tie(monkeypatch) -> None:
    # prefer_human=False -> honest top-1, which after the stable sort is the lower-CURIE tie member.
    a = {"id": "CHEBI:200", "score": 0.8, "name": "q", "prefixes": ["CHEBI"]}
    b = {"id": "CHEBI:100", "score": 0.8, "name": "q", "prefixes": ["CHEBI"]}

    def commit(rows: list[dict]) -> dict:
        ann = KestrelHybridSearchAnnotator()
        monkeypatch.setattr(ann, "_kestrel_hybrid_search", lambda t, c, p, limit=1: {t: rows})
        return ann.get_annotations(
            {"name": "q"}, name_field="name", category="biolink:SmallMolecule", prefer_human=False
        )[ann.slug]

    left, right = commit([a, b]), commit([b, a])
    assert left == right
    assert "100" in left["CHEBI"]
