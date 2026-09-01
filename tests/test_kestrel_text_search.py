"""KestrelTextSearchAnnotator: scan the ranked window for the first on-category node.

The endpoint's `category_filter` is advisory, so an off-category row (e.g. a Consumer Health Vocab
node literally named "glutamate") can outrank the intended structural node. The old `limit=1` path
committed/refused on rank-1 alone and returned nothing for such queries; the fix pulls a window and
takes the first on-category candidate, while the category guard still refuses when NOTHING in the
window is on-category. Network is monkeypatched — no live Kestrel call.
"""

from __future__ import annotations

import pandas as pd

from biomapper2.core.annotators.kestrel_text import KestrelTextSearchAnnotator

ACCEPTED = {"biolink:SmallMolecule"}
OFF = {"id": "CHV:0000005585", "score": 0.96, "categories": ["biolink:InformationContentEntity"]}
ON = {"id": "CHEBI:16015", "score": 0.65, "categories": ["biolink:SmallMolecule"]}


def _annotator(monkeypatch, rows_by_term):
    ann = KestrelTextSearchAnnotator()
    captured = {}

    def _fake(search_text, category, prefixes, limit=1):
        captured["limit"] = limit
        terms = [search_text] if isinstance(search_text, str) else list(search_text)
        return {t: rows_by_term.get(t, []) for t in terms}

    monkeypatch.setattr(ann, "_kestrel_text_search", _fake)
    return ann, captured


def test_scans_past_off_category_top1(monkeypatch):
    ann, captured = _annotator(monkeypatch, {"glutamate": [OFF, ON]})
    out = ann.get_annotations(
        {"name": "glutamate"}, name_field="name", category="biolink:SmallMolecule", accepted_categories=ACCEPTED
    )
    assert out[ann.slug] == {"CHEBI": {"16015": {"score": 0.65}}}
    assert captured["limit"] > 1  # pulls a window, not just the top hit


def test_refuses_when_window_all_off_category(monkeypatch):
    # Positive control: the category guard must still refuse when nothing in the window is on-category.
    ann, _ = _annotator(monkeypatch, {"glutamate": [OFF, {**OFF, "id": "GO:0006536"}]})
    out = ann.get_annotations(
        {"name": "glutamate"}, name_field="name", category="biolink:SmallMolecule", accepted_categories=ACCEPTED
    )
    assert out[ann.slug] == {}


def test_empty_results_yield_no_annotation(monkeypatch):
    ann, _ = _annotator(monkeypatch, {"glutamate": []})
    out = ann.get_annotations(
        {"name": "glutamate"}, name_field="name", category="biolink:SmallMolecule", accepted_categories=ACCEPTED
    )
    assert out[ann.slug] == {}


def test_bulk_path_applies_the_same_rerank(monkeypatch):
    ann, captured = _annotator(monkeypatch, {"glutamate": [OFF, ON], "urea": [OFF]})
    df = pd.DataFrame({"name": ["glutamate", "urea"]})
    col = ann.get_annotations_bulk(
        df, name_field="name", category="biolink:SmallMolecule", accepted_categories=ACCEPTED
    )
    assert col.iloc[0][ann.slug] == {"CHEBI": {"16015": {"score": 0.65}}}  # scanned past OFF
    assert col.iloc[1][ann.slug] == {}  # urea window all off-category → refused
    assert captured["limit"] > 1


def test_cache_miss_does_not_crash():
    # Regression: a truthy cache lacking the term yields term_results=None; the window scan must not
    # iterate None (the bug the None-guard fixes). No monkeypatch needed — the cache path is taken.
    ann = KestrelTextSearchAnnotator()
    out = ann.get_annotations(
        {"name": "glutamate"},
        name_field="name",
        category="biolink:SmallMolecule",
        accepted_categories=ACCEPTED,
        cache={"other": [ON]},
    )
    assert out[ann.slug] == {}
