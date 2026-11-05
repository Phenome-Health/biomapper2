"""Case-robust name lookup: NECS's lower-cased spellings must resolve like their Metabolon twins.

The defect: RefMet ``/match`` and Goslin read metabolite names case-sensitively. NECS ships lower-cased
names, so ``acetylcarnitine (c2)`` (NECS) and ``acetylcarnitine (C2)`` (Arivale) got different RefMet
answers, resolved to different KG nodes, and never linked across cohorts: 24 NECSxArivale pairs in the
2026-11-03 prod run (11 RefMet no-match, 13 a less specific RefMet record) plus 3 NECSxXu oxylipins that
Goslin parses only in its own casing.

Offline only: RefMet ``/match`` answers are replayed from ``fixtures/case_split_necs_pairs.json``
(recorded live, see its ``sources``), and Goslin runs locally.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from circuitbreaker import CircuitBreakerError, CircuitBreakerMonitor

from biomapper2.core.annotators import refmet_snapshot, refmet_store
from biomapper2.core.annotators.base import AVAILABILITY_NO_MATCH, AVAILABILITY_UNAVAILABLE, AVAILABILITY_VOTED
from biomapper2.core.annotators.goslin_grammar import LipidGrammar
from biomapper2.core.annotators.goslin_lipid import GoslinLipidAnnotator
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator
from biomapper2.core.lipid_structure_resolver import LipidStructureResolver
from biomapper2.core.name_case import canonical_case_variant, canonical_query

pytestmark = pytest.mark.unit

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "case_split_necs_pairs.json").read_text())
PAIRS = FIXTURE["pairs"]
RECORDED = FIXTURE["refmet_match_responses"]
_PAIR_IDS = [f"{p['cohort']}:{p['necs_name']}" for p in PAIRS]
_BREAKER_NAME = "MetabolomicsWorkbenchAnnotator._do_refmet_request"


@pytest.fixture(autouse=True)
def _reset_breaker():
    breaker = CircuitBreakerMonitor.get(_BREAKER_NAME)
    if breaker is not None:
        breaker.reset()
    yield
    if breaker is not None:
        breaker.reset()


class _Resp:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.from_cache = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


class _ReplaySession:
    """Serves recorded RefMet /match answers; an unrecorded query is a RefMet no-match ("-")."""

    NO_MATCH = {"refmet_id": "-", "refmet_name": "-"}

    def __init__(self, responses: dict[str, Any], status: int = 200) -> None:
        self._responses = responses
        self._status = status
        self.queries: list[str] = []

    def get(self, url: str, timeout: float | None = None) -> _Resp:  # noqa: ARG002
        name = urllib.parse.unquote(url.split("/refmet/match/", 1)[1])
        self.queries.append(name)
        return _Resp(self._responses.get(name, self.NO_MATCH), status=self._status)


def _mw(session: _ReplaySession, monkeypatch: pytest.MonkeyPatch, mode: str = "off") -> MetabolomicsWorkbenchAnnotator:
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode=mode, sleep=lambda _s: None, max_retries=0)
    monkeypatch.setattr(ann, "_session", session)
    return ann


def _refmet_id(ann: MetabolomicsWorkbenchAnnotator, name: str) -> str | None:
    result = ann._fetch_refmet_data(name)
    return (result.data or {}).get("refmet_id") if result.status == AVAILABILITY_VOTED else None


class _RecordingBinder:
    """Goslin's RefMet binder stand-in: records the canonical level names Goslin queried."""

    slug = "metabolomics-workbench"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def get_annotations(self, entity, name_field, category, prefixes=None, **kwargs):  # noqa: ARG002
        self.seen.append(entity.get(name_field))
        return {self.slug: {}}


_GRAMMAR = LipidGrammar()


def _goslin_queries(name: str) -> list[str]:
    binder = _RecordingBinder()
    ann = GoslinLipidAnnotator(binder=binder)  # pyright: ignore[reportArgumentType]
    ann.get_annotations({"name": name}, name_field="name", category="biolink:SmallMolecule")
    return binder.seen


# -- 1. the rule: canonical_query ------------------------------------------------------------------


@pytest.mark.parametrize("pair", PAIRS, ids=_PAIR_IDS)
def test_each_necs_name_canonicalizes_to_its_cohort_twin(pair):
    assert canonical_query(pair["necs_name"]) == pair["cohort_name"]


@pytest.mark.parametrize("pair", PAIRS, ids=_PAIR_IDS)
def test_a_name_already_in_canonical_case_is_queried_unchanged(pair):
    # Every Metabolon spelling is a fixed point, so the fix cannot move a name that resolves today.
    assert canonical_query(pair["cohort_name"]) is None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("3-hydroxyisovalerylcarnitine (c5-oh)", "3-hydroxyisovalerylcarnitine (C5-OH)"),
        ("glutarylcarnitine (c5-dc)", "glutarylcarnitine (C5-DC)"),
        ("tiglylcarnitine (c5:1)", "tiglylcarnitine (C5:1)"),
        (  # only the ether letter changes; "n3" stays lower case
            "1-(1-enyl-stearoyl)-2-docosapentaenoyl-gpe (p-18:0/22:5n3)*",
            "1-(1-enyl-stearoyl)-2-docosapentaenoyl-GPE (P-18:0/22:5n3)*",
        ),
        ("1-palmityl-gpc (o-16:0)", "1-palmityl-GPC (O-16:0)"),
        ("1-stearoyl-gpi (18:0)", "1-stearoyl-GPI (18:0)"),
        ("14,15-eet", "14,15-EET"),
        ("9-hode", "9-HODE"),
        ("12,13-DIHOME", "12,13-DiHOME"),
        # Only the token letters change; the rest of the name keeps its case, stereo prefixes included.
        ("L-carnitine (c2)", "L-carnitine (C2)"),
        ("Acetylcarnitine (c2)", "Acetylcarnitine (C2)"),
    ],
)
def test_only_the_notation_tokens_change(name: str, expected: str):
    assert canonical_query(name) == expected
    assert canonical_case_variant(expected) == expected


MUST_NOT_TOUCH = [
    "c1ccccc1",  # SMILES: lower case marks aromatic atoms
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "InChI=1S/C9H17NO4/c1-7(11)14-8(5-9(12)13)6-10(2,3)4/h8H,5-6H2,1-4H3/t8-/m1/s1",
    "RDHQFKQIGNGIED-MRVPVSSYSA-N",  # InChIKey
    "HMDB:HMDB0000201",  # CURIE
    "D-glucose",
    "d-glucose",  # rotation vs configuration: case is the claim, never rewritten
    "l-carnitine",
    "12(S)-HETE",
    "12(s)-hete",  # stereo-bearing oxylipin: not a bare position locant, left as given
    "(2S,3R)-2-amino-3-hydroxybutanoate",
    "N-acetylglycine",
    "S-adenosylmethionine",
    "PC O-16:0/18:1",  # ether letter outside a chain parenthetical
    "pc 16:0/18:1",  # lipid-class shorthand: out of scope
    "PC 16:0/18:1",
    "TP53",
    "Trp53",
    "CoA",
    "Co",
    "CO",
    "carbon monoxide (co)",
    "home",
    "glucose",
    "ascorbic acid (vitamin c)",
]


@pytest.mark.parametrize("name", MUST_NOT_TOUCH)
def test_case_meaningful_names_are_never_rewritten(name: str):
    assert canonical_query(name) is None


# -- 2. RefMet: replay the recorded /match answers ------------------------------------------------


@pytest.mark.parametrize("pair", PAIRS, ids=_PAIR_IDS)
def test_necs_name_gets_its_twins_refmet_answer(pair, monkeypatch: pytest.MonkeyPatch):
    ann = _mw(_ReplaySession(RECORDED), monkeypatch)
    twin = _refmet_id(ann, pair["cohort_name"])
    assert twin is not None  # the Metabolon spelling votes in the recording
    assert _refmet_id(ann, pair["necs_name"]) == twin


def test_the_recording_reproduces_the_bug_for_the_raw_spellings():
    # Guard on the fixture itself: sent as given, the 24 NECSxArivale names really do get a different
    # answer from their twins (11 no-match, 13 a different record), so the test above is not vacuous.
    def rid(name: str) -> str:
        return RECORDED[name]["refmet_id"]

    arivale = [p for p in PAIRS if p["cohort"] == "arivale"]
    assert len(arivale) == 24
    differ = [p for p in arivale if rid(p["necs_name"]) != rid(p["cohort_name"])]
    no_match = [p for p in differ if rid(p["necs_name"]) == "-"]
    assert (len(differ), len(no_match)) == (24, 11)


@pytest.mark.parametrize(
    "lower,upper",
    [
        ("acetylcarnitine (c2)", "acetylcarnitine (C2)"),
        ("1-(1-enyl-palmitoyl)-2-oleoyl-gpc (p-16:0/18:1)*", "1-(1-enyl-palmitoyl)-2-oleoyl-GPC (P-16:0/18:1)*"),
    ],
)
def test_case_variants_get_identical_refmet_votes(lower: str, upper: str, monkeypatch: pytest.MonkeyPatch):
    ann = _mw(_ReplaySession(RECORDED), monkeypatch)
    votes_lower = ann.get_annotations({"name": lower}, name_field="name", category="biolink:SmallMolecule")
    votes_upper = ann.get_annotations({"name": upper}, name_field="name", category="biolink:SmallMolecule")
    assert votes_lower == votes_upper
    assert votes_upper[ann.slug]  # a real vote, not two empty ones


def test_canonical_spelling_is_sent_first(monkeypatch: pytest.MonkeyPatch):
    session = _ReplaySession(RECORDED)
    ann = _mw(session, monkeypatch)
    ann._fetch_refmet_data("1-(1-enyl-oleoyl)-gpe (p-18:1)*")
    # Canonical-first, not retry-on-miss: the raw spelling would have matched an ambiguous record.
    assert session.queries == ["1-(1-enyl-oleoyl)-GPE (P-18:1)*"]


def test_raw_spelling_is_the_fallback_on_a_no_match(monkeypatch: pytest.MonkeyPatch):
    session = _ReplaySession({"weirdname (c2)": {"refmet_id": "RM1", "refmet_name": "x"}})
    ann = _mw(session, monkeypatch)
    assert _refmet_id(ann, "weirdname (c2)") == "RM1"
    assert session.queries == ["weirdname (C2)", "weirdname (c2)"]


def test_an_outage_is_not_retried_with_the_raw_spelling(monkeypatch: pytest.MonkeyPatch):
    session = _ReplaySession({}, status=503)
    ann = _mw(session, monkeypatch)
    assert ann._fetch_refmet_data("acetylcarnitine (c2)").status == AVAILABILITY_UNAVAILABLE
    assert session.queries == ["acetylcarnitine (C2)"]


def test_a_name_without_tokens_is_sent_exactly_once_as_given(monkeypatch: pytest.MonkeyPatch):
    session = _ReplaySession({})
    ann = _mw(session, monkeypatch)
    assert ann._fetch_refmet_data("c1ccccc1").status == AVAILABILITY_NO_MATCH
    assert session.queries == ["c1ccccc1"]


def test_the_canonical_query_is_issued_once_per_batch(monkeypatch: pytest.MonkeyPatch):
    session = _ReplaySession(RECORDED)
    ann = _mw(session, monkeypatch)
    cache = ann._fetch_all(["acetylcarnitine (c2)", "acetylcarnitine (C2)", "acetylcarnitine (C2)"])
    assert {(r.data or {}).get("refmet_id") for r in cache.values()} == {"RM0154009"}
    assert session.queries.count("acetylcarnitine (C2)") == 1


# -- 3. live_backup: the store answers for the query actually sent, never a case twin's ---------


@pytest.fixture
def store_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    refmet_snapshot.reset()
    db = tmp_path / "store.sqlite"
    monkeypatch.setenv("REFMET_STORE_PATH", str(db))
    monkeypatch.delenv("REFMET_SNAPSHOT_PATH", raising=False)
    yield db
    refmet_snapshot.reset()


def _breaker_open(*_args, **_kwargs):
    raise CircuitBreakerError("simulated open breaker")


def test_write_back_is_keyed_by_the_canonical_query_sent(store_db: Path, monkeypatch: pytest.MonkeyPatch):
    ann = _mw(_ReplaySession(RECORDED), monkeypatch, mode="live_backup")
    ann._fetch_refmet_data("acetylcarnitine (c2)")
    conn = sqlite3.connect(str(store_db))
    try:
        rows = conn.execute("SELECT query, status, refmet_id FROM refmet_by_query").fetchall()
    finally:
        conn.close()
    assert rows == [("acetylcarnitine (C2)", "voted", "RM0154009")]


def test_outage_serves_the_lowercase_name_its_canonical_answer(store_db: Path, monkeypatch: pytest.MonkeyPatch):
    ann = _mw(_ReplaySession(RECORDED), monkeypatch, mode="live_backup")
    ann._fetch_refmet_data("acetylcarnitine (C2)")  # live: learned under the exact query
    monkeypatch.setattr(ann, "_do_refmet_request", _breaker_open)  # RefMet goes down
    for name in ("acetylcarnitine (C2)", "acetylcarnitine (c2)"):
        result = ann._fetch_refmet_data(name)
        assert (result.status, result.source, (result.data or {}).get("refmet_id")) == (
            AVAILABILITY_VOTED,
            "freeze_backup",
            "RM0154009",
        )


def test_outage_never_serves_a_case_twins_answer(store_db: Path, monkeypatch: pytest.MonkeyPatch):
    # A case pair with no recognized token: the two spellings are distinct queries. Under the old
    # case-folded key the outage answer for one was whatever the other had last written.
    session = _ReplaySession({"Retinol": {"refmet_id": "RM0000001", "refmet_name": "Retinol"}})
    ann = _mw(session, monkeypatch, "live_backup")
    assert _refmet_id(ann, "Retinol") == "RM0000001"
    assert ann._fetch_refmet_data("RETINOL").status == AVAILABILITY_NO_MATCH  # live no-match, written back
    monkeypatch.setattr(ann, "_do_refmet_request", _breaker_open)
    assert _refmet_id(ann, "Retinol") == "RM0000001"
    assert ann._fetch_refmet_data("RETINOL").status == AVAILABILITY_NO_MATCH
    assert ann._fetch_refmet_data("retinol").status == AVAILABILITY_UNAVAILABLE  # never learned: no backup
    assert refmet_store.get("retinol") is None


# -- 4. Goslin: annotator and the Tier B lipid structure resolver -----------------------------------


@pytest.mark.parametrize("pair", PAIRS, ids=_PAIR_IDS)
def test_necs_name_reaches_its_twins_goslin_parse(pair):
    assert _goslin_queries(pair["necs_name"]) == _goslin_queries(pair["cohort_name"])


@pytest.mark.parametrize(
    "lower,upper",
    [("12,13-dihome", "12,13-DiHOME"), ("9,10-dihome", "9,10-DiHOME"), ("12-hete", "12-HETE")],
)
def test_oxylipin_case_variants_parse(lower: str, upper: str):
    assert _GRAMMAR.parse(lower) is None  # the bug: Goslin rejects the lower-cased spelling
    queried = _goslin_queries(upper)
    assert queried  # it parsed and bound
    assert _goslin_queries(lower) == queried


class _RecordingEnricher:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def candidates_checked(self, canonical_name: str) -> tuple[list[dict[str, str]], bool]:
        self.seen.append(canonical_name)
        return [{"inchi_key": "CQSLTKIXAJTQGA-FLIBITNWSA-N"}], True


def test_lipid_structure_resolver_is_case_robust():
    enricher = _RecordingEnricher()
    resolver = LipidStructureResolver(grammar=_GRAMMAR, enricher=enricher)
    lower = resolver.resolve("12,13-dihome")
    upper = resolver.resolve("12,13-DiHOME")
    assert enricher.seen == ["FA 18:1;O2", "FA 18:1;O2"]
    assert lower.inchikey_block == upper.inchikey_block is not None
