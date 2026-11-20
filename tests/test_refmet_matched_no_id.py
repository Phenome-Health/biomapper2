"""A RefMet match with no retrievable id is a per-name outcome, not a service failure.

``/match`` can match a name and the follow-up ``/refmet/name/{name}/refmet_id`` lookup can still
return no id (observed live for names with an apostrophe, such as ``5'-Methylthioadenosine``). That
answer is about the name. It used to be raised inside the circuit-decorated call, so three such
names opened the breaker and every other name in the worker read ``unavailable`` for five minutes.

These tests pin:
- a matched name with no id reads ``matched_no_id``, never ``unavailable``, and is not counted by
  the breaker, so it changes nothing for any other name;
- a ``/match`` reply whose ``refmet_name`` is a message (``'C' is ambiguous: ...``) is a no-match and
  never reaches the id lookup;
- a real 5xx or timeout on ``/match`` or on ``/name`` still counts toward the breaker;
- the new value reaches the certificate, the API response and the batch summary;
- ``live_backup`` serves the stored vote for it (source ``kept_live_matched_no_id``);
- breaker-open skips are logged at INFO with a count per batch.

Fake session only. The breaker is process-global, so it is reset around every test.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
from circuitbreaker import CircuitBreakerMonitor
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomapper2.core.annotation_engine import AnnotationEngine
from biomapper2.core.annotators import refmet_store
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator
from biomapper2.core.certificate import REFMET_ANNOTATOR
from biomapper2.core.resolver import Resolver
from biomapper2.utils import quote_name_path_segment

pytestmark = pytest.mark.unit

CATEGORY = "biolink:SmallMolecule"
MATCHED_NO_ID = "matched_no_id"
KEPT_NO_ID = "kept_live_matched_no_id"
_BREAKER_NAME = "MetabolomicsWorkbenchAnnotator._do_refmet_request"
MATCH_URL = MetabolomicsWorkbenchAnnotator.BASE_URL
NAME_URL = MetabolomicsWorkbenchAnnotator.NAME_URL

# What live RefMet returned on 2026-11-20 for the trigger names in the issue this fixes.
AMBIGUOUS_MESSAGE = {
    "refmet_name": "'C' is ambiguous:Use 'FA'(fatty acids) or 'CAR'(acylcarnitines)*",
    "formula": "-",
    "exactmass": "-",
    "super_class": "-",
    "main_class": "-",
    "sub_class": "-",
}
NO_MATCH_BODY = {k: "-" for k in AMBIGUOUS_MESSAGE}

# query -> (refmet_name /match returns, refmet_id /name returns or None for "no id")
MATCHES: dict[str, tuple[str, str | None]] = {
    "5'-Methylthioadenosine": ("5'-Methylthioadenosine", None),
    "3'-O-Methylguanosine": ("3'-O-Methylguanosine", None),
    "8-Oxo-2-deoxyadenosine": ("8-Oxo-2'-deoxyadenosine", None),
    "Riboflavin": ("Riboflavin", "RM0135919"),
    "Theophylline": ("Theophylline", "RM0136130"),
    "2-hydroxyoleate": ("2-Hydroxyoleic acid", "RM0153366"),
    "CAR 10:0": ("CAR 10:0", "RM0153849"),
}
MESSAGE_QUERIES = ["C0", "C10", "C2"]
NO_ID_QUERIES = ["5'-Methylthioadenosine", "3'-O-Methylguanosine", "8-Oxo-2-deoxyadenosine"]
NORMAL_QUERIES = ["Riboflavin", "Theophylline", "2-hydroxyoleate", "CAR 10:0"]


def _breaker():
    return CircuitBreakerMonitor.get(_BREAKER_NAME)


@pytest.fixture(autouse=True)
def _reset_breaker():
    breaker = _breaker()
    if breaker is not None:
        breaker.reset()
    yield
    if breaker is not None:
        breaker.reset()


class _Resp:
    def __init__(self, body: Any, status: int = 200) -> None:
        self._body = body
        self.status_code = status
        self.from_cache = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._body


class _RefMet:
    """A fake RefMet: ``/match`` and ``/name`` answered from the tables above; every URL is recorded."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.match_fault: Any = None  # a status code or an exception to raise on every /match
        self.name_fault: Any = None  # same, for /name
        self._by_match_url = {f"{MATCH_URL}/{quote_name_path_segment(q)}": q for q in MATCHES}
        self._by_match_url.update({f"{MATCH_URL}/{quote_name_path_segment(q)}": q for q in MESSAGE_QUERIES})
        self._by_name_url = {
            f"{NAME_URL}/{quote_name_path_segment(refmet_name)}/refmet_id": (refmet_name, refmet_id)
            for refmet_name, refmet_id in MATCHES.values()
        }

    def name_calls(self) -> list[str]:
        return [u for u in self.calls if u.startswith(NAME_URL)]

    @staticmethod
    def _fault(fault: Any) -> _Resp:
        if isinstance(fault, Exception):
            raise fault
        return _Resp({}, status=fault)

    def get(self, url: str, timeout: float | None = None) -> _Resp:  # noqa: ARG002
        self.calls.append(url)
        if url.startswith(NAME_URL):
            if self.name_fault is not None:
                return self._fault(self.name_fault)
            refmet_name, refmet_id = self._by_name_url.get(url, ("", None))
            # Live RefMet answers a name it has no id for with an empty JSON array.
            return _Resp({"name": refmet_name, "refmet_id": refmet_id} if refmet_id else [])
        if self.match_fault is not None:
            return self._fault(self.match_fault)
        query = self._by_match_url.get(url)
        if query in MESSAGE_QUERIES:
            return _Resp(dict(AMBIGUOUS_MESSAGE))
        if query in MATCHES:
            return _Resp({"refmet_name": MATCHES[query][0], "formula": "C1H1", "exactmass": "1.0"})
        return _Resp(dict(NO_MATCH_BODY))


def _annotator(freeze_mode: str = "off") -> tuple[MetabolomicsWorkbenchAnnotator, _RefMet]:
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode=freeze_mode, max_retries=0, sleep=lambda _s: None)
    session = _RefMet()
    ann._session = session  # type: ignore[assignment]
    return ann, session


def _batch(ann: MetabolomicsWorkbenchAnnotator, names: list[str]) -> dict[str, tuple[str, str, Any]]:
    cache = ann.build_availability_cache(pd.DataFrame({"name": names}), "name")
    return {name: (r.status, r.source, r.data and r.data.get("refmet_id")) for name, r in cache.items()}


# --------------------------------------------------------------------------------------------
# The per-name outcome and the breaker
# --------------------------------------------------------------------------------------------


def test_three_no_id_names_then_a_normal_name_the_normal_name_is_fetched_and_votes() -> None:
    ann, session = _annotator()
    for name in NO_ID_QUERIES:
        result = ann._fetch_refmet_data(name)
        assert (result.status, result.source, result.data) == (MATCHED_NO_ID, "live_api", None), name

    breaker = _breaker()
    assert breaker.failure_count == 0 and not breaker.opened

    before = len(session.calls)
    normal = ann._fetch_refmet_data("Riboflavin")
    assert len(session.calls) == before + 2  # /match and /name were both requested
    assert normal.status == "voted" and normal.data is not None and normal.data["refmet_id"] == "RM0135919"
    votes = ann.get_annotations({"name": "Riboflavin"}, name_field="name", category=CATEGORY)
    assert "RM0135919" in votes[ann.slug]["refmet_id"]


def test_a_matched_name_with_no_id_casts_no_vote() -> None:
    ann, _ = _annotator()
    entity = {"name": "5'-Methylthioadenosine"}
    assert ann.get_annotations(entity, name_field="name", category=CATEGORY) == {ann.slug: {}}
    assert ann.get_availability(entity, "name") == {ann.slug: MATCHED_NO_ID}
    assert ann.get_source(entity, "name") == {ann.slug: "live_api"}


@pytest.mark.parametrize("status", [400, 404])
def test_a_per_name_client_error_on_the_id_lookup_is_also_a_per_name_outcome(status: int) -> None:
    ann, session = _annotator()
    session.name_fault = status
    for name in NORMAL_QUERIES:
        assert ann._fetch_refmet_data(name).status == MATCHED_NO_ID
    assert _breaker().failure_count == 0


@pytest.mark.parametrize("query", MESSAGE_QUERIES)
def test_a_match_message_reply_is_a_no_match_with_no_name_call(query: str) -> None:
    ann, session = _annotator()
    result = ann._fetch_refmet_data(query)
    assert (result.status, result.source, result.data) == ("no_match", "live_api", None)
    assert session.name_calls() == []
    assert _breaker().failure_count == 0


@pytest.mark.parametrize("endpoint", ["match", "name"])
@pytest.mark.parametrize(
    "fault",
    [500, 503, requests.Timeout("timed out"), requests.ConnectionError("connection reset")],
    ids=["http-500", "http-503", "timeout", "transport"],
)
def test_a_real_service_failure_still_counts_toward_the_breaker(endpoint: str, fault: Any) -> None:
    ann, session = _annotator()
    setattr(session, f"{endpoint}_fault", fault)

    for i, name in enumerate(NORMAL_QUERIES[:3], start=1):
        result = ann._fetch_refmet_data(name)
        assert (result.status, result.source) == ("unavailable", "unavailable"), name
        assert _breaker().failure_count == i

    assert _breaker().opened
    before = len(session.calls)
    skipped = ann._fetch_refmet_data(NORMAL_QUERIES[3])
    assert (skipped.status, skipped.source) == ("unavailable", "unavailable")
    assert len(session.calls) == before  # breaker open: no request made


def test_a_batch_with_trigger_names_gives_every_other_name_the_same_result_as_without_them() -> None:
    triggers = MESSAGE_QUERIES + NO_ID_QUERIES
    ann, _ = _annotator()
    without = _batch(ann, NORMAL_QUERIES)
    _breaker().reset()

    ann, _ = _annotator()
    with_triggers = _batch(ann, triggers + NORMAL_QUERIES)

    assert {name: with_triggers[name] for name in NORMAL_QUERIES} == without
    assert all(without[name][0] == "voted" for name in NORMAL_QUERIES)
    assert {with_triggers[name][0] for name in MESSAGE_QUERIES} == {"no_match"}
    assert {with_triggers[name][0] for name in NO_ID_QUERIES} == {MATCHED_NO_ID}
    assert not _breaker().opened


def test_the_same_batch_twice_in_a_row_gives_the_same_result() -> None:
    names = MESSAGE_QUERIES + NO_ID_QUERIES + NORMAL_QUERIES
    ann, _ = _annotator()
    first = _batch(ann, names)
    assert _batch(ann, names) == first
    assert "unavailable" not in {status for status, _, _ in first.values()}


def test_a_new_annotator_does_not_inherit_a_deadline_left_armed_under_its_id() -> None:
    # The deadline is keyed by id(self) in a context variable and CPython reuses ids. Seed the entry
    # a collected instance would leave behind, directly under this instance's id, before __init__.
    ann = MetabolomicsWorkbenchAnnotator.__new__(MetabolomicsWorkbenchAnnotator)
    ann._batch_deadline = 0.0  # long past on the monotonic clock
    assert ann._batch_deadline == 0.0

    ann.__init__(freeze_mode="off", max_retries=0, sleep=lambda _s: None)  # type: ignore[misc]
    session = _RefMet()
    ann._session = session  # type: ignore[assignment]

    assert ann._batch_deadline is None
    result = ann._fetch_refmet_data("Riboflavin")
    assert result.status == "voted"
    assert len(session.calls) == 2  # the lookup reached the session: /match and /name


# --------------------------------------------------------------------------------------------
# The value reaches the certificate, the API response and the run summaries
# --------------------------------------------------------------------------------------------


def _engine_with(ann: MetabolomicsWorkbenchAnnotator) -> AnnotationEngine:
    bc = MagicMock()
    bc.get_descendants.side_effect = lambda c: {c}
    engine = AnnotationEngine(biolink_client=bc)
    engine.annotator_registry[REFMET_ANNOTATOR] = ann
    return engine


def _offline_mapper(ann: MetabolomicsWorkbenchAnnotator):
    """Real ``map_entity_to_kg``, real engine and real RefMet annotator (fake session); mocked KG."""
    from biomapper2.mapper import Mapper

    engine = _engine_with(ann)
    m = Mapper.__new__(Mapper)
    m.biolink_client = MagicMock()
    m.biolink_client.standardize_entity_type = lambda t: CATEGORY
    m.biolink_client.get_descendants = lambda c: {CATEGORY}
    m.annotation_engine = MagicMock()
    m.annotation_engine.annotate = lambda **k: engine.annotate(**{**k, "mode": "all", "annotators": [REFMET_ANNOTATOR]})
    m.normalizer = MagicMock()
    m.normalizer.get_standard_prefix = lambda v: None
    m.normalizer.normalize = lambda **k: pd.Series(dtype=object)
    m.linker = MagicMock()
    m.linker.link = lambda s: pd.Series({"kg_ids": {}, "kg_ids_provided": {}, "kg_ids_assigned": {}})
    m.linker.get_equivalent_ids_checked = lambda ids, prefixes=None: ({}, True)
    m.tier_b = None
    m.resolver = Resolver(linker=m.linker, biolink_client=m.biolink_client)
    m.resolver.structure_resolver = MagicMock()
    return m


def test_the_engine_surfaces_the_outcome_with_an_empty_vote() -> None:
    ann, _ = _annotator()
    result = _engine_with(ann).annotate(
        item={"name": "5'-Methylthioadenosine"},
        name_field="name",
        provided_id_fields=[],
        category=CATEGORY,
        prefixes=[],
        mode="all",
        annotators=[REFMET_ANNOTATOR],
    )
    assert isinstance(result, pd.Series)
    assert result["annotator_availability"][REFMET_ANNOTATOR] == MATCHED_NO_ID
    assert result["annotator_source"][REFMET_ANNOTATOR] == "live_api"
    assert result["assigned_ids"][REFMET_ANNOTATOR] == {}


def test_the_outcome_reaches_the_result_row_and_the_certificate() -> None:
    ann, _ = _annotator()
    out = _offline_mapper(ann).map_entity_to_kg(
        {"name": "5'-Methylthioadenosine"}, name_field="name", provided_id_fields=[], entity_type="metabolite"
    )
    assert isinstance(out, dict)
    assert out["refmet_availability"] == MATCHED_NO_ID
    assert out["refmet_source"] == "live_api"
    certificate = out["resolution_certificate"]
    assert certificate["refmet_availability"] == MATCHED_NO_ID
    assert certificate["refmet_source"] == "live_api"


def test_the_outcome_appears_in_the_api_response_and_its_own_summary_count() -> None:
    from biomapper2.api.main import app

    names = ["5'-Methylthioadenosine", "Riboflavin", "C0"]
    ann, _ = _annotator()
    with TestClient(app) as client:
        fa = client.app
        assert isinstance(fa, FastAPI)
        original = fa.state.mapper
        fa.state.mapper = _offline_mapper(ann)
        try:
            response = client.post(
                "/api/v1/map/batch",
                json={"entities": [{"name": name, "entity_type": "metabolite"} for name in names]},
            )
        finally:
            fa.state.mapper = original
    assert response.status_code == 200
    body = response.json()
    by_name = {r["name"]: r for r in body["results"]}
    assert [by_name[n]["refmet_availability"] for n in names] == [MATCHED_NO_ID, "voted", "no_match"]
    assert by_name[names[0]]["resolution_certificate"]["refmet_availability"] == MATCHED_NO_ID
    assert by_name[names[0]]["refmet_source"] == "live_api"
    assert body["summary"]["refmet_matched_no_id"] == 1
    assert body["summary"]["refmet_unavailable"] == 0


def test_the_response_models_document_the_new_values() -> None:
    from biomapper2.api.models.responses import EntityMappingResult

    schema = EntityMappingResult.model_json_schema()
    assert MATCHED_NO_ID in schema["properties"]["refmet_availability"]["description"]
    assert KEPT_NO_ID in schema["properties"]["refmet_source"]["description"]


# --------------------------------------------------------------------------------------------
# live_backup: the stored vote is served, the store is left alone
# --------------------------------------------------------------------------------------------


@pytest.fixture
def store_db(monkeypatch, tmp_path):
    monkeypatch.setenv("REFMET_STORE_PATH", str(tmp_path / "store.sqlite"))
    monkeypatch.delenv("REFMET_SNAPSHOT_PATH", raising=False)
    return tmp_path / "store.sqlite"


def test_live_backup_serves_the_stored_vote_for_a_matched_name_with_no_id(store_db) -> None:
    name = "5'-Methylthioadenosine"
    refmet_store.upsert(name, "voted", refmet_id="RM0135763", snapshot_version="mw_match_fixture")
    ann, _ = _annotator("live_backup")

    result = ann._fetch_refmet_data(name)

    assert (result.status, result.source, result.data) == ("voted", KEPT_NO_ID, {"refmet_id": "RM0135763"})
    assert result.version == "mw_match_fixture"
    hit = refmet_store.get(name)
    assert hit is not None and (hit.status, hit.refmet_id, hit.live_status) == ("voted", "RM0135763", None)
    assert refmet_store.list_changes() == []
    assert _breaker().failure_count == 0


@pytest.mark.parametrize("stored", [None, "no_match"])
def test_live_backup_with_no_stored_vote_reports_the_outcome_and_writes_nothing(store_db, stored) -> None:
    name = "3'-O-Methylguanosine"
    if stored is not None:
        refmet_store.upsert(name, stored)
    ann, _ = _annotator("live_backup")

    result = ann._fetch_refmet_data(name)

    assert (result.status, result.source, result.data) == (MATCHED_NO_ID, "live_api", None)
    hit = refmet_store.get(name)
    assert (hit.status if hit is not None else None) == stored
    assert refmet_store.list_changes() == []


# --------------------------------------------------------------------------------------------
# Breaker-open skips are visible at INFO, with a count per batch
# --------------------------------------------------------------------------------------------


def test_breaker_open_skips_are_logged_at_info_with_a_count_per_batch(caplog) -> None:
    ann, session = _annotator()
    session.match_fault = 503
    for name in NO_ID_QUERIES:  # three real failures open the breaker
        ann._fetch_refmet_data(name)
    assert _breaker().opened

    with caplog.at_level("INFO"):
        _batch(ann, NORMAL_QUERIES)
        _batch(ann, NORMAL_QUERIES[:1])

    summaries = [r for r in caplog.records if r.levelname == "INFO" and "circuit open" in r.getMessage()]
    assert len(summaries) == 2  # one line per batch, not one per name
    assert "4 name(s)" in summaries[0].getMessage()
    assert "1 name(s)" in summaries[1].getMessage()
    assert "Riboflavin" in summaries[0].getMessage()


def test_a_breaker_open_skip_outside_a_batch_is_logged_at_info(caplog) -> None:
    ann, session = _annotator()
    session.match_fault = 503
    for name in NO_ID_QUERIES:
        ann._fetch_refmet_data(name)

    with caplog.at_level("INFO"):
        ann._fetch_refmet_data("Riboflavin")

    assert any(
        r.levelname == "INFO" and "circuit open" in r.getMessage() and "Riboflavin" in r.getMessage()
        for r in caplog.records
    )
