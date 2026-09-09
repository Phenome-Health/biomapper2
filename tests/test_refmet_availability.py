"""RefMet annotator: availability status + bounded resilience (U0 + U1).

Mocked HTTP only — never a live call. The circuit breaker is process-global (decorated once at class
definition), so it is reset before each test or one test's trip leaks into the next.

The honesty property this file guards (U0, in its regression form): once the breaker trips mid-batch,
the remaining names are reported UNAVAILABLE, NOT silently blanked as a no-match. On the pre-fix code
there was no availability signal at all, so ``test_mid_batch_unavailable_is_not_blanked_as_no_match``
fails if the annotator change is reverted.
"""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
from circuitbreaker import CircuitBreakerMonitor

from biomapper2.core.annotators.base import (
    AVAILABILITY_NO_MATCH,
    AVAILABILITY_NOT_QUERIED,
    AVAILABILITY_UNAVAILABLE,
    AVAILABILITY_VOTED,
)
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator

SLUG = MetabolomicsWorkbenchAnnotator.slug
CATEGORY = "biolink:SmallMolecule"
_BREAKER_NAME = "MetabolomicsWorkbenchAnnotator._do_refmet_request"


def _breaker():
    return CircuitBreakerMonitor.get(_BREAKER_NAME)


@pytest.fixture(autouse=True)
def _reset_breaker():
    """Close the shared breaker and zero its failure count around every test."""
    breaker = _breaker()
    if breaker is not None:
        breaker.reset()
    yield
    if breaker is not None:
        breaker.reset()


def _annotator(**kwargs) -> MetabolomicsWorkbenchAnnotator:
    """An annotator with a no-op sleep so tests never block; other tunables overridable."""
    kwargs.setdefault("sleep", lambda _seconds: None)
    return MetabolomicsWorkbenchAnnotator(**kwargs)


def _raise_timeout(metabolite_name):
    raise requests.exceptions.Timeout("slow")


def test_mid_batch_unavailable_is_not_blanked_as_no_match():
    """After the breaker trips, the tail is UNAVAILABLE (detectable), not an empty no-match vote.

    Without the availability side-channel these tail rows return ``{slug: {}}`` — byte-identical to a
    genuine no-match — which is exactly the silent blanking this change removes.
    """
    ann = _annotator(max_retries=0)  # one attempt per name, so three failures trip the breaker
    failing = {"n0", "n1", "n2"}
    ann._request_once = lambda metabolite_name: (
        _raise_timeout(metabolite_name) if metabolite_name in failing else {"refmet_id": f"RM_{metabolite_name}"}
    )

    df = pd.DataFrame({"name": [f"n{i}" for i in range(6)]})
    cache = ann.build_availability_cache(df, "name")
    votes = ann.get_annotations_bulk(df, "name", CATEGORY, cache=cache)

    # n5 would have voted, but the breaker opened after the first three failures.
    tail = df.iloc[5]
    assert ann.get_availability(tail, "name", cache=cache) == {SLUG: AVAILABILITY_UNAVAILABLE}
    # The vote itself is empty — indistinguishable from a no-match, which is why availability is separate.
    assert votes.iloc[5] == {SLUG: {}}


def test_genuine_dash_is_no_match_not_unavailable():
    """A 200 whose refmet_id is '-' (surfaced as a None body) is a no-match, not an outage."""
    ann = _annotator()
    ann._request_once = lambda metabolite_name: None
    result = ann._fetch_refmet_data("NonexistentMetabolite")
    assert result.status == AVAILABILITY_NO_MATCH
    assert result.data is None


def test_one_off_slow_call_succeeds_on_retry_and_breaker_stays_zero():
    """A single transient failure is rescued by the in-decorator retry; no failure is counted."""
    ann = _annotator(max_retries=1)
    attempts = itertools.count()

    def flaky(metabolite_name):
        if next(attempts) == 0:
            raise requests.exceptions.Timeout("slow")
        return {"refmet_id": "RM1"}

    ann._request_once = flaky
    result = ann._fetch_refmet_data("Carnitine")
    assert result.status == AVAILABILITY_VOTED
    assert result.data == {"refmet_id": "RM1"}
    assert _breaker().failure_count == 0


def test_ultimate_failure_increments_breaker_by_exactly_one():
    """Both attempts fail, so the decorated call raises once — exactly one breaker failure, not two."""
    ann = _annotator(max_retries=1)
    ann._request_once = _raise_timeout
    result = ann._fetch_refmet_data("Carnitine")
    assert result.status == AVAILABILITY_UNAVAILABLE
    assert _breaker().failure_count == 1


def test_per_batch_deadline_marks_the_tail_unavailable_without_network():
    """Past the hard deadline, remaining names are UNAVAILABLE with no network call."""
    # deadline read once, then one clock read per name; advance past the deadline after the first name.
    ticks = itertools.chain([0.0, 0.0, 100.0, 100.0], itertools.repeat(100.0))
    ann = _annotator(batch_deadline_s=10.0, clock=lambda: next(ticks))
    called: list[str] = []

    def record(metabolite_name):
        called.append(metabolite_name)
        return {"refmet_id": "RM"}

    ann._request_once = record
    df = pd.DataFrame({"name": ["a", "b", "c"]})
    cache = ann.build_availability_cache(df, "name")

    assert cache["a"].status == AVAILABILITY_VOTED
    assert cache["b"].status == AVAILABILITY_UNAVAILABLE
    assert cache["c"].status == AVAILABILITY_UNAVAILABLE
    assert called == ["a"]  # b and c never reached the network


def test_voted_result_carries_the_id_in_both_the_vote_and_availability():
    ann = _annotator()
    ann._request_once = lambda metabolite_name: {"refmet_id": "RM0008606"}
    df = pd.DataFrame({"name": ["Carnitine"]})
    cache = ann.build_availability_cache(df, "name")
    votes = ann.get_annotations_bulk(df, "name", CATEGORY, cache=cache)
    assert votes.iloc[0] == {SLUG: {"refmet_id": {"RM0008606": {}}}}
    assert ann.get_availability(df.iloc[0], "name", cache=cache) == {SLUG: AVAILABILITY_VOTED}
    # A row with no name is never queried.
    assert ann.get_availability(pd.Series({"name": None}), "name", cache=cache) == {SLUG: AVAILABILITY_NOT_QUERIED}


def test_non_dict_success_is_unavailable_not_no_match():
    # A 200 whose JSON body is not a dict (null/array/proxy blob) is a DEGRADED response, not RefMet's
    # dash negative — it must classify UNAVAILABLE, never a genuine no-match.
    ann = _annotator(max_retries=0)
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: []  # valid JSON, wrong shape
    ann._session = MagicMock()
    ann._session.get = lambda url, timeout: resp
    result = ann._fetch_refmet_data("weird")
    assert result.status == AVAILABILITY_UNAVAILABLE


def test_armed_deadline_bounds_a_per_entity_loop_without_network():
    # The API /batch route arms the shared deadline once, then maps entities one at a time. Once the
    # deadline passes, each subsequent per-entity fetch is UNAVAILABLE with NO network call.
    ticks = itertools.chain([0.0, 0.0, 100.0, 100.0], itertools.repeat(100.0))
    ann = _annotator(batch_deadline_s=10.0, clock=lambda: next(ticks))
    called: list[str] = []
    ann._request_once = lambda metabolite_name: (called.append(metabolite_name), {"refmet_id": "RM"})[1]
    ann.arm_batch_deadline()  # arm read: 0.0 -> deadline 10.0
    try:
        first = ann._fetch_refmet_data("a")  # clock 0.0 < 10 -> network
        second = ann._fetch_refmet_data("b")  # clock 100 >= 10 -> UNAVAILABLE, no network
    finally:
        ann.disarm_batch_deadline()
    assert first.status == AVAILABILITY_VOTED
    assert second.status == AVAILABILITY_UNAVAILABLE
    assert called == ["a"]
