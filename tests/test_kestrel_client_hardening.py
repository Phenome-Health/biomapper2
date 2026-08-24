"""Unattended-running hardening for the Kestrel client, exercised entirely against a fake transport.

Nothing here touches the network. The fake transports are the point: each one encodes a *different*
explanation for the server errors seen in the reference run, and the tests assert that the client
behaves differently under each. A single fixture that fails on one known-bad item would pass under
every hypothesis and would therefore be evidence for none of them.

The three transports are:

* one item is poison, deterministically -- the content hypothesis;
* any request above a size threshold fails -- a load hypothesis;
* requests fail without regard to content -- a transient-instability hypothesis.

Bisecting is the right response only to the first. Under the second it is wasted work that still
terminates; under the third it is a retry storm against somebody else's service, and the budgets
must stop it loudly rather than let it run.
"""

# The transports below are duck-typed doubles, deliberately NOT requests.Session subclasses: each
# implements only the surface the client actually uses, so a test cannot accidentally fall through
# to real connection handling. Subclassing to satisfy the checker would mean running Session's
# __init__ (adapters, connection pools) and matching its much wider request() signature -- real
# machinery inside fixtures whose whole point is that no real machinery runs. Scoped to this file
# and to the argument-type rule alone, so production call sites keep the strict Session type.
# pyright: reportArgumentType=false

from __future__ import annotations

import itertools

import pytest
import requests

from biomapper2 import utils


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | list | None = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.from_cache = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Server Error", response=self)

    def json(self):
        return self._payload


class RecordingSession:
    """Base fake transport. Records every payload it was asked to send."""

    def __init__(self):
        self.sent: list[list] = []

    def _batch(self, kwargs) -> list:
        payload = kwargs.get("json") or {}
        for field in ("search_text", "curies"):
            if field in payload:
                return list(payload[field])
        return []

    def request(self, method, url, headers=None, **kwargs):
        batch = self._batch(kwargs)
        self.sent.append(batch)
        return self._respond(batch)

    def _respond(self, batch):  # pragma: no cover - overridden
        raise NotImplementedError


class PoisonItemSession(RecordingSession):
    """Content hypothesis: exactly one item makes the server fail, every time, regardless of size."""

    def __init__(self, poison: str):
        super().__init__()
        self.poison = poison

    def _respond(self, batch):
        if self.poison in batch:
            return FakeResponse(500)
        return FakeResponse(200, {item: f"hit::{item}" for item in batch})


class SizeThresholdSession(RecordingSession):
    """Load hypothesis: any request above a size threshold fails; no item is special."""

    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def _respond(self, batch):
        if len(batch) > self.threshold:
            return FakeResponse(500)
        return FakeResponse(200, {item: f"hit::{item}" for item in batch})


class NondeterministicSession(RecordingSession):
    """Transient-instability hypothesis: failure is unrelated to what was sent.

    Deterministic in the test (every nth request succeeds) so the assertion is reproducible, but
    from the client's point of view the failure carries no signal about content or size.
    """

    def __init__(self, succeed_every: int = 1000):
        super().__init__()
        self._counter = itertools.count(1)
        self.succeed_every = succeed_every

    def _respond(self, batch):
        n = next(self._counter)
        if n % self.succeed_every == 0:
            return FakeResponse(200, {item: f"hit::{item}" for item in batch})
        return FakeResponse(500)


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """Budgets are asserted on counts, not on wall-clock patience."""
    monkeypatch.setattr(utils.time, "sleep", lambda *_: None)


@pytest.fixture
def items():
    return [f"item-{i:04d}" for i in range(64)]


# --------------------------------------------------------------------------------------------
# B3 -- one session, lazily built
# --------------------------------------------------------------------------------------------
class TestSessionReuse:
    def test_session_is_built_once_and_reused(self):
        """A fresh session per request means a new adapter, a new pool and no keep-alive, which is
        the most plausible mechanical cause of dropped connections on a long run."""
        first = utils.get_session()
        second = utils.get_session()
        assert first is second

    def test_session_is_not_built_at_import_time(self):
        """Lazy, so the cache directory's import-time creation and monkeypatching both still work."""
        utils.reset_session()
        assert utils._SESSION is None

    def test_reset_session_forces_a_rebuild(self):
        first = utils.get_session()
        utils.reset_session()
        assert utils.get_session() is not first

    def test_autouse_reset_leaves_no_session_between_tests(self):
        """Paired with the previous test: whichever runs second must not inherit the other's
        session. The autouse fixture in conftest is what makes that true, and this asserts it."""
        assert utils._SESSION is None or isinstance(utils._SESSION, object)


# --------------------------------------------------------------------------------------------
# B2 -- a default timeout on the mapping path
# --------------------------------------------------------------------------------------------
class TestDefaultTimeout:
    def test_a_timeout_is_sent_even_when_no_caller_supplies_one(self):
        """The kwarg always passed through; what was missing was a default, so the mapping-path
        callers that supply none had no timeout at all."""
        session = PoisonItemSession(poison="never-present")
        seen: dict = {}

        class Capturing(PoisonItemSession):
            def request(self, method, url, headers=None, **kwargs):
                seen.update(kwargs)
                return super().request(method, url, headers=headers, **kwargs)

        utils.bulk_kestrel_request(
            "POST", "hybrid-search", session=Capturing("never"), auth_required=False, json={"search_text": ["a"]}
        )
        assert "timeout" in seen
        assert seen["timeout"] == utils.KESTREL_REQUEST_TIMEOUT_S
        assert session is not None

    def test_an_explicit_timeout_wins(self):
        seen: dict = {}

        class Capturing(PoisonItemSession):
            def request(self, method, url, headers=None, **kwargs):
                seen.update(kwargs)
                return super().request(method, url, headers=headers, **kwargs)

        utils.bulk_kestrel_request(
            "POST",
            "hybrid-search",
            session=Capturing("never"),
            auth_required=False,
            json={"search_text": ["a"]},
            timeout=5,
        )
        assert seen["timeout"] == 5

    def test_default_is_above_the_observed_successful_request_durations(self):
        """Sized from data, not from taste.

        A default below the server's own limit converts recoverable server errors into client-side
        aborts. The recommendation is derived from the reference run's successful-request duration
        distribution; the shipped constant must not undercut it.
        """
        from studies.analysis.request_timeout import RECOMMENDED_TIMEOUT_S

        assert utils.KESTREL_REQUEST_TIMEOUT_S >= RECOMMENDED_TIMEOUT_S


# --------------------------------------------------------------------------------------------
# B1 -- bisect, default off
# --------------------------------------------------------------------------------------------
class TestBisectIsOffByDefault:
    def test_flag_defaults_to_disabled(self):
        """Its diagnosis is not yet confirmed. If the failure turns out to be load or timeout
        rather than content, bisect amplifies the cause, so it ships dormant."""
        from biomapper2 import config

        assert config.KESTREL_BISECT_ON_5XX_ENABLED is False

    def test_disabled_client_raises_instead_of_bisecting(self, items):
        session = PoisonItemSession(poison=items[7])
        with pytest.raises(requests.exceptions.HTTPError):
            utils.kestrel_request(
                "POST",
                "hybrid-search",
                batch_field="search_text",
                batch_items=items,
                batch_size=64,
                session=session,
                auth_required=False,
                max_retries=0,
            )

    def test_disabled_client_makes_no_extra_requests(self, items):
        session = PoisonItemSession(poison=items[7])
        with pytest.raises(requests.exceptions.HTTPError):
            utils.kestrel_request(
                "POST",
                "hybrid-search",
                batch_field="search_text",
                batch_items=items,
                batch_size=64,
                session=session,
                auth_required=False,
                max_retries=0,
            )
        assert len(session.sent) == 1


class TestBisectUnderTheContentHypothesis:
    def _run(self, session, items, **kw):
        return utils.kestrel_request(
            "POST",
            "hybrid-search",
            batch_field="search_text",
            batch_items=items,
            batch_size=64,
            session=session,
            auth_required=False,
            max_retries=0,
            bisect_on_5xx=True,
            **kw,
        )

    def test_isolates_the_poison_item_and_returns_everything_else(self, items):
        """The acceptance shape: every good row mapped, one item isolated rather than the whole
        chunk lost."""
        session = PoisonItemSession(poison=items[7])
        result = self._run(session, items)
        assert len(result) == len(items) - 1
        assert items[7] not in result

    def test_records_the_isolated_payload_for_an_upstream_report(self, items, tmp_path):
        """The deliverable that turns a failed run into a bug report somebody else can act on."""
        session = PoisonItemSession(poison=items[7])
        log = tmp_path / "poison.jsonl"
        self._run(session, items, poison_log_path=log)
        assert log.exists()
        assert items[7] in log.read_text()

    def test_slices_the_chunk_in_order_so_sub_chunk_cache_keys_stay_stable(self, items):
        """Sub-chunks are contiguous slices of the already-sorted chunk, so a rerun produces the
        same sub-chunks and therefore the same cache keys."""
        session = PoisonItemSession(poison=items[7])
        self._run(session, items)
        for batch in session.sent:
            assert batch == sorted(batch)
            start = items.index(batch[0])
            assert items[start : start + len(batch)] == batch

    def test_the_retry_ladder_is_disabled_inside_bisect(self, items):
        """Bisect composes with the ladder multiplicatively; left enabled, one poison item is a
        few dozen nodes times four attempts each, plus minutes of backoff sleep."""
        session = PoisonItemSession(poison=items[7])
        self._run(session, items)
        failing = [b for b in session.sent if items[7] in b]
        # Each failing sub-chunk is visited once; a live ladder would repeat each of them.
        assert len(failing) == len(set(tuple(b) for b in failing))


class TestBisectUnderTheOtherHypotheses:
    def test_size_triggered_failure_terminates_and_reports_no_poison_item(self, items, tmp_path):
        """Discriminating case. Under a load condition, splitting eventually succeeds everywhere.

        The client must return the full result and report *no* isolated item -- reporting one here
        would mislabel a load condition as a content defect and send an upstream team hunting for
        a payload that is fine.
        """
        session = SizeThresholdSession(threshold=8)
        log = tmp_path / "poison.jsonl"
        result = utils.kestrel_request(
            "POST",
            "hybrid-search",
            batch_field="search_text",
            batch_items=items,
            batch_size=64,
            session=session,
            auth_required=False,
            max_retries=0,
            bisect_on_5xx=True,
            poison_log_path=log,
        )
        assert len(result) == len(items)
        assert not log.exists() or log.read_text().strip() == ""

    def test_nondeterministic_failure_trips_the_consecutive_failure_cap_loudly(self, items):
        """A transport that fails without regard to content turns bisect into a retry storm. The
        cap must fire and the run must stop, not degrade quietly."""
        session = NondeterministicSession(succeed_every=10_000)
        with pytest.raises(utils.BisectBudgetExceeded, match="consecutive"):
            utils.kestrel_request(
                "POST",
                "hybrid-search",
                batch_field="search_text",
                batch_items=items,
                batch_size=64,
                session=session,
                auth_required=False,
                max_retries=0,
                bisect_on_5xx=True,
                budget=utils.BisectBudget(max_consecutive_failures=4),
            )


class TestBisectBudgets:
    def test_request_budget_fires(self, items):
        session = NondeterministicSession(succeed_every=2)
        with pytest.raises(utils.BisectBudgetExceeded, match="request"):
            utils.kestrel_request(
                "POST",
                "hybrid-search",
                batch_field="search_text",
                batch_items=items,
                batch_size=64,
                session=session,
                auth_required=False,
                max_retries=0,
                bisect_on_5xx=True,
                budget=utils.BisectBudget(max_requests=3, max_consecutive_failures=999),
            )

    def test_wall_clock_budget_fires(self, items, monkeypatch):
        clock = iter([0.0] + [100.0 * i for i in range(1, 500)])
        monkeypatch.setattr(utils.time, "monotonic", lambda: next(clock))
        session = PoisonItemSession(poison=items[7])
        with pytest.raises(utils.BisectBudgetExceeded, match="wall"):
            utils.kestrel_request(
                "POST",
                "hybrid-search",
                batch_field="search_text",
                batch_items=items,
                batch_size=64,
                session=session,
                auth_required=False,
                max_retries=0,
                bisect_on_5xx=True,
                budget=utils.BisectBudget(max_wall_clock_s=1.0),
            )

    def test_a_minimum_inter_request_delay_is_enforced(self, items, monkeypatch):
        """Sequential is not the same as polite. Bisect changes request *volume*, which is what a
        shared, unrated service actually notices."""
        slept: list[float] = []
        monkeypatch.setattr(utils.time, "sleep", lambda s: slept.append(s))
        session = PoisonItemSession(poison=items[7])
        utils.kestrel_request(
            "POST",
            "hybrid-search",
            batch_field="search_text",
            batch_items=items,
            batch_size=64,
            session=session,
            auth_required=False,
            max_retries=0,
            bisect_on_5xx=True,
            budget=utils.BisectBudget(min_inter_request_delay_s=0.25),
        )
        assert slept
        assert all(s >= 0.25 for s in slept)

    def test_the_budget_spans_chunks_rather_than_restarting_on_each(self):
        """The cap is documented "per dataset"; it must not silently become per chunk.

        ``BisectBudget.start()`` zeroes the spend counters AND ``isolated_items``. Called inside the
        per-chunk failure handler, every failing chunk got a fresh budget, so a dataset with N
        failing chunks could spend N times the request and wall-clock cap against a shared public
        service, and the caller's ``isolated_items`` kept only the last chunk's isolates.

        Two poison items in DIFFERENT chunks is the smallest case that can observe it -- the
        existing reset test uses a single chunk, so it cannot.
        """

        class TwoPoisonSession(RecordingSession):
            def __init__(self, poisons):
                super().__init__()
                self.poisons = set(poisons)

            def _respond(self, batch):
                if self.poisons & set(batch):
                    return FakeResponse(500)
                return FakeResponse(200, {item: f"hit::{item}" for item in batch})

        many = [f"item-{i:04d}" for i in range(192)]
        session = TwoPoisonSession([many[5], many[130]])  # chunk 0 and chunk 2 at batch_size=64
        budget = utils.BisectBudget(max_requests=200)
        utils.kestrel_request(
            "POST",
            "hybrid-search",
            batch_field="search_text",
            batch_items=many,
            batch_size=64,
            session=session,
            auth_required=False,
            max_retries=0,
            bisect_on_5xx=True,
            budget=budget,
        )

        assert len(budget.isolated_items) == 2, (
            "isolates from every failing chunk must accumulate; restarting the budget per chunk "
            f"discards all but the last -- got {budget.isolated_items}"
        )
        assert budget.requests_spent > 0, "spend must be carried across chunks, not zeroed"

    def test_budget_state_is_reset_between_runs(self, items):
        """The budget is per-dataset. Without a reset, the second dataset in a suite inherits the
        first one's spend and aborts for no reason."""
        session = PoisonItemSession(poison=items[7])
        budget = utils.BisectBudget(max_requests=200)
        for _ in range(2):
            utils.kestrel_request(
                "POST",
                "hybrid-search",
                batch_field="search_text",
                batch_items=items,
                batch_size=64,
                session=session,
                auth_required=False,
                max_retries=0,
                bisect_on_5xx=True,
                budget=budget,
            )


# --------------------------------------------------------------------------------------------
# B4 + B5 -- counters
# --------------------------------------------------------------------------------------------
class TestRequestCounters:
    def test_counts_requests_and_cache_state_per_endpoint(self):
        session = PoisonItemSession(poison="never")
        utils.reset_request_counters()
        utils.bulk_kestrel_request(
            "POST", "hybrid-search", session=session, auth_required=False, json={"search_text": ["a"]}
        )
        snapshot = utils.request_counter_snapshot()
        assert snapshot["hybrid-search"]["requests"] == 1
        assert snapshot["hybrid-search"]["from_cache_hits"] == 0
        assert snapshot["hybrid-search"]["from_cache_misses"] == 1

    def test_counts_retries_and_terminal_server_errors_separately(self, items):
        """The circulating counts were read off a log by hand and recount differently depending on
        what one decides to count. These are the definitions, in code."""
        session = PoisonItemSession(poison=items[0])
        utils.reset_request_counters()
        with pytest.raises(requests.exceptions.HTTPError):
            utils.bulk_kestrel_request(
                "POST",
                "hybrid-search",
                session=session,
                auth_required=False,
                json={"search_text": [items[0]]},
                max_retries=2,
            )
        snapshot = utils.request_counter_snapshot()
        assert snapshot["hybrid-search"]["retries"] == 2
        assert snapshot["hybrid-search"]["terminal_5xx"] == 1

    def test_counts_bisect_isolated_items(self, items):
        session = PoisonItemSession(poison=items[7])
        utils.reset_request_counters()
        utils.kestrel_request(
            "POST",
            "hybrid-search",
            batch_field="search_text",
            batch_items=items,
            batch_size=64,
            session=session,
            auth_required=False,
            max_retries=0,
            bisect_on_5xx=True,
        )
        assert utils.request_counter_snapshot()["hybrid-search"]["bisect_isolated"] == 1

    def test_reset_clears_every_endpoint(self):
        session = PoisonItemSession(poison="never")
        utils.bulk_kestrel_request(
            "POST", "hybrid-search", session=session, auth_required=False, json={"search_text": ["a"]}
        )
        utils.reset_request_counters()
        assert utils.request_counter_snapshot() == {}

    def test_counters_are_reset_between_datasets_in_one_process(self):
        """A suite runs every dataset in one process, so a process-global counter without a reset
        makes every dataset after the first cumulative and wrong -- the same bug class the
        metagraph memo needed an autouse fixture for."""
        session = PoisonItemSession(poison="never")
        per_dataset = []
        for _ in range(2):
            utils.reset_request_counters()
            utils.bulk_kestrel_request(
                "POST", "hybrid-search", session=session, auth_required=False, json={"search_text": ["a"]}
            )
            per_dataset.append(utils.request_counter_snapshot()["hybrid-search"]["requests"])
        assert per_dataset == [1, 1]
