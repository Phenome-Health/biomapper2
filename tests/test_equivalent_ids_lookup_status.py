"""A Kestrel outage must not be indistinguishable from "the graph asserts no structure".

``Linker.get_equivalent_ids`` returns ``{}`` on any exception and only logs a warning, so a
transient ``/get-nodes`` failure would mark every row of a run ``structure_absent`` -> ``unavailable``
and an offline rerun on the resulting TSV could never detect it. The certificate therefore needs the
success flag, not just the payload.
"""

from __future__ import annotations

import pytest

from biomapper2.core.linker import Linker


def test_checked_lookup_reports_success_and_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "biomapper2.core.linker.kestrel_request",
        lambda **_: {"CHEBI:15365": {"equivalent_ids": ["HMDB:HMDB0001879", "INCHIKEY:BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]}},
    )
    mapping, ok = Linker.get_equivalent_ids_checked(["CHEBI:15365"])
    assert ok is True
    assert mapping["CHEBI:15365"]["INCHIKEY"] == ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]


def test_checked_lookup_reports_failure_distinctly_from_an_empty_graph_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(**_):
        raise RuntimeError("kestrel is down")

    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", _boom)
    mapping, ok = Linker.get_equivalent_ids_checked(["CHEBI:15365"])
    assert mapping == {}
    assert ok is False

    # A node the graph genuinely knows nothing about is a SUCCESSFUL lookup with no payload.
    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", lambda **_: {})
    mapping, ok = Linker.get_equivalent_ids_checked(["CHEBI:15365"])
    assert mapping == {}
    assert ok is True


def test_no_nodes_to_look_up_is_a_successful_no_op() -> None:
    assert Linker.get_equivalent_ids_checked([]) == ({}, True)


def test_legacy_entry_point_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing signature keeps its contract; the flag is additive."""

    def _boom(**_):
        raise RuntimeError("kestrel is down")

    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", _boom)
    assert Linker.get_equivalent_ids(["CHEBI:15365"]) == {}
