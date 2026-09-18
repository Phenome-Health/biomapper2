"""LIPID MAPS multi-row parsing (F4). The REST endpoint returns a flat object for a single hit but a
``{"Row1": {...}, "Row2": {...}, ...}`` envelope for species / molecular-species queries, and the row
order is NOT stable between calls. ``enrich_checked`` read only the top-level ``lm_id``/``inchi_key``,
so every multi-row response silently became ``{}`` (a false unresolvable), and taking ``Row1`` would be
order-dependent. ``candidates_checked`` parses either shape into an order-independent, deterministically
sorted list of candidates.
"""

from __future__ import annotations

from typing import Any

from biomapper2.core.annotators.lipidmaps_rest import LipidMapsRestEnricher


class _FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self._status = status

    def get(self, url: str, timeout: float | None = None) -> _FakeResponse:  # noqa: ARG002
        return _FakeResponse(self._payload, self._status)


_ROW_A = {"lm_id": "LMGP01010005", "name": "PC 16:0/18:1", "abbrev": "PC 34:1", "inchi_key": "AAA-B-N"}
_ROW_B = {"lm_id": "LMGP01010999", "name": "PC 18:1/16:0", "abbrev": "PC 34:1", "inchi_key": "CCC-D-N"}


def _enricher(payload: Any, status: int = 200) -> LipidMapsRestEnricher:
    return LipidMapsRestEnricher(session=_FakeSession(payload, status))


def test_single_flat_object_yields_one_candidate() -> None:
    cands, ok = _enricher(_ROW_A).candidates_checked("PC 16:0/18:1")
    assert ok is True
    assert [c["lm_id"] for c in cands] == ["LMGP01010005"]


def test_multi_row_envelope_yields_the_full_set() -> None:
    cands, ok = _enricher({"Row1": _ROW_A, "Row2": _ROW_B}).candidates_checked("PC 34:1")
    assert ok is True
    assert {c["lm_id"] for c in cands} == {"LMGP01010005", "LMGP01010999"}


def test_row_order_does_not_change_the_output() -> None:
    forward = _enricher({"Row1": _ROW_A, "Row2": _ROW_B}).candidates_checked("PC 34:1")[0]
    reversed_ = _enricher({"Row1": _ROW_B, "Row2": _ROW_A}).candidates_checked("PC 34:1")[0]
    # Deterministic sort (by lm_id) makes the two permutations identical; Row1 is never privileged.
    assert [c["lm_id"] for c in forward] == [c["lm_id"] for c in reversed_]
    assert [c["lm_id"] for c in forward] == ["LMGP01010005", "LMGP01010999"]


def test_empty_body_is_a_clean_no_match() -> None:
    cands, ok = _enricher({}).candidates_checked("not a lipid")
    assert ok is True and cands == []


def test_server_error_is_a_lookup_failure() -> None:
    cands, ok = _enricher({}, status=503).candidates_checked("PC 34:1")
    assert ok is False and cands == []
