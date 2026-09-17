"""LIPID MAPS REST enrichment (ENRICHMENT ONLY; the circular path against LMSD).

Goslin emits no InChIKey/id, so binding a canonical shorthand to a LIPID MAPS ``LM_ID``/InChIKey is
a DATABASE LOOKUP. Because LMSD's gold *is* LIPID MAPS, this path is circular against the LMSD arm
and MUST stay off in any accuracy configuration. It is provided only as an opt-in enrichment seam:
a caller injects it, and whether it fired is recorded so any number it touched is flagged as
coverage, not independent accuracy. Fail-soft on every error (returns ``{}``).
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

_LIPIDMAPS_REST = "https://www.lipidmaps.org/rest/compound/abbrev/{name}/all/json"
# LIPID MAPS returns a flat object for one hit but a ``{"Row1": {...}, "Row2": {...}}`` envelope for
# many, and the row order is not stable between calls, so a row key is matched by shape, never Row1.
_ROW_KEY_RE = re.compile(r"^Row\d+$")


@runtime_checkable
class LipidEnricher(Protocol):
    def enrich(self, canonical_name: str) -> dict[str, str]: ...

    def enrich_checked(self, canonical_name: str) -> tuple[dict[str, str], bool]:
        """As :meth:`enrich`, plus whether the lookup actually reached a clean answer.

        ``(mapping, ok)`` -- ``ok`` is False ONLY when the service failed (a 5xx or a transport
        error), never for a clean "this registry does not know this lipid" (a 200-empty or a 4xx).
        The two failure shapes are otherwise identical downstream, and a structure resolver that
        cannot tell them apart would cache a transient outage as "the lipid has no known structure".
        """
        ...

    def candidates_checked(self, canonical_name: str) -> tuple[list[dict[str, str]], bool]:
        """Every candidate the registry returned, order-independent (see
        :meth:`LipidMapsRestEnricher.candidates_checked`). Required by the resolver, so it is part of
        the contract, not just the concrete class."""
        ...


class LipidMapsRestEnricher:
    """Bind a Goslin-canonical shorthand -> LIPID MAPS ``LM_ID`` + InChIKey via the REST API."""

    def __init__(self, *, session: Any | None = None, timeout: float = 20.0) -> None:
        self._timeout = timeout
        if session is not None:
            self._session = session
        else:
            import requests

            self._session = requests.Session()

    def enrich(self, canonical_name: str) -> dict[str, str]:
        """Fail-soft bind, returning ``{}`` on any failure. The GoslinLipidAnnotator consumer relies
        on this: it does not catch, so a raise here would break annotation on a transient 5xx."""
        return self.enrich_checked(canonical_name)[0]

    def enrich_checked(self, canonical_name: str) -> tuple[dict[str, str], bool]:
        """Bind and report whether the service actually answered (see :class:`LipidEnricher`).

        Returns a single mapping ONLY when the registry returned exactly one candidate. Zero (no
        match) or many (ambiguous) both return ``{}`` here; a caller that needs the set uses
        :meth:`candidates_checked`. ``ok`` is False only on a network failure (5xx or transport),
        which is what lets :class:`LipidStructureResolver` keep ``lookup_failed`` apart from
        ``unresolvable``.
        """
        candidates, ok = self.candidates_checked(canonical_name)
        if not ok:
            return {}, False
        if len(candidates) != 1:
            return {}, True
        row = candidates[0]
        out: dict[str, str] = {}
        if row["lm_id"]:
            out["LIPIDMAPS"] = row["lm_id"]
        if row["inchi_key"]:
            out["INCHIKEY"] = row["inchi_key"]
        return out, True

    def candidates_checked(self, canonical_name: str) -> tuple[list[dict[str, str]], bool]:
        """Every candidate LIPID MAPS returned, order-independent and deterministically sorted.

        Parses both the flat single-object response and the ``{"Row1": {...}, ...}`` multi-row
        envelope into a list of ``{lm_id, name, abbrev, abbrev_chains, inchi_key}``. Never privileges
        ``Row1`` (the order is not stable). ``ok`` is False only on a network failure; a clean 200
        with no match, or a 4xx, is ``([], True)`` -- the registry saying "unknown lipid".
        """
        if not canonical_name or not str(canonical_name).strip():
            return [], True
        data, ok = self._fetch_json(canonical_name)
        if not ok:
            return [], False
        if not isinstance(data, dict) or not data:
            return [], True
        rows = [v for k, v in data.items() if _ROW_KEY_RE.match(str(k)) and isinstance(v, dict)]
        if not rows:
            rows = [data]  # a flat single-object response
        candidates: list[dict[str, str]] = []
        for row in rows:
            lm_id = row.get("lm_id") or row.get("regno")
            cand = {
                "lm_id": str(lm_id) if lm_id else "",
                "name": str(row.get("name") or ""),
                "abbrev": str(row.get("abbrev") or ""),
                "abbrev_chains": str(row.get("abbrev_chains") or ""),
                "inchi_key": str(row.get("inchi_key") or row.get("inchikey") or ""),
            }
            if cand["lm_id"] or cand["inchi_key"]:  # a row with neither is not a candidate
                candidates.append(cand)
        candidates.sort(key=lambda c: (c["lm_id"], c["inchi_key"]))
        return candidates, True

    def _fetch_json(self, canonical_name: str) -> tuple[Any, bool]:
        """GET the LIPID MAPS response, splitting a network failure (ok=False) from an unknown-lipid
        answer (ok=True, empty). ``safe=""`` encodes the slash-heavy lipid names into one path
        segment; the endpoint still 404s on the encoded slash, which is a clean unknown-lipid here."""
        url = _LIPIDMAPS_REST.format(name=quote(str(canonical_name).strip(), safe=""))
        try:
            resp = self._session.get(url, timeout=self._timeout)
            status = getattr(resp, "status_code", 200)
            if status is not None and status >= 500:
                return None, False
            if status != 200:
                return None, True
            return resp.json(), True
        except Exception:
            return None, False
