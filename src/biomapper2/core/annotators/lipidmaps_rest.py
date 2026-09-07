"""LIPID MAPS REST enrichment (ENRICHMENT ONLY; the circular path against LMSD).

Goslin emits no InChIKey/id, so binding a canonical shorthand to a LIPID MAPS ``LM_ID``/InChIKey is
a DATABASE LOOKUP. Because LMSD's gold *is* LIPID MAPS, this path is circular against the LMSD arm
and MUST stay off in any accuracy configuration. It is provided only as an opt-in enrichment seam:
a caller injects it, and whether it fired is recorded so any number it touched is flagged as
coverage, not independent accuracy. Fail-soft on every error (returns ``{}``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

_LIPIDMAPS_REST = "https://www.lipidmaps.org/rest/compound/abbrev/{name}/all/json"


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

        A 5xx or a transport exception is a property of the network, so ``ok=False``. A clean 200
        with an empty/unmatched body, or a 4xx, is the registry saying "unknown lipid": ``ok=True``
        with ``{}``. Keeping the two apart is what lets :class:`LipidStructureResolver` distinguish
        ``lookup_failed`` (not cached) from ``unresolvable``.
        """
        if not canonical_name or not str(canonical_name).strip():
            return {}, True
        # safe="" -- LIPID MAPS is path-segment addressed and lipid names are full of slashes
        # ("PC 16:0/18:1"). The default safe="/" leaves the slash intact, which adds a path segment
        # and silently returns nothing. Exposure per arm: artifact field ``slash_bearing_name_rate``.
        url = _LIPIDMAPS_REST.format(name=quote(str(canonical_name).strip(), safe=""))
        try:
            resp = self._session.get(url, timeout=self._timeout)
            status = getattr(resp, "status_code", 200)
            if status is not None and status >= 500:
                # A server error is not "unknown lipid". Surface it so the caller records a
                # lookup_failed rather than caching a transient outage as a durable miss.
                return {}, False
            if status != 200:
                return {}, True  # 4xx / other non-200: a clean "this registry does not know it"
            data = resp.json()
        except Exception:
            return {}, False
        if not isinstance(data, dict):
            return {}, True
        out: dict[str, str] = {}
        lm_id = data.get("lm_id") or data.get("regno")
        inchikey = data.get("inchi_key") or data.get("inchikey")
        if lm_id:
            out["LIPIDMAPS"] = str(lm_id)
        if inchikey:
            out["INCHIKEY"] = str(inchikey)
        return out, True
