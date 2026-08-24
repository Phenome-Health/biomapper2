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
        if not canonical_name or not str(canonical_name).strip():
            return {}
        # safe="" -- LIPID MAPS is path-segment addressed and lipid names are full of slashes
        # ("PC 16:0/18:1"). The default safe="/" leaves the slash intact, which adds a path segment
        # and silently returns nothing. Exposure per arm: artifact field ``slash_bearing_name_rate``.
        url = _LIPIDMAPS_REST.format(name=quote(str(canonical_name).strip(), safe=""))
        try:
            resp = self._session.get(url, timeout=self._timeout)
            if getattr(resp, "status_code", None) != 200:
                return {}
            data = resp.json()
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, str] = {}
        lm_id = data.get("lm_id") or data.get("regno")
        inchikey = data.get("inchi_key") or data.get("inchikey")
        if lm_id:
            out["LIPIDMAPS"] = str(lm_id)
        if inchikey:
            out["INCHIKEY"] = str(inchikey)
        return out
