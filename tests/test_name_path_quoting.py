"""Slash handling for Workbench / LIPID MAPS path lookups (verified live 2026-10-21).

* A slash between lipid chain specs must be sent literally: the Workbench resolves
  ``/refmet/match/PC%2016%3A0/18%3A1`` to RM0015001 and 404s on the ``%2F`` form.
* Any other slash must be encoded: sent literally, the Workbench answers for the text before the slash
  alone (``mannitol/sorbitol`` -> Mannitol, ``vitamin D2/D3`` -> Vitamin D2), a confident wrong answer;
  encoded it 404s (clean no-match).
* For a path it cannot parse (multi-isomer names with several chain slashes) the Workbench's name
  endpoint answers 200 with a plain-text "This output item does not exist". That is a not-found, never a
  lookup failure, so it must not count toward the Tier B circuit breaker.

Fake sessions only; nothing here touches a live service.
"""

from __future__ import annotations

from typing import Any

import pytest

from biomapper2.utils import MW_UNPARSEABLE_PATH_PREFIX, quote_name_path_segment


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PC 16:0/18:1", "PC%2016%3A0/18%3A1"),
        ("PC P-16:0/20:4", "PC%20P-16%3A0/20%3A4"),
        ("TG 16:0/18:1/18:2", "TG%2016%3A0/18%3A1/18%3A2"),
        ("GalCer 20:0;O3/36:0", "GalCer%2020%3A0%3BO3/36%3A0"),
        ("1,2-dilinoleoyl-GPC (18:2/18:2)", "1%2C2-dilinoleoyl-GPC%20%2818%3A2/18%3A2%29"),
        ("sphingomyelin (d18:1/20:0, d16:1/22:0)*", "sphingomyelin%20%28d18%3A1/20%3A0%2C%20d16%3A1/22%3A0%29%2A"),
        ("mannitol/sorbitol", "mannitol%2Fsorbitol"),
        ("butyrate/isobutyrate (4:0)", "butyrate%2Fisobutyrate%20%284%3A0%29"),
        ("14-hdohe/17-hdohe", "14-hdohe%2F17-hdohe"),
        ("pregnanolone/allopregnanolone sulfate", "pregnanolone%2Fallopregnanolone%20sulfate"),
        ("PA 18:1(5Z)/12:0", "PA%2018%3A1%285Z%29/12%3A0"),
        ("glycosyl-n-(2-hydroxynervonoyl)-sphingosine (d18:1/24:1(2oh))*", None),
        ("vitamin D2/D3", "vitamin%20D2%2FD3"),
        ("carnitine", "carnitine"),
    ],
)
def test_chain_slashes_stay_literal_and_other_slashes_are_encoded(name: str, expected: str | None) -> None:
    got = quote_name_path_segment(name)
    if expected is None:  # only the slash's treatment matters here
        assert "%2F" not in got and got.count("/") == name.count("/"), got
    else:
        assert got == expected


def test_a_mixture_name_is_sent_encoded_to_refmet_match() -> None:
    from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator

    calls: list[str] = []

    class _Resp:
        status_code = 404

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {}

    class _Session:
        def get(self, url: str, timeout: Any = None) -> _Resp:
            calls.append(url)
            return _Resp()

    annotator = MetabolomicsWorkbenchAnnotator.__new__(MetabolomicsWorkbenchAnnotator)
    annotator._session = _Session()  # type: ignore[assignment]
    assert annotator._request_once("mannitol/sorbitol") is None
    assert calls[0] == f"{MetabolomicsWorkbenchAnnotator.BASE_URL}/mannitol%2Fsorbitol"


class _TextResponse:
    status_code = 200
    from_cache = False

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        raise ValueError("not JSON")


def test_structure_resolver_treats_the_unparseable_path_page_as_not_found() -> None:
    from unittest.mock import MagicMock

    from biomapper2.core.structure_resolver import StructureResolver

    resolver = StructureResolver(linker=MagicMock())

    class _Session:
        def get(self, url: str, timeout: Any = None) -> _TextResponse:
            return _TextResponse(f"{MW_UNPARSEABLE_PATH_PREFIX}<br> Choose from: 'all','name','inchi_key'")

    resolver._session = _Session()  # type: ignore[assignment]
    assert resolver._fetch_mw_inchikey("sphingomyelin (d18:1/20:2, d18:2/20:1, d16:1/22:2)*") is None


def test_tier_b_treats_the_unparseable_path_page_as_a_clean_unknown() -> None:
    from biomapper2.core.tier_b import IndependentStructureLookup

    class _Session:
        def get(self, url: str, timeout: Any = None) -> _TextResponse:
            return _TextResponse(f"{MW_UNPARSEABLE_PATH_PREFIX}<br> Choose from: 'all','name','inchi_key'")

    lookup = IndependentStructureLookup(session=_Session(), sleep=lambda _s: None, clock=lambda: 0.0)
    key, _cache, failed = lookup._fetch_mw("sphingomyelin (d18:1/20:2, d18:2/20:1, d16:1/22:2)*")
    assert key is None
    assert failed is False  # a name the Workbench cannot address is not a service failure


class _StatusResponse:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")

    def json(self) -> Any:
        return self._payload


def test_pubchem_not_found_is_a_clean_miss_not_an_exception() -> None:
    from unittest.mock import MagicMock

    from biomapper2.core.structure_resolver import StructureResolver

    resolver = StructureResolver(linker=MagicMock())

    class _Session:
        def get(self, url: str, timeout: Any = None) -> _StatusResponse:
            return _StatusResponse(404, {"Fault": {"Code": "PUGREST.NotFound"}})

    resolver._session = _Session()  # type: ignore[assignment]
    assert resolver._fetch_pubchem_inchikey("PC 16:0_18:1") is None


def test_pubchem_miss_still_reaches_the_lipid_resolver_hop() -> None:
    """A PubChem 404 used to raise out of the MW/PubChem try, so _resolve_name_key returned None
    before ever consulting the lipid resolver. It must now fall through to that hop."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from biomapper2.core.certificate import TierBOutcome
    from biomapper2.core.structure_resolver import StructureResolver

    lipid = MagicMock()
    lipid.resolve.return_value = SimpleNamespace(outcome=TierBOutcome.RESOLVED, inchikey_block="WTJKGGKOPKCXLL")
    resolver = StructureResolver(linker=MagicMock(), lipid_resolver=lipid)
    resolver._frozen_inchikey = lambda _name: None  # type: ignore[method-assign]

    class _Session:
        def get(self, url: str, timeout: Any = None) -> _StatusResponse:
            if "metabolomicsworkbench" in url:
                return _StatusResponse(200, [])  # Workbench: no structure for this name
            return _StatusResponse(404, {"Fault": {"Code": "PUGREST.NotFound"}})

    resolver._session = _Session()  # type: ignore[assignment]
    assert resolver._resolve_name_key("PC 16:0_18:1") == "WTJKGGKOPKCXLL"
    lipid.resolve.assert_called_once_with("PC 16:0_18:1")
