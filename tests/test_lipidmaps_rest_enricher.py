from biomapper2.core.annotators.lipidmaps_rest import LipidMapsRestEnricher


class _FakeResp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        return self._resp


def test_enrich_extracts_lm_id_and_inchikey():
    session = _FakeSession(_FakeResp(200, {"lm_id": "LMGP01010001", "inchi_key": "KILNVBDSWZSGLL-KXQOOQHDSA-N"}))
    enr = LipidMapsRestEnricher(session=session)
    out = enr.enrich("PC 16:0/16:0")
    assert out["LIPIDMAPS"] == "LMGP01010001"
    assert out["INCHIKEY"] == "KILNVBDSWZSGLL-KXQOOQHDSA-N"


def test_enrich_is_fail_soft_on_http_error():
    enr = LipidMapsRestEnricher(session=_FakeSession(_FakeResp(500)))
    assert enr.enrich("PC 16:0/16:0") == {}


def test_enrich_is_fail_soft_on_unparseable_body():
    enr = LipidMapsRestEnricher(session=_FakeSession(_FakeResp(200, None)))
    assert enr.enrich("PC 16:0/16:0") == {}


def test_enrich_blank_name_makes_no_call():
    session = _FakeSession(_FakeResp(200, {}))
    enr = LipidMapsRestEnricher(session=session)
    assert enr.enrich("") == {}
    assert session.calls == 0


def test_a_slash_in_the_lipid_name_is_sent_literally():
    """LIPID MAPS answers a literal slash in the abbreviation and 404s on an encoded ``%2F``.

    Live (2026-10-21): ``/rest/compound/abbrev/PC%2016%3A0/18%3A1/all/json`` returns a normal
    JSON answer; the ``%2F`` form returns the web server's 404 page before any lookup.
    """
    from biomapper2.core.annotators.lipidmaps_rest import _LIPIDMAPS_REST, LipidMapsRestEnricher

    calls = []

    class _Resp:
        status_code = 200

        def json(self):
            return {}

    class _Session:
        def get(self, url, timeout=None):
            calls.append(url)
            return _Resp()

    LipidMapsRestEnricher(session=_Session()).enrich("PC 16:0/18:1")

    url = calls[0]
    assert "%2F" not in url.upper(), f"the slash was percent-encoded, which LIPID MAPS 404s: {url}"
    assert url == _LIPIDMAPS_REST.format(name="PC%2016%3A0/18%3A1"), url


# --- Unit 1: failure signal (enrich_checked) ------------------------------------------------
# enrich() must keep its {}-on-any-failure contract (GoslinLipidAnnotator depends on it), while
# enrich_checked() distinguishes a network/5xx FAILURE (ok=False, never cached as "unknown") from a
# clean "this registry does not know this lipid" answer (ok=True, {}).


def test_enrich_checked_known_lipid_returns_ids_and_ok():
    session = _FakeSession(_FakeResp(200, {"lm_id": "LMGP01010001", "inchi_key": "KILNVBDSWZSGLL-KXQOOQHDSA-N"}))
    out, ok = LipidMapsRestEnricher(session=session).enrich_checked("PC 16:0/16:0")
    assert ok is True
    assert out["LIPIDMAPS"] == "LMGP01010001"
    assert out["INCHIKEY"] == "KILNVBDSWZSGLL-KXQOOQHDSA-N"


def test_enrich_checked_unknown_lipid_200_empty_is_clean_not_a_failure():
    # A clean 200 with an empty body means "unknown lipid", not a broken service.
    out, ok = LipidMapsRestEnricher(session=_FakeSession(_FakeResp(200, {}))).enrich_checked("PC 99:9/99:9")
    assert out == {}
    assert ok is True


def test_enrich_checked_5xx_is_a_failure_signal():
    out, ok = LipidMapsRestEnricher(session=_FakeSession(_FakeResp(503))).enrich_checked("PC 16:0/16:0")
    assert out == {}
    assert ok is False, "a 5xx must surface as a failure, not a clean 'unknown lipid'"


def test_enrich_checked_transport_error_is_a_failure_signal():
    class _Boom:
        def get(self, url, timeout=None):
            raise ConnectionError("network down")

    out, ok = LipidMapsRestEnricher(session=_Boom()).enrich_checked("PC 16:0/16:0")
    assert out == {}
    assert ok is False


def test_enrich_checked_4xx_is_clean_not_a_failure():
    out, ok = LipidMapsRestEnricher(session=_FakeSession(_FakeResp(404))).enrich_checked("PC 16:0/16:0")
    assert out == {}
    assert ok is True


def test_enrich_stays_fail_soft_on_5xx_for_the_annotator_consumer():
    # GoslinLipidAnnotator calls enrich() and does not catch — the {}-on-failure contract must hold.
    assert LipidMapsRestEnricher(session=_FakeSession(_FakeResp(503))).enrich("PC 16:0/16:0") == {}
