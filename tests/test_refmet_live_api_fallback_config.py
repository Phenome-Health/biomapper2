"""REFMET_LIVE_API_FALLBACK config toggle + AnnotationEngine wiring.

The RefMet annotator's freeze-miss fallback to live /match is off by default and enabled per-deployment
via the env var. This pins the config reader and confirms the engine constructs the annotator with the
resolved value (so a deployment that pins a corpus-bound freeze can still resolve novel names live).
No network.
"""

from __future__ import annotations

import pytest

from biomapper2.config import get_refmet_live_api_fallback
from biomapper2.core.annotation_engine import AnnotationEngine
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator

pytestmark = pytest.mark.unit


def test_config_reader_parses_truthy_values(monkeypatch):
    monkeypatch.delenv("REFMET_LIVE_API_FALLBACK", raising=False)
    assert get_refmet_live_api_fallback() is False  # unset -> off
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("REFMET_LIVE_API_FALLBACK", truthy)
        assert get_refmet_live_api_fallback() is True
    for falsy in ("0", "false", "", "no"):
        monkeypatch.setenv("REFMET_LIVE_API_FALLBACK", falsy)
        assert get_refmet_live_api_fallback() is False


def test_engine_wires_fallback_from_env(monkeypatch):
    monkeypatch.setenv("REFMET_LIVE_API_FALLBACK", "1")
    ann = AnnotationEngine().annotator_registry[MetabolomicsWorkbenchAnnotator.slug]
    assert isinstance(ann, MetabolomicsWorkbenchAnnotator)
    assert ann.LIVE_API_FALLBACK is True


def test_engine_fallback_off_by_default(monkeypatch):
    monkeypatch.delenv("REFMET_LIVE_API_FALLBACK", raising=False)
    ann = AnnotationEngine().annotator_registry[MetabolomicsWorkbenchAnnotator.slug]
    assert isinstance(ann, MetabolomicsWorkbenchAnnotator)
    assert ann.LIVE_API_FALLBACK is False


def test_goslin_binder_also_honors_the_toggle(monkeypatch):
    # GoslinLipidAnnotator wraps its OWN RefMet binder; the toggle must reach it too, or lipids resolved
    # via Goslin would silently ignore the fallback.
    from biomapper2.core.annotators.goslin_lipid import GoslinLipidAnnotator

    monkeypatch.setenv("REFMET_LIVE_API_FALLBACK", "1")
    goslin = AnnotationEngine().annotator_registry[GoslinLipidAnnotator.slug]
    assert isinstance(goslin, GoslinLipidAnnotator)
    binder = goslin._binder
    assert isinstance(binder, MetabolomicsWorkbenchAnnotator)
    assert binder.LIVE_API_FALLBACK is True
