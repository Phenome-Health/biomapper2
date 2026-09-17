"""Mapper._build_tier_b couples the default-on posture to freeze presence at build time.

The P1 risk this pins: default-on must NOT silently reach live services on a fresh deploy. So the
default with no loadable freeze builds NO Tier B (inert) and logs one prominent warning; an explicit
truthy override builds Tier B against the live path (the supervised-sweep corpus-build) and warns it
is doing so without a freeze; a loadable freeze builds Tier B on the safe freeze-first path; a falsy
override disables silently. No network is touched: constructing IndependentStructureLookup builds a
session but makes no call until a lookup runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from biomapper2.core import tier_b_snapshot
from biomapper2.mapper import Mapper

FIXTURE = Path(__file__).parent / "fixtures" / "tier_b_freeze_fixture.tsv"


@pytest.fixture(autouse=True)
def _reset_snapshot_cache():
    tier_b_snapshot.reset()
    yield
    tier_b_snapshot.reset()


def test_inert_by_default_without_a_freeze_and_warns(monkeypatch, caplog) -> None:
    monkeypatch.delenv("BIOMAPPER2_TIER_B_ENABLED", raising=False)
    monkeypatch.delenv("BIOMAPPER2_TIER_B_SNAPSHOT_PATH", raising=False)
    tier_b_snapshot.reset()
    with caplog.at_level(logging.WARNING):
        built = Mapper._build_tier_b()
    assert built is None
    assert any("inert" in rec.message.lower() for rec in caplog.records), "inert default must warn"


def test_enabled_freeze_builds_on_the_safe_path(monkeypatch) -> None:
    monkeypatch.delenv("BIOMAPPER2_TIER_B_ENABLED", raising=False)
    monkeypatch.setenv("BIOMAPPER2_TIER_B_SNAPSHOT_PATH", str(FIXTURE))
    tier_b_snapshot.reset()
    assert Mapper._build_tier_b() is not None


def test_force_enabled_without_a_freeze_builds_live_and_warns(monkeypatch, caplog) -> None:
    monkeypatch.setenv("BIOMAPPER2_TIER_B_ENABLED", "true")
    monkeypatch.delenv("BIOMAPPER2_TIER_B_SNAPSHOT_PATH", raising=False)
    tier_b_snapshot.reset()
    with caplog.at_level(logging.WARNING):
        built = Mapper._build_tier_b()
    assert built is not None
    assert any("live lookups" in rec.message.lower() for rec in caplog.records), "forced-live must warn"


def test_falsy_disables_silently(monkeypatch, caplog) -> None:
    monkeypatch.setenv("BIOMAPPER2_TIER_B_ENABLED", "false")
    monkeypatch.setenv("BIOMAPPER2_TIER_B_SNAPSHOT_PATH", str(FIXTURE))
    tier_b_snapshot.reset()
    with caplog.at_level(logging.WARNING):
        built = Mapper._build_tier_b()
    assert built is None
    assert not any("tier b" in rec.message.lower() for rec in caplog.records), "disabled must be silent"
