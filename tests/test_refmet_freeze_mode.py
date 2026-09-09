"""REFMET_FREEZE_MODE resolution + its backward-compatible unset default.

The unset default must infer `frozen` when a REFMET_SNAPSHOT_PATH is configured, so a deployment that
set only the snapshot path (pre-mode, freeze-first) does NOT silently revert to live-only on upgrade.
"""

from __future__ import annotations

import pytest

from biomapper2.config import get_refmet_freeze_mode

pytestmark = pytest.mark.unit


def test_unset_and_no_snapshot_is_off(monkeypatch):
    monkeypatch.delenv("REFMET_FREEZE_MODE", raising=False)
    monkeypatch.delenv("REFMET_SNAPSHOT_PATH", raising=False)
    assert get_refmet_freeze_mode() == "off"


def test_unset_but_snapshot_configured_infers_frozen(monkeypatch, tmp_path):
    # Backward compat: a box that set only REFMET_SNAPSHOT_PATH keeps freeze behavior.
    monkeypatch.delenv("REFMET_FREEZE_MODE", raising=False)
    monkeypatch.setenv("REFMET_SNAPSHOT_PATH", str(tmp_path / "freeze.tsv"))
    assert get_refmet_freeze_mode() == "frozen"


def test_explicit_value_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("REFMET_SNAPSHOT_PATH", str(tmp_path / "freeze.tsv"))
    for mode in ("off", "frozen", "live_backup"):
        monkeypatch.setenv("REFMET_FREEZE_MODE", mode)
        assert get_refmet_freeze_mode() == mode


def test_unknown_value_falls_back_to_unset_default(monkeypatch, tmp_path):
    # A typo uses the unset default (frozen here, since a snapshot is configured) rather than a
    # surprising live-only revert.
    monkeypatch.setenv("REFMET_SNAPSHOT_PATH", str(tmp_path / "freeze.tsv"))
    monkeypatch.setenv("REFMET_FREEZE_MODE", "frozn")
    assert get_refmet_freeze_mode() == "frozen"
    monkeypatch.delenv("REFMET_SNAPSHOT_PATH", raising=False)
    assert get_refmet_freeze_mode() == "off"
