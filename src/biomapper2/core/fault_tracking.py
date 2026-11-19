"""Row-scoped fault attribution and tracking for Kestrel interactions."""

from __future__ import annotations

import contextvars
from collections import defaultdict
from typing import Any

import pandas as pd


class KestrelFaultError(RuntimeError):
    """Raised when an entity or row fails to resolve due to a Kestrel fault."""

    pass


_ACTIVE_ROW_FAULTS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_ACTIVE_ROW_FAULTS", default=None
)

_ACTIVE_DATASET_FAULTS: contextvars.ContextVar[dict[Any, list[str]] | None] = contextvars.ContextVar(
    "_ACTIVE_DATASET_FAULTS", default=None
)


def start_row_fault_recording() -> tuple[list[str], contextvars.Token]:
    """Begin recording faults for the current execution context."""
    faults: list[str] = []
    token = _ACTIVE_ROW_FAULTS.set(faults)
    return faults, token


def end_row_fault_recording(token: contextvars.Token) -> None:
    """End recording faults for the current execution context."""
    _ACTIVE_ROW_FAULTS.reset(token)


def start_dataset_fault_recording() -> tuple[dict[Any, list[str]], contextvars.Token]:
    """Begin recording faults mapped by row index for dataset mapping."""
    fault_map: dict[Any, list[str]] = defaultdict(list)
    token = _ACTIVE_DATASET_FAULTS.set(fault_map)
    return fault_map, token


def end_dataset_fault_recording(token: contextvars.Token) -> None:
    """End recording faults for dataset mapping."""
    _ACTIVE_DATASET_FAULTS.reset(token)


def record_kestrel_fault(target: Any, fault_description: str) -> None:
    """Record a Kestrel fault on the target and in active contexts."""
    active_row = _ACTIVE_ROW_FAULTS.get()
    if active_row is not None:
        active_row.append(fault_description)

    active_ds = _ACTIVE_DATASET_FAULTS.get()
    if active_ds is not None:
        if isinstance(target, pd.Series) and target.name is not None:
            active_ds[target.name].append(fault_description)
        elif isinstance(target, (int, str)):
            active_ds[target].append(fault_description)

    if isinstance(target, dict):
        target.setdefault("_kestrel_faults", []).append(fault_description)
    elif isinstance(target, pd.Series):
        existing = target.get("_kestrel_faults")
        if not isinstance(existing, list):
            existing = []
            target["_kestrel_faults"] = existing
        existing.append(fault_description)


def get_kestrel_faults(target: Any = None, row_idx: Any = None) -> list[str]:
    """Get all recorded Kestrel faults for the target or row index."""
    faults: list[str] = []

    if row_idx is not None:
        active_ds = _ACTIVE_DATASET_FAULTS.get()
        if active_ds and row_idx in active_ds:
            faults.extend(active_ds[row_idx])

    active_row = _ACTIVE_ROW_FAULTS.get()
    if active_row:
        for item in active_row:
            if item not in faults:
                faults.append(item)

    if isinstance(target, pd.Series):
        active_ds = _ACTIVE_DATASET_FAULTS.get()
        if active_ds and target.name in active_ds:
            for item in active_ds[target.name]:
                if item not in faults:
                    faults.append(item)
        f = target.get("_kestrel_faults")
        if isinstance(f, list):
            for item in f:
                if item not in faults:
                    faults.append(item)
    elif isinstance(target, dict):
        f = target.get("_kestrel_faults")
        if isinstance(f, list):
            for item in f:
                if item not in faults:
                    faults.append(item)

    return faults


def has_kestrel_fault(target: Any = None, row_idx: Any = None) -> bool:
    """Check if any Kestrel fault has been recorded on target or in context."""
    if row_idx is not None:
        active_ds = _ACTIVE_DATASET_FAULTS.get()
        if active_ds and active_ds.get(row_idx):
            return True
    active_row = _ACTIVE_ROW_FAULTS.get()
    if active_row:
        return True
    if isinstance(target, pd.Series):
        active_ds = _ACTIVE_DATASET_FAULTS.get()
        if active_ds and active_ds.get(target.name):
            return True
        return bool(target.get("_kestrel_faults"))
    if isinstance(target, dict):
        return bool(target.get("_kestrel_faults"))
    return False
