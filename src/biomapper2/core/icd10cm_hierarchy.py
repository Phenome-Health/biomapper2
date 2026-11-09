"""Pinned CDC ICD-10-CM hierarchy snapshot loader (Phase 1b: the parent walk-up).

Why this exists
---------------
When neither ``ICD10CM:X`` nor its ``ICD10:X`` twin links, the fallback climbs to a less specific code
(``E11.65`` -> ``E11.6`` -> ``E11``). Only codes that EXIST in a pinned CDC order file are candidates:
naive truncation would invent codes such as the placeholder intermediate ``T36.0X``.

Format (tab-separated, with header)::

    code    billable
    E11     0
    E11.6   0
    E11.65  1

Codes are dotted after the third character; ``billable`` is the CDC header/billable flag (``0``/``1``).
A ``<file>.version`` sidecar (first non-empty line, e.g. ``FY2027/sha256:93e3ad6004ba/built:2026-11-08``)
is REQUIRED: it is stamped on every row the walk-up serves. Both files are written by
``scripts/build_icd10cm_hierarchy.py``.

The snapshot ships inside the package (``biomapper2/data/icd10cm/``) and is read through
``importlib.resources``, so wheel and git-dependency installs find it. ``config.ICD10CM_HIERARCHY_PATH``
overrides it. A missing or malformed snapshot is rejected WHOLE and raises
:class:`Icd10cmHierarchyError`; nothing here ever degrades to "no parent".

Loading happens once per resolved source and is cached; ``reset()`` clears the cache (tests).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from .. import config

if TYPE_CHECKING:  # importlib.resources.abc is 3.11+; annotations only (requires-python >= 3.10)
    from importlib.resources.abc import Traversable

PACKAGED_SNAPSHOT = ("data", "icd10cm", "icd10cm_order_FY2027.tsv")
HEADER = ("code", "billable")
_PLACEHOLDER = "X"
_CODE = re.compile(r"^[A-Z][0-9A-Z]{2}(\.[0-9A-Z]{1,4})?$")


class Icd10cmHierarchyError(RuntimeError):
    """The hierarchy snapshot is missing or malformed while the walk-up is enabled."""


def undot(code: str) -> str:
    return code.replace(".", "")


def redot(code: str) -> str:
    """``E1165`` -> ``E11.65``; codes of three characters or fewer are returned as-is."""
    code = undot(code.strip())
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


@dataclass(frozen=True)
class Hierarchy:
    """A parsed, immutable snapshot: the set of existing dotted codes and the snapshot version."""

    codes: frozenset[str]
    version: str
    source: str

    def contains(self, code: str) -> bool:
        return code in self.codes

    def ancestors(self, code: str) -> list[str]:
        """Existing ancestors of ``code``, most specific first, down to three characters.

        Truncating one character at a time drops the 7th character first. Truncations absent from the
        snapshot are skipped, and so are placeholder-X intermediates (``T36.0X``): CDC lists them as
        headers, but they only pad the code to the 7th-character position and repeat the parent's
        title, so asking for them spends batch slots on no new concept. A code not in the snapshot has
        no ancestors.
        """
        if code not in self.codes:
            return []
        bare = undot(code)
        found: list[str] = []
        for length in range(len(bare) - 1, 2, -1):
            prefix = bare[:length]
            if length > 3 and prefix.endswith(_PLACEHOLDER):
                continue
            candidate = redot(prefix)
            if candidate in self.codes:
                found.append(candidate)
        return found


def _parse(text: str, version_text: str | None, source: str) -> Hierarchy:
    version = next((ln.strip() for ln in (version_text or "").splitlines() if ln.strip()), None)
    if not version:
        raise Icd10cmHierarchyError(f"ICD-10-CM hierarchy {source} has no .version sidecar (or it is empty)")
    lines = text.splitlines()
    if not lines or tuple(lines[0].rstrip("\r").split("\t")) != HEADER:
        raise Icd10cmHierarchyError(f"ICD-10-CM hierarchy {source}: expected the header {HEADER!r}")
    codes: set[str] = set()
    for lineno, raw in enumerate(lines[1:], start=2):
        fields = raw.rstrip("\r").split("\t")
        if len(fields) != 2 or fields[1] not in {"0", "1"} or not _CODE.match(fields[0]):
            raise Icd10cmHierarchyError(f"ICD-10-CM hierarchy {source}: malformed row {lineno}: {raw!r}")
        if fields[0] in codes:
            raise Icd10cmHierarchyError(f"ICD-10-CM hierarchy {source}: duplicate code {fields[0]!r} (row {lineno})")
        codes.add(fields[0])
    if not codes:
        raise Icd10cmHierarchyError(f"ICD-10-CM hierarchy {source} holds no codes")
    return Hierarchy(codes=frozenset(codes), version=version, source=source)


def _read(resource: Path | Traversable, sidecar: Path | Traversable, source: str) -> Hierarchy:
    if not resource.is_file():
        raise Icd10cmHierarchyError(f"ICD-10-CM hierarchy snapshot not found at {source}")
    version_text = sidecar.read_text(encoding="utf-8") if sidecar.is_file() else None
    return _parse(resource.read_text(encoding="utf-8"), version_text, source)


def load(path: Path | str) -> Hierarchy:
    """Parse a snapshot file (and its sidecar) without touching the cache."""
    path = Path(path)
    return _read(path, path.with_name(path.name + ".version"), str(path))


_CACHE: dict[str, Hierarchy] = {}


def _source() -> tuple[str, Path | Traversable, Path | Traversable]:
    """``(cache key, snapshot, sidecar)`` for the configured path, else the packaged resource."""
    if config.ICD10CM_HIERARCHY_PATH:
        path = Path(config.ICD10CM_HIERARCHY_PATH)
        return str(path), path, path.with_name(path.name + ".version")
    folder = resources.files("biomapper2").joinpath(*PACKAGED_SNAPSHOT[:-1])
    name = PACKAGED_SNAPSHOT[-1]
    return "<packaged:" + "/".join(PACKAGED_SNAPSHOT) + ">", folder.joinpath(name), folder.joinpath(name + ".version")


def load_or_raise() -> Hierarchy:
    """The configured (or packaged) snapshot, cached per source. Raises when missing or malformed."""
    key, resource, sidecar = _source()
    if key not in _CACHE:
        _CACHE[key] = _read(resource, sidecar, key)
    return _CACHE[key]


def is_present() -> bool:
    try:
        load_or_raise()
    except Icd10cmHierarchyError:
        return False
    return True


def version() -> str | None:
    return load_or_raise().version if is_present() else None


def contains(code: str) -> bool:
    return load_or_raise().contains(code)


def ancestors(code: str) -> list[str]:
    return load_or_raise().ancestors(code)


def reset() -> None:
    """Drop the load cache (tests repointing ``ICD10CM_HIERARCHY_PATH``)."""
    _CACHE.clear()
