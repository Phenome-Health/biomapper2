"""Every ``requests_cache`` session construction site under ``src/`` must pass
``ignored_parameters=CACHE_IGNORED_PARAMETERS``.

``requests_cache`` redacts a default list of secret-ish parameter names, but matches them
case-sensitively, so a site that omits the explicit list will persist an API key or Authorization
header in cleartext into the on-disk cache (the Kestrel incident,
docs/solutions/security-issues/kestrel-api-key-persisted-in-cleartext-to-http-cache-2026-08-05.md).
This is a static AST guard so a new site cannot silently reintroduce the leak. It covers both direct
``CachedSession(...)`` calls AND construction of any subclass of ``CachedSession`` defined under
``src/`` (for example ``_KestrelCachedSession`` in ``utils.py``, the factory that actually carries
the Kestrel key), so removing the redaction list from a subclass factory is caught too. The at-rest
byte check for the Kestrel session lives in ``test_kestrel_auth_header.py``; this test covers the
enumeration of construction sites.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "biomapper2"


def _callee_name(func: ast.expr) -> str | None:
    """The simple name of a call/base target: ``CachedSession`` for both ``CachedSession`` and
    ``requests_cache.CachedSession``."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _cache_session_class_names() -> set[str]:
    """``CachedSession`` plus every subclass of it (transitively) defined under ``src/``.

    A subclass like ``_KestrelCachedSession`` constructs a real cache session, so its instantiation
    sites must redact too. Resolving subclasses by name is sufficient here: the repo does not alias
    ``CachedSession`` under another name.
    """
    class_defs: list[tuple[str, set[str]]] = []
    for py in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {name for b in node.bases if (name := _callee_name(b)) is not None}
                class_defs.append((node.name, bases))
    names = {"CachedSession"}
    changed = True
    while changed:
        changed = False
        for cname, bases in class_defs:
            if cname not in names and (bases & names):
                names.add(cname)
                changed = True
    return names


def _cache_session_calls() -> list[tuple[Path, ast.Call]]:
    """Every call whose callee constructs a ``CachedSession`` (direct or via a discovered subclass).

    A ``class Foo(requests_cache.CachedSession)`` base is a ClassDef, not a Call, so subclass
    definitions themselves are not flagged, only their construction sites.
    """
    names = _cache_session_class_names()
    calls: list[tuple[Path, ast.Call]] = []
    for py in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee_name(node.func) in names:
                calls.append((py, node))
    return calls


def test_at_least_one_cache_session_site_is_scanned() -> None:
    """Guard the guard: if the enumeration finds nothing, the test would pass vacuously."""
    assert _cache_session_calls(), "no CachedSession construction sites found under src/biomapper2"


def test_kestrel_subclass_factory_is_in_scope() -> None:
    """The Kestrel subclass factory is the one that actually carries a credential; make sure the
    subclass discovery keeps it in scope (regression for the gap Greptile flagged on PR #76)."""
    assert "_KestrelCachedSession" in _cache_session_class_names()


def test_every_cache_session_passes_ignored_parameters() -> None:
    offenders: list[str] = []
    for py, node in _cache_session_calls():
        kw = {k.arg: k.value for k in node.keywords}
        value = kw.get("ignored_parameters")
        rel = py.relative_to(SRC_ROOT.parents[1])
        if value is None:
            offenders.append(f"{rel}:{node.lineno} missing ignored_parameters")
        elif not (isinstance(value, ast.Name) and value.id == "CACHE_IGNORED_PARAMETERS"):
            offenders.append(f"{rel}:{node.lineno} ignored_parameters is not CACHE_IGNORED_PARAMETERS")
    assert not offenders, "CachedSession sites persist credentials in cleartext:\n" + "\n".join(offenders)
