"""Case-robust name lookup for the two case-sensitive name readers (RefMet ``/match`` and Goslin).

Both read a metabolite NAME case-sensitively, so the same analyte spelled with different case can
resolve to different records and never link across cohorts. NECS ships lower-cased names, so its
spellings differ from Arivale's / Xu's Metabolon spellings only in the case of a few notation tokens:

* RefMet ``/match`` answers ``acetylcarnitine (C2)`` (CAR 2:0) and ``1-palmitoyl-2-oleoyl-GPC (16:0/18:1)``
  (PC 16:0/18:1) but returns no match for ``acetylcarnitine (c2)`` or ``1-palmitoyl-2-oleoyl-gpc (16:0/18:1)``.
* For lower-case plasmalogens it returns a DIFFERENT, less specific record rather than none:
  ``1-(1-enyl-palmitoyl)-2-oleoyl-gpc (p-16:0/18:1)*`` matches ``PC P-16:0/18:1 or PC O-16:1/18:1`` where
  the Metabolon spelling ``...-GPC (P-16:0/18:1)*`` matches ``PC P-16:0/18:1``. A retry that fires only on
  a miss cannot reach this case, which is why the restored spelling is queried FIRST.
* Goslin parses ``12,13-DiHOME`` and ``12-HETE`` but not ``12,13-dihome`` or ``12-hete``.

The rule
--------
:func:`canonical_query` upper-cases ONLY the letters of four narrow, recognized notation tokens, in the
name as given. Every other character, including its case, is left exactly as it was:

1. Acyl chain notation in a parenthetical: ``(c2)`` -> ``(C2)``, ``(c5:1)`` -> ``(C5:1)``,
   ``(c4-oh)`` -> ``(C4-OH)``, ``(c5-dc)`` -> ``(C5-DC)``.
2. The ether / plasmalogen letter that opens a chain parenthetical: ``(p-16:0)`` -> ``(P-16:0)``,
   ``(p-16:0/20:4)`` -> ``(P-16:0/20:4)``, ``(o-18:1)`` -> ``(O-18:1)``. Only the letter changes.
3. Glycerophospholipid head-group abbreviations as whole tokens: ``gpc`` / ``gpe`` / ``gpi`` / ``gps`` /
   ``gpg`` / ``gpa`` -> ``GPC`` ... ``GPA``.
4. Position-prefixed oxylipin names in Goslin's casing: ``12,13-dihome`` -> ``12,13-DiHOME``,
   ``12-hete`` -> ``12-HETE``, ``14,15-eet`` -> ``14,15-EET``. Only after a numeric position locant, so
   an ordinary word (``home``) is never touched.

It returns None when nothing changes, so a name already in canonical case (every Metabolon spelling,
and every name without these tokens) is queried exactly as before: the fix cannot move a name that
resolves today. Callers query the canonical spelling first and fall back to the name as given only when
it is a no-match (never on an outage, which is not an answer).

Deliberately NOT touched (case carries meaning, or the notation is outside this fix): the rest of the
name; identifiers and CURIE local parts; SMILES (lower case marks aromatic atoms); InChI / InChIKey;
stereo and configuration descriptors (``D-``/``L-`` vs ``d-``/``l-``, ``R``/``S``, ``E``/``Z``, ``sn-``);
``O-``/``P-`` letters outside a chain parenthetical (``PC O-16:0/18:1``); ``N-``/``S-`` locants; element
symbols (``Co`` vs ``CO``); gene and protein symbols; and lipid-class shorthand such as ``pc 16:0/18:1``
(Goslin rejects that too, but class case is a separate change).

Two spellings that differ in case ONLY inside these tokens send the identical first query. Spellings
that also differ in case elsewhere are not unified here: nothing shows RefMet folds case in the prose
part, and folding it would rewrite text this module has no business rewriting.
"""

from __future__ import annotations

import re
from typing import Protocol, TypeVar

_T = TypeVar("_T", covariant=True)

# (c2) (c5:1) (c4-oh) (c3-dc) (c10:2) (c6-dc-m) ... : acyl chain length [+ unsaturation] [+ suffixes]
_ACYL_PAREN = re.compile(r"\(c(\d+(?::\d+)?(?:-[a-z]{1,3})*)\)", re.IGNORECASE)
# (p-16:0) (p-16:0/20:4) (o-18:1): the ether / plasmalogen letter opening a chain parenthetical
_ETHER_PAREN = re.compile(r"\(([op])-(?=\d+:\d+)", re.IGNORECASE)
# whole-token glycerophospholipid head groups: -gpc, -gpe, gpi ...
_GP_TOKEN = re.compile(r"(?<![a-z])gp([acegis])(?![a-z])", re.IGNORECASE)

# Goslin's casing for oxylipin trivial-name tokens (pygoslin FattyAcids grammar / lipid-list).
_OXYLIPIN_CASE = {
    "dihotre": "DiHOTrE",
    "dihome": "DiHOME",
    "dihode": "DiHODE",
    "dihete": "DiHETE",
    "hpotre": "HpOTrE",
    "hpome": "HpOME",
    "hpode": "HpODE",
    "hpete": "HpETE",
    "hotre": "HOTrE",
    "hete": "HETE",
    "hode": "HODE",
    "home": "HOME",
    "hepe": "HEPE",
    "eet": "EET",
}
# Longest first so "dihome" is not consumed as "home". Only when preceded by a position locant.
_OXYLIPIN = re.compile(
    r"(?<![a-z0-9])(\d+(?:,\d+)*-)(" + "|".join(sorted(_OXYLIPIN_CASE, key=len, reverse=True)) + r")(?![a-z])",
    re.IGNORECASE,
)


def canonical_case_variant(name: str) -> str:
    """``name`` with the recognized notation tokens in canonical case; every other character unchanged."""
    k = _ACYL_PAREN.sub(lambda m: f"(C{m.group(1).upper()})", name)
    k = _ETHER_PAREN.sub(lambda m: f"({m.group(1).upper()}-", k)
    k = _GP_TOKEN.sub(lambda m: f"GP{m.group(1).upper()}", k)
    k = _OXYLIPIN.sub(lambda m: m.group(1) + _OXYLIPIN_CASE[m.group(2).lower()], k)
    return k


def canonical_query(name: str) -> str | None:
    """The spelling to query BEFORE ``name``, or None to query ``name`` exactly as given.

    None when ``name`` carries none of the recognized notation tokens in non-canonical case, i.e. when
    :func:`canonical_case_variant` would return it unchanged.
    """
    variant = canonical_case_variant(name)
    return None if variant == name else variant


class _Parser(Protocol[_T]):
    def parse(self, name: str) -> _T | None: ...


def parse_case_robust(grammar: _Parser[_T], name: str) -> _T | None:
    """Goslin parse, canonical-case spelling first, then the name as given (same order as RefMet)."""
    canonical = canonical_query(name)
    parsed = grammar.parse(canonical) if canonical is not None else None
    if parsed is None:
        parsed = grammar.parse(name)
    return parsed
