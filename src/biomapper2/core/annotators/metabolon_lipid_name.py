"""Vendor (Metabolon-style) lipid names -> a Goslin parse at the level their alternatives share.

The Goslin annotator only engages when pygoslin parses the WHOLE name. Metabolon names are a common name
plus a chain parenthetical, ``sphingomyelin (d18:1/17:0, d17:1/18:0, d19:1/16:0)``, so they never take
the lipid path, and Kestrel's text search collapses every such sphingomyelin onto one generic
``Sphingomyelin`` node. Many of these names list ALTERNATIVE structures: isomers with one shared sum
composition (comma-separated chain lists), omega-position alternatives (``20:3n3 or n6``) or alternative
analytes (``butyrate/isobutyrate (4:0)``). A single listed isomer is not what was measured, so the honest
query is the level the alternatives share.

This module is pure and offline: it reads the name, builds Goslin shorthand per listed alternative, and
asks the injected grammar to parse it. It never queries a database.

Rules (each is a refusal unless every condition holds; a refusal returns None and the name keeps today's
behavior exactly):

R1  Route only names that list alternatives (:func:`name_is_mixture`). A single-isomer prose name
    (``palmitate (16:0)``, ``stearoyl sphingomyelin (d18:1/18:0)``) is NOT routed: RefMet ``/match`` already
    answers those at their asserted level, and a second, Goslin-capped vote could only compete with it.
R2  The chain parenthetical is the LAST ``(...)`` group, holds only chain notation, and only Metabolon's
    confidence markers (``*``/``**``), peak markers (``[1]``) or a closing ``]`` follow it. A nested
    parenthetical (``16:0(OH)``) is refused.
R3  The lipid class comes from the common-name head (the text before the parenthetical, cut at the first
    ``[``), matched EXACTLY against :data:`_HEAD_CLASSES` or one of three narrow patterns:
      * ``...-GPC/-GPE/-GPI/-GPS/-GPG`` (case-insensitive): one chain -> the lyso class (LPC ...), two -> PC ...;
        a ``1-``/``2-`` acyl position on a lyso name is kept as a ``0:0`` slot (``LPC 20:3/0:0``).
      * ``...carnitine`` with ``C``-prefixed chains -> CAR (one chain).
      * a fatty acid: every ``/``-separated head part ends in ``ate`` and the parenthetical is one bare chain
        -> FA. Heads naming a branch / substituent (``methyl``, ``hydroxy``, ``oxo``, ``keto``, ``amino``,
        ``sulfate``, ``carnitine``, ``glycerol``) are refused (the sum composition would merge them with the
        straight-chain acid).
    Any other head is refused (glycerol esters named by position, e.g. ``1-linoleoylglycerol``, included).
R4  Chain tokens: ``[dtm]?`` long-chain-base prefix (sphingolipids only, required there), optional
    ``P-``/``O-`` ether prefix (glycerophospholipids only), ``C:D``, optional omega locant ``n3``, optional
    omega alternative `` or n6`` / `` or 3n6`` / `` or 6``. Branch prefixes (``a15:0``/``i15:0``), functional
    groups (``16:0-OH``) and anything else are refused.
R5  Every alternative must parse with Goslin and all must render the SAME species (sum composition) and
    class. Comma-separated alternatives are then resolved at SPECIES only (never at one listed isomer).
    Omega-only alternatives keep the parse of the omega-stripped shorthand, whose every level is shared by
    all of them (the omega position is exactly what was dropped).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Protocol

from .goslin_grammar import LipidParse

# Exact common-name heads (lower-cased) -> (Goslin class, required chain count per alternative).
_HEAD_CLASSES: dict[str, tuple[str, int]] = {
    "sphingomyelin": ("SM", 2),
    "ceramide": ("Cer", 2),
    "glycosyl ceramide": ("HexCer", 2),
    "diacylglycerol": ("DG", 2),
    "phosphatidylcholine": ("PC", 2),
    "phosphatidylethanolamine": ("PE", 2),
    "phosphatidylinositol": ("PI", 2),
    "phosphatidylserine": ("PS", 2),
    "phosphatidylglycerol": ("PG", 2),
}
_SPHINGOLIPID_CLASSES = frozenset({"SM", "Cer", "HexCer"})
_GP_HEAD = re.compile(r"-gp([cegis])$", re.IGNORECASE)
_GP_CLASS = {"c": "PC", "e": "PE", "i": "PI", "s": "PS", "g": "PG"}
# A fatty-acid head part: ends in "ate", and names no branch/substituent that species would erase.
_FA_REFUSED_WORDS = ("methyl", "hydroxy", "oxo", "keto", "amino", "sulfate", "carnitine", "glycerol")

_MARKERS_TAIL = re.compile(r"(?:\s*\[\d+\]|\s*\*+|\s*\])*\s*$")
_PEAK_MARKER = re.compile(r"\s*\[\d+\]")
_LAST_PAREN = re.compile(r"\(([^()]*)\)(?=[^()]*$)")
_OMEGA_ALT = r"(?:\s+or\s+(?:\d+n|n)?\d+)"
_CHAIN = re.compile(
    r"^(?P<carn>[Cc])?(?P<lcb>[dtm])?(?P<ether>[PO]-)?(?P<c>\d+):(?P<d>\d+)(?P<omega>n\d+)?(?P<alt>"
    + _OMEGA_ALT
    + r")?$"
)
_LETTER_SLASH_LETTER = re.compile(r"[A-Za-z]\s*/\s*[A-Za-z]")
_WORD_OR = re.compile(r"\bor\b", re.IGNORECASE)
# A "+" joining two analytes; an optical-rotation sign "(+)" / "(+/-)" is not one.
_PLUS = re.compile(r"(?<!\()\+(?![)/])")
_CHAIN_LIKE = re.compile(r"\d+:\d+")


class _Grammar(Protocol):
    def parse(self, name: str) -> LipidParse | None: ...


@dataclass(frozen=True)
class ParentheticalLipidQuery:
    """A Goslin parse built from a vendor name's chain parenthetical (see module rules)."""

    parse: LipidParse  # reduced to species when the alternatives differ at any finer level
    shorthand: str  # the Goslin shorthand actually parsed (first alternative)
    alternatives: tuple[str, ...]  # per-alternative shorthand, in the order the name lists them
    species_level: bool  # True when comma-separated alternatives forced a species-level query


def _strip_markers(name: str) -> str:
    """Name without Metabolon confidence (``*``/``**``) and peak (``[1]``) markers."""
    return _PEAK_MARKER.sub("", re.sub(r"\*+", "", name)).strip()


def _top_level_text(name: str) -> str:
    """The name with every parenthetical and bracket group removed (head text only)."""
    out = re.sub(r"\([^()]*\)", " ", name)
    out = re.sub(r"\[[^\[\]]*\]", " ", out)
    return out


def _paren_chain_alternatives(name: str) -> list[str]:
    """Comma-separated chain alternatives in the last parenthetical (empty when there are fewer than two)."""
    match = _LAST_PAREN.search(_strip_markers(name))
    if not match:
        return []
    parts = [p.strip() for p in match.group(1).split(",")]
    return parts if len(parts) >= 2 and all(_CHAIN_LIKE.search(p) for p in parts) else []


def name_is_mixture(name: str | None) -> bool:
    """Whether the name lists two or more alternative structures / analytes.

    True on any of: the word ``or`` (``20:3n3 or n6``, ``[cis or trans]``); a ``+`` joining two names
    (``13-HODE + 9-HODE``; an optical-rotation sign ``(+)-`` does not count);
    a ``/`` between letters outside every parenthetical and bracket (``oleate/vaccenate``; chain notation
    ``18:1/20:4`` has digits around its slash and never counts); two or more comma-separated chain lists in
    the parenthetical (``(d18:1/14:0, d16:1/16:0)``). Commas between locants (``12,13-DiHOME``) never count.
    """
    if not name or not str(name).strip():
        return False
    text = str(name)
    if _WORD_OR.search(text) or _PLUS.search(text):
        return True
    if _LETTER_SLASH_LETTER.search(_top_level_text(text)):
        return True
    return bool(_paren_chain_alternatives(text))


def _head_class(head: str, chain_count: int, carnitine_prefixed: bool) -> str | None:
    """Goslin class for the common-name head (rule R3), or None when the head is not supported."""
    key = head.strip().lower()
    if key in _HEAD_CLASSES:
        cls, required = _HEAD_CLASSES[key]
        return cls if chain_count == required else None
    gp = _GP_HEAD.search(key)
    if gp:
        cls = _GP_CLASS[gp.group(1).lower()]
        if chain_count == 1:
            return "L" + cls
        return cls if chain_count == 2 else None
    if key.endswith("carnitine"):
        return "CAR" if chain_count == 1 and carnitine_prefixed else None
    if chain_count == 1 and not carnitine_prefixed:
        parts = [p.strip() for p in key.split("/")]
        if parts and all(p.endswith("ate") for p in parts) and not any(w in key for w in _FA_REFUSED_WORDS):
            return "FA"
    return None


def _alternative_shorthand(cls: str, chains: list[re.Match[str]], head: str = "") -> str | None:
    """Goslin shorthand for one alternative, or None when a chain's prefixes do not fit the class (R4).

    A lyso glycerophospholipid named with its acyl position (``1-dihomo-linolenoyl-GPC``) keeps that
    claim as an empty ``0:0`` slot (``LPC 20:3/0:0``), the form RefMet uses for the same record, so the
    existing sn-position trust policy and RefMet-precedence rule treat it exactly like ``LPC 20:3/0:0``
    written out; dropping the position would read the name as coarser than it is."""
    sphingo = cls in _SPHINGOLIPID_CLASSES
    rendered = []
    for i, chain in enumerate(chains):
        lcb, ether = chain.group("lcb"), chain.group("ether")
        if sphingo:
            # The long-chain base carries the hydroxylation prefix; the N-acyl chain carries none.
            if (i == 0) != bool(lcb) or ether:
                return None
        elif lcb:
            return None
        if ether and cls.lstrip("L") not in ("PC", "PE"):
            return None
        rendered.append(f"{lcb or ''}{ether or ''}{chain.group('c')}:{chain.group('d')}")
    if cls.startswith("L") and len(rendered) == 1:
        position = re.match(r"([12])-", head.strip())
        if position:
            rendered = rendered + ["0:0"] if position.group(1) == "1" else ["0:0"] + rendered
    return f"{cls} {'/'.join(rendered)}"


def parenthetical_lipid_query(name: str | None, grammar: _Grammar) -> ParentheticalLipidQuery | None:
    """A species-honest Goslin parse built from ``name``'s chain parenthetical, or None (see module rules)."""
    if not name or not name_is_mixture(name):
        return None  # R1
    text = str(name).strip()
    tail = _MARKERS_TAIL.search(text)
    core = text[: tail.start()] if tail else text
    match = re.search(r"\(([^()]*)\)$", core)
    if not match:
        return None  # R2: a flat chain parenthetical must close the name (nested "16:0(OH)" never matches)
    inner = _PEAK_MARKER.sub("", match.group(1)).strip()
    head = core[: match.start()].split("[", 1)[0].strip()
    if not head:
        return None

    alternatives_raw = [p.strip() for p in inner.split(",")]
    parsed_alts: list[list[re.Match[str]]] = []
    for alt in alternatives_raw:
        chains = []
        for token in re.split(r"[/_]", alt):
            m = _CHAIN.match(token.strip())
            if m is None:
                return None  # R4
            chains.append(m)
        parsed_alts.append(chains)
    chain_counts = {len(chains) for chains in parsed_alts}
    if len(chain_counts) != 1:
        return None
    chain_count = chain_counts.pop()
    carnitine_prefixed = all(bool(c.group("carn")) for chains in parsed_alts for c in chains)
    if any(bool(c.group("carn")) for chains in parsed_alts for c in chains) and not carnitine_prefixed:
        return None
    cls = _head_class(head, chain_count, carnitine_prefixed)
    if cls is None:
        return None  # R3

    shorthands: list[str] = []
    for chains in parsed_alts:
        shorthand = _alternative_shorthand(cls, chains, head)
        if shorthand is None:
            return None
        shorthands.append(shorthand)

    parses: list[LipidParse] = []
    for shorthand in shorthands:
        parsed = grammar.parse(shorthand)
        if parsed is None:
            return None  # R5
        parses.append(parsed)
    species = {(p.level_names.get("SPECIES") or p.canonical_name) for p in parses}
    classes = {p.level_names.get("CLASS") for p in parses}
    if len(species) != 1 or len(classes) != 1:
        return None  # R5: alternatives with different sum compositions share no species

    first = parses[0]
    species_level = len(parses) > 1
    if species_level:
        first = _reduce_to_species(first)
    return ParentheticalLipidQuery(
        parse=dataclasses.replace(first, input_name=text),
        shorthand=shorthands[0],
        alternatives=tuple(shorthands),
        species_level=species_level,
    )


def _reduce_to_species(parse: LipidParse) -> LipidParse:
    """The parse with every level finer than species removed (no chains, level SPECIES)."""
    keep = {k: v for k, v in parse.level_names.items() if k in ("CATEGORY", "CLASS", "SPECIES")}
    return dataclasses.replace(parse, level="LipidLevel.SPECIES", level_names=keep, chains=())
