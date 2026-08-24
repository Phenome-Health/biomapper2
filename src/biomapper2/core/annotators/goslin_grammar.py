"""Offline pygoslin wrapper: lipid shorthand name -> canonical name + formula + mass + dialect.

Goslin is a GRAMMAR PARSER/NORMALIZER + physicochemical-property calculator. It does NOT emit an
InChIKey/SMILES/structure or any database identifier — the canonical-shorthand -> id step is a
separate database lookup (see ``GoslinLipidAnnotator`` stage 2). This wrapper is deterministic and
network-free, so it is fully unit-testable on literals. Any parse failure/exception degrades to
``None`` (fail-soft): a non-lipid name is simply "not a lipid", never an error.

The dialect-specific parsers are tried in order so the wrapper can RECORD which grammar fired
(the cross-dialect audit signal): a name written in a non-LIPID-MAPS dialect that parses under the
SwissLipids/HMDB grammar proves Goslin did real dialect translation, not an identity pass-through.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LipidParse:
    """The normalized reading of one lipid shorthand name (pygoslin, neutral species level)."""

    input_name: str
    canonical_name: str
    sum_formula: str | None
    monoisotopic_mass: float | None
    level: str | None
    dialect: str


# (dialect label, parser class path). Ordered: Liebisch/Goslin shorthand first (the common metabolomics
# form), then the database dialects. First parser that succeeds without raising wins.
_DIALECTS: tuple[tuple[str, str], ...] = (
    ("Goslin", "GoslinParser"),
    ("LipidMaps", "LipidMapsParser"),
    ("SwissLipids", "SwissLipidsParser"),
    ("HMDB", "HmdbParser"),
)


class LipidGrammar:
    """Parse a lipid shorthand name into a normalized :class:`LipidParse`, or ``None``."""

    def __init__(self) -> None:
        # Import lazily and construct one parser instance per dialect (parsers are reusable).
        from pygoslin.parser.Parser import (
            GoslinParser,
            HmdbParser,
            LipidMapsParser,
            SwissLipidsParser,
        )

        classes = {
            "GoslinParser": GoslinParser,
            "LipidMapsParser": LipidMapsParser,
            "SwissLipidsParser": SwissLipidsParser,
            "HmdbParser": HmdbParser,
        }
        self._parsers: tuple[tuple[str, object], ...] = tuple((dialect, classes[cls]()) for dialect, cls in _DIALECTS)

    def parse(self, name: str) -> LipidParse | None:
        """Normalized reading of ``name``, or ``None`` if no grammar parses it (fail-soft)."""
        if not name or not str(name).strip():
            return None
        text = str(name).strip()
        for dialect, parser in self._parsers:
            adduct = self._try_parse(parser, text)
            if adduct is None:
                continue
            canonical = self._canonical_name(adduct)
            if not canonical:
                continue
            return LipidParse(
                input_name=text,
                canonical_name=canonical,
                sum_formula=self._sum_formula(adduct),
                monoisotopic_mass=self._mass(adduct),
                level=self._level(adduct),
                dialect=dialect,
            )
        return None

    @staticmethod
    def _try_parse(parser: object, text: str) -> object | None:
        """Return the parsed LipidAdduct, or ``None`` on any pygoslin failure (parsers raise)."""
        try:
            adduct = parser.parse(text)  # type: ignore[attr-defined]
        except Exception:
            return None
        return adduct

    @staticmethod
    def _canonical_name(adduct: object) -> str | None:
        """Species-level normalized shorthand; fall back to the default string. Best-effort."""
        try:
            from pygoslin.domain.LipidLevel import LipidLevel

            species = adduct.get_lipid_string(LipidLevel.SPECIES)  # type: ignore[attr-defined]
            if species:
                return str(species)
        except Exception:
            pass
        try:
            return str(adduct.get_lipid_string())  # type: ignore[attr-defined]
        except Exception:
            return None

    @staticmethod
    def _sum_formula(adduct: object) -> str | None:
        try:
            value = adduct.get_sum_formula()  # type: ignore[attr-defined]
            return str(value) if value else None
        except Exception:
            return None

    @staticmethod
    def _mass(adduct: object) -> float | None:
        try:
            value = adduct.get_mass()  # type: ignore[attr-defined]
            return float(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _level(adduct: object) -> str | None:
        """Lipid grammar level (e.g. SPECIES/MOLECULAR_SPECIES). Best-effort; ``None`` if unavailable."""
        try:
            lipid = getattr(adduct, "lipid", None)
            info = getattr(lipid, "info", None)
            level = getattr(info, "level", None)
            return str(level) if level is not None else None
        except Exception:
            return None
