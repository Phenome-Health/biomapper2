import math

from biomapper2.core.annotators.goslin_grammar import LipidGrammar, LipidParse


def test_pygoslin_is_importable_and_parses_shorthand():
    from pygoslin.parser.Parser import GoslinParser

    adduct = GoslinParser().parse("PC 34:1")
    assert adduct is not None
    assert adduct.get_lipid_string() is not None


def test_parses_liebisch_shorthand_and_reports_dialect():
    g = LipidGrammar()
    parsed = g.parse("PC 34:1")
    assert isinstance(parsed, LipidParse)
    assert parsed.input_name == "PC 34:1"
    assert parsed.canonical_name.startswith("PC ")
    assert parsed.dialect == "Goslin"
    assert parsed.sum_formula is not None
    assert parsed.monoisotopic_mass is not None and parsed.monoisotopic_mass > 0


def test_parses_lipidmaps_style_cer():
    parsed = LipidGrammar().parse("Cer(d18:1/12:0)")
    assert parsed is not None
    assert parsed.canonical_name.startswith("Cer ")


def test_formula_and_mass_are_reasonable_for_fa_16_0():
    parsed = LipidGrammar().parse("FA 16:0")
    assert parsed is not None
    # palmitic acid: C16H32O2, monoisotopic ~256.24
    assert parsed.sum_formula == "C16H32O2"
    assert parsed.monoisotopic_mass is not None
    assert math.isclose(parsed.monoisotopic_mass, 256.24, abs_tol=0.1)


def test_non_lipid_returns_none():
    assert LipidGrammar().parse("caffeine") is None
    assert LipidGrammar().parse("D-Glucose") is None


def test_blank_input_returns_none():
    assert LipidGrammar().parse("") is None
    assert LipidGrammar().parse("   ") is None
