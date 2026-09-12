"""Tests for the Normalizer class."""

import pandas as pd
import pytest

from biomapper2.core.normalizer import Normalizer

pytestmark = pytest.mark.unit


class TestParseDelimitedString:
    """Tests for Normalizer._parse_delimited_string method."""

    @pytest.fixture
    def normalizer(self):
        return Normalizer()

    def test_parse_standard_delimited_string(self, normalizer):
        """Handles standard delimiter-separated values."""
        result = normalizer._parse_delimited_string("Q14213_Q8NEV9", ["_"])
        assert result == ["Q14213", "Q8NEV9"]

    def test_parse_comma_delimited_string(self, normalizer):
        """Handles comma-separated values."""
        result = normalizer._parse_delimited_string("5793,79025", [","])
        assert result == ["5793", "79025"]

    def test_parse_list_in_string(self, normalizer):
        """Handles Python list-in-string format like "['Q14213', 'Q8NEV9']"."""
        result = normalizer._parse_delimited_string("['Q14213', 'Q8NEV9']", ["_"])
        assert result == ["Q14213", "Q8NEV9"]

    def test_parse_list_in_string_single_item(self, normalizer):
        """Handles single-item list-in-string format."""
        result = normalizer._parse_delimited_string("['HMDB0000122']", [","])
        assert result == ["HMDB0000122"]

    def test_parse_list_in_string_with_double_quotes(self, normalizer):
        """Handles list-in-string with double quotes."""
        result = normalizer._parse_delimited_string('["Q14213", "Q8NEV9"]', ["_"])
        assert result == ["Q14213", "Q8NEV9"]

    def test_parse_non_string_passthrough(self, normalizer):
        """Non-string values pass through unchanged."""
        result = normalizer._parse_delimited_string(12345, ["_"])
        assert result == 12345

    def test_parse_none_passthrough(self, normalizer):
        """None values pass through unchanged."""
        result = normalizer._parse_delimited_string(None, ["_"])
        assert result is None

    def test_parse_empty_list_in_string(self, normalizer):
        """Handles empty list-in-string format."""
        result = normalizer._parse_delimited_string("[]", [","])
        assert result == []

    def test_parse_tuple_in_string(self, normalizer):
        """Handles Python tuple-in-string format."""
        result = normalizer._parse_delimited_string("('Q14213', 'Q8NEV9')", ["_"])
        assert result == ["Q14213", "Q8NEV9"]

    def test_parse_set_in_string(self, normalizer):
        """Handles Python set-in-string format."""
        result = normalizer._parse_delimited_string("{'Q14213', 'Q8NEV9'}", ["_"])
        # Sets are unordered, so check contents rather than order
        assert set(result) == {"Q14213", "Q8NEV9"}
        assert isinstance(result, list)

    def test_parse_empty_tuple_in_string(self, normalizer):
        """Handles empty tuple-in-string format."""
        result = normalizer._parse_delimited_string("()", [","])
        assert result == []

    def test_parse_pipe_delimiter(self, normalizer):
        """Handles pipe-separated values (common in UniProt)."""
        result = normalizer._parse_delimited_string("Q14213|Q8NEV9", ["|"])
        assert result == ["Q14213", "Q8NEV9"]

    def test_parse_dict_in_string_falls_through(self, normalizer):
        """Dicts are not valid ID lists, fall through to delimiter parsing."""
        # A dict like "{'a': 'b'}" should NOT be treated as an ID list
        result = normalizer._parse_delimited_string("{'key': 'value'}", [","])
        # Falls through to delimiter parsing since dict is not list/tuple/set
        assert result == ["{'key': 'value'}"]


class TestNormalizeIntegration:
    """Integration tests for full normalization flow with list-in-string inputs."""

    @pytest.fixture
    def normalizer(self):
        return Normalizer()

    def test_normalize_entity_with_list_in_string_uniprot(self, normalizer):
        """List-in-string UniProt IDs produce correct curies."""
        entity = pd.Series(
            {
                "name": "Test Protein",
                "UniProt": "['Q14213', 'Q8NEV9']",
            }
        )
        result = normalizer.normalize(
            item=entity,
            provided_id_fields=["UniProt"],
            array_delimiters=["_"],
        )
        # Both IDs should be normalized to curies
        assert "UniProtKB:Q14213" in result["curies_provided"]
        assert "UniProtKB:Q8NEV9" in result["curies_provided"]
        assert len(result["invalid_ids_provided"]) == 0

    def test_normalize_entity_with_tuple_in_string_hmdb(self, normalizer):
        """Tuple-in-string HMDB IDs produce correct curies."""
        entity = pd.Series(
            {
                "name": "Test Metabolite",
                "HMDB": "('HMDB0000122', 'HMDB0000190')",
            }
        )
        result = normalizer.normalize(
            item=entity,
            provided_id_fields=["HMDB"],
            array_delimiters=[","],
        )
        assert "HMDB:HMDB0000122" in result["curies_provided"]
        assert "HMDB:HMDB0000190" in result["curies_provided"]

    def test_normalize_entity_mixed_formats(self, normalizer):
        """Handles mix of list-in-string and delimited formats."""
        entity = pd.Series(
            {
                "name": "Test Entity",
                "UniProt": "['Q14213', 'Q8NEV9']",  # list-in-string
                "HMDB": "HMDB0000122,HMDB0000190",  # comma-delimited
            }
        )
        result = normalizer.normalize(
            item=entity,
            provided_id_fields=["UniProt", "HMDB"],
            array_delimiters=[",", "_"],
        )
        # All four IDs should be present
        assert "UniProtKB:Q14213" in result["curies_provided"]
        assert "UniProtKB:Q8NEV9" in result["curies_provided"]
        assert "HMDB:HMDB0000122" in result["curies_provided"]
        assert "HMDB:HMDB0000190" in result["curies_provided"]


class TestGetCuries:
    """Tests for Normalizer.get_curies method."""

    @pytest.fixture
    def normalizer(self):
        return Normalizer()

    def test_get_curies_all_params(self, normalizer):
        curies, invalid_ids, unrecognized_vocabs = normalizer.get_curies(
            {"unii": "01MP33F412"}, stop_on_invalid_id=False, log_warnings=False, fuzzy_match_vocab=False
        )
        assert curies
        assert not invalid_ids
        assert not unrecognized_vocabs
        assert "UNII:01MP33F412" in curies

    def test_get_curies_uppercase_cleaners(self, normalizer):
        curies, invalid_ids, unrecognized_vocabs = normalizer.get_curies(
            {"UNII": "01mP33f412", "CHEMBL.COMPOUND": "chembl112"}
        )
        assert "UNII:01MP33F412" in curies
        assert "CHEMBL.COMPOUND:CHEMBL112" in curies

    def test_get_curies_array_delimiters_split_compound(self, normalizer):
        """A delimited/compound id string resolves to ALL its codes when array_delimiters is given."""
        curies, invalid_ids, _ = normalizer.get_curies(
            {"ncit": "C34831:C34915"}, array_delimiters=[":"], log_warnings=False, fuzzy_match_vocab=False
        )
        assert set(curies) == {"NCIT:C34831", "NCIT:C34915"}
        assert not invalid_ids
        # Whitespace around delimited codes is cleaned; unparseable fragments are marked invalid
        curies, invalid_ids, _ = normalizer.get_curies(
            {"ncit": "C48660: C37998:C43234(Primary)"},
            array_delimiters=[":"],
            log_warnings=False,
            fuzzy_match_vocab=False,
        )
        assert set(curies) == {"NCIT:C48660", "NCIT:C37998"}
        assert invalid_ids["ncit"] == ["C43234(Primary)"]

    def test_get_curies_prefix_stripping_and_multi_colon(self, normalizer):
        """A single-colon full curie has its prefix stripped (any prefix, not just recognized ones);
        a multi-colon local id is left intact and fails validation rather than resolving to one part."""
        # Recognized prefix is stripped
        curies, _, _ = normalizer.get_curies({"ncit": "NCIT:C34831"}, log_warnings=False, fuzzy_match_vocab=False)
        assert "NCIT:C34831" in curies
        # A nonstandard/aliased prefix is also stripped (we don't require a known prefix)
        curies, _, _ = normalizer.get_curies({"ncit": "foo:C34831"}, log_warnings=False, fuzzy_match_vocab=False)
        assert "NCIT:C34831" in curies
        # Multi-colon (un-split compound) is not silently reduced -> invalid
        curies, invalid_ids, _ = normalizer.get_curies(
            {"ncit": "C34831:C34915:C34916"}, log_warnings=False, fuzzy_match_vocab=False
        )
        assert not curies
        assert invalid_ids["ncit"] == ["C34831:C34915:C34916"]


class TestCleanId:
    """Tests for Normalizer.clean_id -- in particular that a dotted code keeps its '.0'."""

    @pytest.fixture
    def normalizer(self):
        return Normalizer()

    def test_float_whole_number_loses_its_dot_zero(self, normalizer):
        """The artifact we DO mean to undo: pandas types an int column with blanks as float64."""
        assert normalizer.clean_id(12345.0) == "12345"
        assert normalizer.clean_id(250.0) == "250"

    def test_string_keeps_its_dot_zero(self, normalizer):
        """A '.0' inside a STRING is part of the identifier, not a float artifact.

        ICD9 '250.0' (diabetes with coma) is a different code from '250' (diabetes mellitus),
        so stripping it would silently change which entity is referenced.
        """
        assert normalizer.clean_id("250.0") == "250.0"
        assert normalizer.clean_id("12345.0") == "12345.0"
        assert normalizer.clean_id(" 250.0 ") == "250.0"

    def test_non_whole_float_and_nan_are_unharmed(self, normalizer):
        assert normalizer.clean_id(1.5) == "1.5"
        assert normalizer.clean_id(float("nan")) == "nan"

    def test_plain_values_and_dashes(self, normalizer):
        assert normalizer.clean_id("250") == "250"
        assert normalizer.clean_id(12345) == "12345"
        assert normalizer.clean_id("-") == ""

    def test_dotted_code_survives_curie_construction(self, normalizer):
        """End to end: the ICD9 code keeps its '.0' all the way to the curie."""
        curies, _, _ = normalizer.get_curies({"icd9": "250.0"}, log_warnings=False, fuzzy_match_vocab=False)
        assert "ICD9:250.0" in curies
        # ...while a float-typed whole number is still cleaned up.
        curies, _, _ = normalizer.get_curies({"icd9": 250.0}, log_warnings=False, fuzzy_match_vocab=False)
        assert "ICD9:250" in curies


class TestVariantVocabs:
    """CAID and HGVS -- the vocabularies aggregators record as the original endpoints of variant edges."""

    @pytest.fixture
    def normalizer(self):
        return Normalizer()

    def test_caid_ids(self, normalizer):
        curies, _, unrecognized = normalizer.get_curies(
            {"caid": ["CA15984545", "CA321211"]}, log_warnings=False, fuzzy_match_vocab=False
        )
        assert not unrecognized
        assert set(curies) == {"CAID:CA15984545", "CAID:CA321211"}

    def test_caid_is_upper_cased(self, normalizer):
        curies, _, _ = normalizer.get_curies({"caid": "ca15984545"}, log_warnings=False, fuzzy_match_vocab=False)
        assert "CAID:CA15984545" in curies

    def test_caid_rejects_non_ids(self, normalizer):
        curies, invalid, _ = normalizer.get_curies({"caid": "notanid"}, log_warnings=False, fuzzy_match_vocab=False)
        assert not curies
        assert invalid["caid"] == ["notanid"]

    @pytest.mark.parametrize(
        "local_id",
        [
            "NC_000001.11:g.109175441A>G",  # substitution
            "NC_000001.11:g.1398673_1398677del",  # deletion
            "NM_000546.5:c.215C>G",  # coding
            "NP_000537.3:p.Pro72Arg",  # protein
        ],
    )
    def test_hgvs_expressions_keep_their_internal_colon(self, normalizer, local_id):
        """An HGVS local id contains a colon; the reference sequence must not be stripped as a prefix."""
        curies, invalid, unrecognized = normalizer.get_curies(
            {"hgvs": local_id}, log_warnings=False, fuzzy_match_vocab=False
        )
        assert not unrecognized and not invalid
        assert f"HGVS:{local_id}" in curies

    def test_hgvs_full_curie_strips_only_the_real_prefix(self, normalizer):
        curies, _, _ = normalizer.get_curies(
            {"hgvs": "HGVS:NC_000021.9:g.25840043C>G"}, log_warnings=False, fuzzy_match_vocab=False
        )
        assert "HGVS:NC_000021.9:g.25840043C>G" in curies

    def test_hgvs_rejects_non_expressions(self, normalizer):
        curies, invalid, _ = normalizer.get_curies({"hgvs": "garbage"}, log_warnings=False, fuzzy_match_vocab=False)
        assert not curies
        assert invalid["hgvs"] == ["garbage"]

    def test_ordinary_prefix_stripping_is_unaffected(self, normalizer):
        """The colon-aware strip must still remove a genuine leading prefix, and still leave an
        un-split compound intact so it fails validation rather than resolving to one of its parts."""
        curies, _, _ = normalizer.get_curies({"ncit": "NCIT:C34831"}, log_warnings=False, fuzzy_match_vocab=False)
        assert "NCIT:C34831" in curies
        curies, invalid, _ = normalizer.get_curies(
            {"ncit": "C34831:C34915:C34916"}, log_warnings=False, fuzzy_match_vocab=False
        )
        assert not curies
        assert invalid["ncit"] == ["C34831:C34915:C34916"]
