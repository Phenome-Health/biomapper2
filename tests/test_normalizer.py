"""Tests for the Normalizer class."""

import pandas as pd
import pytest

from biomapper2.core.normalizer import Normalizer


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
