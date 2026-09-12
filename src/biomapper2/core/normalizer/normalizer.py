"""
Main normalization logic for converting local IDs to Biolink-standard curies.

Validates local identifiers and constructs standardized curies using Biolink model prefixes.
"""

import ast
import logging
import re
import sys
from collections import defaultdict
from typing import Any

import pandas as pd

from ...biolink_client import BiolinkClient
from ...utils import (
    ALIASES_PROP,
    CLEANER_PROP,
    VALIDATOR_PROP,
    to_list,
)
from . import cleaners
from .vocab_config import load_prefix_info, load_validator_map


class Normalizer:
    """
    Normalizes local identifiers to Biolink-standard curies.

    Validates IDs against regex patterns and constructs properly formatted curies
    using Biolink model prefix mappings.
    """

    def __init__(self, biolink_client: BiolinkClient | None = None, biolink_version: str | None = None):
        self.validator_prop = VALIDATOR_PROP
        self.cleaner_prop = CLEANER_PROP
        self.aliases_prop = ALIASES_PROP

        # Set up biolink client flexibly (kraken uses Normalizer directly, so needs to pass in biolink version)
        if biolink_client:
            self.biolink_client = biolink_client
        elif biolink_version:
            self.biolink_client = BiolinkClient(biolink_version=biolink_version)
        else:
            self.biolink_client = BiolinkClient()

        self.vocab_info_map = load_prefix_info(self.biolink_client)
        self.vocab_validator_map = load_validator_map()
        self.field_name_to_vocab_name_cache: dict[str, set[str]] = dict()
        self.dashes = {"-", "–", "—", "−", "‐", "‑", "‒"}

    def normalize(
        self,
        item: pd.Series | dict[str, Any] | pd.DataFrame,
        provided_id_fields: list[str],
        array_delimiters: list[str],
        stop_on_invalid_id: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Normalize local IDs to Biolink-standard curies, distinguishing 'provided' vs. 'assigned' IDs.

        Args:
            item: Entity or entities containing local IDs
            provided_id_fields: Fields with user-provided (un-normalized) IDs
            array_delimiters: Characters for splitting delimited ID strings
            stop_on_invalid_id: Halt on invalid IDs (default: False)

        Returns:
            Normalization results as a pd.Series (for single entity) or pd.DataFrame (for multiple entities)
            Includes: curies, curies_provided, curies_assigned, invalid_ids, invalid_ids_provided, invalid_ids_assigned
        """
        logging.debug("Beginning ID normalization step..")

        if isinstance(item, pd.DataFrame):
            # Normalize all entities in the dataframe
            return item.apply(
                self._normalize_entity,
                axis=1,
                provided_id_fields=provided_id_fields,
                array_delimiters=array_delimiters,
                result_type="expand",  # Expands Series into columns
            )
        else:
            # Normalize the single input entity
            return self._normalize_entity(item, provided_id_fields, array_delimiters, stop_on_invalid_id)

    def _normalize_entity(
        self,
        entity: pd.Series | dict[str, Any],
        provided_id_fields: list[str],
        array_delimiters: list[str],
        stop_on_invalid_id: bool = False,
    ) -> pd.Series:
        """
        Normalize local IDs to Biolink-standard curies, distinguishing 'provided' vs. 'assigned' IDs.

        Args:
            entity: Entity containing local IDs
            provided_id_fields: Fields with user-provided (un-normalized) IDs
            array_delimiters: Characters for splitting delimited ID strings
            stop_on_invalid_id: Halt on invalid IDs (default: False)

        Returns:
            Series of curies, curies_provided, curies_assigned, invalid_ids, invalid_ids_provided, invalid_ids_assigned
        """
        # Load/clean the provided and assigned local IDs for this item
        # Parse any delimited strings (multiple identifiers in one string)
        provided_ids: dict[str | tuple, Any] = {
            id_field: (
                self._parse_delimited_string(entity[id_field], array_delimiters)
                if array_delimiters
                else entity[id_field]
            )
            for id_field in provided_id_fields
            if pd.notnull(entity[id_field])
        }
        assigned_ids = entity.get("assigned_ids", dict())

        # Get curies for the provided IDs
        curies_provided, invalid_ids_provided, unrecognized_vocabs_provided = self.get_curies(
            provided_ids, stop_on_invalid_id
        )
        if unrecognized_vocabs_provided:
            logging.warning(
                f"Unrecognized vocabs detected - provided IDs will not be fully utilized:"
                f" {unrecognized_vocabs_provided} fields could not be matched to a known vocab."
            )

        # Get curies for the assigned IDs (per annotator, to track provenance)
        curies_assigned, invalid_ids_assigned, unrecognized_vocabs_assigned = dict(), dict(), set()
        for annotator_slug, annotator_assigned_ids in assigned_ids.items():
            annotator_curies, annotator_invalid_ids, annotator_unrecognized_vocabs = self.get_curies(
                annotator_assigned_ids, stop_on_invalid_id
            )
            curies_assigned[annotator_slug] = list(annotator_curies)
            if annotator_invalid_ids:
                invalid_ids_assigned[annotator_slug] = annotator_invalid_ids
            unrecognized_vocabs_assigned |= annotator_unrecognized_vocabs

        if unrecognized_vocabs_assigned:
            logging.warning(
                f"Unrecognized vocabs detected - assigned IDs will not be fully utilized:"
                f" {unrecognized_vocabs_assigned} fields could not be matched to a known vocab."
            )

        # Form final overall combined set of curies
        curies = set(curies_provided) | set().union(*curies_assigned.values())

        # Return a named Series
        return pd.Series(
            {
                "curies": list(curies),
                "curies_provided": list(curies_provided),
                "curies_assigned": curies_assigned,
                "invalid_ids_provided": invalid_ids_provided,
                "invalid_ids_assigned": invalid_ids_assigned,
                "unrecognized_vocabs_provided": list(unrecognized_vocabs_provided),
                "unrecognized_vocabs_assigned": list(unrecognized_vocabs_assigned),
            }
        )

    def get_curies(
        self,
        local_ids_dict: dict[Any, Any],
        stop_on_invalid_id: bool = False,
        log_warnings: bool = True,
        fuzzy_match_vocab: bool = True,
        array_delimiters: list[str] | None = None,
    ) -> tuple[dict[str, str], dict[str | tuple, list[str]], set[str]]:
        """
        Convert local IDs to curies for all fields in dictionary.

        Args:
            local_ids_dict: Dictionary mapping vocab field names to local IDs
            stop_on_invalid_id: Halt on invalid IDs (default: False)
            log_warnings: Whether to log warnings for invalid IDs
            fuzzy_match_vocab: Whether to get more creative when identifying the ID's vocab
            array_delimiters: Optional characters for splitting a single delimited/compound ID string
                (e.g. "C1:C2") into multiple local IDs, each resolved independently. When omitted,
                each value is treated as a single ID. (The higher-level normalize() splits before
                calling this, so it does not pass this through.)

        Returns:
            Tuple of (valid_curies_dict_with_iris, invalid_ids_dict, unrecognized_vocabs_set)
        """
        curies = dict()
        invalid_ids = defaultdict(list)
        unrecognized_vocabs = set()
        for id_field_name, local_ids_entry in local_ids_dict.items():
            local_ids = to_list(local_ids_entry)
            if array_delimiters:
                # Split any delimited/compound strings into individual local IDs before resolving
                split_ids: list[Any] = []
                for local_id in local_ids:
                    parsed = self._parse_delimited_string(local_id, array_delimiters)
                    split_ids.extend(parsed if isinstance(parsed, list) else [parsed])
                local_ids = split_ids
            id_field_names = [id_field_name] if isinstance(id_field_name, str) else id_field_name
            vocab_names = set()
            for field_name in id_field_names:
                matching_vocabs = self.determine_vocab(field_name, do_fuzzy_matching=fuzzy_match_vocab)
                if matching_vocabs:
                    vocab_names |= matching_vocabs
                else:
                    unrecognized_vocabs.add(field_name)

            logging.debug(f"Matching vocabs are: {vocab_names}")
            if vocab_names:
                for local_id in local_ids:
                    # Make sure the local ID is a nice clean string (not int or float)
                    local_id = self.clean_id(local_id)
                    # Get the curie for this local ID
                    if local_id:  # Sometimes cleaning the local ID can make it empty (like if it was just a space)
                        curie, iri = self._construct_curie(
                            local_id, list(vocab_names), stop_on_failure=stop_on_invalid_id, log_warnings=log_warnings
                        )
                        if curie:
                            curies[curie] = iri
                        else:
                            invalid_ids[id_field_name].append(local_id)

        return curies, dict(invalid_ids), unrecognized_vocabs

    def determine_vocab(self, id_field_name: str, do_fuzzy_matching: bool = True) -> set[str] | None:
        """
        Determine which vocabulary corresponds to an ID field/column name.

        Uses heuristic matching against known vocab names and aliases.

        Args:
            id_field_name: Name of ID field/column (e.g., "CHEBI ID", "Labcorp LOINC id")
            do_fuzzy_matching: Whether to get more creative in identifying a vocab (look for substrings)

        Returns:
            Set of matching vocabulary names (in standardized form)
        """
        logging.debug(f"Determining which vocab corresponds to field '{id_field_name}'")
        field_name_underscored = re.sub(
            r"[-\s]+", "_", id_field_name
        ).lower()  # Replace spaces, hyphens with underscores
        field_name_words = field_name_underscored.split("_")
        field_name_rejoined = "".join(
            [word for word in field_name_words if word not in {"id", "ids", "code", "cid", "codes", "list"}]
        )
        field_name_cleaned = cleaners.clean_vocab_prefix(field_name_rejoined)
        logging.debug(f"Field name cleaned is: {field_name_cleaned}")

        if field_name_cleaned in self.vocab_validator_map:
            # We have an exact match, so we return it
            return {field_name_cleaned}
        elif field_name_cleaned in self.field_name_to_vocab_name_cache:
            # We've already processed this field name before, so we return the cached mapping
            return self.field_name_to_vocab_name_cache[field_name_cleaned]
        else:
            # Check explicit and implicit aliases
            matches_on_alias = set()
            for vocab, info in self.vocab_validator_map.items():
                if info.get(self.aliases_prop) and field_name_cleaned in info[self.aliases_prop]:
                    # This field matches an explicit alias (defined in the vocab_validator_map)
                    matches_on_alias.add(vocab)
                elif "." in vocab and vocab.split(".")[0] == field_name_cleaned:
                    # This field matches implicitly, based on the 'root' vocab name (e.g., 'kegg' for 'kegg.compound')
                    matches_on_alias.add(vocab)
                elif vocab.replace(".", "") == field_name_cleaned:
                    # This field matches implicitly, after removing periods (e.g., 'keggcompound' for 'kegg.compound')
                    matches_on_alias.add(vocab)

            if matches_on_alias:
                self.field_name_to_vocab_name_cache[field_name_cleaned] = matches_on_alias
                return matches_on_alias

            if do_fuzzy_matching:
                # Final tier: check if any known vocab name appears within the field name
                # This handles cases like "labcorploincid" -> "loinc"
                matches_on_substring = set()
                for vocab in self.vocab_validator_map:
                    # Use the root vocab name for substring matching
                    vocab_root = vocab.split(".")[0] if "." in vocab else vocab
                    if vocab_root in field_name_cleaned:
                        matches_on_substring.add(vocab)

                if matches_on_substring:
                    logging.debug(f"Found substring match(es) for '{id_field_name}': {matches_on_substring}")
                    self.field_name_to_vocab_name_cache[field_name_cleaned] = matches_on_substring
                    return matches_on_substring

            return None

    def get_standard_prefix(self, vocab: str | list[str] | None) -> list[str]:
        logging.info(f"Determining standard prefix for input vocab(s): {vocab}")
        vocabs: list[str] = to_list(vocab)
        prefixes = set()
        for vocab_name in vocabs:
            matching_normalized_vocabs = self.determine_vocab(vocab_name)
            if matching_normalized_vocabs:
                logging.info(f"Matching normalized names for vocab '{vocab_name}' are: {matching_normalized_vocabs}")
                for matching_vocab in matching_normalized_vocabs:
                    prefix = self.vocab_info_map[matching_vocab]["prefix"]
                    prefixes.add(prefix)
                    logging.info(f"Standardized prefix for normalized vocab '{matching_vocab}' is: {prefix}")
            else:
                raise ValueError(
                    f"Failed to standardize vocab name '{vocab_name}' - valid vocabs are: "
                    f"{self.vocab_validator_map.keys()}"
                )
        return list(prefixes)

    def is_valid_id(self, local_id: str, vocab_name_cleaned: str) -> tuple[bool, str]:
        """
        Validate local ID for specified vocabulary.

        Args:
            local_id: Local identifier to validate
            vocab_name_cleaned: Lowercase vocabulary name

        Returns:
            Tuple of (is_valid, cleaned_local_id)
        """
        # Grab the proper validation and cleaning functions
        validator = self.vocab_validator_map[vocab_name_cleaned][self.validator_prop]
        cleaner = self.vocab_validator_map[vocab_name_cleaned].get(self.cleaner_prop)

        # Clean the local ID if necessary
        if cleaner:
            local_id = cleaner(local_id)

        # Then determine whether it's valid for the specified vocabulary
        return validator(local_id), local_id

    def _construct_curie(
        self,
        local_id: str,
        vocab_name_cleaned: str | list[str],
        stop_on_failure: bool = False,
        log_warnings: bool = True,
    ) -> tuple[str, str]:
        """
        Construct standardized curie from local ID and vocabulary.

        Args:
            local_id: Local identifier
            vocab_name_cleaned: Vocabulary name(s) to try
            stop_on_failure: Halt on validation failure (default: False)

        Returns:
            Tuple of (curie, iri) - empty strings if validation fails
        """
        # Construct a standardized curie for the given local ID and vocab (or list of vocabs; first valid kept)
        prefixes_lowercase = [vocab_name_cleaned] if isinstance(vocab_name_cleaned, str) else vocab_name_cleaned
        candidate_local_ids = self._candidate_local_ids(local_id)
        curie = ""
        iri = ""
        for prefix_lowercase in prefixes_lowercase:
            is_valid_id, cleaned_local_id = False, local_id
            for candidate in candidate_local_ids:
                is_valid_id, cleaned_local_id = self.is_valid_id(candidate, prefix_lowercase)
                if is_valid_id:
                    break
            if is_valid_id:
                # Return the standardized curie and its corresponding IRI
                prefix_normalized = self.vocab_info_map[prefix_lowercase]["prefix"]
                iri_root = self.vocab_info_map[prefix_lowercase]["iri"]
                iri = f"{iri_root}{cleaned_local_id}" if iri_root else ""
                curie = f"{prefix_normalized}:{cleaned_local_id}"
                iri = iri
            if curie:
                break  # Stop at the first prefix we find that doesn't fail curie construction

        if not curie:
            # The local ID did not pass validation for its corresponding vocab(s)
            if stop_on_failure:
                logging.error(f"Local id '{local_id}' is invalid for {vocab_name_cleaned}")
                sys.exit(1)
            elif log_warnings:
                logging.warning(f"Local id '{local_id}' is invalid for {vocab_name_cleaned}. Skipping.")

        return curie, iri

    def _candidate_local_ids(self, local_id: str) -> list[str]:
        """The forms of ``local_id`` to try validating, most literal first.

        Most vocabularies' ids carry no colon, so a leading "PREFIX:" on one is a full curie whose
        prefix should come off -- and it comes off whether or not we recognize the prefix, since
        sources use nonstandard and aliased ones ("foo:C34831" is still NCIT's C34831).

        But a few vocabularies' ids DO contain a colon -- an HGVS expression is
        "NC_000001.11:g.109175441A>G", where "NC_000001.11" is a reference sequence, not a prefix.
        Blind stripping would leave a meaningless fragment, so the id AS GIVEN is always tried
        first: if it validates for the target vocabulary, that is what it is.

        An un-split compound ("C1:C2:C3") yields no valid candidate and so fails validation rather
        than resolving to one of its parts -- unless its leading segment names a known vocabulary,
        which makes it a full curie wrapping a colon-bearing local id
        ("HGVS:NC_000021.9:g.25840043C>G"). Genuine compounds should be split via get_curies'
        array_delimiters, which runs before this.
        """
        candidates = [local_id]
        if local_id.startswith("http") or ":" not in local_id:
            return candidates
        prefix, remainder = local_id.split(":", 1)
        if local_id.count(":") == 1 or cleaners.clean_vocab_prefix(prefix) in self.vocab_validator_map:
            candidates.append(remainder)
        return candidates

    @staticmethod
    def _parse_delimited_string(value: Any, array_delimiters: list[str]) -> Any:
        if isinstance(value, str):
            # Check for Python list/tuple/set formats
            if (
                (value.startswith("[") and value.endswith("]"))
                or (value.startswith("(") and value.endswith(")"))
                or (value.startswith("{") and value.endswith("}"))
            ):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, (list, tuple, set)):
                        return list(parsed)
                except (ValueError, SyntaxError):
                    pass  # Fall through to delimiter-based parsing
            return [local_id for local_id in re.split(f"[{''.join(array_delimiters)}]", value)]
        else:
            return value

    def clean_id(self, local_id: str | float | int) -> str:
        """Convert numeric IDs to strings, strip whitespace, removing the trailing '.0' a whole
        number picks up when it arrives as a float.

        The '.0' is stripped ONLY for an actual float input, never for a string. A float is the
        spreadsheet artifact we mean to undo (pandas types an integer column containing blanks as
        float64, so the code 12345 arrives as 12345.0), whereas in a string the '.0' is part of the
        identifier -- ICD9 '250.0' (diabetes with coma) is a DIFFERENT code from '250' (diabetes
        mellitus), so stripping it silently changes the entity being referenced. Anything that can
        reach us as text (an ICD/OMOP/HCPCS-style dotted code) is therefore left intact.
        """
        # NaN is never a whole number (and int(nan) raises), so screen it out before the int compare.
        if isinstance(local_id, float) and local_id == local_id and local_id == int(local_id):
            return str(int(local_id))
        local_id = str(local_id).strip()
        if local_id in self.dashes:
            return ""
        else:
            return local_id
