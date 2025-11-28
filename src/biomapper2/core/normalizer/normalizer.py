"""
Main normalization logic for converting local IDs to Biolink-standard curies.

Validates local identifiers and constructs standardized curies using Biolink model prefixes.
"""

import logging
import re
import sys
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List, Set

import pandas as pd

from . import validators, cleaners
from .vocab_config import load_prefix_info, load_validator_map
from ...config import BIOLINK_VERSION_DEFAULT


class Normalizer:
    """
    Normalizes local identifiers to Biolink-standard curies.

    Validates IDs against regex patterns and constructs properly formatted curies
    using Biolink model prefix mappings.
    """

    def __init__(self, biolink_version: Optional[str] = None):
        self.validator_prop = "validator"
        self.cleaner_prop = "cleaner"
        self.aliases_prop = "aliases"
        self.biolink_version = biolink_version if biolink_version else BIOLINK_VERSION_DEFAULT
        self.vocab_info_map = load_prefix_info(self.biolink_version)
        self.vocab_validator_map = load_validator_map()
        self.field_name_to_vocab_name_cache: Dict[str, Set[str]] = dict()
        self.dashes = {"-", "–", "—", "−", "‐", "‑", "‒"}

    def normalize(
        self,
        item: pd.Series | Dict[str, Any] | pd.DataFrame,
        provided_id_fields: List[str],
        array_delimiters: List[str],
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
        logging.debug(f"Beginning ID normalization step..")

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
        entity: pd.Series | Dict[str, Any],
        provided_id_fields: List[str],
        array_delimiters: List[str],
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
        provided_ids: Dict[str | tuple, Any] = {
            id_field: (
                self._parse_delimited_string(entity[id_field], array_delimiters)
                if array_delimiters
                else entity[id_field]
            )
            for id_field in provided_id_fields
            if pd.notnull(entity[id_field])
        }
        assigned_ids = entity.get("assigned_ids", dict())

        assigned_ids_flat: Dict[str, Set[Any]] = defaultdict(set)
        for annotator, annotator_assigned_ids in assigned_ids.items():
            for vocab, local_ids_dict in annotator_assigned_ids.items():
                assigned_ids_flat[vocab] |= set(local_ids_dict)

        # Get curies for the provided/assigned IDs
        curies_provided, invalid_ids_provided = self.get_curies(provided_ids, stop_on_invalid_id)
        curies_assigned, invalid_ids_assigned = self.get_curies(assigned_ids_flat, stop_on_invalid_id)

        # Form final result
        curies = set(curies_provided) | set(curies_assigned)
        invalid_ids = {
            id_field: invalid_ids_provided.get(id_field, []) + invalid_ids_assigned.get(id_field, [])
            for id_field in set(invalid_ids_provided) | set(invalid_ids_assigned)
        }

        # Return a named Series
        return pd.Series(
            {
                "curies": list(curies),
                "curies_provided": list(curies_provided),
                "curies_assigned": list(curies_assigned),
                "invalid_ids": invalid_ids,
                "invalid_ids_provided": invalid_ids_provided,
                "invalid_ids_assigned": invalid_ids_assigned,
            }
        )

    def get_curies(
        self, local_ids_dict: Dict[Any, Any], stop_on_invalid_id: bool = False
    ) -> Tuple[Dict[str, str], Dict[str | tuple, List[str]]]:
        """
        Convert local IDs to curies for all fields in dictionary.

        Args:
            local_ids_dict: Dictionary mapping vocab field names to local IDs
            stop_on_invalid_id: Halt on invalid IDs (default: False)

        Returns:
            Tuple of (valid_curies_dict_with_iris, invalid_ids_dict)
        """
        curies = dict()
        invalid_ids = defaultdict(list)
        for id_field_name, local_ids_entry in local_ids_dict.items():
            local_ids = [local_ids_entry] if isinstance(local_ids_entry, str) else local_ids_entry
            id_field_names = [id_field_name] if isinstance(id_field_name, str) else id_field_name
            vocab_names = set()
            for field_name in id_field_names:
                vocab_names |= self.determine_vocab(field_name)

            logging.debug(f"Matching vocabs are: {vocab_names}")
            for local_id in local_ids:
                # Make sure the local ID is a nice clean string (not int or float)
                local_id = self.clean_id(local_id)
                # Get the curie for this local ID
                if local_id:  # Sometimes cleaning the local ID can make it empty (like if it was just a space)
                    curie, iri = self._construct_curie(local_id, list(vocab_names), stop_on_failure=stop_on_invalid_id)
                    if curie:
                        curies[curie] = iri
                    else:
                        invalid_ids[id_field_name].append(local_id)
        return curies, dict(invalid_ids)

    def determine_vocab(self, id_field_name: str) -> Set[str]:
        """
        Determine which vocabulary corresponds to an ID field/column name.

        Uses heuristic matching against known vocab names and aliases.

        Args:
            id_field_name: Name of ID field/column

        Returns:
            List of matching vocabulary names (in standardized form)
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
            # Otherwise we inspect vocab aliases (explicit and implicit ones) to try to find a match
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
                # Cache this mapping for quick lookup later
                self.field_name_to_vocab_name_cache[field_name_cleaned] = matches_on_alias
                return matches_on_alias
            else:
                valid_vocab_names = ", ".join(
                    [
                        f"{vocab} (or: {', '.join(info[self.aliases_prop])})" if info.get(self.aliases_prop) else vocab
                        for vocab, info in self.vocab_validator_map.items()
                    ]
                )
                error_message = (
                    f"Could not determine vocab for field '{id_field_name}'. "
                    f"Valid vocab names are: {valid_vocab_names}"
                )
                logging.error(error_message)
                raise ValueError(error_message)

    def is_valid_id(self, local_id: str, vocab_name_cleaned: str) -> Tuple[bool, str]:
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
        self, local_id: str, vocab_name_cleaned: str | List[str], stop_on_failure: bool = False
    ) -> Tuple[str, str]:
        """
        Construct standardized curie from local ID and vocabulary.

        Args:
            local_id: Local identifier
            vocab_name_cleaned: Vocabulary name(s) to try
            stop_on_failure: Halt on validation failure (default: False)

        Returns:
            Tuple of (curie, iri) - empty strings if validation fails
        """
        # First, if this is a proper curie - remove its prefix
        local_id = local_id.split(":")[1] if ":" in local_id and not local_id.startswith("http") else local_id
        # Construct a standardized curie for the given local ID and vocabulary (or list of vocabularies; first valid kept)
        prefixes_lowercase = [vocab_name_cleaned] if isinstance(vocab_name_cleaned, str) else vocab_name_cleaned
        curie = ""
        iri = ""
        for prefix_lowercase in prefixes_lowercase:
            is_valid_id, cleaned_local_id = self.is_valid_id(local_id, prefix_lowercase)
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
            else:
                logging.warning(f"Local id '{local_id}' is invalid for {vocab_name_cleaned}. Skipping.")

        return curie, iri

    @staticmethod
    def _parse_delimited_string(value: Any, array_delimiters: List[str]) -> Any:
        if isinstance(value, str):
            return [local_id for local_id in re.split(f"[{''.join(array_delimiters)}]", value)]
        else:
            return value

    def clean_id(self, local_id: str | float | int) -> str:
        """Convert numeric IDs to strings, strip whitespace, removing trailing .0 for whole numbers..."""
        local_id = str(local_id).strip()
        try:
            if local_id.endswith(".0") and float(local_id) == int(float(local_id)):
                return local_id.removesuffix(".0")
        except (ValueError, TypeError):
            pass
        if local_id in self.dashes:
            return ""
        else:
            return local_id
