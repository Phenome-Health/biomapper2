import json
import logging
from collections.abc import Iterable
from typing import cast

import inflect
import requests
import yaml
from bmt import Toolkit

from .config import BIOLINK_VERSION_DEFAULT, CACHE_DIR
from .utils import setup_logging, to_set

setup_logging()


class BiolinkClient:
    """Client for Biolink Model Toolkit operations (with caching)."""

    def __init__(self, biolink_version: str | None = None):
        self.biolink_version = biolink_version if biolink_version else BIOLINK_VERSION_DEFAULT
        biolink_url = (
            f"https://raw.githubusercontent.com/biolink/biolink-model/"
            f"refs/tags/v{self.biolink_version}/biolink-model.yaml"
        )
        logging.info("Initializing bmt (Biolink Model Toolkit)...")
        self.bmt = Toolkit(schema=biolink_url)
        self.biolink_ancestors_cache = dict()
        self.biolink_descendants_cache = dict()
        logging.info(f"Initialized BiolinkClient with version {biolink_version}")

    def get_ancestors(self, items: str | Iterable[str] | None) -> set[str]:
        item_set = to_set(items)
        all_ancestors = set()
        for item in item_set:
            if item not in self.biolink_ancestors_cache:
                ancestors = set(self.bmt.get_ancestors(item, formatted=True, mixin=True, reflexive=True))
                self.biolink_ancestors_cache[item] = ancestors
            all_ancestors |= self.biolink_ancestors_cache[item]

        return all_ancestors

    def get_descendants(self, items: str | Iterable[str] | None) -> set[str]:
        item_set = to_set(items)
        all_descendants = set()
        for item in item_set:
            if item not in self.biolink_descendants_cache:
                descendants = set(self.bmt.get_descendants(item, formatted=True, mixin=True, reflexive=True))
                self.biolink_descendants_cache[item] = descendants
            all_descendants |= self.biolink_descendants_cache[item]

        return all_descendants

    def get_prefix_map(self) -> dict[str, str]:
        logging.debug(f"Grabbing biolink prefix map for version: {self.biolink_version}")
        url = (
            f"https://raw.githubusercontent.com/biolink/biolink-model/refs/tags/v{self.biolink_version}/"
            f"project/prefixmap/biolink-model-prefix-map.json"
        )
        prefix_to_iri_map = self._load_biolink_file(url)
        return prefix_to_iri_map

    def _load_biolink_file(self, url: str) -> dict:
        """
        Download and cache Biolink model file (or load from cache if already exists).

        Args:
            url: URL to Biolink JSON/YAML file

        Returns:
            Parsed JSON content
        """
        file_name = url.split("/")[-1]
        file_name_json = file_name.split(".")[0] + f"_{self.biolink_version}" + ".json"
        local_path = CACHE_DIR / file_name_json
        logging.debug(f"Local file path is: {local_path}")

        # Download the file if we don't already have it cached
        if not local_path.exists():
            logging.info(f"Downloading YAML file from {url}. local path is: {local_path}")
            response = requests.get(url)
            response.raise_for_status()
            if file_name.endswith(".yaml"):
                response_json = yaml.safe_load(response.text)
            else:
                response_json = response.json()

            # Cache the response
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(local_path, "w+") as cache_file:
                json.dump(response_json, cache_file, indent=2)

        # Read and return the cached JSON
        with open(local_path) as cache_file:
            contents = json.load(cache_file)
            return contents

    def standardize_entity_type(self, entity_type: str) -> str:
        # Map any aliases to their corresponding biolink category
        entity_type_singular = self.singularize(entity_type.removeprefix("biolink:"))
        entity_type_cleaned = "".join(entity_type_singular.lower().split())
        aliases = {
            "metabolite": "SmallMolecule",
            "lipid": "SmallMolecule",
            "clinicallab": "ClinicalFinding",
            "lab": "ClinicalFinding",
        }
        category_raw = aliases.get(entity_type_cleaned, entity_type_cleaned)

        if self.bmt.is_category(category_raw):
            category_element = self.bmt.get_element(category_raw)
            if category_element:
                category = category_element["class_uri"]
                logging.info(f"Biolink category for entity type '{entity_type}' is: {category}")
                return category

        message = (
            f"Could not find valid Biolink category for entity type '{entity_type}'. "
            f"Valid entity types are: {self.get_descendants('NamedThing')}. "
            f"Or accepted aliases are: {aliases}. Will proceed with top-level Biolink category "
            f"of NamedThing (Annotators may be over-selected/not used ideally)."
        )
        logging.warning(message)
        return "biolink:NamedThing"

    @staticmethod
    def singularize(phrase: str) -> str:
        """Singularize the last word of a phrase.

        Examples:
            "metabolites" -> "metabolite"
            "amino acids" -> "amino acid"
            "classes" -> "class"
        """
        _inflect_engine = inflect.engine()

        words = phrase.split()
        if not words:
            return phrase

        last_word = words[-1]
        singular = _inflect_engine.singular_noun(cast(inflect.Word, last_word))

        # singular_noun returns False if the word is already singular
        if singular:
            words[-1] = singular

        return " ".join(words)
