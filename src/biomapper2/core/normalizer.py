"""
ID normalization module for converting local IDs to Biolink-standard curies.

Validates local identifiers and constructs standardized curies using Biolink model prefixes.
"""
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple, List, Set

import requests
import yaml
import pandas as pd

from ..config import BIOLINK_VERSION


# TODO: add in the string parsing and other bits necessary from the extractor module in phenome-kg
# TODO: handle fuzzy id prop names (figure out best match)


class Normalizer:
    """
    Normalizes local identifiers to Biolink-standard curies.

    Validates IDs against regex patterns and constructs properly formatted curies
    using Biolink model prefix mappings.
    """
    def __init__(self):
        self.validator_prop = 'validator'
        self.cleaner_prop = 'cleaner'
        self.aliases_prop = 'aliases'
        self.vocab_info_map = self._load_prefix_info(BIOLINK_VERSION)
        self.vocab_validator_map = self._load_validator_map()


    def normalize(self,
                  item: pd.Series | Dict[str, Any],
                  provided_id_fields: List[str],
                  array_delimiters: List[str],
                  stop_on_invalid_id: bool = False) -> Tuple[List[str], List[str], List[str], Dict[str, list], Dict[str, list], Dict[str, list]]:
        """
        Normalize local IDs to Biolink-standard curies.

        Args:
            item: Entity containing local IDs
            provided_id_fields: Fields with user-provided IDs
            array_delimiters: Characters for splitting delimited ID strings
            stop_on_invalid_id: Halt on invalid IDs (default: False)

        Returns:
            Tuple of (curies, curies_provided, curies_assigned, invalid_ids, invalid_ids_provided, invalid_ids_assigned)
        """
        logging.debug(f"Beginning ID normalization step..")
        # Load/clean the provided and assigned local IDs for this item
        if array_delimiters:
            # Parse any delimited strings (multiple identifiers in one string)
            provided_ids = {id_field: [local_id for local_id in re.split(f"[{''.join(array_delimiters)}]", item[id_field])]
                             for id_field in provided_id_fields if pd.notnull(item[id_field])}
        else:
            provided_ids = {id_field: item[id_field] for id_field in provided_id_fields if pd.notnull(item[id_field])}
        assigned_ids = item['assigned_ids']

        # Get curies for the provided/assigned IDs
        curies_provided, invalid_ids_provided = self.get_curies(provided_ids, stop_on_invalid_id)
        curies_assigned, invalid_ids_assigned = self.get_curies(assigned_ids, stop_on_invalid_id)

        # Form final result
        curies = curies_provided | curies_assigned
        invalid_ids = {id_field: invalid_ids_provided.get(id_field, []) + invalid_ids_assigned.get(id_field, [])
                       for id_field in set(invalid_ids_provided) | set(invalid_ids_assigned)}

        return list(curies), list(curies_provided), list(curies_assigned), invalid_ids, invalid_ids_provided, invalid_ids_assigned


    def get_curies(self, local_ids_dict: Dict[str, Any], stop_on_invalid_id: bool = False) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Convert local IDs to curies for all fields in dictionary.

        Args:
            local_ids_dict: Dictionary mapping field names to local IDs
            stop_on_invalid_id: Halt on invalid IDs (default: False)

        Returns:
            Tuple of (valid_curies_set, invalid_ids_dict)
        """
        curies = set()
        invalid_ids = defaultdict(list)
        for id_field_name, local_ids_entry in local_ids_dict.items():
            local_ids = local_ids_entry if isinstance(local_ids_entry, list) else [local_ids_entry]
            vocab_names = self.determine_vocab(id_field_name)
            logging.debug(f"Matching vocabs are: {vocab_names}")
            for local_id in local_ids:
                # Make sure the local ID is a nice clean string
                local_id = self.clean_numeric_id(local_id)
                # Get the curie for this local ID
                curie, iri = self.construct_curie(local_id, vocab_names, stop_on_failure=stop_on_invalid_id)
                if curie:
                    curies.add(curie)
                else:
                    invalid_ids[id_field_name].append(local_id)
        return curies, dict(invalid_ids)


    def determine_vocab(self, column_name: str, entity_type: Optional[str] = None) -> List[str]:
        """
        Determine which vocabulary/prefix corresponds to a column name.

        Uses heuristic matching against known vocab names and aliases.

        Args:
            column_name: Name of ID column
            entity_type: Optional entity type hint

        Returns:
            List of matching vocabulary names
        """
        logging.debug(f"Determining which vocab corresponds to column '{column_name}'")
        col_name_underscored = re.sub(r'[-\s]+', '_', column_name).lower()  # Replace spaces, hyphens with underscores
        col_name_words = col_name_underscored.split('_')
        col_name_rejoined = ''.join([word for word in col_name_words if word not in {'id', 'ids', 'code', 'codes', 'list'}])
        col_name_cleaned = self.clean_vocab_prefix(col_name_rejoined)
        logging.debug(f"Column name cleaned is: {col_name_cleaned}")

        # If we have an exact match, return it
        if col_name_cleaned in self.vocab_validator_map:
            return [col_name_cleaned]
        else:
            # Inspect vocab aliases (explicit and implicit ones) to try to find a match
            matches_on_alias = set()
            for vocab, info in self.vocab_validator_map.items():
                if info.get(self.aliases_prop) and col_name_cleaned in info[self.aliases_prop]:
                    # This column matches an explicit alias (defined in the vocab_validator_map)
                    matches_on_alias.add(vocab)
                elif '.' in vocab and vocab.split('.')[0] == col_name_cleaned:
                    # This column matches implicitly, based on the 'root' vocab name (e.g., 'kegg' for 'kegg.compound')
                    matches_on_alias.add(vocab)
                elif vocab.replace('.', '') == col_name_cleaned:
                    # This column matches implicitly, after removing periods (e.g., 'keggcompound' for 'kegg.compound')
                    matches_on_alias.add(vocab)
            if matches_on_alias:
                return list(matches_on_alias)
            else:
                valid_vocab_names = ', '.join([f"{vocab} (or: {', '.join(info[self.aliases_prop])})"
                                               if info.get(self.aliases_prop) else vocab
                                               for vocab, info in self.vocab_validator_map.items()])
                error_message = (f"Could not determine vocab for column '{column_name}'. "
                                 f"Valid vocab names are: {valid_vocab_names}")
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


    def construct_curie(self, local_id: str, vocab_name_cleaned: str | List[str], stop_on_failure: bool = False) -> Tuple[str, str]:
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
        local_id = local_id.split(':')[1] if ':' in local_id and not local_id.startswith('http') else local_id
        # Construct a standardized curie for the given local ID and vocabulary (or list of vocabularies; first valid kept)
        prefixes_lowercase = [vocab_name_cleaned] if isinstance(vocab_name_cleaned, str) else vocab_name_cleaned
        curie = ''
        iri = ''
        for prefix_lowercase in prefixes_lowercase:
            is_valid_id, cleaned_local_id = self.is_valid_id(local_id, prefix_lowercase)
            if is_valid_id:
                # Return the standardized curie and its corresponding IRI
                prefix_normalized = self.vocab_info_map[prefix_lowercase]['prefix']
                iri_root = self.vocab_info_map[prefix_lowercase]['iri']
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


    # ------------------------------------------- INSTANTIATION HELPERS  --------------------------------------------- #


    def _load_prefix_info(self, biolink_version: str) -> Dict[str, Dict[str, str]]:
        """
        Load Biolink model prefix map and add custom entries.

        Args:
            biolink_version: Biolink model version string

        Returns:
            Dictionary mapping lowercase prefixes to {prefix, iri}
        """
        logging.debug(f"Grabbing biolink prefix map for version: {biolink_version}")
        url = f"https://raw.githubusercontent.com/biolink/biolink-model/refs/tags/v{biolink_version}/project/prefixmap/biolink-model-prefix-map.json"
        prefix_to_iri_map = self._load_biolink_file(url, biolink_version)

        # Remove prefixes as needed
        if 'KEGG' in prefix_to_iri_map:
            del prefix_to_iri_map['KEGG']  # We want to use only KEGG.COMPOUND, KEGG.REACTION, etc.

        # Add prefixes as needed (ones we're making up, that don't exist in biolink)
        prefix_to_iri_map['USZIPCODE'] = "https://www.unitedstateszipcodes.org/"
        prefix_to_iri_map['SMILES'] = "https://pubchem.ncbi.nlm.nih.gov/compound/"
        prefix_to_iri_map['CVCL'] = "https://web.expasy.org/cellosaurus/CVCL_"
        prefix_to_iri_map['VESICLEPEDIA'] = "http://microvesicles.org/exp_summary?exp_id="
        prefix_to_iri_map['NDFRT'] = "http://purl.bioontology.org/ontology/NDFRT/"
        prefix_to_iri_map['BVBRC'] = "https://www.bv-brc.org/view/Genome/"
        prefix_to_iri_map['GeoNames'] = "http://www.geonames.org/search.html?q="  # Note: this doesn't go exactly to page for item, but closest I could find
        prefix_to_iri_map['NHANES'] = "https://dsld.od.nih.gov/label/"  # These IRIs work, but weirdly SPOKE's identifiers for these nodes don't match what they have..
        prefix_to_iri_map['MIRDB'] = "https://mirdb.org/cgi-bin/mature_mir.cgi?name="  # Not sure if it's right to use a 'mature' iri like this for all...
        prefix_to_iri_map['CYTOBAND'] = ""  # Haven't found good iri for these yet..
        prefix_to_iri_map['CHR'] = ""  # Country Health Rankings.. Haven't found good iri for these yet
        prefix_to_iri_map['AHRQ'] = ""  # AHRQ SDOH Database
        prefix_to_iri_map['HPS'] = ""  # Household Pulse Survey
        prefix_to_iri_map['mirbase'] = "https://mirbase.org/hairpin/"  # Biolink has mirbase in here, but their iri doesn't work
        prefix_to_iri_map['metacyc.pathway'] = "https://metacyc.org/pathway?orgid=META&id="  # Biolink has metacyc.reaction, but not pathway
        prefix_to_iri_map['metacyc.ec'] = "https://biocyc.org/META/NEW-IMAGE?type=EC-NUMBER&object=EC-"  # Biolink has metacyc.reaction, but not ec (these are like provisional ec codes, not yet in explorenz)
        prefix_to_iri_map['FIPS.PLACE'] = ""
        prefix_to_iri_map['FIPS.STATE'] = ""
        prefix_to_iri_map['PHARMVAR'] = ""  # This wants a number rather than the symbol..
        prefix_to_iri_map['CDCSVI'] = ""  # CDC Social Vulnerability Index
        prefix_to_iri_map['LM'] = "https://www.lipidmaps.org/databases/lmsd/LM"
        prefix_to_iri_map['SLM'] = "https://www.swisslipids.org/#/entity/SLM:"
        prefix_to_iri_map['LIPIDBANK'] = ""  # Could look harder for this iri..
        prefix_to_iri_map['PLANTFA'] = ""  # Could look harder for this iri..
        prefix_to_iri_map['RM'] = "https://www.metabolomicsworkbench.org/databases/refmet/refmet_details.php?REFMET_ID=RM"


        # Override prefixes as needed (if Biolink's iri is broken)
        prefix_to_iri_map['OMIM'] = "https://omim.org/entry/"
        prefix_to_iri_map['REACT'] = "https://reactome.org/content/detail/" # Works for Complexes and Pathways (I think)


        # Return a mapping of lowercase prefixes to their normalized form (varying capitalization) and IRIs
        vocab_info_map = {self.clean_vocab_prefix(prefix): {'prefix': prefix, 'iri': iri}
                          for prefix, iri in prefix_to_iri_map.items()}

        return vocab_info_map


    def _load_validator_map(self) -> Dict[str, Dict[str, Callable | List[str]]]:
        """
        Load vocabulary validator/cleaner function mappings.

        Returns:
            Dictionary mapping vocab names to validator, cleaner, and alias configs
        """
        validator = self.validator_prop
        cleaner = self.cleaner_prop
        aliases = self.aliases_prop
        return {
            'ahrq': {validator: self.is_ahrq_id},
            'bfo': {validator: self.is_bfo_id},
            'bvbrc': {validator: self.is_bvbrc_id},
            'cas': {validator: self.is_cas_id},
            'cdcsvi': {validator: self.is_cdcsvi_id},
            'chebi': {validator: self.is_chebi_id},
            'chembl.compound': {validator: self.is_chembl_compound_id},
            'chembl.target': {validator: self.is_chembl_target_id},
            'chr': {validator: self.is_chr_id},
            'cl': {validator: self.is_cl_id},
            'clo': {validator: self.is_clo_id, aliases: ['celllineontology']},
            'complexportal': {validator: self.is_complexportal_id},
            'cvcl': {validator: self.is_cellosaurus_id},
            'cytoband': {validator: self.is_cytoband_id},
            'dbsnp': {validator: self.is_dbsnp_id},
            'doid': {validator: self.is_doid_id},
            'drugbank': {validator: self.is_drugbank_id},
            'ec': {validator: self.is_ec_id, aliases: ['explorenz']},
            'efo': {validator: self.is_efo_id},
            'ensembl': {validator: self.is_ensembl_gene_id},
            'envo': {validator: self.is_envo_id},
            'fips.place': {validator: self.is_fips_compound_id},
            'fips.state': {validator: self.is_fips_state_id},
            'geonames': {validator: self.is_geonames_id},
            'go': {validator: self.is_go_id},
            'hmdb': {validator: self.is_hmdb_id, cleaner: self.clean_hmdb_id},
            'hps': {validator: self.is_hps_id},
            'icd9': {validator: self.is_icd9_id},
            'icd10': {validator: self.is_icd10_id},
            'inchikey': {validator: self.is_inchikey_id},
            'kegg.compound': {validator: self.is_kegg_compound_id},
            'kegg.drug': {validator: self.is_kegg_drug_id},
            'kegg.reaction': {validator: self.is_kegg_reaction_id},
            'lipidbank': {validator: self.is_lipidbank_id},
            'loinc': {validator: self.is_loinc_id},
            'lm': {validator: self.is_lipidmaps_id, cleaner: lambda x: x.removeprefix('LM'), aliases: ['lipidmaps']},
            'mesh': {validator: self.is_mesh_id},
            'metacyc.ec': {validator: self.is_metacyc_ec_id},
            'metacyc.pathway': {validator: self.is_metacyc_pathway_id},
            'metacyc.reaction': {validator: self.is_metacyc_reaction_id},
            'mirbase': {validator: self.is_mirbase_id},
            'mirdb': {validator: self.is_mirdb_id},
            'mondo': {validator: self.is_mondo_id},
            'ncbigene': {validator: self.is_ncbigene_id, aliases: ['entrez', 'entrezgene']},
            'ncbitaxon': {validator: self.is_ncbitaxon_id, aliases: ['ncbitaxonomy']},
            'ndfrt': {validator: self.is_ndfrt_id},
            'nhanes': {validator: self.is_nhanes_id},
            'omim': {validator: self.is_omim_id},
            'pfam': {validator: self.is_pfam_id},
            'pharmvar': {validator: self.is_pharmvar_id},
            'pubchem.compound': {validator: self.is_pubchem_compound_id},
            'plantfa': {validator: self.is_plantfa_id},
            'react': {validator: self.is_reactome_id, aliases: ['reactome']},
            'rm': {validator: self.is_refmet_id, cleaner: lambda x: x.removeprefix('RM'), aliases: ['refmet']},
            'slm': {validator: self.is_slm_id, cleaner: lambda x: x.removeprefix('SLM:')},
            'smiles': {validator: self.is_smiles_string},
            'snomedct': {validator: self.is_snomedct_id, cleaner: self.clean_snomed_id, aliases: ['snomed']},
            'uberon': {validator: self.is_uberon_id},
            'umls': {validator: self.is_umls_id},
            'uniprotkb': {validator: self.is_uniprot_id, aliases: ['uniprot']},
            'uszipcode': {validator: self.is_uszipcode_id, cleaner: self.clean_zipcode},
            'vesiclepedia': {validator: self.is_vesiclepedia_id},
            'wikipathways': {validator: self.is_wikipathways_id, cleaner: self.clean_wikipathways_id},
        }

    @staticmethod
    def _load_biolink_file(url: str, biolink_version: str) -> dict:
        """
        Download and cache Biolink model file.

        Args:
            url: URL to Biolink JSON/YAML file
            biolink_version: Version string for cache naming

        Returns:
            Parsed JSON content
        """
        project_root = Path(__file__).parents[2]

        cache_dir = project_root / 'cache'
        file_name = url.split('/')[-1]
        file_name_json = file_name.split('.')[0] + f"_{biolink_version}" + '.json'
        local_path = cache_dir / file_name_json
        logging.debug(f"Local file path is: {local_path}")

        # Download the file if we don't already have it cached
        if not local_path.exists():
            logging.info(f"Downloading YAML file from {url}. local path is: {local_path}")
            response = requests.get(url)
            response.raise_for_status()
            if file_name.endswith('.yaml'):
                response_json = yaml.safe_load(response.text)
            else:
                response_json = response.json()

            # Cache the response
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'w+') as cache_file:
                json.dump(response_json, cache_file, indent=2)

        # Read and return the cached JSON
        with open(local_path, 'r') as cache_file:
            contents = json.load(cache_file)
            return contents


    # -------------------------------------------------- CLEANERS ---------------------------------------------------- #

    @staticmethod
    def clean_vocab_prefix(vocab: str) -> str:
        """Remove non-alphanumeric characters (except periods) from vocabulary name."""
        return re.sub(r'[^a-z0-9.]', '', vocab.lower())

    @staticmethod
    def clean_numeric_id(local_id: str | float | int) -> str:
        """Convert numeric IDs to strings, removing trailing .0 for whole numbers."""
        try:
            float_val = float(local_id)
            # If it's a whole number, return as int string (removes .0)
            if float_val == int(float_val):
                return str(int(float_val))
        except (ValueError, TypeError):
            pass
        # Return as-is for non-numeric or decimal values
        return str(local_id)

    def clean_snomed_id(self, local_id: str) -> str:
        return self.clean_numeric_id(local_id)

    @staticmethod
    def clean_zipcode(local_id: str) -> str:
        # Strip prefix off of zipcodes like AZ-85039
        return local_id.split('-')[-1]

    @staticmethod
    def clean_wikipathways_id(local_id: str) -> str:
        # Get rid of version suffix info, like in WP5395_r126912
        return local_id.split('_')[0]

    @staticmethod
    def clean_hmdb_id(local_id: str) -> str:
        # Remove any double-HMDB prefix
        if local_id.startswith('HMDBHMDB'):
            local_id = local_id.removeprefix('HMDB')

        # Convert any 5-digit HMDB local IDs to current 7-digit form
        if local_id.startswith('HMDB') and len(local_id) == 9:
            id_digits = local_id.removeprefix('HMDB')
            return f"HMDB00{id_digits}"
        else:
            return local_id



    # ----------------------------------------------- VALIDATORS ---------------------------------------------------- #

    @staticmethod
    def is_loinc_id(local_id: str) -> bool:
        # LOINC codes: digits followed by dash and check digit (e.g., 27858-0)
        # or LP codes: LP followed by digits and dash-digit (e.g., LP32606-3)
        return bool(re.match(r'^(LP)?\d+-\d$', local_id))

    @staticmethod
    def is_lipidbank_id(local_id: str) -> bool:
        # Allows: 3 uppercase letters followed by exactly 4 digits
        # Examples: XPR4101, DFA8145
        return bool(re.match(r'^[A-Z]{3}\d{4}$', local_id))

    @staticmethod
    def is_lipidmaps_id(local_id: str) -> bool:
        # Allows: 2 uppercase letters followed by a mix of uppercase letters and digits
        # Examples: ST02030282, PR0103110003, SP0501AA01
        return bool(re.match(r'^[A-Z]{2}[A-Z0-9]+$', local_id))

    @staticmethod
    def is_mesh_id(local_id: str) -> bool:
        # Allows: D, C, or M followed by one or more digits
        return bool(re.match(r'^[DCM]\d+$', local_id))

    @staticmethod
    def is_metacyc_ec_id(local_id: str) -> bool:
        # Allows three digits groups, then a final group of alphanumeric characters
        return bool(re.match(r'^\d+\.\d+\.\d+\.[a-zA-Z0-9]+$', local_id))

    @staticmethod
    def is_metacyc_reaction_id(local_id: str) -> bool:
        # Allows: Hyphen-separated uppercase/numeric or capitalized alpha parts; must contain 'RXN' somewhere
        # e.g., 3.2.1.68-RXN, TRANS-RXN0-593, CYPRIDINA-LUCIFERIN-2-MONOOXYGENASE-RXN, RXN0-5258-Yeast
        has_valid_chars = bool(re.match(r'^[A-Za-z0-9-.+]+$', local_id))
        parts = local_id.split('-')
        has_proper_capitalization = all(
            part.isupper() or (not any(char.isalpha() for char in part)) or (part.isalpha()) for part in parts)
        return has_valid_chars and has_proper_capitalization and 'RXN' in local_id

    @staticmethod
    def is_metacyc_pathway_id(local_id: str) -> bool:
        # MetaCyc pathway IDs: examples: PWY-#### or PWY0-#### or DESCRIPTIVE-NAME-PWY or PWY18C3-9
        has_valid_chars = bool(re.match(r'^[A-Z0-9-+]+$', local_id))
        is_metacyc_id = has_valid_chars and local_id.isupper()
        if is_metacyc_id:
            return True
        else:
            return False

    @staticmethod
    def is_snomedct_id(local_id: str) -> bool:
        # SNOMED CT IDs are numeric strings
        return local_id.isdigit()

    @staticmethod
    def is_cellosaurus_id(local_id: str) -> bool:
        # Allows: Exactly 4 digits or uppercase letters
        return bool(re.match(r'^[A-Z0-9]{4}$', local_id))

    @staticmethod
    def is_cytoband_id(local_id: str) -> bool:
        # Allows: chromosome (number or X/Y), arm (p/q), band, and optional sub-band e.g., 1p36.33
        return bool(re.match(r'^(\d{1,2}|[XYxy])[pq]\d+(\.\d+)?$', local_id))

    @staticmethod
    def is_mirbase_id(local_id: str) -> bool:
        # Allows: MI or MIMAT followed by exactly 7 digits
        return bool(re.match(r'^(MI|MIMAT)\d{7}$', local_id))

    @staticmethod
    def is_mirdb_id(local_id: str) -> bool:
        # Allows: 3 lowercase letters, an optional miR-, followed by a mix of lowercase letters, digits, and hyphens
        return bool(re.match(r'^[a-z]{3}-(miR-)?[-a-z0-9]+$', local_id))

    @staticmethod
    def is_mondo_id(local_id: str) -> bool:
        # MONDO local IDs (e.g., 0005070 from MONDO:0005070) are 7 digits
        return bool(re.match(r'^[0-9]{7}$', local_id))

    @staticmethod
    def is_uberon_id(local_id: str) -> bool:
        # Allows: digits only (e.g., 0003233 from UBERON:0003233)
        return bool(re.match(r'^[0-9]+$', local_id))

    @staticmethod
    def is_dbsnp_id(local_id: str) -> bool:
        # Allows: rs followed by digits, with an optional version suffix (e.g., .1)
        return bool(re.match(r'^rs[0-9]+(\.\d+)?$', local_id))

    @staticmethod
    def is_ec_id(local_id: str) -> bool:
        # Allows: EC number format (e.g., 3.1.7.2, 1.14.13.M81)
        parts = local_id.split('.')
        if len(parts) < 1 or len(parts) > 4:
            return False

        # Check each part
        for part in parts:
            # Each part must be either:
            # - A number (including 0)
            # - A letter followed by numbers (like M81, B1)
            # - Just a dash (for unspecified sub-subclasses)
            if not re.match(r'^([0-9]+|[A-Z]+[0-9]*|-)$', part):
                return False

        return True

    @staticmethod
    def is_efo_id(local_id: str) -> bool:
        # EFO local IDs (e.g., 0000400 from EFO:0000400) are 7 digits
        return bool(re.match(r'^[0-9]{7}$', local_id))

    @staticmethod
    def is_ensembl_gene_id(local_id: str) -> bool:
        # Allows: ENSG followed by exactly 11 digits
        # Example: ENSG00000138675
        return bool(re.match(r'^ENSG\d{11}$', local_id))

    @staticmethod
    def is_envo_id(local_id: str) -> bool:
        # Allows: one or more digits
        return bool(re.match(r'^\d+$', local_id))

    @staticmethod
    def is_plantfa_id(local_id: str) -> bool:
        # Allows: exactly 5 digits
        # Examples: 10162, 10457
        return bool(re.match(r'^\d{5}$', local_id))

    @staticmethod
    def is_reactome_id(local_id: str) -> bool:
        # Allows: R-HSA-digits (e.g., R-HSA-162582)
        return bool(re.match(r'^R-[A-Z]{3}-[0-9]+$', local_id))

    @staticmethod
    def is_refmet_id(local_id: str) -> bool:
        # Allows: exactly 7 digits
        return bool(re.match(r'^\d{7}$', local_id))

    @staticmethod
    def is_slm_id(local_id: str) -> bool:
        # Allows: a string of one or more digits
        # Examples: 000399049, 00048749
        return bool(re.match(r'^\d+$', local_id))

    @staticmethod
    def is_kegg_reaction_id(local_id: str) -> bool:
        # Allows: R followed by exactly 5 digits
        return bool(re.match(r'^R\d{5}$', local_id))

    @staticmethod
    def is_kegg_drug_id(local_id: str) -> bool:
        # Allows: D followed by exactly 5 digits
        return bool(re.match(r'^D\d{5}$', local_id))

    @staticmethod
    def is_kegg_compound_id(local_id: str) -> bool:
        # Allows: C followed by exactly 5 digits
        return bool(re.match(r'^C\d{5}$', local_id))

    @staticmethod
    def is_pubchem_compound_id(local_id: str) -> bool:
        # Allows: positive integers (PubChem CIDs are numeric)
        return local_id.isdigit() and int(local_id) > 0

    @staticmethod
    def is_smiles_string(local_id: str) -> bool:
        """
        A simple, permissive SMILES validator. It uses a regex to check for
        a valid set of characters and ensures at least one letter is present.
        """
        allowed_chars_pattern = r'^[a-zA-Z0-9\[\]\(\){}=\#\%+\\\/\@\.\-\*:]+$'
        if not re.match(allowed_chars_pattern, local_id):
            return False
        # Ensure there is at least one letter (a SMILES string must represent atoms).
        return any(c.isalpha() for c in local_id)

    @staticmethod
    def is_wikipathways_id(local_id: str) -> bool:
        # Allows: WP followed by digits
        return bool(re.match(r'^WP[0-9]+$', local_id))

    @staticmethod
    def is_vesiclepedia_id(local_id: str) -> bool:
        # Allows: one or more digits
        return bool(re.match(r'^\d+$', local_id))

    @staticmethod
    def is_doid_id(local_id: str) -> bool:
        # Allows: digits only (e.g., 0070557 from DOID:0070557)
        return bool(re.match(r'^[0-9]+$', local_id))

    @staticmethod
    def is_drugbank_id(local_id: str) -> bool:
        # Allows: DB followed by exactly 5 digits
        return bool(re.match(r'^DB\d{5}$', local_id))

    @staticmethod
    def is_ncbigene_id(local_id: str) -> bool:
        # Allows: pure digits (Entrez Gene IDs)
        return bool(re.match(r'^[0-9]+$', local_id))

    @staticmethod
    def is_ncbitaxon_id(local_id: str) -> bool:
        # NCBI Taxonomy IDs are positive integers
        return local_id.isdigit() and int(local_id) > 0

    @staticmethod
    def is_omim_id(local_id: str) -> bool:
        # OMIM IDs are 6-digit numbers
        return local_id.isdigit() and len(local_id) == 6

    @staticmethod
    def is_pfam_id(local_id: str) -> bool:
        # Allows: PF or CL followed by digits
        return bool(re.match(r'^(PF|CL)\d+$', local_id))

    @staticmethod
    def is_pharmvar_id(local_id: str) -> bool:
        # Allows: Gene symbol, asterisk, allele number, and optional sub-allele - e.g., CYP26A1*1.001
        return bool(re.match(r'^[A-Z0-9]+\*\d+(\.\d+)?$', local_id))

    @staticmethod
    def is_umls_cui(local_id: str) -> bool:
        # UMLS CUI: C followed by 7 digits
        return bool(re.match(r'^C\d{7}$', local_id))

    @staticmethod
    def is_umls_mthu_id(local_id: str) -> bool:
        # UMLS MTHU identifiers: MTHU followed by 6 digits
        return bool(re.match(r'^MTHU\d{6}$', local_id))

    def is_umls_id(self, local_id: str) -> bool:
        # UMLS CUI or MTHU identifiers
        return self.is_umls_cui(local_id) or self.is_umls_mthu_id(local_id)

    @staticmethod
    def is_uniprot_protein_id(local_id: str) -> bool:
        # Allows: Base ID (6 or 10 chars) with an optional isoform suffix (e.g., -2)
        # The base ID must still contain at least one letter and one digit.

        # 1. Check the overall format (base ID + optional isoform part)
        if not re.match(r'^([A-Z0-9]{6}|[A-Z0-9]{10})(-\d+)?$', local_id):
            return False

        # 2. Isolate the base ID to check its content
        base_id = local_id.split('-')[0]

        # 3. Ensure the base ID has both letters and digits
        has_letter = any(c.isalpha() for c in base_id)
        has_digit = any(c.isdigit() for c in base_id)

        return has_letter and has_digit

    @staticmethod
    def is_uniprot_feature_id(local_id: str) -> bool:
        # Allows: UniProt ID, hyphen, then PRO_ and digits
        pattern = r'^([A-Z0-9]{6}|[A-Z0-9]{10})-PRO_\d+$'
        return bool(re.match(pattern, local_id))

    def is_uniprot_id(self, local_id: str) -> bool:
        # Allows: Regular uniprot protein IDs or the special 'feature' ids
        return self.is_uniprot_protein_id(local_id) or self.is_uniprot_feature_id(local_id)

    @staticmethod
    def is_inchikey_id(local_id: str) -> bool:
        # Allows: standard InChI key format (e.g., AMOFQIUOTAJRKS-UHFFFAOYSA-N)
        return bool(re.match(r'^([A-Z]{14}|[A-Z]{12})-[A-Z]{10}-[A-Z]$', local_id))

    @staticmethod
    def is_icd10_id(local_id: str) -> bool:
        # ICD-10 codes: letter followed by 2 letters or digits, optional dot and alphanumeric
        return bool(re.match(r'^[A-Z][A-Z0-9]{2}(\.[A-Z0-9]+)?$', local_id))

    @staticmethod
    def is_go_id(local_id: str) -> bool:
        # Allows: exactly 7 digits
        return bool(re.match(r'^\d{7}$', local_id))

    @staticmethod
    def is_hmdb_id(local_id: str) -> bool:
        # Allows: HMDB followed by 5 or 7 digits
        # Examples: HMDB10418, HMDB0046334
        return bool(re.match(r'^HMDB(\d{5}|\d{7})$', local_id))

    @staticmethod
    def is_hps_id(local_id: str) -> bool:
        # Allows: one or more alphabetic characters
        return bool(re.match(r'^[a-zA-Z_]+$', local_id))

    @staticmethod
    def is_icd9_id(local_id: str) -> bool:
        # ICD-9 codes: single code or range (e.g., 344.81 or 317-319.99)
        if '-' in local_id:
            # Range format: XXX-XXX.XX or XXX.XX-XXX.XX
            parts = local_id.split('-')
            if len(parts) != 2:
                return False
            return all(re.match(r'^\d{3}(\.\d{1,2})?$', part) for part in parts)
        else:
            # Single code format: XXX.XX
            return bool(re.match(r'^\d{3}(\.\d{1,2})?$', local_id))

    @staticmethod
    def is_chr_id(local_id: str) -> bool:
        # Allows: lowercase letters, digits, and underscores; requires at least one letter
        has_valid_chars = bool(re.match(r'^[a-z0-9_/-]+$', local_id))
        return has_valid_chars and any(char.isalpha() for char in local_id)

    @staticmethod
    def is_cl_id(local_id: str) -> bool:
        # Allows: digits only (e.g., 0000540 from CL:0000540)
        return bool(re.match(r'^[0-9]+$', local_id))

    @staticmethod
    def is_clo_id(local_id: str) -> bool:
        # Allows: Exactly 7 digits
        return bool(re.match(r'^\d{7}$', local_id))

    @staticmethod
    def is_complexportal_id(local_id: str) -> bool:
        # Allows: CPX- followed by one or more digits
        return bool(re.match(r'^CPX-\d+$', local_id))

    @staticmethod
    def is_ahrq_id(local_id: str) -> bool:
        # Allows: uppercase letters, digits, and underscores
        return bool(re.match(r'^[A-Z0-9_]+$', local_id))

    @staticmethod
    def is_bfo_id(local_id: str) -> bool:
        # Allows: one or more digits
        return bool(re.match(r'^\d+$', local_id))

    @staticmethod
    def is_bvbrc_id(local_id: str) -> bool:
        # Allows: digits, a period, and more digits
        return bool(re.match(r'^\d+\.\d+$', local_id))

    @staticmethod
    def is_cas_id(local_id: str) -> bool:
        # Allows: 2-7 digits, hyphen, 2 digits, hyphen, 1 digit
        # Examples: 2906-39-0, 124-20-9, 54-16-0
        return bool(re.match(r'^\d{2,7}-\d{2}-\d$', local_id))

    @staticmethod
    def is_cdcsvi_id(local_id: str) -> bool:
        # Allows: one or more uppercase alphabetic characters
        return bool(re.match(r'^[A-Z]+$', local_id))

    @staticmethod
    def is_chebi_id(local_id: str) -> bool:
        # ChEBI IDs are positive integers
        return local_id.isdigit() and int(local_id) > 0

    @staticmethod
    def is_chembl_compound_id(local_id: str) -> bool:
        # Allows: CHEMBL followed by digits
        return bool(re.match(r'^CHEMBL\d+$', local_id))

    @staticmethod
    def is_chembl_target_id(local_id: str) -> bool:
        # Allows: CHEMBL followed by one or more digits
        return bool(re.match(r'^CHEMBL\d+$', local_id))

    @staticmethod
    def is_hpo_id(local_id: str) -> bool:
        # Allows: digits only (e.g., 0001234 from HP:0001234)
        return bool(re.match(r'^[0-9]+$', local_id))

    @staticmethod
    def is_uszipcode_id(local_id: str) -> bool:
        # Allows: 5-digit US ZIP codes
        return bool(re.match(r'^[0-9]{5}$', local_id)) or local_id == 'US'

    @staticmethod
    def is_fips_compound_id(local_id: str) -> bool:
        # Allows: 6, 7, 11, or 12 digit FIPS-like codes
        return bool(re.match(r'^(\d{6}|\d{7}|\d{11}|\d{12})$', local_id))

    @staticmethod
    def is_fips_state_id(local_id: str) -> bool:
        # Allows: exactly 2 digits
        return bool(re.match(r'^\d{2}$', local_id))

    @staticmethod
    def is_geonames_id(local_id: str) -> bool:
        # Allows: 2-letter country code OR 2 letters followed by one or more dot-separated alphanumeric segments
        return bool(re.match(r'^[A-Z]{2}$', local_id)) or bool(re.match(r'^[A-Z]{2}(\.[A-Z0-9]+)+$', local_id))

    @staticmethod
    def is_ndfrt_id(local_id: str) -> bool:
        # Allows: N followed by exactly 10 digits
        return bool(re.match(r'^N\d{10}$', local_id))

    @staticmethod
    def is_nhanes_id(local_id: str) -> bool:
        # Allows: one or more digits
        return bool(re.match(r'^\d+$', local_id))

    @staticmethod
    def is_sider_id(local_id: str) -> bool:
        # Allows: SIDER identifiers (typically alphanumeric with possible special chars)
        return bool(re.match(r'^[A-Z0-9._-]+$', local_id))