"""Validation functions for biological identifier vocabularies."""

import re


def is_loinc_id(local_id: str) -> bool:
    """LOINC codes: digits followed by dash and check digit (e.g., 27858-0)
    or LP codes: LP followed by digits and dash-digit (e.g., LP32606-3)"""
    return bool(re.match(r"^(LP)?\d+-\d$", local_id))


def is_lipidbank_id(local_id: str) -> bool:
    """Allows: 3 uppercase letters followed by exactly 4 digits
    Examples: XPR4101, DFA8145"""
    return bool(re.match(r"^[A-Z]{3}\d{4}$", local_id))


def is_lipidmaps_id(local_id: str) -> bool:
    """Allows: 2 uppercase letters followed by a mix of uppercase letters and digits
    Examples: ST02030282, PR0103110003, SP0501AA01"""
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9]+$", local_id))


def is_mesh_id(local_id: str) -> bool:
    """Allows: D, C, or M followed by one or more digits"""
    return bool(re.match(r"^[DCM]\d+$", local_id))


def is_metacyc_ec_id(local_id: str) -> bool:
    """Allows three digits groups, then a final group of alphanumeric characters"""
    return bool(re.match(r"^\d+\.\d+\.\d+\.[a-zA-Z0-9]+$", local_id))


def is_metacyc_reaction_id(local_id: str) -> bool:
    """Allows: Hyphen-separated uppercase/numeric or capitalized alpha parts; must contain 'RXN' somewhere
    e.g., 3.2.1.68-RXN, TRANS-RXN0-593, CYPRIDINA-LUCIFERIN-2-MONOOXYGENASE-RXN, RXN0-5258-Yeast"""
    has_valid_chars = bool(re.match(r"^[A-Za-z0-9-.+]+$", local_id))
    parts = local_id.split("-")
    has_proper_capitalization = all(
        part.isupper() or (not any(char.isalpha() for char in part)) or (part.isalpha()) for part in parts
    )
    return has_valid_chars and has_proper_capitalization and "RXN" in local_id


def is_metacyc_pathway_id(local_id: str) -> bool:
    """MetaCyc pathway IDs: examples: PWY-#### or PWY0-#### or DESCRIPTIVE-NAME-PWY or PWY18C3-9"""
    has_valid_chars = bool(re.match(r"^[A-Z0-9-+]+$", local_id))
    is_metacyc_id = has_valid_chars and local_id.isupper()
    if is_metacyc_id:
        return True
    else:
        return False


def is_snomedct_id(local_id: str) -> bool:
    """SNOMED CT IDs are numeric strings"""
    return local_id.isdigit()


def is_cellosaurus_id(local_id: str) -> bool:
    """Allows: Exactly 4 digits or uppercase letters"""
    return bool(re.match(r"^[A-Z0-9]{4}$", local_id))


def is_cytoband_id(local_id: str) -> bool:
    """Allows: chromosome (number or X/Y), arm (p/q), band, and optional sub-band e.g., 1p36.33"""
    return bool(re.match(r"^(\d{1,2}|[XYxy])[pq]\d+(\.\d+)?$", local_id))


def is_mirbase_id(local_id: str) -> bool:
    """Allows: MI or MIMAT followed by exactly 7 digits"""
    return bool(re.match(r"^(MI|MIMAT)\d{7}$", local_id))


def is_mirdb_id(local_id: str) -> bool:
    """Allows: 3 lowercase letters, an optional miR-, followed by a mix of lowercase letters, digits, and hyphens"""
    return bool(re.match(r"^[a-z]{3}-(miR-)?[-a-z0-9]+$", local_id))


def is_mondo_id(local_id: str) -> bool:
    """MONDO local IDs (e.g., 0005070 from MONDO:0005070) are 7 digits"""
    return bool(re.match(r"^[0-9]{7}$", local_id))


def is_uberon_id(local_id: str) -> bool:
    """Allows: digits only (e.g., 0003233 from UBERON:0003233)"""
    return bool(re.match(r"^[0-9]+$", local_id))


def is_dbsnp_id(local_id: str) -> bool:
    """Allows: rs followed by digits, with an optional version suffix (e.g., .1)"""
    return bool(re.match(r"^rs[0-9]+(\.\d+)?$", local_id))


def is_ec_id(local_id: str) -> bool:
    """Allows: EC number format (e.g., 3.1.7.2, 1.14.13.M81)"""
    parts = local_id.split(".")
    if len(parts) < 1 or len(parts) > 4:
        return False

    # Check each part
    for part in parts:
        # Each part must be either:
        # - A number (including 0)
        # - A letter followed by numbers (like M81, B1)
        # - Just a dash (for unspecified sub-subclasses)
        if not re.match(r"^([0-9]+|[A-Z]+[0-9]*|-)$", part):
            return False

    return True


def is_efo_id(local_id: str) -> bool:
    """EFO local IDs (e.g., 0000400 from EFO:0000400) are 7 digits"""
    return bool(re.match(r"^[0-9]{7}$", local_id))


def is_ensembl_gene_id(local_id: str) -> bool:
    """Allows: ENSG followed by exactly 11 digits
    Example: ENSG00000138675"""
    return bool(re.match(r"^ENSG\d{11}$", local_id))


def is_envo_id(local_id: str) -> bool:
    """Allows: one or more digits"""
    return bool(re.match(r"^\d+$", local_id))


def is_plantfa_id(local_id: str) -> bool:
    """Allows: exactly 5 digits
    Examples: 10162, 10457"""
    return bool(re.match(r"^\d{5}$", local_id))


def is_reactome_id(local_id: str) -> bool:
    """Allows: R-HSA-digits (e.g., R-HSA-162582)"""
    return bool(re.match(r"^R-[A-Z]{3}-[0-9]+$", local_id))


def is_refmet_id(local_id: str) -> bool:
    """Allows: exactly 7 digits"""
    return bool(re.match(r"^\d{7}$", local_id))


def is_slm_id(local_id: str) -> bool:
    """Allows: a string of one or more digits
    Examples: 000399049, 00048749"""
    return bool(re.match(r"^\d+$", local_id))


def is_kegg_reaction_id(local_id: str) -> bool:
    """Allows: R followed by exactly 5 digits"""
    return bool(re.match(r"^R\d{5}$", local_id))


def is_kegg_drug_id(local_id: str) -> bool:
    """Allows: D followed by exactly 5 digits"""
    return bool(re.match(r"^D\d{5}$", local_id))


def is_kegg_compound_id(local_id: str) -> bool:
    """Allows: C followed by exactly 5 digits"""
    return bool(re.match(r"^C\d{5}$", local_id))


def is_pubchem_compound_id(local_id: str) -> bool:
    """Allows: positive integers (PubChem CIDs are numeric)"""
    return local_id.isdigit() and int(local_id) > 0


def is_smiles_string(local_id: str) -> bool:
    """A simple, permissive SMILES validator. It uses a regex to check for
    a valid set of characters and ensures at least one letter is present."""
    allowed_chars_pattern = r"^[a-zA-Z0-9\[\]\(\){}=\#\%+\\\/\@\.\-\*:]+$"
    if not re.match(allowed_chars_pattern, local_id):
        return False
    # Ensure there is at least one letter (a SMILES string must represent atoms).
    return any(c.isalpha() for c in local_id)


def is_wikipathways_id(local_id: str) -> bool:
    """Allows: WP followed by digits"""
    return bool(re.match(r"^WP[0-9]+$", local_id))


def is_vesiclepedia_id(local_id: str) -> bool:
    """Allows: one or more digits"""
    return bool(re.match(r"^\d+$", local_id))


def is_doid_id(local_id: str) -> bool:
    """Allows: digits only (e.g., 0070557 from DOID:0070557)"""
    return bool(re.match(r"^[0-9]+$", local_id))


def is_drugbank_id(local_id: str) -> bool:
    """Allows: DB followed by exactly 5 digits"""
    return bool(re.match(r"^DB\d{5}$", local_id))


def is_ncbigene_id(local_id: str) -> bool:
    """Allows: pure digits (Entrez Gene IDs)"""
    return bool(re.match(r"^[0-9]+$", local_id))


def is_ncbitaxon_id(local_id: str) -> bool:
    """NCBI Taxonomy IDs are positive integers"""
    return local_id.isdigit() and int(local_id) > 0


def is_ncit_id(local_id: str) -> bool:
    """NCIT IDs (C-codes) consist of the letter 'C' followed by digits"""
    return bool(re.match(r"^C\d+$", local_id))


def is_umls_cui(local_id: str) -> bool:
    """UMLS CUI: C followed by 7 digits"""
    return bool(re.match(r"^C\d{7}$", local_id))


def is_umls_mthu_id(local_id: str) -> bool:
    """UMLS MTHU identifiers: MTHU followed by 6 digits"""
    return bool(re.match(r"^MTHU\d{6}$", local_id))


def is_umls_id(local_id: str) -> bool:
    """UMLS CUI or MTHU identifiers"""
    return is_umls_cui(local_id) or is_umls_mthu_id(local_id)


def is_omim_id(local_id: str) -> bool:
    """Allows canonical 6-digit IDs OR MTHU-prefixed IDs (from UMLS/MeSH, but in OMIM, like: OMIM:MTHU067886)"""
    return (local_id.isdigit() and len(local_id) == 6) or is_umls_mthu_id(local_id)


def is_pfam_id(local_id: str) -> bool:
    """Allows: PF or CL followed by digits"""
    return bool(re.match(r"^(PF|CL)\d+$", local_id))


def is_pharmvar_id(local_id: str) -> bool:
    """Allows: Gene symbol, asterisk, allele number, and optional sub-allele - e.g., CYP26A1*1.001"""
    return bool(re.match(r"^[A-Z0-9]+\*\d+(\.\d+)?$", local_id))


def is_uniprot_protein_id(local_id: str) -> bool:
    """Allows: Base ID (6 or 10 chars) with an optional isoform suffix (e.g., -2)
    The base ID must still contain at least one letter and one digit."""
    # 1. Check the overall format (base ID + optional isoform part)
    if not re.match(r"^([A-Z0-9]{6}|[A-Z0-9]{10})(-\d+)?$", local_id):
        return False

    # 2. Isolate the base ID to check its content
    base_id = local_id.split("-")[0]

    # 3. Ensure the base ID has both letters and digits
    has_letter = any(c.isalpha() for c in base_id)
    has_digit = any(c.isdigit() for c in base_id)

    return has_letter and has_digit


def is_uniprot_feature_id(local_id: str) -> bool:
    """Allows: UniProt ID, hyphen, then PRO_ and digits"""
    pattern = r"^([A-Z0-9]{6}|[A-Z0-9]{10})-PRO_\d+$"
    return bool(re.match(pattern, local_id))


def is_uniprot_id(local_id: str) -> bool:
    """Allows: Regular uniprot protein IDs or the special 'feature' ids"""
    return is_uniprot_protein_id(local_id) or is_uniprot_feature_id(local_id)


def is_inchikey_id(local_id: str) -> bool:
    """Allows: standard InChI key format (e.g., AMOFQIUOTAJRKS-UHFFFAOYSA-N)"""
    return bool(re.match(r"^([A-Z]{14}|[A-Z]{12})-[A-Z]{10}-[A-Z]$", local_id))


def is_icd10_id(local_id: str) -> bool:
    """ICD-10 codes: letter followed by 2 letters or digits, optional dot and alphanumeric"""
    return bool(re.match(r"^[A-Z][A-Z0-9]{2}(\.[A-Z0-9]+)?$", local_id))


def is_go_id(local_id: str) -> bool:
    """Allows: exactly 7 digits"""
    return bool(re.match(r"^\d{7}$", local_id))


def is_hmdb_id(local_id: str) -> bool:
    """Allows: HMDB followed by 5 or 7 digits
    Examples: HMDB10418, HMDB0046334"""
    return bool(re.match(r"^HMDB(\d{5}|\d{7})$", local_id))


def is_hps_id(local_id: str) -> bool:
    """Allows: one or more alphabetic characters"""
    return bool(re.match(r"^[a-zA-Z_]+$", local_id))


def is_icd9_id(local_id: str) -> bool:
    """ICD-9 codes: single code or range (e.g., 344.81 or 317-319.99)"""
    if "-" in local_id:
        # Range format: XXX-XXX.XX or XXX.XX-XXX.XX
        parts = local_id.split("-")
        if len(parts) != 2:
            return False
        return all(re.match(r"^\d{3}(\.\d{1,2})?$", part) for part in parts)
    else:
        # Single code format: XXX.XX
        return bool(re.match(r"^\d{3}(\.\d{1,2})?$", local_id))


def is_chr_id(local_id: str) -> bool:
    """Allows: lowercase letters, digits, and underscores; requires at least one letter"""
    has_valid_chars = bool(re.match(r"^[a-z0-9_/-]+$", local_id))
    return has_valid_chars and any(char.isalpha() for char in local_id)


def is_cl_id(local_id: str) -> bool:
    """Allows: digits only (e.g., 0000540 from CL:0000540)"""
    return bool(re.match(r"^[0-9]+$", local_id))


def is_clo_id(local_id: str) -> bool:
    """Allows: Exactly 7 digits"""
    return bool(re.match(r"^\d{7}$", local_id))


def is_complexportal_id(local_id: str) -> bool:
    """Allows: CPX- followed by one or more digits"""
    return bool(re.match(r"^CPX-\d+$", local_id))


def is_ahrq_id(local_id: str) -> bool:
    """Allows: uppercase letters, digits, and underscores"""
    return bool(re.match(r"^[A-Z0-9_]+$", local_id))


def is_bfo_id(local_id: str) -> bool:
    """Allows: one or more digits"""
    return bool(re.match(r"^\d+$", local_id))


def is_bvbrc_id(local_id: str) -> bool:
    """Allows: digits, a period, and more digits"""
    return bool(re.match(r"^\d+\.\d+$", local_id))


def is_cas_id(local_id: str) -> bool:
    """Allows: 2-7 digits, hyphen, 2 digits, hyphen, 1 digit
    Examples: 2906-39-0, 124-20-9, 54-16-0"""
    return bool(re.match(r"^\d{2,7}-\d{2}-\d$", local_id))


def is_cdcsvi_id(local_id: str) -> bool:
    """Allows: one or more uppercase alphabetic characters"""
    return bool(re.match(r"^[A-Z]+$", local_id))


def is_chebi_id(local_id: str) -> bool:
    """ChEBI IDs are positive integers"""
    return local_id.isdigit() and int(local_id) > 0


def is_chembl_compound_id(local_id: str) -> bool:
    """Allows: CHEMBL followed by digits"""
    return bool(re.match(r"^CHEMBL\d+$", local_id))


def is_chembl_target_id(local_id: str) -> bool:
    """Allows: CHEMBL followed by one or more digits"""
    return bool(re.match(r"^CHEMBL\d+$", local_id))


def is_hpo_id(local_id: str) -> bool:
    """Allows: digits only (e.g., 0001234 from HP:0001234)"""
    return bool(re.match(r"^[0-9]+$", local_id))


def is_uszipcode_id(local_id: str) -> bool:
    """Allows: 5-digit US ZIP codes"""
    return bool(re.match(r"^[0-9]{5}$", local_id)) or local_id == "US"


def is_fips_compound_id(local_id: str) -> bool:
    """Allows: 6, 7, 11, or 12 digit FIPS-like codes"""
    return bool(re.match(r"^(\d{6}|\d{7}|\d{11}|\d{12})$", local_id))


def is_fips_state_id(local_id: str) -> bool:
    """Allows: exactly 2 digits"""
    return bool(re.match(r"^\d{2}$", local_id))


def is_geonames_id(local_id: str) -> bool:
    """Allows: 2-letter country code OR 2 letters followed by one or more dot-separated alphanumeric segments"""
    return bool(re.match(r"^[A-Z]{2}$", local_id)) or bool(re.match(r"^[A-Z]{2}(\.[A-Z0-9]+)+$", local_id))


def is_ndfrt_id(local_id: str) -> bool:
    """Allows: N followed by exactly 10 digits"""
    return bool(re.match(r"^N\d{10}$", local_id))


def is_nhanes_id(local_id: str) -> bool:
    """Allows: one or more digits"""
    return bool(re.match(r"^\d+$", local_id))


def is_sider_id(local_id: str) -> bool:
    """Allows: SIDER identifiers (typically alphanumeric with possible special chars)"""
    return bool(re.match(r"^[A-Z0-9._-]+$", local_id))
