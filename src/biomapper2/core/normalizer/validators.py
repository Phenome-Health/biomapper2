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


# =============================================================================
# KRAKEN Vocab Validators (Issue #12)
# =============================================================================


def is_numeric_id(local_id: str) -> bool:
    """Generic validator for pure numeric identifiers (positive integers).
    Used by: HGNC, MGI, RGD, RxNorm, RXCUI, DrugCentral, RHEA, orphanet, FMA, etc."""
    return local_id.isdigit() and int(local_id) > 0


def is_seven_digit_id(local_id: str) -> bool:
    """Allows: exactly 7 digits (zero-padded).
    Used by: HP, PATO, SO, NBO, OBI, UO, AEO, BSPO, FAO, DDANAT, GENEPIO, MAXO, etc."""
    return bool(re.match(r"^\d{7}$", local_id))


# --- Tier 1: Core Metabolomics/Proteomics/Drugs ---


def is_atc_id(local_id: str) -> bool:
    """ATC drug classification codes: letter, 2 digits, 2 letters, 2 digits.
    Examples: N02AX05, C09DB06, G04BE09"""
    return bool(re.match(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$", local_id))


def is_unii_id(local_id: str) -> bool:
    """FDA UNII identifiers: exactly 10 alphanumeric characters.
    Examples: 4XQ51KS2JU, 99R7V50C6Y"""
    return bool(re.match(r"^[A-Z0-9]{10}$", local_id))


def is_omim_ps_id(local_id: str) -> bool:
    """OMIM Phenotype Series IDs: exactly 6 digits.
    Examples: 220150, 145600"""
    return bool(re.match(r"^\d{6}$", local_id))


def is_pr_id(local_id: str) -> bool:
    """Protein Ontology IDs: UniProt-style (6 alphanumeric) or 9-digit numeric.
    Examples: Q9BY49, P12345, 000007707"""
    # UniProt-style: 6 alphanumeric with at least one letter and one digit
    if re.match(r"^[A-Z0-9]{6}$", local_id):
        has_letter = any(c.isalpha() for c in local_id)
        has_digit = any(c.isdigit() for c in local_id)
        return has_letter and has_digit
    # 9-digit numeric
    return bool(re.match(r"^\d{9}$", local_id))


def is_smpdb_id(local_id: str) -> bool:
    """SMPDB pathway IDs: SMP followed by 7 digits.
    Examples: SMP0032202, SMP0086506"""
    return bool(re.match(r"^SMP\d{7}$", local_id))


def is_kegg_glycan_id(local_id: str) -> bool:
    """KEGG glycan IDs: G followed by 5 digits.
    Examples: G04638, G02524"""
    return bool(re.match(r"^G\d{5}$", local_id))


# --- Tier 2: Anatomy/Phenotype Ontologies ---


def is_fbbt_id(local_id: str) -> bool:
    """FlyBase anatomy IDs: exactly 8 digits.
    Examples: 00001059, 00050048"""
    return bool(re.match(r"^\d{8}$", local_id))


def is_zfa_id(local_id: str) -> bool:
    """Zebrafish anatomy IDs: exactly 7 digits.
    Examples: 0001617, 0000110"""
    return bool(re.match(r"^\d{7}$", local_id))


def is_mod_id(local_id: str) -> bool:
    """Protein modification ontology IDs: exactly 5 digits.
    Examples: 01160, 00046"""
    return bool(re.match(r"^\d{5}$", local_id))


def is_mi_id(local_id: str) -> bool:
    """Molecular interactions ontology IDs: 4 digits.
    Examples: 2133, 0001"""
    return bool(re.match(r"^\d{4}$", local_id))


# --- Tier 3: Specialized/Medical ---


def is_pathwhiz_id(local_id: str) -> bool:
    """PathWhiz pathway IDs: PW followed by 6 digits.
    Examples: PW050892, PW056905"""
    return bool(re.match(r"^PW\d{6}$", local_id))


def is_meddra_id(local_id: str) -> bool:
    """MedDRA IDs: exactly 8 digits.
    Examples: 10011730, 10000001"""
    return bool(re.match(r"^\d{8}$", local_id))


def is_icd10pcs_id(local_id: str) -> bool:
    """ICD-10 Procedure Coding System IDs: 7 alphanumeric characters.
    Examples: 0LPY4JZ, 02100Z9"""
    return bool(re.match(r"^[A-Z0-9]{7}$", local_id))


def is_hcpcs_id(local_id: str) -> bool:
    """Healthcare Common Procedure Coding System IDs: letter followed by 4 digits.
    Examples: A9551, J0171"""
    return bool(re.match(r"^[A-Z]\d{4}$", local_id))


def is_vandf_id(local_id: str) -> bool:
    """VA National Drug File IDs: numeric.
    Examples: 4040230"""
    return local_id.isdigit() and int(local_id) > 0


def is_gtopdb_id(local_id: str) -> bool:
    """Guide to Pharmacology IDs: numeric.
    Examples: 7484"""
    return local_id.isdigit() and int(local_id) > 0


def is_pdq_id(local_id: str) -> bool:
    """NCI PDQ IDs: CDR followed by 10 digits.
    Examples: CDR0000770458"""
    return bool(re.match(r"^CDR\d{10}$", local_id))


def is_chv_id(local_id: str) -> bool:
    """Consumer Health Vocabulary IDs: exactly 10 digits.
    Examples: 0000006350"""
    return bool(re.match(r"^\d{10}$", local_id))


def is_foodon_id(local_id: str) -> bool:
    """Food Ontology IDs: exactly 8 digits.
    Examples: 03541961"""
    return bool(re.match(r"^\d{8}$", local_id))


def is_ttd_target_id(local_id: str) -> bool:
    """Therapeutic Target Database IDs: alphanumeric with optional hyphens.
    Examples: CY-1503, T12345"""
    return bool(re.match(r"^[A-Za-z0-9_-]+$", local_id))


def is_kegg_pathway_id(local_id: str) -> bool:
    """KEGG pathway IDs: 5 digits (general pathways) or hsa/mmu + 5 digits.
    Examples: 04966, hsa04110"""
    return bool(re.match(r"^([a-z]{3})?\d{5}$", local_id))


# --- Tier 4: Model Organism Databases ---


def is_flybase_id(local_id: str) -> bool:
    """FlyBase IDs: FB prefix + type code (gn/tr/pp/cl/ab/ba/rf) + digits.
    Examples: FBgn0019985, FBtr0073412, FBpp0080851"""
    return bool(re.match(r"^FB(gn|tr|pp|cl|ab|ba|rf)\d+$", local_id))


def is_wormbase_gene_id(local_id: str) -> bool:
    """WormBase gene IDs: WBGene followed by 8 digits.
    Examples: WBGene00012992, WBGene00010912"""
    return bool(re.match(r"^WBGene\d{8}$", local_id))


def is_zfin_id(local_id: str) -> bool:
    """ZFIN zebrafish IDs: ZDB-TYPE-digits-digits.
    Examples: ZDB-GENE-130109-1, ZDB-GENE-041014-10"""
    return bool(re.match(r"^ZDB-[A-Z]+-\d+-\d+$", local_id))


def is_sgd_id(local_id: str) -> bool:
    """SGD yeast IDs: S followed by 9 digits.
    Examples: S000004291, S000004559"""
    return bool(re.match(r"^S\d{9}$", local_id))


def is_pombase_id(local_id: str) -> bool:
    """PomBase fission yeast IDs: SP + alphanumeric + dot + digits + optional 'c'.
    Examples: SPAC6F12.09, SPAP8A3.14c, SPBC1289.02"""
    return bool(re.match(r"^SP[A-Z0-9]+\.\d+c?$", local_id))


def is_dictybase_id(local_id: str) -> bool:
    """DictyBase IDs: DDB_G followed by 7 digits.
    Examples: DDB_G0293130"""
    return bool(re.match(r"^DDB_G\d{7}$", local_id))


def is_dictybase_gene_id(local_id: str) -> bool:
    """DictyBase gene IDs (short form): G followed by 7 digits.
    Examples: G0281589"""
    return bool(re.match(r"^G\d{7}$", local_id))


def is_araport_id(local_id: str) -> bool:
    """Arabidopsis (AraPort) IDs: AT + chromosome (1-5, M, C) + G + 5 digits.
    Examples: AT1G27500, AT5G10140"""
    return bool(re.match(r"^AT[1-5MC]G\d{5}$", local_id))


def is_ecogene_id(local_id: str) -> bool:
    """E. coli EcoGene IDs: EG followed by digits.
    Examples: EG12315"""
    return bool(re.match(r"^EG\d+$", local_id))


def is_ensemblgenomes_id(local_id: str) -> bool:
    """Ensembl Genomes IDs: uppercase letters followed by digits.
    Examples: BMEI0545"""
    return bool(re.match(r"^[A-Z]+\d+$", local_id))


# --- Edge Cases: Final 4 Vocabulary Validators ---


def is_kegg_generic_id(local_id: str) -> bool:
    """Generic KEGG IDs: 5 digits (pathways) OR letter + 5 digits (compounds/drugs).
    Examples: 04966, 04024, 00590, C00031, D00001
    Note: This is flexible to handle both KRAKEN pathway IDs and user-provided compound IDs."""
    return bool(re.match(r"^(\d{5}|[A-Z]\d{5})$", local_id))


def is_chembl_mechanism_id(local_id: str) -> bool:
    """CHEMBL mechanism IDs: lowercase alphanumeric with underscores.
    Examples: mitochondrial_complex_i_(nadh_dehydrogenase)_inhibitor"""
    # Allow lowercase letters, digits, underscores, hyphens, and parentheses
    return bool(re.match(r"^[a-z0-9_(),-]+$", local_id))


def is_oba_id(local_id: str) -> bool:
    """Ontology for Biomedical Annotations IDs: 7 digits.
    Examples: 2044301, 2053738, 2042686"""
    return bool(re.match(r"^\d{7}$", local_id))


def is_obo_id(local_id: str) -> bool:
    """Open Biological Ontology cross-references: variable patterns.
    Examples: APOLLO_SV_00000031, INO_0000018, EnsemblBacteria#_SAOUHSC_02706"""
    # Allow uppercase letters, digits, underscores, hashes, and colons
    return bool(re.match(r"^[A-Za-z0-9_#:]+$", local_id))
