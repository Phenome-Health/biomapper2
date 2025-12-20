"""Unit tests for KRAKEN vocab validators (Issue #12).

Tests cover representative validators from each tier to ensure KRAKEN
vocabulary prefixes can be properly validated by the Normalizer.
"""

import pytest

from biomapper2.core.normalizer import validators


class TestKrakenValidators:
    """Tests for KRAKEN vocabulary validators."""

    # =========================================================================
    # Shared Validators
    # =========================================================================

    def test_numeric_id_validators(self):
        """Test validators for pure numeric IDs (HGNC, RxNorm, etc.)."""
        # Valid numeric IDs
        assert validators.is_numeric_id("25111")  # HGNC example
        assert validators.is_numeric_id("1550048")  # RxNorm example
        assert validators.is_numeric_id("1")  # Minimum valid

        # Invalid numeric IDs
        assert not validators.is_numeric_id("0")  # Zero not allowed
        assert not validators.is_numeric_id("")  # Empty
        assert not validators.is_numeric_id("HGNC:25111")  # With prefix
        assert not validators.is_numeric_id("abc")  # Letters

    # =========================================================================
    # Tier 1: Core Metabolomics/Proteomics/Drugs
    # =========================================================================

    def test_atc_drug_classification(self):
        """Test ATC drug classification code validator."""
        # Valid ATC codes
        assert validators.is_atc_id("N02AX05")  # Tramadol
        assert validators.is_atc_id("C09DB06")  # Losartan combination
        assert validators.is_atc_id("G04BE09")  # Vardenafil

        # Invalid ATC codes
        assert not validators.is_atc_id("N02AX0")  # Too short
        assert not validators.is_atc_id("N02AX050")  # Too long
        assert not validators.is_atc_id("102AX05")  # Starts with digit
        assert not validators.is_atc_id("n02ax05")  # Lowercase

    def test_unii_fda_identifiers(self):
        """Test FDA UNII identifier validator."""
        # Valid UNII IDs
        assert validators.is_unii_id("4XQ51KS2JU")  # Example UNII
        assert validators.is_unii_id("99R7V50C6Y")  # Another example

        # Invalid UNII IDs
        assert not validators.is_unii_id("4XQ51KS2J")  # 9 chars (too short)
        assert not validators.is_unii_id("4XQ51KS2JUX")  # 11 chars (too long)
        assert not validators.is_unii_id("4xq51ks2ju")  # Lowercase

    # =========================================================================
    # Tier 2: Anatomy/Phenotype Ontologies (7-digit shared pattern)
    # =========================================================================

    def test_seven_digit_ontology_ids(self):
        """Test 7-digit ontology ID validators (HP, PATO, SO, etc.)."""
        # Valid 7-digit IDs
        assert validators.is_seven_digit_id("0410133")  # HP example
        assert validators.is_seven_digit_id("0040056")  # PATO example
        assert validators.is_seven_digit_id("0000001")  # Minimum

        # Invalid 7-digit IDs
        assert not validators.is_seven_digit_id("410133")  # 6 digits
        assert not validators.is_seven_digit_id("04101330")  # 8 digits
        assert not validators.is_seven_digit_id("041013A")  # Contains letter

    # =========================================================================
    # Tier 3: Specialized/Medical Vocabularies
    # =========================================================================

    def test_pathwhiz_pathway_ids(self):
        """Test PathWhiz pathway ID validator."""
        # Valid PathWhiz IDs
        assert validators.is_pathwhiz_id("PW050892")
        assert validators.is_pathwhiz_id("PW056905")
        assert validators.is_pathwhiz_id("PW000001")

        # Invalid PathWhiz IDs
        assert not validators.is_pathwhiz_id("050892")  # Missing prefix
        assert not validators.is_pathwhiz_id("PW05089")  # Only 5 digits
        assert not validators.is_pathwhiz_id("PW0508920")  # 7 digits

    def test_smpdb_pathway_ids(self):
        """Test SMPDB pathway ID validator."""
        # Valid SMPDB IDs
        assert validators.is_smpdb_id("SMP0032202")
        assert validators.is_smpdb_id("SMP0086506")
        assert validators.is_smpdb_id("SMP0000001")

        # Invalid SMPDB IDs
        assert not validators.is_smpdb_id("0032202")  # Missing prefix
        assert not validators.is_smpdb_id("SMP003220")  # Only 6 digits
        assert not validators.is_smpdb_id("SMP00322020")  # 8 digits

    def test_invalid_ids_rejected(self):
        """Test that various invalid ID formats are rejected."""
        # Empty strings
        assert not validators.is_atc_id("")
        assert not validators.is_unii_id("")
        assert not validators.is_smpdb_id("")

        # Whitespace
        assert not validators.is_atc_id(" N02AX05")
        assert not validators.is_atc_id("N02AX05 ")

        # Special characters where not allowed
        assert not validators.is_numeric_id("-123")
        assert not validators.is_numeric_id("123.45")

    def test_tier1_additional_validators(self):
        """Test additional Tier 1 validators: KEGG.GLYCAN, OMIM.PS, PR."""
        # KEGG Glycan: G followed by 5 digits
        assert validators.is_kegg_glycan_id("G04638")
        assert validators.is_kegg_glycan_id("G02524")
        assert not validators.is_kegg_glycan_id("G0463")  # Too short
        assert not validators.is_kegg_glycan_id("C04638")  # Wrong prefix

        # OMIM Phenotype Series: exactly 6 digits
        assert validators.is_omim_ps_id("220150")
        assert validators.is_omim_ps_id("145600")
        assert not validators.is_omim_ps_id("22015")  # 5 digits
        assert not validators.is_omim_ps_id("2201500")  # 7 digits

        # Protein Ontology: UniProt-style (6 chars) or 9-digit numeric
        assert validators.is_pr_id("Q9BY49")  # UniProt-style
        assert validators.is_pr_id("P12345")  # UniProt-style
        assert validators.is_pr_id("000007707")  # 9-digit
        assert not validators.is_pr_id("123456")  # All digits, 6 chars (no letter)
        assert not validators.is_pr_id("ABCDEF")  # All letters, no digit

    def test_tier2_anatomy_ontologies(self):
        """Test Tier 2 anatomy/phenotype ontology validators."""
        # FBbt (FlyBase anatomy): exactly 8 digits
        assert validators.is_fbbt_id("00001059")
        assert validators.is_fbbt_id("00050048")
        assert not validators.is_fbbt_id("0001059")  # 7 digits
        assert not validators.is_fbbt_id("000010590")  # 9 digits

        # ZFA (Zebrafish anatomy): exactly 7 digits
        assert validators.is_zfa_id("0001617")
        assert validators.is_zfa_id("0000110")
        assert not validators.is_zfa_id("001617")  # 6 digits

        # MOD (Protein modification): exactly 5 digits
        assert validators.is_mod_id("01160")
        assert validators.is_mod_id("00046")
        assert not validators.is_mod_id("1160")  # 4 digits
        assert not validators.is_mod_id("011600")  # 6 digits

        # MI (Molecular interactions): 4 digits
        assert validators.is_mi_id("2133")
        assert validators.is_mi_id("0001")
        assert not validators.is_mi_id("133")  # 3 digits

    def test_tier3_medical_vocabs(self):
        """Test Tier 3 medical vocabulary validators."""
        # MedDRA: exactly 8 digits
        assert validators.is_meddra_id("10011730")
        assert validators.is_meddra_id("10000001")
        assert not validators.is_meddra_id("1001173")  # 7 digits

        # ICD-10 PCS: 7 alphanumeric characters
        assert validators.is_icd10pcs_id("0LPY4JZ")
        assert validators.is_icd10pcs_id("02100Z9")
        assert not validators.is_icd10pcs_id("0LPY4J")  # 6 chars

        # HCPCS: letter followed by 4 digits
        assert validators.is_hcpcs_id("A9551")
        assert validators.is_hcpcs_id("J0171")
        assert not validators.is_hcpcs_id("9551A")  # Wrong order
        assert not validators.is_hcpcs_id("A955")  # Too short

        # PDQ: CDR followed by 10 digits
        assert validators.is_pdq_id("CDR0000770458")
        assert not validators.is_pdq_id("CDR000077045")  # 9 digits

        # CHV: exactly 10 digits
        assert validators.is_chv_id("0000006350")
        assert not validators.is_chv_id("000000635")  # 9 digits

        # FOODON: exactly 8 digits
        assert validators.is_foodon_id("03541961")
        assert not validators.is_foodon_id("0354196")  # 7 digits

    # =========================================================================
    # Tier 4: Model Organism Databases
    # =========================================================================

    def test_tier4_model_organism_ids(self):
        """Test Tier 4 model organism database validators."""
        # FlyBase: FB + type code + digits
        assert validators.is_flybase_id("FBgn0019985")
        assert validators.is_flybase_id("FBtr0073412")
        assert validators.is_flybase_id("FBpp0080851")
        assert not validators.is_flybase_id("FB0019985")  # Missing type code
        assert not validators.is_flybase_id("FBxx0019985")  # Invalid type code

        # WormBase: WBGene + 8 digits
        assert validators.is_wormbase_gene_id("WBGene00012992")
        assert validators.is_wormbase_gene_id("WBGene00010912")
        assert not validators.is_wormbase_gene_id("WBGene0001299")  # 7 digits

        # ZFIN: ZDB-TYPE-digits-digits
        assert validators.is_zfin_id("ZDB-GENE-130109-1")
        assert validators.is_zfin_id("ZDB-GENE-041014-10")
        assert not validators.is_zfin_id("ZDB-130109-1")  # Missing TYPE

        # SGD: S + 9 digits
        assert validators.is_sgd_id("S000004291")
        assert validators.is_sgd_id("S000004559")
        assert not validators.is_sgd_id("S00000429")  # 8 digits

        # PomBase: SP + letters + digits + dot + digits + optional 'c'
        assert validators.is_pombase_id("SPAC6F12.09")
        assert validators.is_pombase_id("SPAP8A3.14c")
        assert not validators.is_pombase_id("SPAC6F12")  # Missing dot portion

        # DictyBase: DDB_G + 7 digits
        assert validators.is_dictybase_id("DDB_G0293130")
        assert not validators.is_dictybase_id("DDB_G029313")  # 6 digits

        # AraPort: AT + chromosome + G + 5 digits
        assert validators.is_araport_id("AT1G27500")
        assert validators.is_araport_id("AT5G10140")
        assert validators.is_araport_id("ATMG00010")  # Mitochondrial
        assert not validators.is_araport_id("AT6G27500")  # Invalid chromosome

        # EcoGene: EG + digits
        assert validators.is_ecogene_id("EG12315")
        assert not validators.is_ecogene_id("12315")  # Missing prefix

    # =========================================================================
    # Edge Cases: Final 4 Vocabulary Validators
    # =========================================================================

    def test_final4_edge_case_validators(self):
        """Test final 4 edge case validators: KEGG, CHEMBL.MECHANISM, OBA, OBO."""
        # KEGG generic IDs: 5 digits (pathways) OR letter + 5 digits (compounds/drugs)
        assert validators.is_kegg_generic_id("04966")  # Pathway
        assert validators.is_kegg_generic_id("04024")  # Pathway
        assert validators.is_kegg_generic_id("00590")  # Pathway
        assert validators.is_kegg_generic_id("C00031")  # Compound
        assert validators.is_kegg_generic_id("D00001")  # Drug
        assert not validators.is_kegg_generic_id("0496")  # 4 digits
        assert not validators.is_kegg_generic_id("049660")  # 6 digits
        assert not validators.is_kegg_generic_id("C0003")  # Too short

        # CHEMBL.MECHANISM: lowercase with underscores, parens, hyphens
        assert validators.is_chembl_mechanism_id("mitochondrial_complex_i_(nadh_dehydrogenase)_inhibitor")
        assert validators.is_chembl_mechanism_id("integrin_beta-7_antagonist")
        assert validators.is_chembl_mechanism_id("inducible_t-cell_costimulator_inhibitor")
        assert not validators.is_chembl_mechanism_id("UPPERCASE_NOT_ALLOWED")  # Uppercase

        # OBA: exactly 7 digits
        assert validators.is_oba_id("2044301")
        assert validators.is_oba_id("2053738")
        assert validators.is_oba_id("2042686")
        assert not validators.is_oba_id("204430")  # 6 digits
        assert not validators.is_oba_id("20443010")  # 8 digits

        # OBO: alphanumeric with underscores, hashes, colons
        assert validators.is_obo_id("APOLLO_SV_00000031")
        assert validators.is_obo_id("INO_0000018")
        assert validators.is_obo_id("EnsemblBacteria#_SAOUHSC_02706")
        assert not validators.is_obo_id("has spaces")  # Spaces not allowed

    # =========================================================================
    # Integration Tests
    # =========================================================================

    @pytest.mark.integration
    def test_normalizer_handles_kraken_vocabs(self):
        """Integration test: Normalizer can process KRAKEN vocab IDs."""
        from biomapper2.core.normalizer.normalizer import Normalizer

        normalizer = Normalizer()

        # Test that KRAKEN vocabs are recognized
        vocab_map = normalizer.vocab_validator_map

        # Check Tier 1 vocabs are registered
        assert "hgnc" in vocab_map
        assert "atc" in vocab_map
        assert "unii" in vocab_map
        assert "smpdb" in vocab_map

        # Check Tier 2 vocabs are registered
        assert "pato" in vocab_map
        assert "so" in vocab_map
        assert "hp" in vocab_map

        # Check Tier 3 vocabs are registered
        assert "pathwhiz" in vocab_map
        assert "meddra" in vocab_map

        # Check Tier 4 vocabs are registered
        assert "fb" in vocab_map
        assert "zfin" in vocab_map

        # Check Final 4 edge case vocabs are registered
        assert "kegg" in vocab_map
        assert "chembl.mechanism" in vocab_map
        assert "oba" in vocab_map
        assert "obo" in vocab_map

        # Verify aliases work
        assert "hpo" in vocab_map.get("hp", {}).get("aliases", [])
        assert "flybase" in vocab_map.get("fb", {}).get("aliases", [])

    @pytest.mark.integration
    def test_mapper_end_to_end(self):
        """End-to-end test: Full pipeline with Mapper.map_entity_to_kg()."""
        from biomapper2.mapper import Mapper

        mapper = Mapper()

        # Test with HGNC ID (a KRAKEN vocab)
        entity = {"name": "TP53", "hgnc": "11998"}
        result = mapper.map_entity_to_kg(
            item=entity,
            name_field="name",
            provided_id_fields=["hgnc"],
            entity_type="gene",
        )

        # Verify the entity was processed
        assert "curies" in result
        # HGNC should be normalized to a CURIE
        if result["curies"]:
            assert any("HGNC" in curie for curie in result["curies"])
