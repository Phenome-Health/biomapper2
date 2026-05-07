# Comprehensive Testing Strategy for biomapper2

**Status**: Draft (Revised)
**Date**: 2026-05-05 (Updated from 2026-01-20)
**Authors**: Drew, Claude

---

## Executive Summary

This document outlines a comprehensive testing strategy for biomapper2, addressing the unique challenges of testing a biomedical entity harmonization pipeline that depends on external APIs and evolving knowledge graphs. The strategy emphasizes **discerning specific changes at each pipeline step** for both logical correctness and performance.

---

## Table of Contents

**Part I: Strategic Overview**
1. [Current State Analysis](#1-current-state-analysis)
2. [Gap Analysis](#2-gap-analysis)
3. [Testing Pyramid Vision](#3-testing-pyramid-vision)

**Part II: Test Implementation**
4. [Component Testing Strategy](#4-component-testing-strategy)
5. [Integration Testing Strategy](#5-integration-testing-strategy)
6. [KG Version Regression Testing](#6-kg-version-regression-testing)

**Part III: Validation & Quality**
7. [Performance Testing Strategy](#7-performance-testing-strategy)
8. [Ground Truth & Validation](#8-ground-truth--validation)

**Part IV: Implementation & Operations**
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [CI Policies & Test Metadata](#10-ci-policies--test-metadata)

**Appendix**
- [Appendix: Key Changes Summary](#appendix-key-changes-summary)

---

---

# Part I: Strategic Overview

---

## 1. Current State Analysis

### Existing Test Coverage (193 tests)

| Test File | Count | Markers | Focus Area |
|-----------|-------|---------|------------|
| `test_validators_kraken.py` | 11 | `unit` (2 also `integration`) | Vocabulary ID validation (KEGG, PubChem, HMDB, etc.) |
| `test_normalizer.py` | 15 | `unit` | Delimiter parsing, CURIE normalization |
| `test_entity_kg_mapping.py` | 10 | `integration` `requires_api` | Single entity end-to-end mapping |
| `test_dataset_kg_mapping.py` | 7 | `integration` `requires_api` `slow` | Batch dataset mapping |
| `test_dataset_analysis.py` | 13 | `unit` | Statistics calculation, output files |
| `test_metabolomics_workbench.py` | 9 | mixed (`unit`/`integration`/`third_party`/`requires_api`) | MW annotator integration |
| `test_visualizer.py` | ~65 | `unit` | Heatmap/breakdown rendering, stats aggregation, P/R/F1 plots |
| `test_example_scripts.py` | 1 | `integration` `requires_api` `slow` | Example scripts smoke test |
| `test_batching.py` | ~12 | `unit` | Kestrel API batching (chunk_list, batched_kestrel_request) |
| `test_api_unit.py` | ~15 | `unit` | REST API (mocked Mapper) |
| `test_api.py` | ~20 | `unit` (3 also `integration` `requires_api`) | REST API live integration |
| `test_entity_model.py` | ~15 | `unit` (1 also `integration` `requires_api`) | Pydantic Entity model |
| `test_performance.py` | 3 | `performance` `requires_api` (2 also `slow`) | Per-step timing benchmarks incl. Step 5 |
| `test_equivalent_ids.py` | 7 | `unit` | Equivalent IDs enrichment (Step 5) - mocked Kestrel |

### Current Strengths

- **Shared mapper fixture** (`conftest.py`) - Session-scoped for efficiency
- **Ground truth validation** - Precision/recall/F1 against `kg_id_groundtruth` column
- **Coverage assertions** - Tests assert expected coverage percentages
- **Per-annotator tracking** - Stats broken down by annotation source

### Architectural Context (from meetings)

The pipeline processes items in bulk at the dataset level for performance:

```
Dataset → [Annotation (all rows)] → [Normalization (all rows)] → [Linking (bulk API)] → [Resolution (all rows)] → [Equivalent IDs (bulk API)] → Mapped Dataset
```

Each step adds specific fields, enabling step-by-step validation:

| Step | Input | Output Fields Added |
|------|-------|---------------------|
| Annotation | `{name, provided_ids}` | `assigned_ids` |
| Normalization | `{..., assigned_ids}` | `curies`, `curies_provided`, `curies_assigned`, `invalid_ids_*` |
| Linking | `{..., curies}` | `kg_ids`, `kg_ids_provided`, `kg_ids_assigned` |
| Resolution | `{..., kg_ids}` | `chosen_kg_id`, `chosen_kg_id_provided`, `chosen_kg_id_assigned` |
| Equivalent IDs | `{..., chosen_kg_id}` | `kg_equivalent_ids` (grouped by CURIE prefix; `{}` on failure — non-critical) |

---

## 2. Gap Analysis

### Critical Missing Tests

| Gap | Impact | Priority | Notes (2026-05-05) |
|-----|--------|----------|-------------------|
| **Per-step output validation** | Can't isolate where failures occur | High | Foundation for all other testing |
| **KG version regression tests** | No detection of KRAKEN changes breaking mappings | High | Critical for Spoke vs non-Spoke comparison |
| **LLM validation layer** | No automated quality assessment for text-based mappings | **High** | **Elevated from Low** - Now P0 for questionnaires/demographics |
| **Vector search validation** | No isolated testing of embedding quality | High | Per Trent: can validate independently |
| **Human curation integration** | No systematic ground truth collection | **High** | Trent's Replit app needs integration |
| **Step 5 full-pipeline degradation** | No test that `map_dataset_to_kg` succeeds when Step 5 API fails | Medium | Step 5 is non-critical by design — should not block result |
| **Entity-level validation** | Only dataset-level metrics exist | Medium | |
| **Concurrent/load testing** | Unknown behavior at scale | Medium | Related to #44 (batch chunking) |

### Gaps Identified in Team Meetings

From **dh-25-11-5** (Amy/Drew/Trent):
> "A full end-to-end integration test... there's a lot of profiling that has to happen to see where things are taking time because presumably this could be a really long process."

> "What if we just exposed a different endpoint with a version number... change an element in the URL for the Kestrel API and get different running versions."

From **dh-25-11-7** (Amy demo):
> "These are like per_provided_ids metrics... even though we have some datasets that are designated ground truth, for any dataset that has provided IDs, we can use those to evaluate how correct the assigned IDs are."

From **dh-25-12-9** (Trent vectors):
> "The thing with the vector databases is I feel like we could almost come up with a way to validate them independently of all this."

From **dh-26-01-13** (Task prioritization):
> Focus on Spoke vs non-Spoke comparison - maintaining two versions of KRAKEN for benchmarking.

From **dh-26-01-20** (Project review):
> Trent building human validation loop via Replit app - "you sign in with your Phenome Health Google account and like anybody can go in whenever they have a few minutes and validate some stuff."

> LLM validation for questionnaires is P0: "We definitely got to figure out how to validate this LLM stuff."

> Separate validation campaigns needed: natural language mappings vs code-based mappings.

---

## 3. Testing Pyramid Vision

```
                    ┌─────────────────────────────────────┐
                    │         E2E / Acceptance            │  ← Few, slow, comprehensive
                    │   (Full pipeline with real APIs)    │
                    └─────────────────────────────────────┘
               ┌─────────────────────────────────────────────────┐
               │              Integration Tests                   │  ← Pipeline step interactions
               │  (Step A→B transitions, API mocking optional)   │
               └─────────────────────────────────────────────────┘
          ┌─────────────────────────────────────────────────────────────┐
          │                    Component Tests                           │  ← Each pipeline step isolated
          │  (Normalizer, Linker, Resolver, each Annotator)             │
          └─────────────────────────────────────────────────────────────┘
     ┌─────────────────────────────────────────────────────────────────────────┐
     │                          Unit Tests                                      │  ← Many, fast, isolated
     │  (Validators, cleaners, parsers, utilities)                             │
     └─────────────────────────────────────────────────────────────────────────┘
```

### Determinism Tiers

All tests are categorized by determinism tier, which drives CI behavior:

| Tier | Description | CI Behavior | Examples |
|------|-------------|-------------|----------|
| **A** | Fully offline/mocked, deterministic | Required, per-commit | Unit tests, Pandera schemas, Hypothesis with `derandomize=True` |
| **B** | Environment-dependent | Nightly CI | Performance benchmarks, memory profiling |
| **C** | Live external services | Opt-in only | Kestrel API integration, LLM validation |

**Critical Rule**: Tier C failures never block merges unless explicitly enabled for a release.

### Test Markers Reference

Each pyramid level maps to pytest markers. Additional markers support cross-cutting concerns (performance, regression, API dependency).

```ini
# pyproject.toml [tool.pytest.ini_options]
markers = [
    # Pyramid levels
    "unit: Tier A - Pure functions, no external dependencies",
    "component: Tier A - Single pipeline step, may mock dependencies",
    "integration: Tier B/C - Multiple steps or real API calls",
    "e2e: Tier C - Full end-to-end pipeline with live KG",
    # Cross-cutting markers
    "performance: Tier B - Timing and memory benchmarks",
    "kg_regression: Tier B/C - Knowledge graph version change detection",
    "slow: Tier B - Tests taking >10 seconds",
    "requires_api: Tier C - Requires live Kestrel API connection",
]
```

| Level | Marker | Tier | CI Behavior |
|-------|--------|------|-------------|
| Unit | `@pytest.mark.unit` | A | Always run |
| Component | `@pytest.mark.component` | A | Always run |
| Integration | `@pytest.mark.integration` | B/C | Nightly or opt-in |
| E2E | `@pytest.mark.e2e` | C | Opt-in only |
| Performance | `@pytest.mark.performance` | B | Nightly |
| KG Regression | `@pytest.mark.kg_regression` | B/C | Nightly or opt-in |
| Slow | `@pytest.mark.slow` | B | Nightly |
| Requires API | `@pytest.mark.requires_api` | C | Opt-in only |

### CI Gating Commands

```bash
# Per-commit CI (Tier A only) - FAST, required for merge
uv run pytest -m "not slow and not performance and not requires_api"

# Nightly CI (Tier A + B) - includes performance benchmarks
uv run pytest -m "not requires_api"

# Opt-in CI (Tier C) - live API tests, manual trigger only
RUN_LIVE_API_TESTS=1 uv run pytest -m requires_api

# Specific test categories
uv run pytest -m integration                    # Integration tests
uv run pytest -m performance --benchmark-only   # Benchmarks with stats
uv run pytest -m kg_regression                  # KG version regression
uv run pytest -m kg_regression --kestrel-url=https://staging.example.com/api  # Against staging KG
```

---

---

# Part II: Test Implementation

---

## 4. Component Testing Strategy

### 4.1 Annotation Engine Tests

**Goal**: Validate each annotator in isolation, then orchestration logic.

```python
# tests/test_annotation_engine.py

class TestAnnotatorSelection:
    """Test that correct annotators are selected per entity type."""

    def test_metabolite_selects_mw_and_kestrel(self):
        """metabolite → [metabolomics-workbench, kestrel-hybrid-search]"""

    def test_protein_selects_kestrel_only(self):
        """protein → [kestrel-hybrid-search]"""

    def test_disease_selects_kestrel_text(self):
        """disease → [kestrel-text-search] (fuzzy)"""

class TestAnnotationModes:
    """Test 'all', 'missing', 'none' annotation modes."""

    def test_mode_all_annotates_everything(self):
        """Even items with provided IDs get annotated."""

    def test_mode_missing_skips_items_with_ids(self):
        """Items with provided IDs are not annotated."""

    def test_mode_none_returns_empty(self):
        """No annotation occurs, assigned_ids empty."""
```

**Per-Annotator Test Pattern**:

```python
# tests/annotators/test_kestrel_vector.py

class TestKestrelVectorAnnotator:
    """Isolated tests for vector search annotator."""

    @pytest.fixture
    def annotator(self):
        return KestrelVectorAnnotator()

    def test_exact_synonym_match(self, annotator):
        """Query exactly matching a synonym returns correct HMDB ID."""
        result = annotator.get_annotations({"name": "glucose"}, "name")
        assert "HMDB:HMDB0000122" in result.get("hmdb", [])

    def test_fuzzy_match_tolerance(self, annotator):
        """10% character mutation still finds match (from Trent's testing)."""
        # "glucos" (missing 'e') should still match
        result = annotator.get_annotations({"name": "glucos"}, "name")
        assert result  # Should find glucose

    def test_no_match_returns_empty(self, annotator):
        """Nonsense query returns empty dict."""
        result = annotator.get_annotations({"name": "xyzzy123notametabolite"}, "name")
        assert result == {}

    def test_multi_vector_disambiguation(self, annotator):
        """Ambiguous terms (e.g., 'triglyceride') handled correctly."""
        # Per meeting: triglyceride has ~40k HMDB IDs as synonyms
        result = annotator.get_annotations({"name": "triglyceride"}, "name")
        # Should return abstract node, not random specific triglyceride
```

### 4.2 Normalizer Tests

**Existing coverage is good**, expand with:

```python
class TestNormalizerEdgeCases:
    """Edge cases for ID normalization."""

    def test_multiple_delimiters_in_single_field(self):
        """'C00487;C00308,C00123' parses correctly with [',', ';']."""

    def test_already_curie_format_passthrough(self):
        """'KEGG:C00487' not double-prefixed."""

    def test_unrecognized_vocab_tracked(self):
        """Unknown vocab name goes to unrecognized_vocabs list."""
```

### 4.3 Linker Tests

```python
class TestLinker:
    """Tests for CURIE → KG node linking."""

    def test_bulk_link_batching(self, shared_mapper):
        """Verify bulk API call is made for efficiency."""
        # Mock or spy on Kestrel API call count

    def test_curie_not_in_kg_tracked(self):
        """Valid CURIE not in KRAKEN goes to curie_misses."""

    def test_multiple_kg_nodes_for_curie(self):
        """One CURIE → multiple KG nodes handled."""
```

### 4.4 Resolver Tests

```python
class TestResolver:
    """Tests for one-to-many resolution."""

    def test_voting_selects_majority(self):
        """3 CURIEs → KG_A, 1 CURIE → KG_B → chooses KG_A."""

    def test_tie_breaking_deterministic(self):
        """Equal votes → deterministic selection (alphabetical?)."""

    def test_single_kg_id_no_resolution_needed(self):
        """Only one KG ID → chosen directly."""
```

---

## 5. Integration Testing Strategy

### 5.1 Step Transition Tests

**Goal**: Verify output of step N is valid input for step N+1.

```python
# tests/test_pipeline_integration.py

class TestPipelineStepTransitions:
    """Test that each step's output feeds correctly into the next."""

    def test_annotation_to_normalization(self, shared_mapper):
        """Annotation output has fields required by Normalizer."""
        item = {"name": "carnitine"}

        # Step 1: Annotate
        annotated = shared_mapper.annotation_engine.annotate(
            item=item, name_field="name", provided_id_fields=[],
            category="biolink:SmallMolecule", prefixes=None, mode="all"
        )

        # Verify annotation output schema
        assert "assigned_ids" in annotated.index

        # Step 2: Normalize - should not fail
        merged = item | annotated.to_dict()
        normalized = shared_mapper.normalizer.normalize(
            item=pd.Series(merged), provided_id_fields=[], array_delimiters=[",", ";"]
        )

        assert "curies_assigned" in normalized.index

    def test_normalization_to_linking(self, shared_mapper):
        """Normalization output has CURIEs required by Linker."""
        # Similar pattern...

    def test_linking_to_resolution(self, shared_mapper):
        """Linking output has kg_ids required by Resolver."""
        # Similar pattern...

    def test_resolution_to_equivalent_ids(self, shared_mapper):
        """Step 4 output provides chosen_kg_id for Step 5 enrichment."""
        item = {"name": "aspirin"}

        # Run Steps 1–4 via map_entity_to_kg (which also runs Step 5)
        result = shared_mapper.map_entity_to_kg(
            item=item, name_field="name", provided_id_fields=[], entity_type="metabolite"
        )

        # If a KG match was found, kg_equivalent_ids must be a dict (never None)
        assert isinstance(result["kg_equivalent_ids"], dict)
        if result["chosen_kg_id"] is not None:
            # At least some prefix groupings should be present for a known metabolite
            assert len(result["kg_equivalent_ids"]) > 0

    def test_step5_graceful_degradation(self, shared_mapper):
        """Pipeline result is valid even when Step 5 Kestrel call fails."""
        from unittest.mock import patch

        with patch("biomapper2.core.linker.kestrel_request", side_effect=Exception("timeout")):
            result = shared_mapper.map_entity_to_kg(
                item={"name": "glucose", "kegg": "C00031"},
                name_field="name", provided_id_fields=["kegg"], entity_type="metabolite"
            )

        # Steps 1–4 must still produce a valid KG mapping
        assert result["chosen_kg_id"] is not None
        # Step 5 failed gracefully — field present but empty
        assert result["kg_equivalent_ids"] == {}
```

### 5.2 Field Provenance Tests

**Goal**: Track that provided vs assigned IDs are correctly attributed throughout.

```python
class TestFieldProvenance:
    """Verify provided/assigned ID tracking through pipeline."""

    def test_provided_ids_tracked_separately(self, shared_mapper):
        """IDs from input columns tracked in *_provided fields."""
        item = {"name": "glucose", "kegg": "C00031"}
        result = shared_mapper.map_entity_to_kg(
            item=item, name_field="name", provided_id_fields=["kegg"],
            entity_type="metabolite"
        )

        assert "KEGG:C00031" in result["curies_provided"]
        assert "KEGG:C00031" not in result.get("curies_assigned", {})

    def test_assigned_ids_tracked_separately(self, shared_mapper):
        """IDs from annotation tracked in *_assigned fields."""
        item = {"name": "glucose"}  # No provided IDs
        result = shared_mapper.map_entity_to_kg(
            item=item, name_field="name", provided_id_fields=[],
            entity_type="metabolite", annotation_mode="all"
        )

        assert result["assigned_ids"]  # Should have annotations
        assert result["curies_assigned"]
        assert not result["curies_provided"]
```

### 5.3 API Integration Tests

```python
@pytest.mark.integration
class TestKestrelAPIIntegration:
    """Tests requiring live Kestrel API."""

    def test_api_connectivity(self):
        """Verify Kestrel API is reachable."""

    def test_api_timeout_handling(self):
        """Graceful handling of API timeout."""

    def test_api_rate_limiting(self):
        """Behavior under rate limiting."""
```

---

## 6. KG Version Regression Testing

### The Challenge

From meetings: KRAKEN is large (~9M nodes) and evolves. New versions may:
- Add/remove node equivalences
- Change canonical IDs for entities
- Update prefix mappings
- Modify edge relationships

**Active Comparison (2026-01-20)**: The team is evaluating KRAKEN with and without Spoke due to licensing concerns. This requires systematic comparison infrastructure.

### 6.0 Milestone Runner Integration (NEW)

The existing `scripts/run_milestone_datasets.py` provides the foundation for KG regression testing. It runs 11 datasets across 3 sources (Arivale, UKBB, HPP) and 4 entity types.

**Current State**:
```python
# scripts/run_milestone_datasets.py
kg_name = "kraken"  # Can be parameterized
results_dir = datasets_dir / "results" / kg_name
```

**Enhanced for Regression Testing**:

```python
# scripts/run_kg_comparison.py

import argparse
from pathlib import Path

def run_comparison(kg_configs: list[dict]) -> dict:
    """
    Run milestone datasets against multiple KG configurations.

    Args:
        kg_configs: List of {"name": "kraken-with-spoke", "url": "https://..."}

    Returns:
        Comparison report with deltas
    """
    all_results = {}

    for config in kg_configs:
        # Temporarily override Kestrel URL
        os.environ["KESTREL_API_URL"] = config["url"]
        mapper = Mapper()

        results_dir = PROJECT_ROOT / "data" / "milestone" / "results" / config["name"]
        all_results[config["name"]] = run_all_datasets(mapper, results_dir)

    return generate_comparison_report(all_results)

# Usage:
# python scripts/run_kg_comparison.py \
#   --kg kraken-with-spoke https://kestrel.phenomehealth.org/api \
#   --kg kraken-no-spoke https://kestrel-staging.phenomehealth.org/api
```

**Output Structure**:
```
data/milestone/results/
├── kraken-with-spoke/
│   ├── arivale_proteins_MAPPED.tsv
│   ├── arivale_proteins_MAPPED_a_summary_stats.json
│   ├── ...
│   └── viz/
│       ├── heatmap_all.png
│       └── breakdown_all.png
├── kraken-no-spoke/
│   └── (same structure)
└── comparison/
    ├── delta_report.json          # Differences between KG versions
    ├── affected_entities.tsv      # Entities that mapped differently
    └── recommendation.md          # Human-readable summary
```

### 6.1 Snapshot Testing

**Concept**: Capture expected outputs for fixed inputs, detect when they change.

```python
# tests/test_kg_regression.py

@pytest.mark.kg_regression
class TestKGVersionRegression:
    """Detect changes in KG behavior across versions."""

    # Snapshot file: tests/snapshots/kg_mappings_v4.2.5.json
    SNAPSHOT_FILE = PROJECT_ROOT / "tests" / "snapshots" / "kg_mappings_baseline.json"

    @pytest.fixture
    def baseline_mappings(self):
        """Load baseline expected mappings."""
        with open(self.SNAPSHOT_FILE) as f:
            return json.load(f)

    def test_housekeeping_entities_unchanged(self, shared_mapper, baseline_mappings):
        """
        'Housekeeping' entities (Andrew's suggestion) should always map consistently.
        These are well-known, stable entities unlikely to change.
        """
        housekeeping = [
            {"name": "glucose", "kegg": "C00031", "expected_kg": "CHEBI:17234"},
            {"name": "ATP", "kegg": "C00002", "expected_kg": "CHEBI:30616"},
            {"name": "insulin", "uniprot": "P01308", "expected_kg": "UniProtKB:P01308"},
        ]

        for entity in housekeeping:
            result = shared_mapper.map_entity_to_kg(
                item={k: v for k, v in entity.items() if k != "expected_kg"},
                name_field="name",
                provided_id_fields=[k for k in entity if k not in ("name", "expected_kg")],
                entity_type="metabolite" if "kegg" in entity else "protein"
            )
            assert result["chosen_kg_id"] == entity["expected_kg"], \
                f"{entity['name']} mapped to {result['chosen_kg_id']}, expected {entity['expected_kg']}"

    def test_groundtruth_dataset_stable(self, shared_mapper, baseline_mappings):
        """Full ground truth dataset should have stable precision/recall."""
        _, stats = shared_mapper.map_dataset_to_kg(
            dataset=PROJECT_ROOT / "data" / "groundtruth" / "diseases_handcrafted.tsv",
            entity_type="disease", name_column="name", provided_id_columns=[]
        )

        baseline = baseline_mappings["diseases_handcrafted"]

        # Allow small variance (±2%) but flag significant changes
        assert abs(stats["performance"]["overall"]["per_groundtruth"]["precision"]
                   - baseline["precision"]) < 0.02, "Precision changed significantly"
        assert abs(stats["performance"]["overall"]["per_groundtruth"]["recall"]
                   - baseline["recall"]) < 0.02, "Recall changed significantly"
```

### 6.2 KG Version Parameterization

**Concept**: Run same tests against different Kestrel endpoints.

```python
# conftest.py addition

def pytest_addoption(parser):
    parser.addoption(
        "--kestrel-url",
        default="https://kestrel.nathanpricelab.com/api",
        help="Kestrel API URL to test against"
    )
    parser.addoption(
        "--kg-version",
        default="production",
        help="KG version label for reporting"
    )

@pytest.fixture
def kestrel_url(request):
    return request.config.getoption("--kestrel-url")
```

**Usage**:
```bash
# Test against production
uv run pytest tests/test_kg_regression.py

# Test against staging/new KG version
uv run pytest tests/test_kg_regression.py --kestrel-url=https://kestrel-staging.nathanpricelab.com/api --kg-version=kraken-2.1
```

### 6.3 Diff Reporting

```python
def generate_kg_diff_report(baseline_stats: dict, current_stats: dict) -> str:
    """Generate human-readable diff of KG performance changes."""
    report = []

    for metric in ["precision", "recall", "f1_score", "coverage"]:
        baseline_val = baseline_stats.get(metric, 0)
        current_val = current_stats.get(metric, 0)
        delta = current_val - baseline_val
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "="

        report.append(f"{metric}: {baseline_val:.3f} → {current_val:.3f} ({direction} {abs(delta):.3f})")

    return "\n".join(report)
```

---

---

# Part III: Validation & Quality

---

## 7. Performance Testing Strategy

### 7.0 Metrics Collection Architecture (NEW)

**Goal**: Establish consistent, structured metrics collection that can feed into visualizations (#46), CI pipelines, and regression detection.

#### Pipeline Instrumentation Checkpoints

Each pipeline step should emit structured metrics at specific checkpoints:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           METRICS COLLECTION POINTS                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT                                                                           │
│    ↓                                                                             │
│  ┌──────────────────┐                                                           │
│  │   ANNOTATION     │ ─→ Checkpoint A1: Per-annotator timing                    │
│  │                  │ ─→ Checkpoint A2: Per-annotator hit rate                  │
│  │                  │ ─→ Checkpoint A3: API latency (external calls)            │
│  └────────┬─────────┘                                                           │
│           ↓                                                                      │
│  ┌──────────────────┐                                                           │
│  │  NORMALIZATION   │ ─→ Checkpoint N1: Validation pass/fail counts             │
│  │                  │ ─→ Checkpoint N2: Vocab matching success rate             │
│  │                  │ ─→ Checkpoint N3: CURIE construction timing               │
│  └────────┬─────────┘                                                           │
│           ↓                                                                      │
│  ┌──────────────────┐                                                           │
│  │     LINKING      │ ─→ Checkpoint L1: Kestrel API latency (bulk)              │
│  │                  │ ─→ Checkpoint L2: Cache hit rate                          │
│  │                  │ ─→ Checkpoint L3: KG miss rate (CURIEs not in KG)         │
│  └────────┬─────────┘                                                           │
│           ↓                                                                      │
│  ┌──────────────────┐                                                           │
│  │   RESOLUTION     │ ─→ Checkpoint R1: One-to-many frequency                   │
│  │                  │ ─→ Checkpoint R2: Voting margin distribution              │
│  │                  │ ─→ Checkpoint R3: Resolution timing                       │
│  └────────┬─────────┘                                                           │
│           ↓                                                                      │
│  ┌──────────────────┐                                                           │
│  │ EQUIVALENT IDs   │ ─→ Checkpoint E1: Kestrel /get-nodes latency (bulk)       │
│  │   (Step 5)       │ ─→ Checkpoint E2: Unique KG IDs deduplicated (efficiency) │
│  │                  │ ─→ Checkpoint E3: API failure rate (non-critical step)     │
│  └────────┬─────────┘                                                           │
│           ↓                                                                      │
│  OUTPUT                                                                          │
│    ↓                                                                             │
│  ┌──────────────────┐                                                           │
│  │    ANALYSIS      │ ─→ Checkpoint X1: Coverage metrics                        │
│  │                  │ ─→ Checkpoint X2: Precision/Recall/F1 (if ground truth)   │
│  │                  │ ─→ Checkpoint X3: Per-annotator attribution               │
│  └──────────────────┘                                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Structured Metrics Output Format

```python
@dataclass
class PipelineMetrics:
    """Structured metrics from a single pipeline run."""

    # Identification
    dataset_name: str
    entity_type: str
    kg_version: str
    timestamp: datetime

    # Timing (all in milliseconds) — keys: annotation, normalization, linking, resolution, equivalent_ids
    timing: dict[str, float]  # {"annotation": 1234, "normalization": 567, "equivalent_ids": 89, ...}
    timing_per_item: dict[str, float]  # Same keys, per-item averages

    # Counts
    items_total: int
    items_mapped: int
    items_unmapped: int
    items_one_to_many: int

    # Per-annotator breakdown
    annotator_stats: dict[str, AnnotatorMetrics]

    # Quality metrics (if ground truth available)
    precision: float | None
    recall: float | None
    f1_score: float | None

    # API metrics
    api_calls: int
    api_latency_total_ms: float
    cache_hit_rate: float

    def to_json(self) -> str:
        """Export for CI/visualization consumption."""
        ...
```

#### Integration with Milestone Runner

The existing `scripts/run_milestone_datasets.py` can be enhanced to emit metrics:

```python
# Enhanced milestone runner with metrics collection
metrics_collector = MetricsCollector(output_dir=results_dir / "metrics")

for dataset_filename, params in datasets.items():
    with metrics_collector.track(dataset_filename) as tracker:
        mapper.map_dataset_to_kg(
            dataset=tsv_path,
            metrics_tracker=tracker,  # New parameter
            ...
        )

# Output: results/kraken/metrics/pipeline_metrics.jsonl
```

### 7.1 Per-Step Timing

The `clear_kestrel_cache` autouse fixture is critical: other test files run alphabetically before `test_performance.py` and warm the `requests_cache` SQLite file, causing API calls to be served from disk (~25× faster than real network). The fixture deletes the cache file so all timings reflect real Kestrel latency.

```python
# tests/test_performance.py (actual implementation)

import time
from dataclasses import dataclass
from pathlib import Path

from biomapper2.config import CACHE_DIR, PROJECT_ROOT

_KESTREL_CACHE = Path(CACHE_DIR) / "kestrel_http.sqlite"


@pytest.fixture(scope="session", autouse=True)
def clear_kestrel_cache():
    """Delete the Kestrel HTTP cache before performance tests run."""
    if _KESTREL_CACHE.exists():
        _KESTREL_CACHE.unlink()
    yield


@dataclass
class StepTiming:
    step: str
    duration_ms: float
    items_processed: int

    @property
    def ms_per_item(self) -> float:
        return self.duration_ms / self.items_processed if self.items_processed else 0.0


def _time_pipeline_steps(mapper, df, entity_type, name_field, provided_id_fields, array_delimiters):
    timings = []
    n = len(df)
    category = mapper.biolink_client.standardize_entity_type(entity_type)
    prefixes = mapper.normalizer.get_standard_prefix(None)  # None = no prefix filter

    # Step 1: Annotation
    t0 = time.perf_counter()
    annotation_df = mapper.annotation_engine.annotate(
        item=df, name_field=name_field, provided_id_fields=provided_id_fields,
        category=category, prefixes=prefixes, mode="missing",
    )
    timings.append(StepTiming("annotation", (time.perf_counter() - t0) * 1000, n))
    df = df.join(annotation_df)

    # Step 2: Normalization
    t0 = time.perf_counter()
    normalization_df = mapper.normalizer.normalize(
        item=df, provided_id_fields=provided_id_fields, array_delimiters=array_delimiters,
    )
    timings.append(StepTiming("normalization", (time.perf_counter() - t0) * 1000, n))
    df = df.join(normalization_df)

    # Step 3: Linking (Kestrel API call)
    t0 = time.perf_counter()
    linked_df = mapper.linker.link(df)
    timings.append(StepTiming("linking", (time.perf_counter() - t0) * 1000, n))
    df = df.join(linked_df)

    # Step 4: Resolution
    t0 = time.perf_counter()
    resolved_df = mapper.resolver.resolve(df)
    timings.append(StepTiming("resolution", (time.perf_counter() - t0) * 1000, n))
    df = df.join(resolved_df)

    # Step 5: Equivalent IDs enrichment (bulk /get-nodes call for unique chosen_kg_ids)
    unique_kg_ids = [kid for kid in df["chosen_kg_id"].dropna().unique()]
    t0 = time.perf_counter()
    mapper.linker.get_equivalent_ids(unique_kg_ids)
    timings.append(StepTiming("equivalent_ids", (time.perf_counter() - t0) * 1000, n))

    return timings


@pytest.mark.performance
@pytest.mark.slow
class TestPipelinePerformance:
    def test_step_timings_olink_proteins(self, shared_mapper):
        """Profile each step for the OLink protein dataset (~2900 items)."""
        df = pd.read_csv(DATA_DIR / "olink_protein_metadata.tsv", sep="\t")
        timings = _time_pipeline_steps(
            mapper=shared_mapper, df=df, entity_type="protein",
            name_field="Assay", provided_id_fields=["UniProt"], array_delimiters=["_"],
        )
        by_step = {t.step: t for t in timings}
        assert by_step["annotation"].duration_ms < 10_000
        assert by_step["normalization"].duration_ms < 5_000
        assert by_step["linking"].duration_ms < 10_000
        assert by_step["resolution"].duration_ms < 2_000
        assert by_step["equivalent_ids"].duration_ms < 10_000
```

**Observed cold-cache timings (2923 proteins, 2026-05-05):**

| Step | Time (ms) | ms/item | Notes |
|------|-----------|---------|-------|
| annotation | ~500 | 0.17 | Kestrel vector search |
| normalization | ~150 | 0.05 | Local CPU only |
| linking | ~666 | 0.23 | Batched `/link` call |
| resolution | ~50 | 0.02 | Local CPU only |
| equivalent_ids | ~1594 | 0.55 | Two batched `/get-nodes` calls; dominant step |
| **TOTAL** | **~2960** | **1.01** | |

Step 5 (`equivalent_ids`) is the current bottleneck at cold start, making two `/get-nodes` API calls for ~2921 unique KG IDs.

### 7.2 pytest-benchmark Integration

```python
# Using pytest-benchmark for statistical rigor

@pytest.mark.performance
def test_normalizer_throughput(benchmark, shared_mapper):
    """Benchmark normalizer throughput."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "examples" / "metabolites_synthetic.tsv", sep="\t")

    result = benchmark(
        shared_mapper.normalizer.normalize,
        item=df,
        provided_id_fields=["INCHIKEY", "HMDB", "KEGG", "PUBCHEM", "CHEBI"],
        array_delimiters=[",", ";"]
    )

    # Benchmark automatically reports stats
```

### 7.3 Memory Profiling

```python
@pytest.mark.performance
def test_memory_usage_large_dataset(shared_mapper):
    """Monitor memory usage for large dataset processing."""
    import tracemalloc

    tracemalloc.start()

    # Load large dataset
    df = pd.read_csv(PROJECT_ROOT / "data" / "examples" / "olink_protein_metadata.tsv", sep="\t")

    # Process
    _, stats = shared_mapper.map_dataset_to_kg(
        dataset=df, entity_type="protein", name_column="Assay",
        provided_id_columns=["UniProt"], array_delimiters=["_"]
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nMemory: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")

    # Threshold (adjust based on available resources)
    assert peak < 500 * 1024 * 1024, f"Peak memory {peak/1024/1024:.1f}MB exceeds 500MB limit"
```

### 7.4 Metrics Parsing & Analysis (TBD)

> **Status**: Design phase. Relates to Issue #46 (visualization).

Section 7.0 defines metrics *collection* via `PipelineMetrics` and checkpoint instrumentation. This section addresses metrics *consumption*—how collected metrics are parsed, stored, and analyzed.

#### Open Questions

1. **Storage format**: JSON lines (`pipeline_metrics.jsonl`) vs SQLite vs time-series DB?
2. **Retention policy**: How long to keep historical metrics? Per-run? Per-day aggregates?
3. **Regression detection**: Automated alerting when step latency exceeds baseline by X%?
4. **Visualization integration**: How does #46 consume these metrics?

#### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        METRICS LIFECYCLE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  COLLECTION (Section 7.0)           CONSUMPTION (This Section)              │
│  ─────────────────────────          ────────────────────────────            │
│                                                                              │
│  Pipeline run                                                                │
│       ↓                                                                      │
│  Checkpoint instrumentation                                                  │
│       ↓                                                                      │
│  PipelineMetrics.to_json()                                                   │
│       ↓                                                                      │
│  results/{kg}/metrics/pipeline_metrics.jsonl                                 │
│       │                                                                      │
│       ├──→ MetricsLoader.load_run(path) ──→ Single run analysis             │
│       │                                                                      │
│       ├──→ MetricsLoader.load_history(dir) ──→ Time-series analysis         │
│       │         ↓                                                            │
│       │    Regression detection (latency drift, coverage drops)             │
│       │                                                                      │
│       └──→ Visualization (#46)                                               │
│                 ↓                                                            │
│            Dashboard: step timing, annotator breakdown, KG comparison        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Placeholder Implementation

```python
# src/biomapper2/core/metrics_loader.py (TBD)

from pathlib import Path
import json
from dataclasses import dataclass

@dataclass
class MetricsLoader:
    """Load and parse pipeline metrics for analysis."""

    @staticmethod
    def load_run(metrics_file: Path) -> "PipelineMetrics":
        """Load metrics from a single pipeline run."""
        # TBD: Parse JSONL, reconstruct PipelineMetrics
        ...

    @staticmethod
    def load_history(metrics_dir: Path, days: int = 30) -> list["PipelineMetrics"]:
        """Load historical metrics for trend analysis."""
        # TBD: Glob for *.jsonl, filter by date, return sorted list
        ...

    @staticmethod
    def detect_regression(
        current: "PipelineMetrics",
        baseline: "PipelineMetrics",
        threshold_pct: float = 20.0
    ) -> dict[str, float]:
        """
        Compare current run to baseline, flag steps exceeding threshold.

        Returns:
            Dict of {step_name: percent_change} for flagged steps
        """
        # TBD: Compare timing_per_item, coverage, etc.
        ...
```

#### Integration with Visualization (#46)

The visualization system (Ashen's work) should consume metrics via `MetricsLoader`:

```python
# Example: Generate step timing chart
metrics = MetricsLoader.load_run(results_dir / "metrics" / "pipeline_metrics.jsonl")

# Feed to visualization
visualizer.plot_step_timing(
    steps=list(metrics.timing.keys()),
    durations=list(metrics.timing.values()),
    title=f"{metrics.dataset_name} - {metrics.kg_version}"
)
```

**Next Steps**:
- [ ] Finalize storage format decision
- [ ] Implement `MetricsLoader` with basic parsing
- [ ] Define regression thresholds with team
- [ ] Coordinate with #46 on visualization interface

---

## 8. Ground Truth & Validation

### 8.0 Human Curation Loop (NEW - P0)

**Current State**: Trent is building a Replit-based validation app that allows team members to validate mappings in their spare time.

**Workflow**:
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        HUMAN CURATION WORKFLOW                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. GENERATE CANDIDATES                                                       │
│     ┌─────────────────┐                                                      │
│     │ Run biomapper2  │ ──→ Mappings with confidence/source metadata         │
│     │ on dataset      │                                                      │
│     └─────────────────┘                                                      │
│              ↓                                                                │
│  2. SAMPLE FOR VALIDATION                                                     │
│     ┌─────────────────┐                                                      │
│     │ Random subset   │ ──→ 50-100 mappings per campaign                     │
│     │ (stratified)    │     Separate campaigns for:                          │
│     └─────────────────┘       • Natural language → KG (questionnaires)       │
│              ↓                • Code-based → KG (LOINC, ICD-10)              │
│                               • ID-based → KG (metabolites, proteins)        │
│  3. HUMAN VALIDATION (Replit App)                                            │
│     ┌─────────────────┐                                                      │
│     │ Curator sees:   │                                                      │
│     │ • Source label  │                                                      │
│     │ • Mapped KG ID  │ ──→ Clicks: ✓ Match | ✗ No Match | ? Unsure         │
│     │ • Link to KG    │                                                      │
│     └─────────────────┘                                                      │
│              ↓                                                                │
│  4. AGGREGATE & EXPORT                                                        │
│     ┌─────────────────┐                                                      │
│     │ Consolidate     │ ──→ data/groundtruth/{campaign}_curated.tsv          │
│     │ responses       │     Include: curator_id, timestamp, decision         │
│     └─────────────────┘                                                      │
│              ↓                                                                │
│  5. COMPUTE INTER-RATER RELIABILITY                                           │
│     ┌─────────────────┐                                                      │
│     │ Multiple curators │ ──→ Cohen's kappa / Fleiss' kappa                  │
│     │ per mapping       │                                                    │
│     └─────────────────┘                                                      │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions**:
- **Authentication**: Phenome Health Google accounts (audit trail)
- **Campaign separation**: Natural language vs code-based mappings require different expertise
- **Target curator**: Lee Row (primary), but system supports multiple curators
- **Sample size**: Start with 50-100 per campaign, expand based on error rates

**Integration with Testing**:
```python
# tests/test_groundtruth_validation.py

def test_curated_groundtruth_accuracy(shared_mapper):
    """
    Run mappings against human-curated ground truth.
    This test fails if accuracy drops below threshold.
    """
    curated = load_curated_groundtruth("questionnaires_curated.tsv")

    results = []
    for item in curated:
        mapped = shared_mapper.map_entity_to_kg(...)
        results.append({
            "expected": item["kg_id_groundtruth"],
            "actual": mapped["chosen_kg_id"],
            "match": mapped["chosen_kg_id"] == item["kg_id_groundtruth"]
        })

    accuracy = sum(r["match"] for r in results) / len(results)
    assert accuracy >= 0.85, f"Accuracy {accuracy:.1%} below 85% threshold"
```

### 8.1 Ground Truth Dataset Schema

```
data/groundtruth/
├── metabolites_aerovail.tsv           # Real-world metabolomics dataset
├── proteins_olink_subset.tsv          # Curated protein mappings
├── diseases_handcrafted.tsv           # Manually verified disease mappings
├── questionnaires_curated.tsv         # NEW: Human-validated questionnaire mappings
├── demographics_curated.tsv           # NEW: Human-validated demographic mappings
├── curation_metadata.json             # NEW: Curator info, inter-rater stats
└── README.md                          # Documentation of ground truth sources
```

**Required columns**:
- Entity identifier columns (name, provided IDs)
- `kg_id_groundtruth` - The known-correct KG node ID
- `mapping_source` - How the ground truth was determined (manual, API, etc.)
- `curator_id` - (NEW) Who validated this mapping
- `curation_date` - (NEW) When the validation occurred
- `confidence` - (NEW) Curator's confidence level (high/medium/low)

### 8.2 Ground Truth Validation Tests

```python
class TestGroundTruthIntegrity:
    """Validate ground truth datasets themselves."""

    def test_groundtruth_kg_ids_exist_in_kraken(self, shared_mapper):
        """All ground truth KG IDs should exist in current KRAKEN."""
        df = pd.read_csv(PROJECT_ROOT / "data" / "groundtruth" / "diseases_handcrafted.tsv", sep="\t")

        groundtruth_ids = df["kg_id_groundtruth"].dropna().unique()

        # Query KRAKEN for each ID
        missing = []
        for kg_id in groundtruth_ids:
            if not shared_mapper.linker.kg_id_exists(kg_id):
                missing.append(kg_id)

        assert not missing, f"Ground truth KG IDs not in KRAKEN: {missing}"
```

### 8.3 Per-Provided Validation (from meetings)

> "Even though we have some datasets that are designated ground truth, for any dataset that has provided IDs, we can use those to evaluate how correct the assigned IDs are."

```python
def test_assigned_vs_provided_accuracy(shared_mapper):
    """
    For datasets WITH provided IDs, test if annotation engine
    would have found the same IDs.
    """
    _, stats = shared_mapper.map_dataset_to_kg(
        dataset=PROJECT_ROOT / "data" / "examples" / "metabolites_synthetic.tsv",
        entity_type="metabolite",
        name_column="name",
        provided_id_columns=["INCHIKEY", "HMDB", "KEGG", "PUBCHEM", "CHEBI"],
        annotation_mode="all",  # Force annotation even though IDs provided
    )

    # Compare assigned IDs to provided IDs
    per_provided = stats["performance"]["assigned_ids"].get("per_provided_ids")
    if per_provided:
        print(f"Assigned matches provided: {per_provided['precision']:.2%}")
```

### 8.4 Schema-Driven Validation with Pandera

> **Source**: Incorporates recommendations from `TESTING_STRATEGY_DEPS.md`

Instead of brittle snapshot tests that break when KG canonical IDs change, use **schema contracts** that validate structure:

```python
# src/biomapper2/core/schemas.py
import pandera as pa
from pandera.typing import DataFrame, Series

class BaseEntitySchema(pa.DataFrameModel):
    """Base schema for all entity dataframes."""
    name: Series[str] = pa.Field(coerce=True)
    class Config:
        strict = False  # Allow extra columns

class NormalizedSchema(BaseEntitySchema):
    """Output of Normalizer - validates CURIE format without asserting specific values."""
    curies_provided: Series[object]
    curies_assigned: Series[object]

    @pa.check("curies_provided")
    def validate_curies_format(cls, series):
        """Validate CURIEs have prefix:id format without checking specific values."""
        return series.apply(lambda x: all(":" in c for c in x) if isinstance(x, list) else True)

class ResolvedSchema(NormalizedSchema):
    """Final output - schema validates structure, not exact KG IDs."""
    chosen_kg_id: Series[str] = pa.Field(nullable=True, str_matches=r"^[A-Z0-9]+:[A-Za-z0-9_]+$")
```

**Key Benefit**: Schemas validate *structure*, not *specific values* that may shift when KRAKEN evolves.

### 8.5 Property-Based Testing with Hypothesis

Use property-based testing to fuzz the Normalizer without live API calls:

```python
# tests/property/test_normalizer_prop.py
from hypothesis import given, strategies as st, settings

@settings(derandomize=True)  # Tier A: deterministic in CI
@given(st.lists(st.text(min_size=1, max_size=50)))
def test_normalize_never_crashes(input_list):
    """The normalizer should never raise an exception for string inputs."""
    try:
        normalize_ids(input_list)
    except Exception as e:
        pytest.fail(f"Crashed on input {input_list}: {e}")
```

**CI Configuration** (add to `conftest.py`):
```python
from hypothesis import settings
settings.register_profile("ci", derandomize=True, deadline=None, print_blob=True)
settings.register_profile("dev", max_examples=100)

import os
settings.load_profile("ci" if os.environ.get("CI") else "dev")
```

### 8.6 Validation Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
test = [
    "pytest",
    "pytest-benchmark",
    "pandera>=0.24.0",
    "hypothesis>=6.0.0",
]
```

---

---

# Part IV: Implementation & Operations

---

## 9. Implementation Roadmap

> **Updated 2026-01-20**: Priorities revised based on Q1 milestone focus (questionnaires, demographics, LLM validation) and active Spoke comparison work.

### Phase 1: Foundation (Current → +2 weeks)

| Task | Priority | Effort | Assignee | Status | Notes |
|------|----------|--------|----------|--------|-------|
| Add pytest markers (full set) | High | Low | Drew | ✅ **DONE** (2026-04-14) | All test files annotated |
| Add `--kestrel-url` CLI option to tests | Medium | Low | Drew | ✅ **DONE** (2026-04-14) | Also `--kg-version`, `test_run_metadata` fixture |
| Implement step timing instrumentation | High | Medium | Drew | ✅ **DONE** (2026-04-14) | `tests/test_performance.py` with per-step timings |
| Batch chunking for Kestrel API (#44) | High | Low | Drew | ✅ **DONE** (2026-01-22) | See `docs/plans/2026-01-22-kestrel-api-batching.md` |
| Create baseline KG snapshot for regression | High | Medium | Drew | ⬜ TODO | Support Spoke comparison |
| Enhanced milestone runner with KG param | Medium | Medium | Drew | ⬜ TODO | `scripts/run_kg_comparison.py` |

**Deliverables**:
- ✅ `pyproject.toml` with full marker definitions (unit, component, integration, e2e, performance, kg_regression, slow, requires_api, third_party)
- ✅ `tests/test_performance.py` with per-step timing benchmarks
- ✅ `tests/conftest.py` with `--kestrel-url`, `--kg-version`, `test_run_metadata`
- ⬜ `tests/snapshots/kg_mappings_baseline.json`
- ⬜ Enhanced `scripts/run_milestone_datasets.py` / `scripts/run_kg_comparison.py`

### Phase 2: Validation Infrastructure (+2 → +4 weeks)

| Task | Priority | Effort | Assignee | Notes |
|------|----------|--------|----------|-------|
| Human curation export integration | **High** | Medium | Trent → Drew | Consume Replit app output |
| Ground truth test harness | **High** | Medium | Drew | Run against curated datasets |
| Per-step transition tests | High | Medium | Drew | Isolate failure points |
| Spoke vs non-Spoke comparison script | High | Medium | Drew/Amy | `scripts/run_kg_comparison.py` |
| Field provenance tests | Medium | Low | Drew | Track provided vs assigned |

**Deliverables**:
- `tests/test_groundtruth_validation.py`
- `scripts/run_kg_comparison.py`
- `data/groundtruth/questionnaires_curated.tsv` (first batch from Trent)
- `tests/test_pipeline_integration.py`

### Phase 3: Performance & Regression (+4 → +6 weeks)

| Task | Priority | Effort | Assignee | Notes |
|------|----------|--------|----------|-------|
| KG version diff reporting | High | Medium | Drew | Generate comparison reports |
| Metrics collection architecture | High | Medium | Drew | Feed Visualizer (P/R/F1 heatmaps now shipped; see `visualizer.py`) |
| Step 5 full-pipeline degradation test | Medium | Low | Drew | See Section 5.1 — `test_step5_graceful_degradation` |
| pytest-benchmark integration | Medium | Low | Drew | Statistical rigor for benchmarks |
| Memory profiling tests | Medium | Medium | Drew | Large dataset handling |
| CI performance tracking | Medium | High | Drew | GitHub Actions workflow |

**Deliverables**:
- `tests/test_kg_regression.py` with diff reporting
- `src/biomapper2/core/metrics.py` (structured metrics output)
- ✅ P/R/F1 visualizations (`render_metric_heatmaps`, `render_pr_scatter`) — shipped in PR #65
- GitHub Actions workflow for benchmark tracking
- Performance baseline documentation

### Phase 4: LLM Validation & Advanced (+6 → +8 weeks)

| Task | Priority | Effort | Assignee | Notes |
|------|----------|--------|----------|-------|
| LLM validation layer | **High** | High | Trent → Drew | **Elevated** - P0 for questionnaires |
| Vector search isolated validation | High | High | Trent | Per meeting: validate independently |
| Housekeeping entity test suite | Medium | Low | Drew | Stable entities for baseline |
| Load/stress testing | Medium | Medium | Drew | Post-batch-chunking validation |

**Deliverables**:
- `src/biomapper2/core/llm_validator.py` (proof-of-concept)
- `tests/test_vector_search.py`
- `data/groundtruth/housekeeping_entities.json`
- Load testing documentation

### Dependencies & Critical Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CRITICAL PATH                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1                    Phase 2                    Phase 3-4            │
│  ────────                   ────────                   ─────────            │
│                                                                              │
│  pytest markers ──────────→ transition tests ────────→ regression tests     │
│       │                          │                          │               │
│       │                          ↓                          ↓               │
│       └──────────────────→ ground truth harness ────→ LLM validation        │
│                                  ↑                                          │
│  batch chunking (#44) ──→ large dataset testing                             │
│                                  ↑                                          │
│  step timing ───────────→ metrics architecture ────→ CI tracking            │
│                                  │                                          │
│                                  ↓                                          │
│                           Ashen's visualizations (#46)                      │
│                                                                              │
│  KG snapshot ───────────→ Spoke comparison ────────→ diff reporting         │
│                                                                              │
│  Trent's Replit app ────→ curation export ─────────→ ground truth tests     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Quick Wins (Can Start Immediately)

1. ~~**Add pytest markers**~~ ✅ Done (2026-04-14)
2. ~~**Batch chunking (#44)**~~ ✅ Done (2026-01-22)
3. ~~**Step timing / performance skeleton**~~ ✅ Done (2026-04-14)
4. **KG snapshot creation** - Run milestone runner against production, save results as `tests/snapshots/kg_mappings_baseline.json`
5. **Pydantic models (#2)** - Next assigned issue

---

## 10. CI Policies & Test Metadata

> **Source**: Incorporates policies from `TESTING_PRINCIPLES_ADDENDUM.md`

Determinism tiers, marker definitions, and CI gating commands are defined in **[Section 3: Testing Pyramid Vision](#3-testing-pyramid-vision)**.

### 10.0 Current CI Pipeline (As Implemented)

The GitHub Actions CI workflow (`.github/workflows/ci.yml`) runs on every PR to `main`:

```yaml
# Actual CI steps (Python 3.10 and 3.12)
- name: Run ruff check        # Linting
  run: uv run ruff check

- name: Run black check       # Formatting verification
  run: uv run black --check .

- name: Run pyright           # Type checking
  run: uv run pyright

- name: Run tests             # Tier A tests only (fast, no third_party deps)
  run: uv run pytest -v -m "not requires_api and not third_party and not slow and not performance"
```

**Local equivalent** (run before committing):
```bash
./scripts/check.sh    # Runs all 4 checks in same order
./scripts/fix.sh      # Auto-fix ruff + black issues
```

**Current markers** (as of 2026-04-14 — full set implemented):
```ini
# pyproject.toml [tool.pytest.ini_options]
markers = [
    "unit: Tier A - Pure functions, no external dependencies",
    "component: Tier A - Single pipeline step, may mock dependencies",
    "integration: Tier B/C - Multiple steps or real API calls",
    "e2e: Tier C - Full end-to-end pipeline with live KG",
    "performance: Tier B - Timing and memory benchmarks",
    "kg_regression: Tier B/C - Knowledge graph version change detection",
    "slow: Tier B - Tests taking >10 seconds",
    "requires_api: Tier C - Requires live Kestrel API connection",
    "third_party: Tier C - Requires third-party APIs we don't own or control",
]
```

> **Status**: Full marker set and per-file annotations implemented 2026-04-14. CI command updated to use marker-based filtering.
> See `tests/conftest.py` for `--kestrel-url` / `--kg-version` CLI options and `test_run_metadata` fixture.

### 10.1 Test Report Metadata

Every test run should capture:

```python
# tests/conftest.py
import pytest
import subprocess

@pytest.fixture(scope="session", autouse=True)
def test_run_metadata(request):
    """Capture metadata for test reports."""
    metadata = {
        "kg_version": os.environ.get("KG_VERSION", "unknown"),
        "kestrel_url": os.environ.get("KESTREL_API_URL", "default"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "timestamp": datetime.utcnow().isoformat(),
    }
    # Attach to pytest report
    request.config._test_metadata = metadata
```

### 10.2 LLM Validation Requirements (P0)

When implementing `ValidationJudge`, MUST include:

```python
class ValidationJudge:
    # REQUIRED: Pin these for reproducibility
    MODEL_NAME = "claude-3-5-sonnet-20241022"
    PROMPT_VERSION = "v1.0.0"

    PROMPT_TEMPLATE = """
    Entity: {entity_name}
    Mapped KG Node: {kg_id}
    KG Node Label: {kg_label}

    Rate the semantic match from 0.0 (no match) to 1.0 (exact match).
    Respond with only a number.
    """

    def evaluate_mapping(self, entity_name: str, kg_id: str) -> float:
        # Log model, prompt version, inputs for audit trail
        ...
```

**Gating**: LLM validation tests are Tier C unless run against recorded/cached responses.

---

## Appendix: Key Changes Summary

### 2026-01-20 Revision

| Section | Change | Rationale |
|---------|--------|-----------|
| 2. Gap Analysis | LLM validation elevated to High priority | Q1 focus on questionnaires/demographics |
| 6. KG Regression | Added Milestone Runner integration (6.0) | Leverage existing infrastructure for Spoke comparison |
| 7. Performance | Added Metrics Collection Architecture (7.0) | Explicit checkpoints for profiling |
| 8. Ground Truth | Added Human Curation Loop (8.0), Schema-Driven Validation (8.4-8.6) | Document Trent's Replit app; integrate Pandera/Hypothesis |
| 9. Roadmap | Revised with assignees, dependencies, quick wins | Align with current team priorities |
| 10. CI Policies | Consolidated from Addendum | Single source of truth |

### 2026-01-21 Consolidation

- Merged `TESTING_STRATEGY_DEPS.md` content into Section 8.4-8.6
- Merged `TESTING_PRINCIPLES_ADDENDUM.md` content into Section 10
- Consolidated marker definitions to Section 3 (single source of truth)
- Added Part I-IV navigation structure
- Removed duplicate CI commands and marker definitions (~115 lines saved)

### 2026-01-21 Final Consolidation

- **Section 3**: Now single source of truth for determinism tiers, markers, and CI commands (merged from former Appendix A)
- **Section 7.4**: Added Metrics Parsing & Analysis (TBD) to address metrics consumption gap; links to #46 visualization
- **Section 10**: Removed duplicate determinism tiers table; now references Section 3
- **Appendix A**: Removed (content merged into Section 3)
- Renumbered Section 10 subsections (10.2 → 10.1, 10.3 → 10.2)

### 2026-04-14 Markers, Performance Tests, and Conftest Infrastructure

- **Section 1**: Updated test coverage table to 177 tests with per-file marker assignments
- **Section 9**: Marked Phase 1 tasks as complete (markers, step timing, CLI options)
- **Section 10.0**: Updated CI command to use new marker-based filtering; documented full marker set as implemented
- **New**: `tests/test_performance.py` — per-step timing benchmarks for OLink proteins and synthetic metabolites
- **New**: `tests/conftest.py` — `--kestrel-url` / `--kg-version` CLI options, `test_run_metadata` autouse fixture, `shared_mapper` respects URL override
- **Markers added to all test files**: `unit` (normalizer, batching, api_unit, dataset_analysis, visualizer, validators, entity_model, api), `requires_api` (entity_kg_mapping, dataset_kg_mapping, performance, api integration tests, mw end-to-end), `slow` (dataset_kg_mapping, example_scripts, heavy performance tests)

### 2026-05-05 Step 5 Integration and Repo Sync

- **Section 1**: Updated test count to 193; added `test_equivalent_ids.py` row; updated `test_visualizer.py` count for P/R/F1 tests; Step 5 timing noted in `test_performance.py` row
- **Section 1 pipeline diagram**: Extended to 5 steps (added Equivalent IDs enrichment)
- **Section 1 step table**: Added Step 5 row (`kg_equivalent_ids`, non-critical)
- **Section 2**: Removed "Performance profiling" gap (now covered); added "Step 5 full-pipeline degradation" gap
- **Section 5.1**: Added `test_resolution_to_equivalent_ids` and `test_step5_graceful_degradation` to transition tests
- **Section 7.0**: Added checkpoint E1/E2/E3 for Equivalent IDs step; updated `PipelineMetrics.timing` key list
- **Section 7.1**: Replaced aspirational 4-step example with actual implementation; added `clear_kestrel_cache` fixture, Step 5 timing, and observed cold-cache benchmark table
- **Section 9 Phase 3**: Marked P/R/F1 visualizations as shipped (PR #65); added Step 5 degradation test row
- **`tests/test_equivalent_ids.py`**: Added `pytestmark = pytest.mark.unit`
- **`tests/test_performance.py`**: Added `clear_kestrel_cache` autouse fixture (prevents cache warm-up from alphabetically earlier tests inflating timings ~25×); added Step 5 (`equivalent_ids`) timing block and assertions; fixed `get_standard_prefix(None)` bug
- **`tests/test_entity_kg_mapping.py`**: Added `kg_equivalent_ids` assertions — structural check (`isinstance dict`) in `test_map_entity_basic`; non-empty check in `test_map_entity_multiple_identifiers` (aspirin reliably maps); creatinine excluded from non-empty check as it has no KRAKEN node
- **`tests/conftest.py`**: Fixed pyright error (`shared_mapper` return type `Iterator[Mapper]`)
- **Repos**: kestrel and kraken pulled to `origin/main` (kestrel +149 commits incl. subgraph endpoint; kraken +36 commits incl. metagraph and shuffle)
- **Performance finding**: Step 5 is the current cold-start bottleneck at ~1594ms (two batched `/get-nodes` calls for ~2921 unique protein KG IDs), exceeding linking (~666ms)

### 2026-01-22 CI/CD Documentation

- **Section 1**: Added `test_batching.py` to test coverage table (Kestrel API batching)
- **Section 9**: Marked #44 (batch chunking) as IN PROGRESS with link to implementation plan
- **Section 10.0**: NEW - Documented actual CI pipeline (ruff → black → pyright → pytest)
- Clarified that comprehensive markers in Section 3 are aspirational; current markers are `integration` and `third_party` only

---

## Summary

This testing strategy addresses the unique challenges of biomapper2:

1. **Step-by-step visibility** - Tests validate each pipeline step independently
2. **KG evolution** - Regression tests detect when KRAKEN changes affect mappings
3. **Performance monitoring** - Per-step timing with explicit checkpoints identifies bottlenecks
4. **Ground truth management** - Human curation workflow + schema-driven validation
5. **Flexible execution** - Markers with determinism tiers allow fast CI vs thorough testing
6. **LLM validation support** - Infrastructure for questionnaire/demographic mappings (P0)
7. **Metrics architecture** - Structured collection and parsing feeds visualizations (#46) and CI pipelines
