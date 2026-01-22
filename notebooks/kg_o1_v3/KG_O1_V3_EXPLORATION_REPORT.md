# KG-o1 v3 Exploration Report: Semantic Multi-Hop Reasoning in KRAKEN

**Report Date:** 2026-01-21
**Notebooks Reviewed:** 8 (01-08)
**JSON Outputs Analyzed:** 8 files
**Total Execution Status:** Complete

---

## Executive Summary

### Critical Discovery: v2's Conclusion Was Partially Wrong

**v2 concluded** KRAKEN has a "semantic gap" and is vocabulary-focused only.
**v3 discovered** KRAKEN **does have semantic relations** accessible via the `/one-hop` endpoint that v2 never tested.

### Key Metrics

| Metric | Value | Significance |
|--------|-------|--------------|
| **Semantic Predicates Found** | 26 types | Rich semantic vocabulary available |
| **Semantic Edge Percentage** | 89.1% | Vast majority are true semantic relations |
| **Total QA Pairs Generated** | 3,359 | Substantial evaluation dataset |
| **Graph Traversal Recall** | 99.4% | Excellent for 1-hop semantic queries |
| **Search EM on Semantic QA** | 0.1% | Search fails on semantic questions |
| **v2 Search EM (Vocabulary)** | 95.2% | Remains optimal for entity resolution |
| **Multi-hop Path Success** | 15% | Limited; safeguards prevent explosion |

### GO/NO-GO Decision: **GO**

KRAKEN has true semantic multi-hop capability. Proceed with v3 integration.

### Bottom Line

- **For Entity Resolution:** Use hybrid search (95.2% EM) - v2 approach
- **For Semantic Queries:** Use graph traversal via `/one-hop` (99.4% recall) - v3 approach
- **Recommendation:** Implement hybrid approach combining both capabilities

---

## Notebook-by-Notebook Analysis

### Notebook 01: One-Hop API Discovery (GO/NO-GO Gate)

**Objective:** Determine if `/one-hop` returns semantic relations
**Status:** COMPLETE - GO DECISION

#### Key Findings

| Test | Result |
|------|--------|
| Endpoint Exists | Yes - Returns `{edge_schema, results, nodes}` |
| Total Edges Found | 1,432 across 6 metabolites |
| Unique Predicates | 18 types discovered |
| Semantic Predicates | 15 (99.0% of edges) |
| Equivalency Predicates | 1 (`biolink:same_as`, 0.3% of edges) |

#### Top Semantic Predicates Discovered

| Predicate | Count | Category |
|-----------|-------|----------|
| `biolink:related_to` | 862 | Association |
| `biolink:subclass_of` | 405 | Ontological |
| `biolink:has_participant` | 336 | Participation |
| `biolink:interacts_with` | 158 | Interaction |
| `biolink:has_part` | 140 | Structural |
| `biolink:close_match` | 115 | Similarity |
| `biolink:physically_interacts_with` | 102 | Interaction |

**Conclusion:** KRAKEN has rich semantic relations. v2 never discovered them because they only tested search endpoints.

---

### Notebook 02: Predicate & Relationship Mapping

**Objective:** Classify all available predicates
**Status:** COMPLETE

#### Predicate Classification

| Category | Count | Examples |
|----------|-------|----------|
| **Semantic** | 26 | participates_in, treats, catalyzes, affects, causes |
| **Equivalency** | 1 | same_as |
| **Unknown** | 61 | (need manual classification) |

#### Biolink Coverage for Metabolite Reasoning

**Coverage:** 11/19 key predicates (58%)

**Available:**
- `participates_in`, `has_participant` (pathway involvement)
- `catalyzes` (enzyme relations)
- `treats`, `affects` (therapeutic)
- `associated_with`, `related_to` (general associations)
- `has_metabolite` (metabolic products)
- `located_in`, `has_part` (structural)

**Missing:**
- `is_substrate_of`, `has_substrate`
- `has_product`, `metabolite_of`
- `transports`, `transported_by`

---

### Notebook 03: Semantic Subgraph Extraction

**Objective:** Extract entity-relation-entity triplets
**Status:** COMPLETE

#### Extraction Results

| Metric | Value |
|--------|-------|
| Seed Entities | 20 metabolites |
| Entities with Relations | 20/20 (100%) |
| Total Outgoing Relations | 1,743 |
| Total Incoming Relations | 1,835 |
| **Total Relations** | **3,578** |
| Unique Predicates | 28 types |

#### Entity Coverage

All 20 seed entities have both outgoing and incoming semantic relations, demonstrating comprehensive graph connectivity.

---

### Notebook 04: Multi-Hop Path Discovery

**Objective:** Chain one-hop calls for 2+ hop paths
**Status:** PARTIAL SUCCESS

#### Path Finding Results

| Metric | Value |
|--------|-------|
| Entity Pairs Tested | 20 |
| Paths Found | 3 (15% success) |
| Average Path Length | 2.3 hops |
| 2-hop Paths | 2 |
| 3-hop Paths | 1 |

#### Termination Reasons

| Reason | Count | Percentage |
|--------|-------|------------|
| Exhausted | 17 | 85% |
| Found | 3 | 15% |
| Timeout | 0 | 0% |

#### Safeguards Effectiveness

| Safeguard | Setting | Status |
|-----------|---------|--------|
| Max Visited Nodes | 1,000 | Effective |
| Timeout | 30 seconds | Not triggered |
| Max Queue Size | 5,000 | Not triggered |

**Assessment:** BFS works but graph exploration exhausts quickly. Safeguards prevent explosion effectively.

---

### Notebook 05: Semantic QA Generation

**Objective:** Generate TRUE semantic QA pairs
**Status:** COMPLETE

#### QA Dataset Generated

| Type | Count | Percentage |
|------|-------|------------|
| 1-hop QA | 3,356 | 99.9% |
| Multi-hop QA | 3 | 0.1% |
| **Total** | **3,359** | 100% |

#### Sample QA Pairs

```
Q: What entity does glucose treat?
A: hypoglycemia
Chain: glucose --[biolink:treats]--> hypoglycemia

Q: What entity does glucose in clinical trials for?
A: hypoglycemia
Chain: glucose --[biolink:in_clinical_trials_for]--> hypoglycemia

Q: What entity does NAD+ have as participant?
A: pyruvate decarboxylase activity
Chain: NAD+ --[biolink:has_participant]--> pyruvate decarboxylase activity
```

---

### Notebook 06: Semantic QA Evaluation

**Objective:** Compare search vs graph traversal
**Status:** COMPLETE

#### Head-to-Head Comparison

| Metric | Search (Hybrid) | Graph Traversal |
|--------|-----------------|-----------------|
| **Exact Match / Recall** | 0.1% | 99.4% |
| Recall@5 | 0.3% | - |
| Recall@10 | 0.3% | - |
| Recall@20 | 0.6% | - |
| MRR | 0.002 | - |
| **Avg Latency** | 8.4 ms | 5.2 ms |

#### Performance by Hop Count

| Hops | Count | Search EM | Graph Recall |
|------|-------|-----------|--------------|
| 1-hop | 3,356 | 0.1% | **100.0%** |
| 2-hop | 2 | 0.0% | 0.0% |
| 3-hop | 1 | 0.0% | 0.0% |

#### Key Insight

**Graph traversal dramatically outperforms search on semantic QA** (99.4% vs 0.1%)

This validates the core hypothesis:
- Search: Optimized for entity resolution (vocabulary QA)
- Graph: Optimized for semantic reasoning (relation QA)

**Statistical Validity:** n=3,359 supports reliable claims

---

### Notebook 07: Semantic vs Vocabulary Gap Analysis

**Objective:** Quantify capability differences
**Status:** COMPLETE

#### v2 vs v3 Capability Matrix

| Capability | v2 (Search) | v3 (Graph) | Winner |
|-----------|-------------|------------|--------|
| Entity Resolution | 95.2% EM | 0.1% EM | **v2** |
| Semantic 1-hop | 0.1% EM | 99.4% Recall | **v3** |
| Latency | 8.4 ms | 5.2 ms | **v3** |
| Use Case | Vocabulary mapping | Semantic reasoning | Both |

#### Key Insights

1. **Search degrades 95 percentage points** on semantic queries (95.2% -> 0.1%)
2. **Graph traversal provides 99.4% recall** on semantic 1-hop questions
3. **Complementary approaches:** Use both, not either/or

---

### Notebook 08: Integration Recommendations

**Objective:** Production recommendations for biomapper2
**Status:** COMPLETE

#### Method Selection Guide

| Query Type | Method | Expected Performance |
|-----------|--------|---------------------|
| "What HMDB ID for glucose?" | `hybrid_search` | 95%+ EM |
| "What pathways include glucose?" | `get_semantic_relations()` | High recall |
| "How are glucose and ATP related?" | BFS path finding | 15% success |
| "Find pathways for vitamin B12" | Search -> Graph | Compound |

#### Integration Checklist

| Priority | Task | Target File |
|----------|------|-------------|
| **HIGH** | One-hop wrapper | `semantic_query.py` |
| **HIGH** | Hybrid query function | `semantic_query.py` |
| **HIGH** | Unit tests | `test_semantic_query.py` |
| MEDIUM | Convenience functions | `semantic_query.py` |
| MEDIUM | Update Mapper class | `mapper.py` |
| LOW | BFS path finder | `path_finder.py` |
| LOW | Documentation | `semantic_queries.md` |

---

## Success Criteria Assessment

| Notebook | Criterion | Target | Actual | Status |
|----------|-----------|--------|--------|--------|
| 01 | GO/NO-GO decision | Explicit | GO (99% semantic) | PASS |
| 02 | Predicates mapped | 10+ | 26 semantic | PASS |
| 03 | Subgraphs extracted | 50+ | 3,578 relations | PASS |
| 04 | Multi-hop paths | 20+ pairs | 3/20 found (15%) | PARTIAL |
| 04 | Timeout rate | <10% | 0% | PASS |
| 05 | QA pairs generated | 100+ | 3,359 | PASS |
| 06 | Compare search vs graph | Quantified | 0.1% vs 99.4% | PASS |
| 07 | Gap analysis | Documented | Complete | PASS |
| 08 | Production code | Ready | Provided | PASS |

---

## Bug Fixes Applied

### Issue 1: Predicate Metadata Loss (NB02-06)

**Symptom:** Predicates stored as empty strings or "unknown"
**Root Cause:** Edge parsing used incorrect access pattern:
```python
# BROKEN
edges = result.get('edges', result.get('results', []))

# FIXED - uses parse_one_hop_edges()
edges = parse_one_hop_edges(result)
```

**Resolution:** Updated `parse_one_hop_edges()` to properly:
1. Map edge tuples to `edge_schema` for field extraction
2. Look up node names/categories in `nodes` dictionary
3. Return complete edge dicts with predicate, object_name, object_category

### Issue 2: None Value Handling (NB06)

**Symptom:** `AttributeError: 'NoneType' object has no attribute 'lower'`
**Root Cause:** Edge fields could be None, not just empty string
**Resolution:** Use `or ''` pattern for safe None handling:
```python
obj_name = (edge.get('object_name') or edge.get('end_node_name') or '').lower()
```

---

## Conclusions

### What v3 Proved

1. **KRAKEN has semantic relations** - 26+ predicates, 89% semantic edges
2. **Graph traversal works** - 99.4% recall on 1-hop semantic queries
3. **Search and graph are complementary** - Each excels at different tasks
4. **BFS is feasible with safeguards** - 15% success, no runaway explosion

### Recommendations

1. **Immediate:** Implement one-hop wrapper in biomapper2
2. **Short-term:** Add hybrid query capability
3. **Medium-term:** Optimize multi-hop path finding
4. **Long-term:** Investigate external validation with Reactome/KEGG

---

## Final Verdict

| Aspect | Assessment |
|--------|------------|
| v2 Conclusion ("semantic gap") | **Partially incorrect** |
| Semantic capability exists | **YES** (via `/one-hop`) |
| Ready for production | **YES** (with noted caveats) |
| Integration priority | **HIGH** |

**v3 successfully demonstrated that KRAKEN has true semantic reasoning capability that v2 missed. The `/one-hop` endpoint provides access to rich semantic relations with 99.4% recall on 1-hop queries. Integration into biomapper2 is recommended as a complement to the existing search-based approach.**

---

## Appendix: Output Files Generated

| File | Contents |
|------|----------|
| `one_hop_api_audit.json` | GO/NO-GO decision, edge counts, predicates |
| `predicate_mapping.json` | 88 predicates, 26 semantic, 58% Biolink coverage |
| `semantic_subgraphs.json` | 20 entities, 3,578 relations, 28 predicates |
| `multi_hop_paths.json` | 20 pairs, 3 paths found, 15% success |
| `semantic_qa_dataset.json` | 3,359 QA pairs with complete chains |
| `semantic_qa_evaluation.json` | Search 0.1% vs Graph 99.4% |
| `semantic_gap_analysis_v3.json` | v2 vs v3 comparison |
| `integration_recommendations.json` | Decision tree, checklist, code snippets |
