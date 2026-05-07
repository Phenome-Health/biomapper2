"""Performance profiling tests for the biomapper2 pipeline.

These tests time each pipeline step individually to identify bottlenecks.
They are marked `performance` + `requires_api` and run in nightly CI only.

Usage:
    uv run pytest tests/test_performance.py -v -s          # verbose with timing output
    uv run pytest -m performance                            # all performance tests
    uv run pytest tests/test_performance.py --kestrel-url=https://staging.example.com/api
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from biomapper2.config import CACHE_DIR, PROJECT_ROOT
from biomapper2.mapper import Mapper

pytestmark = [pytest.mark.performance, pytest.mark.requires_api]

DATA_DIR = Path(PROJECT_ROOT) / "data" / "examples"

_KESTREL_CACHE = Path(CACHE_DIR) / "kestrel_http.sqlite"


@pytest.fixture(scope="session", autouse=True)
def clear_kestrel_cache():
    """Delete the Kestrel HTTP cache before performance tests run.

    Other test files (test_dataset_kg_mapping, etc.) run alphabetically before
    this one and warm the cache. Without this fixture every 'API call' is a SQLite
    read, making timings meaningless for measuring real network latency.
    """
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

    def __str__(self) -> str:
        return f"{self.step:<20} {self.duration_ms:8.1f}ms  ({self.ms_per_item:.2f}ms/item)"


def _time_pipeline_steps(
    mapper: Mapper,
    df: pd.DataFrame,
    entity_type: str,
    name_field: str,
    provided_id_fields: list[str],
    array_delimiters: list[str],
) -> list[StepTiming]:
    """Run each pipeline step individually and return per-step timings."""
    timings: list[StepTiming] = []
    n = len(df)
    category = mapper.biolink_client.standardize_entity_type(entity_type)
    prefixes = mapper.normalizer.get_standard_prefix(None)

    # Step 1: Annotation
    t0 = time.perf_counter()
    annotation_df = mapper.annotation_engine.annotate(
        item=df,
        name_field=name_field,
        provided_id_fields=provided_id_fields,
        category=category,
        prefixes=prefixes,
        mode="missing",
    )
    timings.append(StepTiming("annotation", (time.perf_counter() - t0) * 1000, n))
    df = df.join(annotation_df)

    # Step 2: Normalization
    t0 = time.perf_counter()
    normalization_df = mapper.normalizer.normalize(
        item=df,
        provided_id_fields=provided_id_fields,
        array_delimiters=array_delimiters,
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


def _print_timing_report(label: str, timings: list[StepTiming]) -> None:
    total_ms = sum(t.duration_ms for t in timings)
    n = timings[0].items_processed if timings else 0
    print(f"\n=== Performance Report: {label} ({n} items) ===")
    for t in timings:
        print(f"  {t}")
    print(f"  {'TOTAL':<20} {total_ms:8.1f}ms  ({total_ms / n:.2f}ms/item)")


def _store_timings(config: pytest.Config, label: str, timings: list[StepTiming]) -> None:
    """Serialize timings onto config for the session-end JSON report."""
    n = timings[0].items_processed if timings else 0
    config._performance_timings[label] = {  # type: ignore[attr-defined]
        "items": n,
        "total_ms": round(sum(t.duration_ms for t in timings), 1),
        "steps": [
            {"step": t.step, "duration_ms": round(t.duration_ms, 1), "ms_per_item": round(t.ms_per_item, 3)}
            for t in timings
        ],
    }


class TestPipelinePerformance:
    """Per-step timing benchmarks for the biomapper2 pipeline."""

    @pytest.mark.slow
    def test_step_timings_olink_proteins(self, request: pytest.FixtureRequest, shared_mapper: Mapper) -> None:
        """Profile each pipeline step for the OLink protein dataset (~2900 items)."""
        tsv_path = DATA_DIR / "olink_protein_metadata.tsv"
        if not tsv_path.exists():
            pytest.skip(f"Dataset not found: {tsv_path}")

        df = pd.read_csv(tsv_path, sep="\t")
        timings = _time_pipeline_steps(
            mapper=shared_mapper,
            df=df,
            entity_type="protein",
            name_field="Assay",
            provided_id_fields=["UniProt"],
            array_delimiters=["_"],
        )
        _print_timing_report("olink_proteins", timings)
        _store_timings(request.config, "olink_proteins", timings)

        by_step = {t.step: t for t in timings}
        assert by_step["annotation"].duration_ms < 10_000, "Annotation >10s for protein dataset"
        assert by_step["normalization"].duration_ms < 5_000, "Normalization >5s for protein dataset"
        assert by_step["linking"].duration_ms < 10_000, "Linking >10s for protein dataset"
        assert by_step["resolution"].duration_ms < 2_000, "Resolution >2s for protein dataset"
        assert by_step["equivalent_ids"].duration_ms < 10_000, "Equivalent IDs >10s for protein dataset"

    @pytest.mark.slow
    def test_step_timings_olink_proteins_name_only(self, request: pytest.FixtureRequest, shared_mapper: Mapper) -> None:
        """Profile each pipeline step for OLink proteins with no provided IDs (annotation runs on all rows)."""
        tsv_path = DATA_DIR / "olink_protein_metadata.tsv"
        if not tsv_path.exists():
            pytest.skip(f"Dataset not found: {tsv_path}")

        df = pd.read_csv(tsv_path, sep="\t")[["Assay"]]
        timings = _time_pipeline_steps(
            mapper=shared_mapper,
            df=df,
            entity_type="protein",
            name_field="Assay",
            provided_id_fields=[],
            array_delimiters=[],
        )
        _print_timing_report("olink_proteins_name_only", timings)
        _store_timings(request.config, "olink_proteins_name_only", timings)

        by_step = {t.step: t for t in timings}
        assert by_step["annotation"].duration_ms < 60_000, "Annotation >60s for name-only protein dataset"
        assert by_step["normalization"].duration_ms < 5_000, "Normalization >5s for protein dataset"
        assert by_step["linking"].duration_ms < 10_000, "Linking >10s for protein dataset"
        assert by_step["resolution"].duration_ms < 2_000, "Resolution >2s for protein dataset"
        assert by_step["equivalent_ids"].duration_ms < 10_000, "Equivalent IDs >10s for protein dataset"

    @pytest.mark.slow
    def test_step_timings_metabolites_synthetic(self, request: pytest.FixtureRequest, shared_mapper: Mapper) -> None:
        """Profile each pipeline step for the synthetic metabolite dataset."""
        tsv_path = DATA_DIR / "metabolites_synthetic.tsv"
        if not tsv_path.exists():
            pytest.skip(f"Dataset not found: {tsv_path}")

        df = pd.read_csv(tsv_path, sep="\t")
        timings = _time_pipeline_steps(
            mapper=shared_mapper,
            df=df,
            entity_type="metabolite",
            name_field="name",
            provided_id_fields=["INCHIKEY", "HMDB", "KEGG", "PUBCHEM", "CHEBI"],
            array_delimiters=[",", ";"],
        )
        _print_timing_report("metabolites_synthetic", timings)
        _store_timings(request.config, "metabolites_synthetic", timings)

        by_step = {t.step: t for t in timings}
        assert by_step["normalization"].duration_ms < 5_000, "Normalization >5s for metabolite dataset"
        assert by_step["linking"].duration_ms < 10_000, "Linking >10s for metabolite dataset"
        assert by_step["equivalent_ids"].duration_ms < 10_000, "Equivalent IDs >10s for metabolite dataset"

    def test_normalizer_throughput_metabolites(self, request: pytest.FixtureRequest, shared_mapper: Mapper) -> None:
        """Normalizer alone on the metabolite dataset — no API calls."""
        tsv_path = DATA_DIR / "metabolites_synthetic.tsv"
        if not tsv_path.exists():
            pytest.skip(f"Dataset not found: {tsv_path}")

        df = pd.read_csv(tsv_path, sep="\t")
        n = len(df)

        t0 = time.perf_counter()
        shared_mapper.normalizer.normalize(
            item=df,
            provided_id_fields=["INCHIKEY", "HMDB", "KEGG", "PUBCHEM", "CHEBI"],
            array_delimiters=[",", ";"],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\nNormalizer throughput: {elapsed_ms:.1f}ms for {n} items ({elapsed_ms / n:.2f}ms/item)")
        _store_timings(
            request.config,
            "normalizer_throughput_metabolites",
            [StepTiming("normalization", elapsed_ms, n)],
        )
        assert elapsed_ms < 5_000, f"Normalizer took {elapsed_ms:.0f}ms — expected <5s for {n} items"
