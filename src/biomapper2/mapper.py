"""
Main mapper module for entity and dataset knowledge graph mapping.

Provides the Mapper class for harmonizing biological entities to knowledge graphs
through annotation, normalization, linking, and resolution steps.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .biolink_client import BiolinkClient
from .config import PROJECT_ROOT, TIER_B_ENABLED, get_kestrel_api_url
from .core.analysis import analyze_dataset_mapping
from .core.annotation_engine import AnnotationEngine
from .core.certificate import (
    ResolutionCertificate,
    derive_chosen_kg_id_review,
    issue,
    node_blocks_from_equivalent_ids,
)
from .core.linker import Linker
from .core.normalizer import Normalizer
from .core.resolver import Resolver
from .models import Entity
from .provenance import build_run_provenance
from .utils import AnnotationMode, setup_logging

setup_logging()


def _scalar_or_none(value: Any) -> Any:
    """Normalize a pandas missing value to None.

    ``pd.NA``/``NaN`` is not ``None``, and the certificate's guards are identity checks against
    None. Without this a missing ``chosen_kg_id`` reads as a committed node whose id happens to be
    NaN, which is the quiet version of the bug the certificate exists to remove.
    """
    return None if value is None or (isinstance(value, float) and value != value) or value is pd.NA else value


class Mapper:
    """
    Maps biological entities and datasets of entities to knowledge graph nodes.

    Performs four-step mapping pipeline:
    1. Annotation - assign additional vocab IDs via external APIs
    2. Normalization - convert un-normalized vocab IDs to Biolink-standard curies
    3. Linking - map curies to knowledge graph nodes
    4. Resolution - resolve one-to-many mappings
    """

    def __init__(self, biolink_version: str | None = None):
        # Instantiate the mapping modules (should only be done once, up front)
        self.biolink_client = BiolinkClient(biolink_version=biolink_version)
        self.annotation_engine = AnnotationEngine(biolink_client=self.biolink_client)
        self.normalizer = Normalizer(biolink_client=self.biolink_client)
        self.linker = Linker()
        self.resolver = Resolver(linker=self.linker, biolink_client=self.biolink_client)
        self.tier_b = self._build_tier_b()

    @staticmethod
    def _build_tier_b():
        """The opt-in independent-structure lookup, or None.

        Constructed only when enabled, so a default run holds no session against Metabolomics
        Workbench or PubChem and cannot drift into making calls.
        """
        if not TIER_B_ENABLED:
            return None
        from .core.tier_b import IndependentStructureLookup

        logging.info("Tier B independent structure evidence is ENABLED for this run")
        return IndependentStructureLookup()

    def _issue_certificate(
        self,
        *,
        query_name: str | None,
        category: str | None,
        chosen_kg_id: str | None,
        kg_equivalent_ids: dict[str, list[str]] | None,
        equivalent_ids_lookup_ok: bool,
        selection_conflict: str | None,
        kg_ids_assigned: dict[str, dict[str, list[str]]] | None,
    ) -> ResolutionCertificate:
        """Assemble one certificate. Shared by both emission paths so they cannot drift apart.

        Tier A reads ``kg_equivalent_ids`` and nothing else — the structure the GRAPH asserts for
        the node the pipeline already committed. Tier B, when enabled, resolves the QUERY NAME (not
        the node's name) against an independent registry.
        """
        assigned = kg_ids_assigned or {}
        # Which annotators supplied the committed node. Needed for the independence claim (L26):
        # a Tier B verdict from the same registry that produced the winning candidate is circular.
        committed_sources = {
            annotator for annotator, kg_ids in assigned.items() if chosen_kg_id is not None and chosen_kg_id in kg_ids
        }
        # Only look up rows the certificate can actually be about. Calling first and discarding
        # later would spend a throttled external round trip on every gene symbol and every
        # uncommitted row, and would put those names into the resolution-rate denominator that
        # must accompany each operating point -- diluting it with names never in scope.
        # Must be the SAME predicate ``issue()`` uses for the population, including the lookup-ok
        # term: during a /get-nodes outage a committed small-molecule row is still out of scope,
        # and looking it up anyway would pay a throttled round trip whose result is discarded and
        # would put the name into the resolution-rate denominator this scoping exists to protect.
        #
        # The blocks term is on the same footing (L21). A committed node the GRAPH asserts no
        # InChIKey for is ``structure_absent`` -> ``unavailable``, and no Tier B verdict can move it:
        # there is nothing to compare against. Looking it up anyway would (a) spend a rate-limited
        # MW/PubChem round trip per unverifiable row, on exactly the population that is largest where
        # the KG is thinnest, (b) put those names into the Tier B resolution-rate denominator that
        # travels with every operating point, and (c) hang ``independent_source`` /
        # ``independent_inchikey_block`` on a row where no structural comparison was made -- a
        # certificate that shows independent evidence beside a state that ignored it.
        is_small_molecule = self.resolver.is_small_molecule(category)
        in_population = (
            is_small_molecule
            and chosen_kg_id is not None
            and equivalent_ids_lookup_ok
            # Same producer ``issue()`` uses, so "the graph asserts a structure" cannot mean two
            # things: a bare ``.get("INCHIKEY")`` truth test would count a whitespace-only key.
            and bool(node_blocks_from_equivalent_ids(kg_equivalent_ids))
        )
        tier_b_result = self.tier_b.lookup(query_name) if (self.tier_b is not None and in_population) else None
        return issue(
            chosen_kg_id=chosen_kg_id,
            is_small_molecule=is_small_molecule,
            kg_equivalent_ids=kg_equivalent_ids,
            equivalent_ids_lookup_ok=equivalent_ids_lookup_ok,
            selection_conflict=selection_conflict,
            tier_b=tier_b_result,
            committed_node_sources=committed_sources,
        )

    def map_entity_to_kg(
        self,
        item: pd.Series | dict[str, Any],
        name_field: str,
        provided_id_fields: list[str],
        entity_type: str,
        vocab: str | list[str] | None = None,
        array_delimiters: list[str] | None = None,
        stop_on_invalid_id: bool = False,
        annotation_mode: AnnotationMode = "missing",
        annotators: list[str] | None = None,
        prefer_human: bool = True,
        prefer_canonical: bool = True,
    ) -> pd.Series | dict[str, Any]:
        """
        Map a single entity to knowledge graph nodes.

        Args:
            item: Entity with name and ID fields
            name_field: Field containing entity name
            provided_id_fields: List of fields containing vocab identifiers
            entity_type: Type of entity (e.g., 'metabolite', 'protein')
            vocab: Allowed vocab name(s) to map to (e.g., 'refmet', 'mondo')
            array_delimiters: Characters used to split delimited ID strings (default: [',', ';'])
            stop_on_invalid_id: Halt execution on invalid IDs (default: False)
            annotation_mode: When to annotate
                - 'all': Annotate all entities
                - 'missing': Only annotate entities without provided_ids (default)
                - 'none': Skip annotation entirely (returns empty)
            annotators: Optional list of annotators to use (by slug). If None, annotators are selected automatically.

        Returns:
            Mapped entity with added fields: curies, kg_ids, chosen_kg_id, etc.
        """
        logging.debug(f"Item at beginning of map_entity_to_kg() is {item}")
        array_delimiters = array_delimiters if array_delimiters is not None else [",", ";"]
        input_is_series = isinstance(item, pd.Series)
        entity = Entity.from_input(item, name_field=name_field)

        # Validate/standardize the input entity type and vocab(s) on Biolink
        entity_type = self.biolink_client.standardize_entity_type(entity_type)
        prefixes = self.normalizer.get_standard_prefix(vocab)

        # Do Step 1: annotate with vocab IDs
        annotation_result = self.annotation_engine.annotate(
            item=entity.to_series(),
            name_field=name_field,
            provided_id_fields=provided_id_fields,
            category=entity_type,
            prefixes=prefixes,
            mode=annotation_mode,
            annotators=annotators,
            prefer_human=prefer_human,
            prefer_canonical=prefer_canonical,
        )
        assert isinstance(annotation_result, pd.Series)
        entity = entity.update_from(annotation_result)

        # Do Step 2: normalize vocab IDs to form proper curies
        normalization_result = self.normalizer.normalize(
            item=entity.to_series(),
            provided_id_fields=provided_id_fields,
            array_delimiters=array_delimiters,
            stop_on_invalid_id=stop_on_invalid_id,
        )
        assert isinstance(normalization_result, pd.Series)
        entity = entity.update_from(normalization_result)

        # Do Step 3: link curies to KG nodes
        linked_result = self.linker.link(entity.to_series())
        assert isinstance(linked_result, pd.Series)
        entity = entity.update_from(linked_result)

        # Do Step 4: resolve one-to-many KG matches
        resolved_result = self.resolver.resolve(entity.to_series(), category=entity_type)
        assert isinstance(resolved_result, pd.Series)
        entity = entity.update_from(resolved_result)

        # Do Step 5: enrich with equivalent IDs from the chosen KG node
        kg_equivalent_ids: dict[str, list[str]] = {}
        equivalent_ids_lookup_ok = True
        if entity.chosen_kg_id is not None:
            equiv_ids, equivalent_ids_lookup_ok = self.linker.get_equivalent_ids_checked([entity.chosen_kg_id])
            kg_equivalent_ids = equiv_ids.get(entity.chosen_kg_id, {})
            entity = entity.update_from(pd.Series({"kg_equivalent_ids": kg_equivalent_ids}))

        # Do Step 6: issue the resolution certificate.
        #
        # Deliberately OUTSIDE the null guard above: the rows the certificate most needs to describe
        # are the ones with no committed node, and building it inside would leave that population
        # undescribed here while the dataset path described it.
        certificate = self._issue_certificate(
            query_name=entity.name,
            category=entity_type,
            chosen_kg_id=entity.chosen_kg_id,
            kg_equivalent_ids=kg_equivalent_ids,
            equivalent_ids_lookup_ok=equivalent_ids_lookup_ok,
            selection_conflict=entity.chosen_kg_id_review,
            kg_ids_assigned=entity.kg_ids_assigned,
        )
        # Emitted as a plain dict, not the dataclass: pydantic rejects a raw dataclass at the
        # response model, and the NDJSON endpoint json.dumps's this value outside its try/except.
        entity = entity.update_from(
            pd.Series(
                {
                    "resolution_certificate": certificate.to_api_dict(),
                    "chosen_kg_id_review": derive_chosen_kg_id_review(certificate),
                }
            )
        )

        if input_is_series:
            return entity.to_series()
        return entity.to_dict()

    def map_dataset_to_kg(
        self,
        dataset: str | Path | pd.DataFrame,
        entity_type: str,
        name_column: str,
        provided_id_columns: list[str],
        vocab: str | list[str] | None = None,
        array_delimiters: list[str] | None = None,
        output_prefix: str | None = None,
        output_dir: str | Path = PROJECT_ROOT / "results",
        annotation_mode: AnnotationMode = "missing",
        annotators: list[str] | None = None,
        prefer_human: bool = True,
        prefer_canonical: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        """
        Map all entities in a dataset to knowledge graph nodes.

        Args:
            dataset: Path to TSV/CSV file or pandas DataFrame for processing
            entity_type: Type of entities (e.g., 'metabolite', 'protein')
            name_column: Column containing entity names
            provided_id_columns: Columns containing (un-normalized) vocab identifiers
            vocab: Allowed vocab name(s) to map to (e.g., 'CHEBI', 'MONDO')
            array_delimiters: Characters used to split delimited ID strings (default: [',', ';'])
            annotation_mode: When to annotate
                - 'all': Annotate all entities
                - 'missing': Only annotate entities without provided_ids (default)
                - 'none': Skip annotation entirely (returns empty)
            output_prefix: Optional prefix for the output TSV file name
            output_dir: Optional path to directory to save output/result files in
            annotators: Optional list of annotators to use (by slug). If None, annotators are selected automatically.

        Returns:
            Tuple of (output_tsv_path, stats_summary)
        """
        logging.info(f"Beginning to map dataset to KG ({dataset})")
        array_delimiters = array_delimiters if array_delimiters is not None else [",", ";"]

        # Validate/standardize the input entity type and vocab(s) on Biolink
        entity_type = self.biolink_client.standardize_entity_type(entity_type)
        prefixes = self.normalizer.get_standard_prefix(vocab)

        # Ensure the results directory for output files exists
        output_dir = Path(output_dir)
        logging.info(f"Output dir path is: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO: how to handle other data types, like .txt?
        # TODO: let file output location be configurable? #11
        # Issue: if dataset is a pandas df, need to create some default filename
        # naively create a default output filename (input_df_MAPPED) if output_prefix not provided

        output_suffix = "_MAPPED.tsv"
        if isinstance(dataset, pd.DataFrame):
            df = dataset
            output_tsv_name = f"input_df{output_suffix}" if output_prefix is None else f"{output_prefix}{output_suffix}"
        elif isinstance(dataset, (str, Path)):
            dataset = str(dataset)
            # Load tsv into pandas
            if output_prefix is not None:
                output_tsv_name = f"{output_prefix}{output_suffix}"
            else:
                output_tsv_name = Path(dataset).name.replace(".tsv", output_suffix).replace(".csv", output_suffix)
            if dataset.endswith(".tsv"):
                df = pd.read_csv(dataset, sep="\t", dtype={id_col: str for id_col in provided_id_columns}, comment="#")
            elif dataset.endswith(".csv"):
                df = pd.read_csv(dataset, dtype={id_col: str for id_col in provided_id_columns}, comment="#")
            else:
                raise ValueError(f"Unsupported file extension for dataset: {dataset}")
        else:
            raise ValueError(
                f"Unsupported type of '{type(dataset)}' for 'dataset' parameter; "
                f"only str, Path, or pd.DataFrame are supported"
            )
        logging.info(f"Output tsv name is: {output_tsv_name}")
        output_tsv_path = output_dir / output_tsv_name
        logging.info(f"output tsv path is: {output_tsv_path}")

        # Do some basic cleanup to try to ensure empty cells are represented consistently
        df[provided_id_columns] = df[provided_id_columns].replace("-", np.nan)
        df[provided_id_columns] = df[provided_id_columns].replace("NO_MATCH", np.nan)
        num_rows_start = len(df)

        # Do Step 1: annotate all rows with vocab IDs
        annotation_df = self.annotation_engine.annotate(
            item=df,
            name_field=name_column,
            provided_id_fields=provided_id_columns,
            category=entity_type,
            prefixes=prefixes,
            mode=annotation_mode,
            annotators=annotators,
            prefer_human=prefer_human,
            prefer_canonical=prefer_canonical,
        )
        df = df.join(annotation_df)
        logging.info(f"After step 1 (annotation), df is: \n{df}")

        # Do Step 2: normalize vocab IDs in all rows to form proper curies
        normalization_df = self.normalizer.normalize(
            item=df, provided_id_fields=provided_id_columns, array_delimiters=array_delimiters
        )
        df = df.join(normalization_df)
        logging.info(f"After step 2 (normalization), df is: \n{df}")

        # Do Step 3: link curies to KG nodes
        linked_df = self.linker.link(df)
        df = df.join(linked_df)
        logging.info(f"After step 3 (linking), df is: \n{df}")

        # Do Step 4: resolve one-to-many KG matches
        resolved_df = self.resolver.resolve(df, category=entity_type)
        df = df.join(resolved_df)
        logging.info(f"After step 4 (resolution), df is: \n{df}")

        # Do Step 5: enrich with equivalent IDs from chosen KG nodes
        unique_kg_ids = [kid for kid in df["chosen_kg_id"].dropna().unique()]
        equivalent_ids_lookup_ok = True
        if unique_kg_ids:
            equiv_map, equivalent_ids_lookup_ok = self.linker.get_equivalent_ids_checked(unique_kg_ids)
            df["kg_equivalent_ids"] = df["chosen_kg_id"].map(lambda kid: {} if pd.isna(kid) else equiv_map.get(kid, {}))
        else:
            df["kg_equivalent_ids"] = pd.Series([{} for _ in range(len(df))], index=df.index)
        logging.info(f"After step 5 (equivalent IDs enrichment), df is: \n{df}")

        # Do Step 6: issue a resolution certificate per row.
        #
        # This is the path every published artifact comes from (the audit, the suite arms, the
        # figure), so it carries the same certificate the single-entity path does — flattened.
        # Flat scalar columns, assembled as plain dicts: an object column surviving to df.to_csv
        # would write `ResolutionCertificate(state=...)` and reintroduce exactly the
        # ast.literal_eval-only column the certificate exists to replace.
        certificate_rows = [
            self._issue_certificate(
                query_name=_scalar_or_none(row.get(name_column)),
                category=entity_type,
                chosen_kg_id=_scalar_or_none(row.get("chosen_kg_id")),
                kg_equivalent_ids=row.get("kg_equivalent_ids") or {},
                equivalent_ids_lookup_ok=equivalent_ids_lookup_ok,
                selection_conflict=_scalar_or_none(row.get("chosen_kg_id_review")),
                kg_ids_assigned=row.get("kg_ids_assigned") or {},
            )
            for _, row in df.iterrows()
        ]
        df = df.join(pd.DataFrame([c.to_flat_columns() for c in certificate_rows], index=df.index))
        # The legacy flag is now DERIVED from the certificate (C4/L20) rather than passed through,
        # so the two can never disagree. Identical for one release; deprecation is a follow-up.
        # object dtype and an explicit index: the review is str | None per row, and a bare list with
        # Nones is both rejected by pandas' typed __setitem__ and, at runtime, coerced to a float NaN
        # column that later identity checks (`is None`) would silently miss.
        df["chosen_kg_id_review"] = pd.Series(
            [derive_chosen_kg_id_review(c) for c in certificate_rows], index=df.index, dtype=object
        )
        logging.info(f"After step 6 (resolution certificate), df is: \n{df}")

        # Do a little validation of results dataframe
        num_rows_end = len(df)
        if num_rows_start != num_rows_end:
            raise ValueError(
                f"At end of map_dataset_to_kg(), dataframe has {num_rows_end} rows but started with {num_rows_start} "
                f"rows. Row count should not change."
            )

        # Dump the final dataframe to a TSV

        logging.info(f"Dumping output TSV to {output_tsv_path}")
        df.to_csv(output_tsv_path, sep="\t", index=False)

        # Stamp KG build + service versions into the stats file. Hits Kestrel /health
        # (timeout 5s); degrades to "unknown" and never raises — see build_run_provenance.
        run_provenance = build_run_provenance(
            kestrel_url=get_kestrel_api_url(),
            run_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        stats_summary = analyze_dataset_mapping(
            output_tsv_path, self.linker, annotation_mode, run_provenance=run_provenance.model_dump()
        )

        return str(output_tsv_path), stats_summary
