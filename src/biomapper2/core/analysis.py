"""
Analysis module for evaluating mapping results.

Generates summary statistics, performance metrics, and diagnostic outputs.
"""

import ast
import json
import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from ..utils import safe_divide


def analyze_dataset_mapping(results_tsv_path: str, linker: Any) -> dict[str, Any]:
    """
    Analyze dataset mapping results and generate summary statistics.

    Args:
        results_tsv_path: Path to mapped dataset TSV
        linker: Linker instance for canonicalizing groundtruth IDs

    Returns:
        Dictionary containing coverage, precision, recall, and F1 metrics
    """
    logging.info(f"Analyzing dataset KG mapping in {results_tsv_path}")

    cols_to_literal_eval = [
        "curies",
        "curies_provided",
        "curies_assigned",
        "invalid_ids_provided",
        "invalid_ids_assigned",
        "kg_ids",
        "kg_ids_provided",
        "kg_ids_assigned",
    ]
    converters = {col: ast.literal_eval for col in cols_to_literal_eval}
    df = pd.read_table(results_tsv_path, converters=converters)

    # Make sure we load any groundtruth column properly
    if "kg_ids_groundtruth" in df.columns:
        df.kg_ids_groundtruth = df.kg_ids_groundtruth.apply(ast.literal_eval)
        # Canonicalize groundtruth IDs
        canonical_map = linker.get_kg_ids(list(set(df.kg_ids_groundtruth.explode().dropna())))
        df["kg_ids_groundtruth_canonical"] = df.apply(
            lambda r: [canonical_map[kg_id] for kg_id in r.kg_ids_groundtruth], axis=1
        )

    # Create reusable masks
    has_valid_ids_mask = df.curies.apply(len) > 0
    has_valid_ids_provided_mask = df.curies_provided.apply(len) > 0
    has_valid_ids_assigned_mask = df.curies_assigned.apply(lambda x: any(len(curies) > 0 for curies in x.values()))
    mapped_to_kg_mask = df.kg_ids.apply(len) > 0
    mapped_to_kg_provided_mask = df.kg_ids_provided.apply(len) > 0
    mapped_to_kg_assigned_mask = df.kg_ids_assigned.apply(lambda x: any(len(kg_ids) > 0 for kg_ids in x.values()))
    not_mapped_to_kg_mask = ~mapped_to_kg_mask
    one_to_many_mask = df.kg_ids.apply(lambda x: len(x) > 1)
    many_to_one_mask = df.chosen_kg_id.notna() & df.chosen_kg_id.duplicated(keep=False)
    has_invalid_ids_mask = df.apply(
        lambda r: len(r.invalid_ids_provided) > 0 or len(r.invalid_ids_assigned) > 0,
        axis=1,
    )
    has_invalid_ids_provided_mask = df.invalid_ids_provided.apply(lambda x: len(x) > 0)
    has_invalid_ids_assigned_mask = df.invalid_ids_assigned.apply(lambda x: len(x) > 0)
    has_no_ids_mask = ~has_valid_ids_mask & ~has_invalid_ids_mask
    assigned_correct_per_provided_mask = df.apply(
        lambda r: len(
            set(r.kg_ids_provided.keys())
            & set().union(*(annotator_kg_ids.keys() for annotator_kg_ids in r.kg_ids_assigned.values()))
        )
        > 0,
        axis=1,
    )
    assigned_correct_per_provided_chosen_mask = (
        (df.chosen_kg_id_provided == df.chosen_kg_id_assigned)
        & df.chosen_kg_id_provided.notna()
        & df.chosen_kg_id_assigned.notna()
    )

    # Calculate some summary stats
    total_items = len(df)
    has_valid_ids = has_valid_ids_mask.sum()
    has_valid_ids_provided = has_valid_ids_provided_mask.sum()
    has_valid_ids_assigned = has_valid_ids_assigned_mask.sum()
    has_only_provided_ids = has_valid_ids - has_valid_ids_assigned
    has_only_assigned_ids = has_valid_ids - has_valid_ids_provided
    has_both_provided_and_assigned_ids = has_valid_ids - has_only_provided_ids - has_only_assigned_ids
    has_no_ids = has_no_ids_mask.sum()
    has_invalid_ids = has_invalid_ids_mask.sum()
    has_invalid_ids_provided = has_invalid_ids_provided_mask.sum()
    has_invalid_ids_assigned = has_invalid_ids_assigned_mask.sum()
    mapped_to_kg = mapped_to_kg_mask.sum()
    mapped_to_kg_provided = mapped_to_kg_provided_mask.sum()
    mapped_to_kg_assigned = mapped_to_kg_assigned_mask.sum()
    mapped_to_kg_both = (mapped_to_kg_provided_mask & mapped_to_kg_assigned_mask).sum()
    assigned_correct_per_provided = assigned_correct_per_provided_mask.sum()
    assigned_correct_per_provided_chosen = assigned_correct_per_provided_chosen_mask.sum()
    has_invalid_ids_and_not_mapped_to_kg = (has_invalid_ids_mask & not_mapped_to_kg_mask).sum()
    one_to_many_mappings = one_to_many_mask.sum()
    many_to_one_mappings = many_to_one_mask.sum()
    multi_mappings = (one_to_many_mask | many_to_one_mask).sum()
    one_to_one_mappings = mapped_to_kg - multi_mappings

    # Do some sanity checks
    assert multi_mappings <= mapped_to_kg
    assert one_to_many_mappings <= multi_mappings
    assert many_to_one_mappings <= multi_mappings
    assert multi_mappings + one_to_one_mappings == mapped_to_kg
    assert has_only_provided_ids + has_only_assigned_ids + has_both_provided_and_assigned_ids == has_valid_ids
    assert assigned_correct_per_provided <= mapped_to_kg_provided

    # Compile final stats summary
    stats = {
        "mapped_dataset": results_tsv_path,
        "total_items": total_items,
        "mapped_to_kg": int(mapped_to_kg),
        "mapped_to_kg_provided": int(mapped_to_kg_provided),
        "mapped_to_kg_assigned": int(mapped_to_kg_assigned),
        "mapped_to_kg_provided_and_assigned": int(mapped_to_kg_both),
        "one_to_one_mappings": int(one_to_one_mappings),
        "multi_mappings": int(multi_mappings),
        "one_to_many_mappings": int(one_to_many_mappings),
        "many_to_one_mappings": int(many_to_one_mappings),
        "has_valid_ids": int(has_valid_ids),
        "has_valid_ids_provided": int(has_valid_ids_provided),
        "has_valid_ids_assigned": int(has_valid_ids_assigned),
        "has_only_provided_ids": int(has_only_provided_ids),
        "has_only_assigned_ids": int(has_only_assigned_ids),
        "has_both_provided_and_assigned_ids": int(has_both_provided_and_assigned_ids),
        "assigned_mappings_correct_per_provided": int(assigned_correct_per_provided),
        "assigned_mappings_correct_per_provided_chosen": int(assigned_correct_per_provided_chosen),
        "has_invalid_ids": int(has_invalid_ids),
        "has_invalid_ids_provided": int(has_invalid_ids_provided),
        "has_invalid_ids_assigned": int(has_invalid_ids_assigned),
        "has_no_ids": int(has_no_ids),
        "has_invalid_ids_and_not_mapped_to_kg": int(has_invalid_ids_and_not_mapped_to_kg),
    }

    # Calculate overall assigned performance
    assigned_performance = _calculate_assigned_performance(
        df,
        assigned_kg_ids_mask=mapped_to_kg_assigned_mask,
        mapped_to_kg_provided_mask=mapped_to_kg_provided_mask,
        mapped_to_kg_provided=mapped_to_kg_provided,
        get_kg_ids_assigned=_get_all_assigned_kg_ids,
        total_items=total_items,
        include_chosen=True,
        chosen_assigned_col="chosen_kg_id_assigned",
    )

    # Get all annotators
    all_annotators = set()
    for kg_ids_assigned in df.kg_ids_assigned:
        all_annotators.update(kg_ids_assigned.keys())

    # Calculate per-annotator performance
    per_annotator_stats = {}
    for annotator in sorted(all_annotators):
        annotator_mask = df.kg_ids_assigned.apply(lambda x, a=annotator: len(x.get(a, {})) > 0)

        per_annotator_stats[annotator] = _calculate_assigned_performance(
            df,
            assigned_kg_ids_mask=annotator_mask,
            mapped_to_kg_provided_mask=mapped_to_kg_provided_mask,
            mapped_to_kg_provided=mapped_to_kg_provided,
            get_kg_ids_assigned=lambda r, a=annotator: r.kg_ids_assigned.get(a, {}).keys(),
            total_items=total_items,
        )

    # Compile performance stats
    performance: dict[str, Any] = {
        "overall": {
            "coverage": _calculate_coverage(mapped_to_kg, total_items),
            "coverage_explanation": f"{mapped_to_kg} / {total_items}",
        },
        "assigned_ids": assigned_performance,
        "per_annotator": per_annotator_stats,
    }

    # Tack the performance metrics onto our other stats
    stats["performance"] = performance

    # Save all result stats
    logging.info(f"Dataset summary stats are: {json.dumps(stats, indent=2)}")
    results_filepath_root = results_tsv_path.replace(".tsv", "")
    with open(f"{results_filepath_root}_a_summary_stats.json", "w+") as stats_file:
        json.dump(stats, stats_file, indent=2)

    # Record the items that had valid curies but that weren't in the KG, for easy reference
    kg_misses = df[has_valid_ids_mask & not_mapped_to_kg_mask]
    kg_misses.to_csv(f"{results_filepath_root}_b_curie_misses.tsv", sep="\t")

    # Record the items that didn't get mapped to the KG, for easy reference
    unmapped = df[not_mapped_to_kg_mask]
    unmapped.to_csv(f"{results_filepath_root}_c_unmapped.tsv", sep="\t")

    # Record the items that DID map to the KG, for easy reference
    mapped = df[mapped_to_kg_mask]
    mapped.to_csv(f"{results_filepath_root}_d_mapped.tsv", sep="\t")

    # Record the items with invalid IDs, for easy reference
    invalid_ids_df = df[has_invalid_ids_mask]
    invalid_ids_df.to_csv(f"{results_filepath_root}_e_invalid_ids.tsv", sep="\t")

    # Record the one-to-many items, for easy reference
    one_to_many_items = df[one_to_many_mask]
    one_to_many_items.to_csv(f"{results_filepath_root}_f_one_to_many.tsv", sep="\t")

    # Record the many-to-one items, for easy reference
    many_to_one_items = df[many_to_one_mask]
    many_to_one_items.to_csv(f"{results_filepath_root}_g_many_to_one.tsv", sep="\t")

    return stats


def _calculate_assigned_performance(
    df: pd.DataFrame,
    assigned_kg_ids_mask: pd.Series,
    mapped_to_kg_provided_mask: pd.Series,
    mapped_to_kg_provided: int,
    get_kg_ids_assigned: Callable,
    total_items: int,
    include_chosen: bool = False,
    chosen_assigned_col: str | None = None,
) -> dict[str, Any]:
    """
    Calculate performance stats for assigned IDs vs provided and groundtruth (if available).
    """
    mapped_to_kg_assigned = assigned_kg_ids_mask.sum()
    mapped_to_kg_both = (assigned_kg_ids_mask & mapped_to_kg_provided_mask).sum()

    correct_per_provided = df.apply(
        lambda r: len(set(_get_provided_kg_ids(r)) & set(get_kg_ids_assigned(r))) > 0,
        axis=1,
    ).sum()

    precision = _calculate_precision(correct_per_provided, mapped_to_kg_both)
    recall = _calculate_recall(correct_per_provided, mapped_to_kg_provided)

    per_provided = {
        "mapped_to_kg_provided_and_assigned": int(mapped_to_kg_both),
        "correct": int(correct_per_provided),
        "precision": precision,
        "precision_explanation": f"{correct_per_provided} / {mapped_to_kg_both}",
        "recall": recall,
        "recall_explanation": f"{correct_per_provided} / {mapped_to_kg_provided}",
        "f1_score": _calculate_f1_score(precision, recall),
    }

    if include_chosen:
        correct_per_provided_chosen = (
            (df["chosen_kg_id_provided"] == df[chosen_assigned_col])
            & df["chosen_kg_id_provided"].notna()
            & df[chosen_assigned_col].notna()
        ).sum()

        precision_chosen = _calculate_precision(correct_per_provided_chosen, mapped_to_kg_both)
        recall_chosen = _calculate_recall(correct_per_provided_chosen, mapped_to_kg_provided)

        per_provided["after_resolving_one_to_manys"] = {
            "correct": int(correct_per_provided_chosen),
            "precision": precision_chosen,
            "precision_explanation": f"{correct_per_provided_chosen} / {mapped_to_kg_both}",
            "recall": recall_chosen,
            "recall_explanation": f"{correct_per_provided_chosen} / {mapped_to_kg_provided}",
            "f1_score": _calculate_f1_score(precision_chosen, recall_chosen),
        }

    result = {
        "mapped_to_kg": int(mapped_to_kg_assigned),
        "coverage": _calculate_coverage(mapped_to_kg_assigned, total_items),
        "coverage_explanation": f"{mapped_to_kg_assigned} / {total_items}",
        "per_provided_ids": per_provided,
    }

    # Add groundtruth comparison if available
    if "kg_ids_groundtruth_canonical" in df.columns:
        has_groundtruth_mask = df.kg_ids_groundtruth_canonical.apply(len) > 0
        groundtruth_count = has_groundtruth_mask.sum()
        mapped_to_kg_both_gt = (assigned_kg_ids_mask & has_groundtruth_mask).sum()

        correct_per_groundtruth = df.apply(
            lambda r: len(set(_get_groundtruth_kg_ids(r)) & set(get_kg_ids_assigned(r))) > 0,
            axis=1,
        ).sum()

        precision_gt = _calculate_precision(correct_per_groundtruth, mapped_to_kg_both_gt)
        recall_gt = _calculate_recall(correct_per_groundtruth, groundtruth_count)

        result["per_groundtruth"] = {
            "mapped_to_kg_groundtruth_and_assigned": int(mapped_to_kg_both_gt),
            "correct": int(correct_per_groundtruth),
            "precision": precision_gt,
            "precision_explanation": f"{correct_per_groundtruth} / {mapped_to_kg_both_gt}",
            "recall": recall_gt,
            "recall_explanation": f"{correct_per_groundtruth} / {groundtruth_count}",
            "f1_score": _calculate_f1_score(precision_gt, recall_gt),
        }

    return result


def _get_all_assigned_kg_ids(r) -> set:
    """Get all kg_ids from all annotators for a row."""
    return set().union(*(kg_ids.keys() for kg_ids in r.kg_ids_assigned.values())) if r.kg_ids_assigned else set()


def _get_provided_kg_ids(r):
    """Get kg_ids from provided for a row."""
    return r.kg_ids_provided.keys()


def _get_groundtruth_kg_ids(r):
    """Get kg_ids from groundtruth for a row."""
    return r.kg_ids_groundtruth_canonical


def _calculate_precision(correct: int, total_predicted: int) -> float | None:
    """Calculate precision, rounded to 4 decimal places."""
    result = safe_divide(correct, total_predicted)
    return round(result, 4) if result is not None else None


def _calculate_recall(correct: int, total_actual: int) -> float | None:
    """Calculate recall, rounded to 4 decimal places."""
    result = safe_divide(correct, total_actual)
    return round(result, 4) if result is not None else None


def _calculate_coverage(mapped: int, total: int) -> float | None:
    """Calculate coverage, rounded to 4 decimal places."""
    result = safe_divide(mapped, total)
    return round(result, 4) if result is not None else None


def _calculate_f1_score(precision: float | None, recall: float | None) -> float | None:
    """Calculate F1 score from precision and recall, rounded to 4 decimal places."""
    if precision is None or recall is None:
        return None
    result = safe_divide(2 * (precision * recall), (precision + recall))
    return round(result, 4) if result is not None else None
