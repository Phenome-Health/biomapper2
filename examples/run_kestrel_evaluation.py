#!/usr/bin/env python3
"""
Kestrel API Performance Evaluation Script

Runs compound names from the HMDB ground truth dataset through Kestrel search endpoints
and saves raw results to JSON for analysis.

Usage:
    # Full run (all endpoints)
    uv run python examples/run_kestrel_evaluation.py

    # Test mode (first 50 queries)
    uv run python examples/run_kestrel_evaluation.py --test

    # Single endpoint
    uv run python examples/run_kestrel_evaluation.py --endpoints text-search

    # Custom batch size and limit
    uv run python examples/run_kestrel_evaluation.py --batch-size 200 --limit 100
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
KESTREL_API_URL = "https://kestrel.nathanpricelab.com/api"
KESTREL_API_KEY = os.getenv("KESTREL_API_KEY")
if not KESTREL_API_KEY:
    raise ValueError("KESTREL_API_KEY environment variable is not set. Check your .env file.")

GROUND_TRUTH_PATH = Path("/home/trentleslie/Insync/projects/biovector-eval/data/hmdb/ground_truth.json")
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "kestrel_evaluation_results.json"
CHECKPOINT_PATH = Path(__file__).parent / "kestrel_evaluation_checkpoint.json"

ENDPOINTS = ["text-search", "vector-search", "hybrid-search"]
CATEGORY_FILTER = "SmallMolecule"  # HMDB compounds are small molecules


def load_ground_truth(path: Path) -> list[dict]:
    """Load ground truth queries from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def call_kestrel_api(
    endpoint: str, search_terms: list[str], limit: int, category_filter: str
) -> tuple[dict, float]:
    """
    Call Kestrel search endpoint and return results with timing.

    Returns:
        Tuple of (response_dict, latency_ms)
    """
    payload = {
        "search_text": search_terms,
        "limit": limit,
        "category_filter": category_filter,
    }

    start_time = time.perf_counter()
    response = requests.post(
        f"{KESTREL_API_URL}/{endpoint}",
        headers={"X-API-Key": KESTREL_API_KEY},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    return response.json(), latency_ms


def process_batch(
    endpoint: str,
    queries: list[dict],
    limit: int,
    category_filter: str,
) -> list[dict]:
    """Process a batch of queries and return results with per-query data."""
    search_terms = [q["query"] for q in queries]

    # Call API
    response, total_latency_ms = call_kestrel_api(endpoint, search_terms, limit, category_filter)

    # Per-query latency estimate (batch latency / num queries)
    per_query_latency = total_latency_ms / len(search_terms) if search_terms else 0

    # Build results for each query
    results = []
    for q in queries:
        query_response = response.get(q["query"], [])
        results.append(
            {
                "query": q["query"],
                "expected": q["expected"],
                "category": q["category"],
                "difficulty": q.get("difficulty", "unknown"),
                "response": query_response,
                "latency_ms": per_query_latency,
                "batch_latency_ms": total_latency_ms,
            }
        )

    return results


def load_checkpoint() -> dict | None:
    """Load checkpoint if exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


def save_checkpoint(data: dict):
    """Save checkpoint for resumption."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f)


def clear_checkpoint():
    """Remove checkpoint file after successful completion."""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


def run_evaluation(
    ground_truth: list[dict],
    endpoints: list[str],
    limit: int,
    batch_size: int,
    output_path: Path,
    resume: bool = True,
) -> dict:
    """
    Run full evaluation across specified endpoints.

    Args:
        ground_truth: List of query dicts with 'query', 'expected', 'category'
        endpoints: List of endpoint names to test
        limit: Max results per query (for recall@k analysis)
        batch_size: Number of queries per API call
        output_path: Path to save results JSON
        resume: Whether to resume from checkpoint

    Returns:
        Results dictionary
    """
    # Check for checkpoint
    checkpoint = load_checkpoint() if resume else None

    if checkpoint:
        print(f"Resuming from checkpoint...")
        results = checkpoint["results"]
        completed_endpoints = checkpoint.get("completed_endpoints", [])
        start_endpoint_idx = checkpoint.get("current_endpoint_idx", 0)
        start_batch_idx = checkpoint.get("current_batch_idx", 0)
    else:
        results = {ep: [] for ep in endpoints}
        completed_endpoints = []
        start_endpoint_idx = 0
        start_batch_idx = 0

    # Process each endpoint
    for ep_idx, endpoint in enumerate(endpoints):
        if ep_idx < start_endpoint_idx:
            continue

        if endpoint in completed_endpoints:
            print(f"Skipping {endpoint} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing endpoint: {endpoint}")
        print(f"{'='*60}")

        # Calculate batches
        num_batches = (len(ground_truth) + batch_size - 1) // batch_size

        # Resume from batch if needed
        batch_start = start_batch_idx if ep_idx == start_endpoint_idx else 0

        for batch_idx in tqdm(range(batch_start, num_batches), desc=endpoint, initial=batch_start, total=num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(ground_truth))
            batch_queries = ground_truth[start_idx:end_idx]

            try:
                batch_results = process_batch(endpoint, batch_queries, limit, CATEGORY_FILTER)
                results[endpoint].extend(batch_results)

                # Save checkpoint every batch
                save_checkpoint(
                    {
                        "results": results,
                        "completed_endpoints": completed_endpoints,
                        "current_endpoint_idx": ep_idx,
                        "current_batch_idx": batch_idx + 1,
                        "metadata": {
                            "limit": limit,
                            "batch_size": batch_size,
                        },
                    }
                )

                # Small delay to avoid overwhelming API
                time.sleep(0.05)

            except requests.exceptions.RequestException as e:
                print(f"\nError at batch {batch_idx}: {e}")
                print("Progress saved. Run again to resume.")
                return results

        completed_endpoints.append(endpoint)
        print(f"Completed {endpoint}: {len(results[endpoint])} results")

    # Build final output
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "ground_truth_source": str(GROUND_TRUTH_PATH),
            "total_queries": len(ground_truth),
            "endpoints_tested": endpoints,
            "limit": limit,
            "batch_size": batch_size,
            "category_filter": CATEGORY_FILTER,
        },
        "results": results,
    }

    # Save final results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Clear checkpoint
    clear_checkpoint()

    return output


def main():
    parser = argparse.ArgumentParser(description="Run Kestrel API performance evaluation")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process first 50 queries",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max results per query for recall@k analysis (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of queries per API call (default: 100)",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=ENDPOINTS,
        choices=ENDPOINTS,
        help=f"Endpoints to test (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint, start fresh",
    )
    args = parser.parse_args()

    # Load ground truth
    print(f"Loading ground truth from: {GROUND_TRUTH_PATH}")
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"Loaded {len(ground_truth)} queries")

    # Test mode: limit queries
    if args.test:
        ground_truth = ground_truth[:50]
        print(f"Test mode: using first {len(ground_truth)} queries")

    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Endpoints: {args.endpoints}")
    print(f"  Limit (results per query): {args.limit}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {args.output}")
    print(f"  Category filter: {CATEGORY_FILTER}")

    # Run evaluation
    results = run_evaluation(
        ground_truth=ground_truth,
        endpoints=args.endpoints,
        limit=args.limit,
        batch_size=args.batch_size,
        output_path=args.output,
        resume=not args.no_resume,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for endpoint in args.endpoints:
        n_results = len(results["results"].get(endpoint, []))
        print(f"  {endpoint}: {n_results} results")


if __name__ == "__main__":
    main()
