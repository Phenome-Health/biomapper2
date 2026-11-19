"""Run provenance: KG build + service versions stamped into analyst stats and test reports.

Conforms to kraken/build_info.schema.json (the cross-repo contract). Independent copy
by design — no kraken/kestrel import.
"""

import importlib.metadata
import logging

import requests
from pydantic import BaseModel, ConfigDict, Field

from .utils import _CIRCUIT_BREAKER, get_current_api_context

__all__ = ["KgBuildInfo", "RunProvenance", "fetch_kg_build_info", "build_run_provenance"]


class KgBuildInfo(BaseModel):
    """KG build metadata as served by Kestrel /health (originates in KRAKEN build_info.json).

    This is a deliberate ANALYST-FACING SUBSET of the contract, not the full schema. The
    build-process fields (steps_run, build_duration_minutes) are intentionally not modelled
    here — they're operational, not reproducibility metadata. extra='allow' lets them (and
    future enrichments like node_count) pass through untyped, so we never drop or reject
    fields the schema gains. See kraken/build_info.schema.json for the authoritative set.
    """

    model_config = ConfigDict(extra="allow")

    kg_version: str = "unknown"
    kraken_package_version: str = "unknown"
    biolink_version: str = "unknown"
    build_timestamp: str = "unknown"
    git_commit: str = "unknown"
    sources: list[str] = []
    kg_label: str | None = None


class RunProvenance(BaseModel):
    """Everything needed to reproduce/interpret a mapping run. Stamped into outputs."""

    biomapper2_version: str
    kestrel_version: str = "unknown"
    kestrel_url: str
    run_timestamp: str  # ISO-8601 UTC
    kg_build: KgBuildInfo = Field(default_factory=KgBuildInfo)


def fetch_kg_build_info(kestrel_url: str) -> tuple[str, KgBuildInfo]:
    """Return (kestrel_version, KgBuildInfo) from Kestrel /health.

    Degrades to ('unknown', empty KgBuildInfo) on any failure, logging a warning — the
    caller always gets a usable object so outputs record that provenance was unavailable
    rather than omitting it.
    """
    api_ctx = get_current_api_context()
    timeout = 5.0
    breaker_active = False
    if api_ctx is not None:
        timeout = min(2.0, api_ctx.timeout_s)
        breaker_active = api_ctx.use_circuit_breaker
        if breaker_active and not _CIRCUIT_BREAKER.can_attempt():
            return "unknown", KgBuildInfo()
    breaker_settled = False
    try:
        resp = requests.get(f"{kestrel_url.rstrip('/')}/health", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if breaker_active:
            _CIRCUIT_BREAKER.record_success()
            breaker_settled = True
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if breaker_active:
            if status is not None and 400 <= status < 500 and status != 429:
                _CIRCUIT_BREAKER.record_success()
            else:
                _CIRCUIT_BREAKER.record_failure()
            breaker_settled = True
        logging.warning(f"Could not fetch Kestrel /health for run provenance ({kestrel_url}): {e}; recording 'unknown'")
        return "unknown", KgBuildInfo()
    except requests.exceptions.RequestException as e:
        if breaker_active:
            _CIRCUIT_BREAKER.record_failure()
            breaker_settled = True
        logging.warning(f"Could not fetch Kestrel /health for run provenance ({kestrel_url}): {e}; recording 'unknown'")
        return "unknown", KgBuildInfo()
    except Exception as e:
        logging.warning(f"Could not fetch Kestrel /health for run provenance ({kestrel_url}): {e}; recording 'unknown'")
        return "unknown", KgBuildInfo()
    finally:
        if breaker_active and not breaker_settled:
            _CIRCUIT_BREAKER.settle_unsettled()

    kestrel_version = data.get("kestrel_version", "unknown")
    kg_build_raw = data.get("kg_build") or {}
    if not kg_build_raw:
        logging.warning(f"Kestrel /health at {kestrel_url} returned empty kg_build — KG provenance unavailable")
    return kestrel_version, KgBuildInfo.model_validate(kg_build_raw)


def build_run_provenance(kestrel_url: str, run_timestamp: str) -> RunProvenance:
    """Assemble a RunProvenance for the current biomapper2 + Kestrel + KG."""
    kestrel_version, kg_build = fetch_kg_build_info(kestrel_url)
    return RunProvenance(
        biomapper2_version=importlib.metadata.version("biomapper2"),
        kestrel_version=kestrel_version,
        kestrel_url=kestrel_url,
        run_timestamp=run_timestamp,
        kg_build=kg_build,
    )
