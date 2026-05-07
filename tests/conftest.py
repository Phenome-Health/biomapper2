import importlib.metadata
import json
import os
import subprocess
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from biomapper2.config import PROJECT_ROOT, get_kestrel_api_url as _get_kestrel_url
from biomapper2.mapper import Mapper
from biomapper2.utils import setup_logging

# Setup logging once for all tests
setup_logging()


def _fetch_kestrel_info(url: str) -> dict:
    """Query Kestrel /health for software and KG build versions. Returns empty values on failure."""
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        data = response.json()
        return {
            "kestrel_version": data.get("kestrel_version", "unknown"),
            "kg_build": data.get("kg_build", {}),
        }
    except Exception:
        return {"kestrel_version": "unknown", "kg_build": {}}


class _ReportCollector:
    """Counts test outcomes across the session for the JSON test report."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call":
            if report.passed:
                self.counts["passed"] += 1
            elif report.failed:
                self.counts["failed"] += 1
            elif report.skipped:
                self.counts["skipped"] += 1
        elif report.when in ("setup", "teardown") and report.failed:
            self.counts["error"] += 1


def _api_tests_selected(request: pytest.FixtureRequest) -> bool:
    """Return True if any collected test item is marked requires_api."""
    return any(item.get_closest_marker("requires_api") is not None for item in request.session.items)


def pytest_configure(config: pytest.Config) -> None:
    config._performance_timings = {}  # type: ignore[attr-defined]
    collector = _ReportCollector()
    config._report_collector = collector  # type: ignore[attr-defined]
    config.pluginmanager.register(collector, name="_report_collector")

    # Apply --kestrel-url here (before any fixture or test body runs) so that
    # Mapper() instances created directly in test bodies — not via shared_mapper —
    # also hit the overridden backend.
    kestrel_url_opt = config.getoption("--kestrel-url", default=None, skip=True)
    if kestrel_url_opt is not None:
        os.environ["KESTREL_API_URL"] = kestrel_url_opt


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    config = session.config
    metadata: dict = getattr(config, "_test_metadata", {})
    if not metadata:
        return

    collector: _ReportCollector | None = getattr(config, "_report_collector", None)
    report = {
        "metadata": metadata,
        "test_counts": collector.counts if collector else {},
        "performance": getattr(config, "_performance_timings", {}),
    }

    ts = metadata.get("timestamp", "").replace(":", "-")[:19]
    tag = metadata.get("tag", "untagged")
    reports_dir = Path(PROJECT_ROOT) / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"{ts}_{tag}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nTest report written: {report_path.name}")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--kestrel-url",
        default=None,
        help="Override Kestrel API URL (e.g. for KG version regression testing)",
    )
    parser.addoption(
        "--tag",
        default="production",
        help="Label for this test run — used in report filename and metadata (e.g. kraken-with-spoke, staging)",
    )


@pytest.fixture(scope="session")
def kestrel_url(request: pytest.FixtureRequest) -> str | None:
    """Kestrel API URL from --kestrel-url option (None = use default from config)."""
    return request.config.getoption("--kestrel-url")


@pytest.fixture(scope="session")
def tag(request: pytest.FixtureRequest) -> str:
    """Run label from --tag option, used in report filenames."""
    return request.config.getoption("--tag")


@pytest.fixture(scope="session", autouse=True)
def test_run_metadata(request: pytest.FixtureRequest) -> dict:
    """Capture metadata for test reports. Attached to request.config._test_metadata."""
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"

    kestrel_url = _get_kestrel_url()
    # Skip the live /health call for offline tiers (e.g. test-fast.sh passes
    # 'not requires_api'); the report will still record which URL would be used.
    kestrel_info = _fetch_kestrel_info(kestrel_url) if _api_tests_selected(request) else {"kestrel_version": "unknown", "kg_build": {}}

    metadata = {
        "biomapper2_version": importlib.metadata.version("biomapper2"),
        "kestrel_version": kestrel_info["kestrel_version"],
        "kg_build": kestrel_info["kg_build"],
        "kestrel_url": kestrel_url,
        "git_commit": git_commit,
        "tag": request.config.getoption("--tag"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    request.config._test_metadata = metadata  # type: ignore[attr-defined]
    return metadata


@pytest.fixture(scope="session")
def shared_mapper() -> Iterator[Mapper]:
    """Single Mapper instance shared across all tests. URL already set by test_run_metadata."""
    mapper = Mapper()
    yield mapper
