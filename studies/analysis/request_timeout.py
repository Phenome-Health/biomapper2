"""Size the client's default request timeout from data rather than from taste.

A default below the server's own limit is worse than no default: it converts a recoverable server
error into a client-side abort, and the run then reports a connection problem that never happened.
So the number is derived from the *successful* request duration distribution in a run log, and the
derivation is committed as an artifact next to the constant it justifies.

The parser reads only a log file already on disk. It makes no requests.

Duration proxy
--------------
The client does not emit per-request timings, so a request's duration is taken as the interval
between the log line announcing it and the next line the process wrote. That overstates duration
whenever downstream work follows immediately, which is the safe direction here: a timeout sized off
an overstated duration is generous, and generosity is exactly what avoids aborting a slow-but-
recoverable request.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"
DERIVATION_PATH = RESULTS_DIR / "request_timeout_derivation.json"

_TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")
_REQUEST_ANNOUNCEMENT = re.compile(r"Getting .*results from Kestrel API for \d+ entities")
_RETRY_WARNING = re.compile(r"Kestrel API (\d{3}) on (\S+)")

# Headroom multiplier applied to the slowest observed successful request. Two, because the
# distribution's tail is long and thin: the median request finishes in a small fraction of the
# slowest one, so anchoring on the maximum and doubling it costs nothing on a healthy run and only
# matters when the backend is already struggling.
HEADROOM_MULTIPLIER = 2.0


def _parse_timestamp(line: str) -> _dt.datetime | None:
    match = _TIMESTAMP.match(line)
    if match is None:
        return None
    return _dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def successful_request_durations(log_text: str) -> list[float]:
    """Seconds between each request announcement and the process's next log line.

    A request whose next line is a retry warning failed, so it is excluded: the whole point is to
    size against the *successful* distribution, and a failing request returns fast.
    """
    lines = log_text.splitlines()
    durations: list[float] = []
    for i, line in enumerate(lines):
        if not _REQUEST_ANNOUNCEMENT.search(line):
            continue
        start = _parse_timestamp(line)
        if start is None:
            continue
        for follower in lines[i + 1 :]:
            end = _parse_timestamp(follower)
            if end is None or end <= start:
                continue
            if _RETRY_WARNING.search(follower):
                break  # this request failed; not part of the successful distribution
            durations.append((end - start).total_seconds())
            break
    return sorted(durations)


def summarize(log_text: str) -> dict[str, Any]:
    """Percentiles of the successful-request distribution plus the timeout they recommend."""
    durations = successful_request_durations(log_text)
    if not durations:
        return {
            "n_successful_requests": 0,
            "percentiles_s": {},
            "max_s": None,
            "headroom_multiplier": HEADROOM_MULTIPLIER,
            "recommended_timeout_s": None,
            "note": "no successful request durations found in the log; no recommendation made",
        }

    def pct(q: float) -> float:
        index = min(len(durations) - 1, int(q * len(durations)))
        return round(durations[index], 3)

    slowest = durations[-1]
    return {
        "n_successful_requests": len(durations),
        "percentiles_s": {"p50": pct(0.5), "p90": pct(0.9), "p95": pct(0.95), "p99": pct(0.99)},
        "max_s": round(slowest, 3),
        "headroom_multiplier": HEADROOM_MULTIPLIER,
        "recommended_timeout_s": int(-(-slowest * HEADROOM_MULTIPLIER // 1)),
        "duration_proxy": (
            "interval between a request's announcement line and the process's next log line; "
            "overstates duration whenever downstream work follows, which is the safe direction"
        ),
    }


def _load_derivation() -> dict[str, Any]:
    if DERIVATION_PATH.exists():
        return json.loads(DERIVATION_PATH.read_text())
    return {}


_DERIVATION = _load_derivation()

# The recommendation the committed derivation artifact carries. The shipped client constant is
# asserted against this in the test suite, so the constant cannot drift away from its evidence
# without turning a test red.
RECOMMENDED_TIMEOUT_S: float = float(_DERIVATION.get("recommended_timeout_s") or 0.0)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="a run log already on disk")
    parser.add_argument("--out", type=Path, default=DERIVATION_PATH)
    args = parser.parse_args(argv)
    summary = summarize(args.log.read_text(errors="replace"))
    summary["source_log"] = str(args.log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
