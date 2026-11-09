"""Build the packaged ICD-10-CM hierarchy snapshot from a pinned CDC order file.

Downloads (or reads) the CDC FY ICD-10-CM code-descriptions zip, verifies its SHA-256 against a pinned
value BEFORE writing anything, re-dots each code after the third character, and writes:

* ``src/biomapper2/data/icd10cm/icd10cm_order_FY<fy>.tsv``: ``code<TAB>billable``, CDC order preserved.
* ``<that file>.version``: ``FY<fy>/sha256:<first 12 hex of the zip>/built:<YYYY-MM-DD>``.

The version line is stamped on every row the walk-up serves (core/icd10cm_hierarchy.py). A refresh for a
new fiscal year is one run of this script with the new URL and SHA, plus a PR.

Usage::

    uv run python scripts/build_icd10cm_hierarchy.py                      # download the pinned FY2027 zip
    uv run python scripts/build_icd10cm_hierarchy.py --zip /path/icd10cm-code-descriptions-2027.zip
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "src" / "biomapper2" / "data" / "icd10cm"

FISCAL_YEAR = "2027"
CDC_ZIP_URL = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/icd10cm-code-descriptions-2027.zip"
)
# The same zip the Unit 1 gate study pinned (studies/analysis/results/icd10cm_twin_control.json).
CDC_ZIP_SHA256 = "93e3ad6004badf470c55bfe679b748ae88fd9b2b421851e409eec382c7713b9a"


class Sha256MismatchError(RuntimeError):
    """The downloaded zip is not the pinned one; nothing was written."""


def redot(code: str) -> str:
    """CDC writes undotted codes (``E1165``); KRAKEN and users write them dotted (``E11.65``)."""
    code = code.strip()
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def parse_order_lines(text: str) -> list[tuple[str, str]]:
    """``(dotted code, billable flag)`` per row of the fixed-width CDC order file (icd10OrderFiles.pdf).

    Columns: order number (1-5), code (7-13), header/billable flag (15). Any other shape raises.
    """
    rows: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        code, flag = line[6:13].strip(), line[14:15]
        if not code or flag not in {"0", "1"}:
            raise ValueError(f"malformed CDC order row: {line!r}")
        rows.append((redot(code), flag))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.chmod(tmp, 0o644)  # mkstemp creates 0600; the packaged data must be world-readable
    os.replace(tmp, path)


def build(zip_path: Path, *, expected_sha256: str, out_dir: Path, fiscal_year: str, build_date: str) -> Path:
    """Verify, parse and write the snapshot + sidecar. Returns the snapshot path."""
    actual = _sha256(zip_path)
    if actual != expected_sha256:
        raise Sha256MismatchError(f"{zip_path}: sha256 {actual} != pinned {expected_sha256}; refusing to write")
    member_name = f"icd10cm-order-{fiscal_year}.txt"
    with zipfile.ZipFile(zip_path) as zf:
        member = next((n for n in zf.namelist() if n.endswith(member_name)), None)
        if member is None:
            raise ValueError(f"{zip_path} has no {member_name}")
        rows = parse_order_lines(zf.read(member).decode("utf-8"))
    codes = [code for code, _ in rows]
    if len(set(codes)) != len(codes):
        raise ValueError(f"{member} repeats codes")
    out = out_dir / f"icd10cm_order_FY{fiscal_year}.tsv"
    _atomic_write(out, "code\tbillable\n" + "".join(f"{code}\t{flag}\n" for code, flag in rows))
    _atomic_write(Path(str(out) + ".version"), f"FY{fiscal_year}/sha256:{actual[:12]}/built:{build_date}\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, help="a local copy of the CDC zip (default: download the pinned URL)")
    parser.add_argument("--url", default=CDC_ZIP_URL)
    parser.add_argument("--sha256", default=CDC_ZIP_SHA256)
    parser.add_argument("--fiscal-year", default=FISCAL_YEAR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--build-date", default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = args.zip
        if zip_path is None:
            zip_path = Path(tmp) / Path(args.url).name
            print(f"downloading {args.url}", file=sys.stderr)
            urllib.request.urlretrieve(args.url, zip_path)  # noqa: S310 (pinned https URL, sha-verified)
        out = build(
            zip_path,
            expected_sha256=args.sha256,
            out_dir=args.out_dir,
            fiscal_year=args.fiscal_year,
            build_date=args.build_date,
        )
    print(f"wrote {out} ({sum(1 for _ in out.open()) - 1} codes) and {out}.version")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
