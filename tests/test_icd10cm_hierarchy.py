"""Unit 4: the pinned CDC ICD-10-CM hierarchy snapshot, its build script and its loader.

The walk-up (Unit 5) climbs from a provided ``ICD10CM:X`` to codes that EXIST in a pinned CDC order
file. The snapshot ships inside the package (read via importlib.resources) so wheel and git-dependency
installs find it; a walk-up that is enabled with a missing or malformed snapshot fails Mapper init.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from biomapper2 import config
from biomapper2.core import icd10cm_hierarchy

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SMALL = REPO_ROOT / "tests" / "fixtures" / "icd10cm_hierarchy" / "small.tsv"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch):
    icd10cm_hierarchy.reset()
    yield
    icd10cm_hierarchy.reset()


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", str(SMALL))
    return SMALL


# ------------------------------------------------------------------------------------------------
# ancestors(): most specific first, existing codes only, stop at 3 characters
# ------------------------------------------------------------------------------------------------
def test_e11_65_walks_to_e11_6_then_e11(small) -> None:
    assert icd10cm_hierarchy.ancestors("E11.65") == ["E11.6", "E11"]


def test_f32_9_walks_to_f32(small) -> None:
    assert icd10cm_hierarchy.ancestors("F32.9") == ["F32"]


def test_seventh_character_dropped_first_and_placeholder_x_intermediate_skipped(small) -> None:
    # T36.0X is a CDC placeholder header (present in the fixture, as in the real file): skipped.
    assert icd10cm_hierarchy.ancestors("T36.0X1A") == ["T36.0X1", "T36.0", "T36"]


def test_s72_chain(small) -> None:
    assert icd10cm_hierarchy.ancestors("S72.001A") == ["S72.001", "S72.00", "S72.0", "S72"]


def test_three_character_code_has_no_ancestors(small) -> None:
    assert icd10cm_hierarchy.ancestors("I10") == []


def test_code_absent_from_snapshot_has_no_ancestors(small) -> None:
    assert icd10cm_hierarchy.contains("E11.649") is False
    assert icd10cm_hierarchy.ancestors("E11.649") == []
    assert icd10cm_hierarchy.contains("E11.65") is True


def test_version_comes_from_the_sidecar(small) -> None:
    assert icd10cm_hierarchy.is_present() is True
    assert icd10cm_hierarchy.version() == "FY2027/sha256:fixture000000/built:2026-11-08"


# ------------------------------------------------------------------------------------------------
# The committed, packaged snapshot
# ------------------------------------------------------------------------------------------------
def test_packaged_snapshot_loads_with_fy_and_sha_in_its_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", None)
    icd10cm_hierarchy.load_or_raise()
    version = icd10cm_hierarchy.version()
    assert version is not None
    assert "FY2027" in version
    assert "sha256:93e3ad6004ba" in version
    assert icd10cm_hierarchy.ancestors("E11.65") == ["E11.6", "E11"]
    assert icd10cm_hierarchy.ancestors("F32.9") == ["F32"]
    assert icd10cm_hierarchy.ancestors("T36.0X1A") == ["T36.0X1", "T36.0", "T36"]


# ------------------------------------------------------------------------------------------------
# Fail loud
# ------------------------------------------------------------------------------------------------
def test_missing_snapshot_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", str(tmp_path / "nope.tsv"))
    assert icd10cm_hierarchy.is_present() is False
    with pytest.raises(icd10cm_hierarchy.Icd10cmHierarchyError, match="nope.tsv"):
        icd10cm_hierarchy.load_or_raise()
    with pytest.raises(icd10cm_hierarchy.Icd10cmHierarchyError):
        icd10cm_hierarchy.ancestors("E11.65")


@pytest.mark.parametrize(
    "body",
    [
        "codes\tflag\nE11\t0\n",  # bad header
        "code\tbillable\nE11\t0\nE11.6\n",  # short row
        "code\tbillable\nE11\t0\nE11.6\t2\n",  # bad flag
        "code\tbillable\nE11\t0\nE1165\t1\n",  # undotted code
        "code\tbillable\nE11\t0\nE11\t0\n",  # duplicate
        "code\tbillable\n",  # empty
    ],
)
def test_malformed_snapshot_is_rejected_whole(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text(body, encoding="utf-8")
    path.with_suffix(".tsv.version").write_text("FY2027/sha256:x/built:y\n", encoding="utf-8")
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", str(path))
    with pytest.raises(icd10cm_hierarchy.Icd10cmHierarchyError):
        icd10cm_hierarchy.load_or_raise()


def test_missing_version_sidecar_is_malformed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "noversion.tsv"
    path.write_text("code\tbillable\nE11\t0\n", encoding="utf-8")
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", str(path))
    with pytest.raises(icd10cm_hierarchy.Icd10cmHierarchyError, match="version"):
        icd10cm_hierarchy.load_or_raise()


def test_mapper_init_raises_when_walkup_enabled_and_snapshot_missing(monkeypatch, tmp_path: Path) -> None:
    from biomapper2.mapper import Mapper

    monkeypatch.setattr(config, "ICD10CM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(config, "ICD10CM_WALKUP_ENABLED", True)
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", str(tmp_path / "missing.tsv"))
    with pytest.raises(icd10cm_hierarchy.Icd10cmHierarchyError):
        Mapper()


def test_mapper_init_raises_when_walkup_enabled_and_snapshot_malformed(monkeypatch, tmp_path: Path) -> None:
    from biomapper2.mapper import Mapper

    path = tmp_path / "bad.tsv"
    path.write_text("not a header\n", encoding="utf-8")
    path.with_suffix(".tsv.version").write_text("v\n", encoding="utf-8")
    monkeypatch.setattr(config, "ICD10CM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(config, "ICD10CM_WALKUP_ENABLED", True)
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", str(path))
    with pytest.raises(icd10cm_hierarchy.Icd10cmHierarchyError):
        Mapper()


def test_mapper_init_does_not_need_the_snapshot_when_walkup_disabled(monkeypatch, tmp_path: Path) -> None:
    from biomapper2.mapper import Mapper

    monkeypatch.setattr(config, "ICD10CM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(config, "ICD10CM_WALKUP_ENABLED", False)
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", str(tmp_path / "missing.tsv"))
    Mapper()


# ------------------------------------------------------------------------------------------------
# Build script
# ------------------------------------------------------------------------------------------------
def _order_line(n: int, code: str, flag: str, title: str) -> str:
    return f"{n:05d} {code:<7} {flag} {title[:60]:<60} {title}"


def _fake_zip(tmp_path: Path) -> Path:
    lines = [
        _order_line(1, "E11", "0", "Type 2 diabetes mellitus"),
        _order_line(2, "E116", "0", "Type 2 diabetes mellitus with other specified complications"),
        _order_line(3, "E1165", "1", "Type 2 diabetes mellitus with hyperglycemia"),
        _order_line(4, "I10", "1", "Essential (primary) hypertension"),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("icd10cm-code-descriptions-2027/icd10cm-order-2027.txt", "\n".join(lines) + "\n")
    path = tmp_path / "cdc.zip"
    path.write_bytes(buf.getvalue())
    return path


def test_build_script_redots_and_writes_snapshot_and_version(tmp_path: Path) -> None:
    from scripts import build_icd10cm_hierarchy as build

    assert build.redot("E1165") == "E11.65"
    assert build.redot("I10") == "I10"
    zip_path = _fake_zip(tmp_path)
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    out = tmp_path / "out"
    written = build.build(zip_path, expected_sha256=sha, out_dir=out, fiscal_year="2027", build_date="2026-11-08")
    assert written.read_text(encoding="utf-8").splitlines() == [
        "code\tbillable",
        "E11\t0",
        "E11.6\t0",
        "E11.65\t1",
        "I10\t1",
    ]
    version = Path(str(written) + ".version").read_text(encoding="utf-8").strip()
    assert version == f"FY2027/sha256:{sha[:12]}/built:2026-11-08"
    # What the script writes, the loader accepts.
    hierarchy = icd10cm_hierarchy.load(written)
    assert hierarchy.ancestors("E11.65") == ["E11.6", "E11"]


def test_build_script_aborts_on_sha_mismatch_without_writing(tmp_path: Path) -> None:
    from scripts import build_icd10cm_hierarchy as build

    zip_path = _fake_zip(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(build.Sha256MismatchError):
        build.build(zip_path, expected_sha256="0" * 64, out_dir=out, fiscal_year="2027", build_date="2026-11-08")
    assert not out.exists() or not any(out.iterdir())


# ------------------------------------------------------------------------------------------------
# Packaging: the snapshot reaches a wheel install outside the repo
# ------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
def test_wheel_install_outside_the_repo_loads_the_snapshot_and_inits_the_mapper(tmp_path: Path) -> None:
    uv = os.environ.get("UV", "uv")
    dist = tmp_path / "dist"
    subprocess.run([uv, "build", "--wheel", "-o", str(dist)], cwd=REPO_ROOT, check=True, capture_output=True)
    wheels = list(dist.glob("biomapper2-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())
    assert "biomapper2/data/icd10cm/icd10cm_order_FY2027.tsv" in names
    assert "biomapper2/data/icd10cm/icd10cm_order_FY2027.tsv.version" in names

    venv = tmp_path / "venv"
    subprocess.run([uv, "venv", "-q", "-p", sys.executable, str(venv)], check=True, capture_output=True)
    python = venv / "bin" / "python"
    subprocess.run(
        [uv, "pip", "install", "-q", "--python", str(python), str(wheels[0])], check=True, capture_output=True
    )
    probe = (
        "import biomapper2, pathlib\n"
        "assert 'site-packages' in str(pathlib.Path(biomapper2.__file__)), biomapper2.__file__\n"
        "from biomapper2.core import icd10cm_hierarchy as h\n"
        "h.load_or_raise()\n"
        "assert h.ancestors('E11.65') == ['E11.6', 'E11']\n"
        "from biomapper2.mapper import Mapper\n"
        "Mapper()\n"
        "print(h.version())\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("BIOMAPPER2_ICD10CM")}
    env.pop("PYTHONPATH", None)
    done = subprocess.run([str(python), "-c", probe], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert "FY2027" in done.stdout
