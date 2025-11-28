"""Tests that example scripts run without errors."""

import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def test_all_example_scripts_run():
    """Test that all example scripts run without errors."""
    example_scripts = list(EXAMPLES_DIR.glob("*.py"))

    failed = []
    for script_path in example_scripts:
        result = subprocess.run(
            [sys.executable, str(script_path)], cwd=script_path.parent, capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            failed.append({"script": script_path.name, "stdout": result.stdout, "stderr": result.stderr})

    assert not failed, f"{len(failed)} example script(s) failed:\n" + "\n".join(
        [f"- {f['script']}: {f['stderr']}" for f in failed]
    )
