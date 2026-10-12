"""
Smoke tests: every tutorial script must run end-to-end with --quick
(a few iterations only). This catches broken imports, shape errors, etc.
It does NOT check accuracy - that needs the full runs.

    pytest -q tests/test_scripts_smoke.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for folder in ROOT.glob("0[1-5]_*") for p in folder.glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_runs_quick(script):
    result = subprocess.run([sys.executable, str(script), "--quick"], cwd=ROOT,
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]
    assert "SUMMARY" in result.stdout
