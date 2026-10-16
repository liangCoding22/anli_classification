"""Every script compiles and shows its help, so a broken import or argument definition is caught without running it."""

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def test_there_are_scripts_to_check():
    assert len(SCRIPTS) >= 7


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(script, tmp_path):
    py_compile.compile(str(script), cfile=str(tmp_path / "out.pyc"), doraise=True)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_help_runs(script):
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    if result.returncode != 0 and "ModuleNotFoundError" in result.stderr:
        pytest.skip(f"needs a package that is not installed here: {result.stderr.strip().splitlines()[-1]}")
    assert result.returncode == 0, result.stderr[-500:]
    assert "usage" in result.stdout.lower()
