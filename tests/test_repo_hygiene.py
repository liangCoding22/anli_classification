"""Nothing personal or secret in tracked files: API keys, real account ids, absolute home paths."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_ACCOUNTS = {"fireworks", "acme", "acct", "a", "account", "your-account"}
KEY = re.compile(r"fw_[A-Za-z0-9]{12,}")
ACCOUNT = re.compile(r"accounts/([A-Za-z0-9_-]+)")
HOME = re.compile(r"/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/")


def tracked_text_files():
    try:
        names = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception:
        pytest.skip("not a git checkout")
    for name in names:
        path = ROOT / name
        if path.suffix in {".gz", ".png", ".pdf"} or not path.is_file():
            continue
        try:
            yield name, path.read_text()
        except UnicodeDecodeError:
            continue


def offences():
    found = []
    for name, text in tracked_text_files():
        if name == "tests/test_repo_hygiene.py":
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if KEY.search(line):
                found.append(f"{name}:{number}: looks like an API key")
            for account in ACCOUNT.findall(line):
                if account not in ALLOWED_ACCOUNTS:
                    found.append(f"{name}:{number}: account id '{account}' (use accounts/<account>)")
            if HOME.search(line):
                found.append(f"{name}:{number}: absolute home path")
    return found


def test_no_api_keys_real_account_ids_or_home_paths_are_tracked():
    assert offences() == []
