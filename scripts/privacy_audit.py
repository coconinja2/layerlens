#!/usr/bin/env python3
"""Fail when publishable LayerLens artifacts contain common secrets or local PII."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AUDIT_FIXTURES = {"scripts/privacy_audit.py", "tests/test_privacy.py"}
TEXT_SUFFIXES = {".html", ".json", ".md", ".py", ".sh", ".toml", ".yml", ".yaml"}
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "macOS home path": re.compile(r"/Users/[^/\s]+/"),
    "Linux home path": re.compile(r"/home/[^/\s]+/"),
    "Windows home path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return [ROOT / name for name in output.splitlines() if name]


def main() -> None:
    findings: list[str] = []
    for path in tracked_files():
        if str(path.relative_to(ROOT)) in AUDIT_FIXTURES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or path.name == "poetry.lock":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: possible {label}")

    showcase_path = ROOT / "docs" / "runs.json"
    if showcase_path.exists():
        showcase = json.loads(showcase_path.read_text(encoding="utf-8"))
        forbidden_keys = {"prompt", "generatedText", "generated_text", "base_url", "run_id"}
        for index, run in enumerate(showcase.get("runs", [])):
            present = forbidden_keys.intersection(run)
            if present:
                findings.append(f"docs/runs.json: run {index} contains {sorted(present)}")

    commit_emails = subprocess.check_output(
        ["git", "log", "--format=%ae"], cwd=ROOT, text=True
    ).splitlines()
    for email in commit_emails:
        if email and not email.endswith("@users.noreply.github.com"):
            findings.append("git history contains a non-noreply author email")
            break

    if findings:
        raise SystemExit("Privacy audit failed:\n- " + "\n- ".join(findings))
    print("Privacy audit passed: no common secrets, home paths, or raw request content found.")


if __name__ == "__main__":
    main()
