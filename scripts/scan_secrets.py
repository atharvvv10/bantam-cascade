"""Scan tracked files and git history for key-shaped strings.

Matched values are never printed; only pattern names, commits, paths, and
counts are reported. Run from the repository root:

    python scripts/scan_secrets.py --history
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PATTERNS: dict[str, re.Pattern[bytes]] = {
    "groq_gsk": re.compile(rb"gsk_[A-Za-z0-9]{20,}"),
    "gemini_aiza": re.compile(rb"AIza[0-9A-Za-z_-]{35,}"),
    "assigned_api_key": re.compile(
        rb"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]"
    ),
}


def find_patterns(data: bytes) -> list[str]:
    """Return the names of key-shaped patterns present in ``data``."""
    return [name for name, pattern in PATTERNS.items() if pattern.search(data)]


def tracked_files() -> list[Path]:
    """Return paths currently tracked by git."""
    proc = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True, text=False)
    return [Path(name.decode("utf-8")) for name in proc.stdout.split(b"\0") if name]


def history_patches() -> bytes:
    """Return the full patch stream for all commits and reflogs."""
    proc = subprocess.run(
        ["git", "log", "--all", "--reflog", "-p", "--", "."],
        capture_output=True,
        check=True,
        text=False,
    )
    return proc.stdout


def main() -> int:
    """Run the scan and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also scan every patch in git history and reflogs.",
    )
    args = parser.parse_args()

    findings: dict[str, set[str]] = {name: set() for name in PATTERNS}
    for path in tracked_files():
        if not path.is_file():
            continue
        for name in find_patterns(path.read_bytes()):
            findings[name].add(f"worktree:{path.as_posix()}")

    if args.history:
        patch = history_patches()
        for name in find_patterns(patch):
            findings[name].add("history:git-log")

    total = 0
    for name in sorted(PATTERNS):
        locations = sorted(findings[name])
        total += len(locations)
        print(f"{name}: {len(locations)} location(s)")
        for location in locations[:20]:
            print(f"  {location}")
    print(f"total: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
