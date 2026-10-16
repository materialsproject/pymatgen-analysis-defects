"""Release helpers for .github/workflows/python-publish.yml (stdlib only).

The range of a release is the commits since the last tag reachable from the
release commit (`git describe`). Tags on side branches are ignored for the range
but still count when picking a free tag name.

    release.py worthy REV            # true/false: feat/fix/perf touched the package
    release.py next-tag [--date D]   # vYYYY.M.D, or vYYYY.M.D.postN if taken
    release.py prepend REV TAG REPO  # prepend TAG's section to CHANGELOG.md
    release.py notes                 # body of the newest CHANGELOG.md section
"""

# ruff: noqa: INP001
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path

CHANGELOG = Path("CHANGELOG.md")
HEADER = "# Changelog\n"
PACKAGE_PATHS = ("pymatgen", "pyproject.toml")
SUBJECT = re.compile(r"^(?P<type>\w+)(?:\([^)]*\))?(?P<bang>!)?: (?P<desc>.+)$")
PR_REF = re.compile(r" \(#(\d+)\)$")
GROUPS = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "perf": "Performance",
    "refactor": "Refactoring",
    "docs": "Documentation",
    "build": "Build",
    "ci": "CI",
    "test": "Tests",
    "chore": "Chores",
}
OTHER = "Other"


def git(*args: str) -> str:
    """Run git and return its stripped stdout."""
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit_range(rev: str) -> str:
    """Commits since the last tag reachable from rev, or all of rev's history."""
    try:
        last = git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", rev)
    except subprocess.CalledProcessError:
        return rev
    return f"{last}..{rev}"


def subjects(rev: str, *paths: str) -> list[str]:
    """First-parent commit subjects in the release range, newest first."""
    out = git("log", "--first-parent", "--format=%s", commit_range(rev), "--", *paths)
    return [s for s in out.splitlines() if s and not s.startswith("chore: release v")]


def worthy(rev: str) -> bool:
    """Whether a feat/fix/perf commit touched the package since the last tag."""
    for subject in subjects(rev, *PACKAGE_PATHS):
        m = SUBJECT.match(subject)
        if m and m["type"] in {"feat", "fix", "perf"}:
            return True
    return False


def next_tag(date: dt.date) -> str:
    """The calver tag for date, with .postN appended if already taken."""
    base = f"v{date.year}.{date.month}.{date.day}"
    taken = set(git("tag", "--list", f"{base}*").splitlines())
    tag, n = base, 0
    while tag in taken:
        n += 1
        tag = f"{base}.post{n}"
    return tag


def section(rev: str, tag: str, repo: str) -> str:
    """The CHANGELOG.md section for the release range, grouped by commit type."""
    groups: dict[str, list[str]] = {}
    for subject in subjects(rev):
        m = SUBJECT.match(subject)
        group = GROUPS.get(m["type"], OTHER) if m else OTHER
        desc = m["desc"] if m else subject
        if m and m["bang"]:
            desc = f"**BREAKING:** {desc}"
        desc = PR_REF.sub(
            lambda pr: f" ([#{pr[1]}](https://github.com/{repo}/pull/{pr[1]}))", desc
        )
        groups.setdefault(group, []).append(f"- {desc}")
    lines = [f"## {tag}", ""]
    for group in [*GROUPS.values(), OTHER]:
        if group in groups:
            lines += [f"### {group}", "", *groups[group], ""]
    return "\n".join(lines)


def prepend(text: str) -> None:
    """Insert a section below the CHANGELOG.md header.

    A newest section whose tag was never created (a merged release PR that did
    not release) is replaced, since the new section covers its commits too.
    """
    old = CHANGELOG.read_text() if CHANGELOG.exists() else HEADER
    body = old.removeprefix(HEADER).lstrip("\n")
    m = re.match(r"## (\S+)\n.*?(?=^## |\Z)", body, re.MULTILINE | re.DOTALL)
    if m and not git("tag", "--list", m[1]):
        body = body[m.end() :]
    CHANGELOG.write_text(f"{HEADER}\n{text}\n{body}".rstrip("\n") + "\n")


def notes() -> str:
    """The body of the newest CHANGELOG.md section, without its heading."""
    m = re.search(
        r"^## .*?\n(.*?)(?=^## |\Z)", CHANGELOG.read_text(), re.MULTILINE | re.DOTALL
    )
    if not m:
        msg = "CHANGELOG.md has no release section"
        raise SystemExit(msg)
    return m[1].strip()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("worthy").add_argument("rev")
    p = sub.add_parser("next-tag")
    p.add_argument(
        "--date", type=dt.date.fromisoformat, default=dt.datetime.now(dt.UTC).date()
    )
    p = sub.add_parser("prepend")
    p.add_argument("rev")
    p.add_argument("tag")
    p.add_argument("repo")
    sub.add_parser("notes")
    args = parser.parse_args()

    if args.cmd == "worthy":
        print(str(worthy(args.rev)).lower())
    elif args.cmd == "next-tag":
        print(next_tag(args.date))
    elif args.cmd == "prepend":
        text = section(args.rev, args.tag, args.repo)
        prepend(text)
        print(text)
    else:
        print(notes())


if __name__ == "__main__":
    main()
