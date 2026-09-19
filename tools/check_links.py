#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "runs", "logs", "data", "__pycache__"}

LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)\)")
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.*)$")


def slugify(heading: str) -> str:
    text = heading.strip().lower().replace("`", "")
    text = re.sub(r"[*_]{1,2}", "", text)
    kept = []
    for char in text:
        if char.isalnum() or char in "-_":
            kept.append(char)
        elif char.isspace():
            kept.append(" ")
    return "".join(kept).replace(" ", "-")


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.md")
        if not any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts)
    )


def anchors_of(path: Path) -> set[str]:
    found: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line)
        if match:
            found.add(slugify(match.group("title")))
    return found


def main() -> int:
    files = markdown_files()
    anchor_cache = {path: anchors_of(path) for path in files}
    problems: list[str] = []
    checked = 0

    for path in files:
        here = path.relative_to(REPO_ROOT)
        for match in LINK_RE.finditer(path.read_text(encoding="utf-8")):
            target = match.group("target")
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue
            checked += 1
            file_part, _, anchor = target.partition("#")

            if file_part:
                resolved = (path.parent / file_part).resolve()
                if not resolved.exists():
                    problems.append(f"{here}: missing file -> {target}")
                    continue
            else:
                resolved = path.resolve()

            if not anchor:
                continue
            if resolved.suffix != ".md":
                continue
            if resolved not in anchor_cache:
                anchor_cache[resolved] = anchors_of(resolved)
            if anchor not in anchor_cache[resolved]:
                problems.append(f"{here}: broken anchor -> {target}")

    if problems:
        print(f"{len(problems)} problem(s) in {len(files)} Markdown files:\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"OK — {checked} relative links across {len(files)} Markdown files resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
