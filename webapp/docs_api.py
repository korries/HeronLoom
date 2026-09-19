"""Renders the project's own Markdown docs (README + docs/*.md) for the
dashboard: a quick in-app "Guide" panel (see _doc_partial.html) and a full
"Docs" reference opened in a new tab (see docs_page.html).

Read-only, same spirit as runs_api.py: an explicit allow-list (DOCS) rather
than an open filesystem path, so a URL can never walk outside the project's
own documentation. Markdown -> HTML uses the stdlib-adjacent `markdown`
package with the same extension set most file-based doc viewers ship
(tables, fenced_code, toc, sane_lists) — see e.g. django-docs-viewer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path

import markdown
from markdown.extensions.toc import TocExtension


class DocNotFound(ValueError):
    pass


@dataclass(frozen=True)
class DocMeta:
    id: str
    title: str
    path: str          # relative to repo_root
    group: str          # sidebar/dropdown section label
    blurb: str = ""     # one line, shown in the Docs dropdown


# Explicit allow-list, in the order they should appear in the "Docs" dropdown
# and sidebar. Keep in sync with README.md's own documentation table.
DOCS: list[DocMeta] = [
    DocMeta("dashboard", "Dashboard guide", "docs/DASHBOARD.md", "Guide",
            "Using the web UI"),
    DocMeta("readme", "README", "README.md", "Project",
            "Overview & quick start"),
    DocMeta("architecture", "Architecture", "docs/ARCHITECTURE.md", "Reference",
            "Programs, run folder, stage flow"),
    DocMeta("algorithms", "Algorithms & citations", "docs/ALGORITHMS.md", "Reference",
            "Clustering, edges, layout math"),
    DocMeta("nova_adept", "Nova & ADEPT", "docs/NOVA_ADEPT.md", "Reference",
            "Narrative subtopics & orphan pools"),
    DocMeta("configuration", "Configuration reference", "docs/CONFIGURATION.md", "Reference",
            "Every config.yaml / advanced.yaml key"),
    DocMeta("cli", "CLI reference", "docs/CLI.md", "Reference",
            "pipeline.py / reload.py / search.py flags"),
    DocMeta("performance", "Performance & cost", "docs/PERFORMANCE.md", "Reference",
            "Runtime and LLM call volume"),
    DocMeta("render", "3D Render & Tree View", "docs/RENDER.md", "Reference",
            "Layout and viewer controls"),
    DocMeta("search", "Search reference", "docs/SEARCH.md", "Reference",
            "Retrieval, prompt contract, validation"),
    DocMeta("security", "Security", "SECURITY.md", "Project",
            "Before exposing the dashboard"),
    DocMeta("contributing", "Contributing", "CONTRIBUTING.md", "Project", ""),
    DocMeta("third_party_notices", "Third-Party Notices", "THIRD_PARTY_NOTICES.md", "Project",
            "Adapted/incorporated code"),
]
_DOCS_BY_ID: dict[str, DocMeta] = {d.id: d for d in DOCS}

# Maps every way a doc might be linked to from *within* another doc's
# Markdown ("CONFIGURATION.md", "docs/CONFIGURATION.md", "../SECURITY.md")
# back to its doc id, so rendered cross-references point at /docs/{id}
# instead of a dead relative .md path. Built once from DOCS itself.
_PATH_TO_ID: dict[str, str] = {}
for _d in DOCS:
    _p = Path(_d.path)
    _PATH_TO_ID[_p.as_posix()] = _d.id          # "docs/CONFIGURATION.md"
    _PATH_TO_ID[_p.name] = _d.id                # "CONFIGURATION.md" (relative, same dir)


def list_docs() -> list[DocMeta]:
    return DOCS


def get_meta(doc_id: str) -> DocMeta:
    try:
        return _DOCS_BY_ID[doc_id]
    except KeyError:
        raise DocNotFound(f"'{doc_id}' is not a known doc") from None


def grouped_docs() -> list[tuple[str, list[DocMeta]]]:
    """Docs grouped for the sidebar/dropdown, in DOCS' own group order."""
    groups: list[tuple[str, list[DocMeta]]] = []
    seen: dict[str, list[DocMeta]] = {}
    for d in DOCS:
        if d.group not in seen:
            seen[d.group] = []
            groups.append((d.group, seen[d.group]))
        seen[d.group].append(d)
    return groups


# ─── Slugify — must match the anchors the docs already use in their own
# cross-references (e.g. "ALGORITHMS.md#adr--adaptive-discriminant-
# refinement", "CONFIGURATION.md#run-mode--module-routing"), which were
# hand-written assuming GitHub's renderer. GitHub strips anything that
# isn't a word char/space/hyphen, lowercases, then swaps each remaining
# whitespace character for a hyphen *without* collapsing runs — so "Run
# mode & module routing" keeps the double space (from the removed "&") as
# a double hyphen. python-markdown's own default slugify doesn't do this,
# so a custom one is passed to the toc extension instead of trusting the
# default.
_SLUG_STRIP_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s")


def _github_slugify(value: str, separator: str) -> str:
    slug = _SLUG_STRIP_RE.sub("", value).strip().lower()
    return _WHITESPACE_RE.sub(separator, slug)


_EXTERNAL_LINK_RE = re.compile(r'href="(https?://[^"]+)"')
_INTERNAL_LINK_RE = re.compile(r'href="([^"#]+\.md)(#[^"]*)?"')
_MERMAID_BLOCK_RE = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.DOTALL
)
_LEADING_H1_RE = re.compile(r'^\s*<h1[^>]*>.*?</h1>\s*', re.DOTALL)


def _rewrite_links(html: str) -> str:
    """External links get target=_blank; .md cross-references get
    resolved to their /docs/{id} route (or left inert with a title if the
    target isn't one of ours — better than a silently dead link)."""
    html = _EXTERNAL_LINK_RE.sub(
        r'href="\1" target="_blank" rel="noopener"', html
    )

    def _swap(m: re.Match) -> str:
        raw_path, anchor = m.group(1), m.group(2) or ""
        # Links are written relative to the linking file; both "X.md" and
        # "docs/X.md" spellings appear across these docs, so try the tail
        # component too before giving up.
        key = raw_path.lstrip("./")
        doc_id = _PATH_TO_ID.get(key) or _PATH_TO_ID.get(Path(key).name)
        if doc_id is None:
            return m.group(0)  # leave as-is; not one of our known docs
        return f'href="/docs/{doc_id}{anchor}"'

    return _INTERNAL_LINK_RE.sub(_swap, html)


def _mermaidify(html: str) -> str:
    """python-markdown's fenced_code extension turns a ```mermaid block
    into <pre><code class="language-mermaid">…</code></pre> with the
    diagram source HTML-escaped. mermaid.js (loaded on docs_page.html)
    looks for <pre class="mermaid"> containing the raw source instead."""
    def _swap(m: re.Match) -> str:
        return f'<pre class="mermaid">{unescape(m.group(1))}</pre>'
    return _MERMAID_BLOCK_RE.sub(_swap, html)


def render_doc(doc_id: str, repo_root: Path) -> dict:
    """Title, rendered HTML body, and a table-of-contents list for one doc.

    Raises DocNotFound if doc_id isn't registered, or the underlying file
    doesn't exist on disk (a doc can be listed but not yet written, e.g.
    docs/RENDER.md, without that being a 500).
    """
    meta = get_meta(doc_id)
    path = (repo_root / meta.path)
    if not path.is_file():
        raise DocNotFound(f"'{meta.path}' does not exist on disk")

    text = path.read_text(encoding="utf-8")

    toc_ext = TocExtension(slugify=_github_slugify, permalink=False, toc_depth="2-3")
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", toc_ext])
    html = md.convert(text)
    html = _LEADING_H1_RE.sub("", html, count=1)
    html = _mermaidify(html)
    html = _rewrite_links(html)

    return {
        "id": meta.id,
        "title": meta.title,
        "path": meta.path,
        "html": html,
        "toc": getattr(md, "toc_tokens", []),
    }
