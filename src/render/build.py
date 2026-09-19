"""Assemble the Three.js render.html from assets/*.{html,css,js} + run data."""

import base64
import json
from pathlib import Path

from run_store import load_config
from utils.logger import get_logger

from .config import resolve_render_config
from .constants import EDGE_COLORS, EDGE_OPACITY, EDGE_WIDTH
from .data import build_json
from .metadata import attach_metadata
from .scene import build_universe_js, compute_scene_bounds
from .treeview import render_treeview_js

log = get_logger(__name__)

_ASSETS_DIR = Path(__file__).parent / "assets"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FAVICON_PATH = _REPO_ROOT / "assets" / "hero.png"
_FAVICON_FALLBACK_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


def _favicon_b64() -> str:
    if _FAVICON_PATH.is_file():
        return base64.b64encode(_FAVICON_PATH.read_bytes()).decode("ascii")
    return _FAVICON_FALLBACK_B64


def _safe_json(obj) -> str:
    """json.dumps() for embedding inside an HTML <script> element.

    Post `content` is arbitrary scraped text — it can contain a literal
    "</script>" (e.g. Reddit's own share-widget embed snippet, which is a
    full "<script ...></script>" block copy-pasted into a post). A plain
    json.dumps() does NOT escape "</", so that substring inside a JSON
    string ends our *own* wrapping <script type="module"> tag early: the
    HTML tokenizer that looks for "</script" doesn't know it's sitting
    inside a JS string. Once that happens, the rest of the JSON + all of
    app.js gets parsed as ordinary page text (visible as a wall of text),
    the module fails with a SyntaxError (truncated mid-object), and any
    *later* post containing the same embed snippet gets parsed as a real
    <script src="..."> tag, so the browser tries to fetch it.

    Escaping "</" as "<\\/" (valid in a JS string; decodes back to "</")
    prevents the HTML tokenizer from ever seeing "</script" without
    changing the decoded value at all. "<!--" is escaped for the same
    reason (can otherwise flip the tokenizer into "script data escaped"
    state).
    """
    return (
        json.dumps(obj, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )


def _json_tokens(data: dict, rcfg, groups: list[str]) -> dict[str, str]:
    return {
        "__RENDER_DATA_JSON__": _safe_json(data),
        "__EDGE_COLORS_JSON__": _safe_json(EDGE_COLORS),
        "__EDGE_WIDTH_JSON__": _safe_json(EDGE_WIDTH),
        "__EDGE_OPACITY_JSON__": _safe_json(EDGE_OPACITY),
        "__NOVA_SPLINE_JSON__": _safe_json(rcfg.nova_spline),
        "__LABEL_FONT_SIZE_JSON__": _safe_json(rcfg.label_font_size),
        "__LABEL_OFFSET_JSON__": _safe_json(rcfg.label_offset),
        "__LABEL_DISTANCE_JSON__": _safe_json(rcfg.label_distance),
        "__LABEL_MIN_FONT_SIZE_JSON__": _safe_json(rcfg.label_min_font_size),
        "__LABEL_SCREEN_PX_JSON__": _safe_json(rcfg.label_screen_px),
        "__LABEL_MAX_VISIBLE_JSON__": _safe_json(rcfg.label_max_visible),
        "__EDGE_GROUPS_JSON__": _safe_json(groups),
    }


def _scalar_tokens(rcfg) -> dict[str, str]:
    return {
        "__CLUSTER_MIN_RADIUS__": f"{rcfg.cluster_min_radius:.4f}",
        "__DASH_SIZE__": f"{rcfg.dash_size:.4f}",
        "__GAP_SIZE__": f"{rcfg.gap_size:.4f}",
        "__FLOW_SPEED__": f"{rcfg.flow_speed:.4f}",
        "__LABEL_BACKDROP_PADX_RATIO__": f"{rcfg.label_backdrop_padx_ratio:.4f}",
        "__LABEL_COLLISION_MARGIN_PX__": f"{rcfg.label_collision_margin_px:.4f}",
        "__LABEL_FADE_SPEED__": f"{rcfg.label_fade_speed:.4f}",
        "__LABEL_GUARANTEED_RADIUS_MULT__": f"{rcfg.label_guaranteed_radius_mult:.4f}",
        "__LABEL_NUDGE_EASE_SPEED__": f"{rcfg.label_nudge_ease_speed:.4f}",
        "__LABEL_NUDGE_ITERATIONS__": str(rcfg.label_nudge_iterations),
        "__LABEL_NUDGE_MAX_PX__": f"{rcfg.label_nudge_max_px:.4f}",
        "__LABEL_NUDGE_ENABLED__": str(rcfg.label_nudge_enabled).lower(),
    }


def _substitute(text: str, tokens: dict[str, str]) -> str:
    for token, value in tokens.items():
        text = text.replace(token, value)
    return text


def render_threejs(posts_df, edges_df, config_path="config.yaml", open_browser=True):
    """Render the interactive Three.js scene and write it to disk as HTML.

    Args:
        posts_df: nodes with computed x/y/z positions.
        edges_df: edges between nodes.
        config_path: path to config.yaml.
        open_browser: open the written file in a browser tab after saving.

    Returns:
        Path to the written render.html file, as a string.
    """
    cfg = load_config(config_path)
    render_cfg = cfg.get("render", {})

    bounds = compute_scene_bounds(posts_df, render_cfg)
    rcfg = resolve_render_config(render_cfg, bounds.diagonal)

    posts_df = attach_metadata(posts_df, edges_df, cfg)

    log.info("%d posts...", len(posts_df))
    data = build_json(posts_df, edges_df, cfg=cfg)
    log.info("%d nodes, %d links", len(data["nodes"]), len(data["links"]))

    groups = sorted({link["group"] for link in data["links"]})
    treeview_js = render_treeview_js(rcfg.tree_dash_size, rcfg.tree_gap_size, rcfg.tree_flow_speed)
    universe_js = build_universe_js(bounds)

    app_js = (_ASSETS_DIR / "app.js").read_text(encoding="utf-8")
    tokens = {
        **_json_tokens(data, rcfg, groups),
        **_scalar_tokens(rcfg),
        "__TREEVIEW_JS__": treeview_js,
        "__UNIVERSE_JS__": universe_js,
    }
    app_js = _substitute(app_js, tokens)

    style_css = (_ASSETS_DIR / "style.css").read_text(encoding="utf-8")
    template = (_ASSETS_DIR / "template.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__STYLE__", style_css)
        .replace("__APP_SCRIPT__", app_js)
        .replace("__THREE_VERSION__", rcfg.three_version)
        .replace("__FAVICON_B64__", _favicon_b64())
    )

    out = Path(cfg["storage"].get("processed_dir", "data/processed")) / "render.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("Saved : %s", out)

    if open_browser:
        import webbrowser
        webbrowser.open(f"file:///{out.resolve()}")

    return str(out)


def run(posts_df, edges_df, config_path="config.yaml", open_browser=True):
    """Module entry point called by the pipeline orchestrator.

    Args:
        posts_df: nodes with computed x/y/z positions.
        edges_df: edges between nodes.
        config_path: path to config.yaml.
        open_browser: forwarded to render_threejs() (default: True).

    Returns:
        Path to the written render.html file, as a string.
    """
    return render_threejs(posts_df, edges_df, config_path, open_browser=open_browser)
