"""Tree-view radial mode JS (assets/treeview.js) — see that file's own header
for the global-scope contract it shares with assets/app.js."""

from pathlib import Path

_TREEVIEW_JS_PATH = Path(__file__).parent / "assets" / "treeview.js"


def render_treeview_js(dash_size: float, gap_size: float, flow_speed: float) -> str:
    js = _TREEVIEW_JS_PATH.read_text(encoding="utf-8").strip("\n")
    return (
        js.replace("__TREE_DASH_SIZE__", repr(float(dash_size)))
        .replace("__TREE_GAP_SIZE__", repr(float(gap_size)))
        .replace("__TREE_FLOW_SPEED__", repr(float(flow_speed)))
    )
