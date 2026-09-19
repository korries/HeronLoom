"""Per-node visual attributes and the nodes/links JSON payload for the template."""

import math

import numpy as np
import pandas as pd

from utils.logger import get_logger

from .cfgutil import cfg_get
from .coerce import sf, si, ss
from .constants import EDGE_COLORS, EDGE_DASHED, EDGE_OPACITY, NODE_ROLES, NODE_ROLES_DEFAULT, SIZE_MULT

log = get_logger(__name__)


def build_color_maps(posts_df: pd.DataFrame) -> dict:
    """One continuous color per cluster, derived from its real 3D centroid.

    No palette, randomness, Leiden or extra graph logic — orients an HSL
    field along the graph's own principal axis (SVD) so color progression
    follows the actual shape of the data.
    """
    valid = posts_df[
        posts_df["cluster_id"].notna()
        & ~posts_df["cluster_id"].astype(str).isin(("nan", "None", ""))
    ]
    if valid.empty:
        return {}

    centers = valid.groupby("cluster_id")[["x", "y", "z"]].mean()
    cids = list(centers.index)
    p = centers.to_numpy(dtype=float)

    if len(p) == 1:
        return {cids[0]: "#5B8FF9"}

    p = p - p.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(p, full_matrices=False)
        p = p @ vh[:3].T
    except np.linalg.LinAlgError:
        pass

    # SVD only returns min(n_clusters, 3) components — with exactly 2
    # clusters that's 2, not 3, and n01(p[:, 2]) below would IndexError.
    # Pad any missing axis with a constant 0 column (n01() maps a
    # constant column to 0.5, same as its existing flat-range case).
    if p.shape[1] < 3:
        p = np.hstack([p, np.zeros((p.shape[0], 3 - p.shape[1]))])

    def n01(v):
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-9:
            return np.full(len(v), 0.5)
        return (v - lo) / (hi - lo)

    x, y, z = n01(p[:, 0]), n01(p[:, 1]), n01(p[:, 2])

    hue = (360.0 * x + 42.0 * (y - 0.5) + 24.0 * (z - 0.5)) % 360.0
    saturation = 58.0 + 14.0 * (y - 0.5)
    lightness = 63.0 + 10.0 * (z - 0.5)

    def hsl_to_hex(h, s, lum):
        h /= 360.0
        s /= 100.0
        lum /= 100.0

        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p

        if s == 0:
            r = g = b = lum
        else:
            qv = lum * (1 + s) if lum < 0.5 else lum + s - lum * s
            pv = 2 * lum - qv
            r = hue_to_rgb(pv, qv, h + 1 / 3)
            g = hue_to_rgb(pv, qv, h)
            b = hue_to_rgb(pv, qv, h - 1 / 3)

        return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"

    return {
        cid: hsl_to_hex(float(hue[i]), float(saturation[i]), float(lightness[i]))
        for i, cid in enumerate(cids)
    }


def compute_cats(posts_df: pd.DataFrame, cfg: dict = None) -> pd.Series:
    """Assign a size bucket ('normal'/'big'/'mega') from engagement percentiles.

    Reads render.big_pct / render.mega_pct from config.yaml (defaults 80/95).
    """
    render_cfg = (cfg or {}).get("render", {})
    big_pct = cfg_get(render_cfg, "big_pct", 80, float)
    mega_pct = cfg_get(render_cfg, "mega_pct", 95, float)
    if big_pct <= 1.0:
        big_pct *= 100
    if mega_pct <= 1.0:
        mega_pct *= 100

    cats = pd.Series("normal", index=posts_df.index)
    eng = posts_df["engagement"].apply(sf)
    values = eng.values
    p_big = float(np.percentile(values, big_pct))
    p_mega = float(np.percentile(values, mega_pct))
    cats[eng >= p_mega] = "mega"
    cats[(eng >= p_big) & (eng < p_mega)] = "big"
    return cats


def adept_vis(adept_role) -> str:
    """Return the geometry key for a node role.

    Looked up from NODE_ROLES, overridable via config.yaml render.node_roles.
    """
    role = ss(adept_role).lower()
    if role == "historical_pioneer":
        role = "pioneer"  # legacy alias

    entry = NODE_ROLES.get(role)
    if not entry:
        return NODE_ROLES_DEFAULT["shape"]
    return entry.get("shape", NODE_ROLES_DEFAULT["shape"])


def build_json(posts_df: pd.DataFrame, edges_df: pd.DataFrame, cfg: dict = None) -> dict:
    """Build the {nodes, links} payload embedded in the Three.js template."""
    if cfg is None:
        cfg = {}
    cmap = build_color_maps(posts_df)
    cats = compute_cats(posts_df, cfg)
    max_eng = max(1.0, posts_df["engagement"].apply(sf).max())

    valid_mask = posts_df["x"].notna() & posts_df["y"].notna() & posts_df["z"].notna()
    n_nan = (~valid_mask).sum()
    if n_nan > 0:
        log.warning("%d nodes skipped (x/y/z NaN — id missing from positions_dict)", n_nan)
    posts_df = posts_df[valid_mask].copy()

    nodes = []
    for idx, row in posts_df.iterrows():
        cid = row.get("cluster_id")
        if cid and str(cid) not in ("nan", "None", ""):
            color = cmap.get(cid, "#555")
        else:
            color = "#555555"

        cat = cats[idx]
        eng = sf(row.get("engagement", 0))
        render_cfg = cfg.get("render", {})
        node_size_mult = cfg_get(render_cfg, "node_size_mult", 3.0, float)
        size = round(
            2.5 * (0.6 + 0.4 * math.sqrt(1 + eng / max_eng)) * SIZE_MULT.get(cat, 1.0) * node_size_mult,
            2,
        )

        adept_role = ss(row.get("adept_role", "")) if "adept_role" in posts_df.columns else ""
        nova_role_val = ss(row.get("nova_role", "")) if "nova_role" in posts_df.columns else ""
        vis_role = "nova" if nova_role_val == "pioneer" else adept_role
        geom = adept_vis(vis_role)

        ts = row.get("timestamp", "")
        ts = ts.isoformat() if hasattr(ts, "isoformat") else ss(str(ts))

        nodes.append({
            "id": str(row["id"]),
            "x": sf(row.get("x")),
            "y": sf(row.get("y")),
            "z": sf(row.get("z")),
            "color": color,
            "size": size,
            "geometry": geom,
            "label": {"mega": "MEGA", "big": "BIG"}.get(cat, ""),
            "content": ss(str(row.get("content", ""))[:300]),
            "fullContent": ss(row.get("content", "")),
            "source": ss(row.get("source", "")),
            "timestamp": ts,
            "engagement": sf(row.get("engagement", 0)),
            "eng_cat": cat,
            "cluster_id": ss(str(cid)),
            "cluster_label": ss(row.get("cluster_label", "")) if "cluster_label" in posts_df.columns else "",
            "adept_role": ss(row.get("adept_role", "")) if "adept_role" in posts_df.columns else "",
            "adept_is_pioneer": bool(row.get("adept_is_pioneer", False)) if "adept_is_pioneer" in posts_df.columns else False,
            "nova_role": ss(row.get("nova_role", "")) if "nova_role" in posts_df.columns else "",
            "subtopic_label": ss(row.get("subtopic_label", "")) if "subtopic_label" in posts_df.columns else "",
            "nova_id": ss(row.get("subtopic_id", "")) if "subtopic_id" in posts_df.columns else "",
            "adept_pool_label": ss(row.get("adept_pool_label", "")) if "adept_pool_label" in posts_df.columns else "",
            "adept_pool_hub_id": ss(row.get("adept_pool_hub_id", "")) if "adept_pool_hub_id" in posts_df.columns else "",
            "pool_reasoning": ss(row.get("pool_reasoning", "")) if "pool_reasoning" in posts_df.columns else "",
            "nova_title": ss(row.get("nova_title", "")) if "nova_title" in posts_df.columns else "",
            "nova_reasoning": ss(row.get("nova_reasoning", "")) if "nova_reasoning" in posts_df.columns else "",
            "nova_sentiment_arc": ss(row.get("nova_sentiment_arc", "")) if "nova_sentiment_arc" in posts_df.columns else "",
            "depth": si(row.get("nova_depth", 0)) if "nova_depth" in posts_df.columns else 0,
        })

    links = []
    for _, e in edges_df.iterrows():
        key = ss(e.get("type", ""))
        links.append({
            "source": str(e["source"]),
            "target": str(e["target"]),
            "color": EDGE_COLORS.get(key, "#888"),
            "opacity": EDGE_OPACITY.get(key, 0.5),
            "force": sf(e.get("force", 0.0)),
            "dashed": key in EDGE_DASHED,
            "group": key,
        })

    return {"nodes": nodes, "links": links}
