"""Resolve config.yaml's `render.*` section into one RenderConfig."""

from dataclasses import dataclass

from utils.logger import get_logger

from .cfgutil import cfg_get, cfg_get_silent
from .constants import EDGE_COLORS, EDGE_OPACITY, EDGE_WIDTH, NODE_ROLES, NODE_ROLES_DEFAULT, THREE_VERSION

log = get_logger(__name__)


def apply_style_overrides(render_cfg: dict) -> None:
    """Merge config.yaml overrides into the EDGE_*/NODE_ROLES module dicts, in place."""
    if render_cfg.get("edge_opacity"):
        EDGE_OPACITY.update(render_cfg["edge_opacity"])
    if render_cfg.get("edge_colors"):
        EDGE_COLORS.update(render_cfg["edge_colors"])
    if render_cfg.get("edge_width"):
        EDGE_WIDTH.update(render_cfg["edge_width"])

    for role_name, role_def in render_cfg.get("node_roles", {}).items():
        merged = dict(NODE_ROLES.get(role_name, NODE_ROLES_DEFAULT))
        merged.update(role_def or {})
        NODE_ROLES[role_name] = merged


@dataclass
class RenderConfig:
    dash_size: float
    gap_size: float
    flow_speed: float
    tree_dash_size: float
    tree_gap_size: float
    tree_flow_speed: float
    nova_spline: dict
    label_font_size: dict
    label_offset: dict
    label_distance: dict
    label_min_font_size: dict
    label_screen_px: dict
    label_collision_margin_px: float
    label_backdrop_padx_ratio: float
    label_guaranteed_radius_mult: float
    label_max_visible: dict
    label_nudge_enabled: bool
    label_nudge_max_px: float
    label_nudge_iterations: int
    label_nudge_ease_speed: float
    label_fade_speed: float
    cluster_min_radius: float
    three_version: str = THREE_VERSION


def resolve_render_config(render_cfg: dict, diagonal: float) -> RenderConfig:
    apply_style_overrides(render_cfg)

    edge_dash_cfg = render_cfg.get("edge_dash", {})

    nova_spline_cfg = render_cfg.get("nova_spline", {})
    nova_spline = {
        "enabled": cfg_get(nova_spline_cfg, "enabled", True, bool),
        "samples": cfg_get(nova_spline_cfg, "samples", 10, int),
        "tension": cfg_get(nova_spline_cfg, "tension", 0.5, float),
    }

    label_font_cfg = render_cfg.get("label_font_size", {})
    label_font_size = {
        "cluster": cfg_get(label_font_cfg, "cluster", 62, int),
        "nova": cfg_get(label_font_cfg, "nova", 22, int),
        "adept": cfg_get(label_font_cfg, "adept", 22, int),
    }

    label_offset_cfg = render_cfg.get("label_offset", {})
    label_offset = {
        "cluster": cfg_get(label_offset_cfg, "cluster", 28, float),
        "nova": cfg_get(label_offset_cfg, "nova", 20, float),
        "adept": cfg_get(label_offset_cfg, "adept", 20, float),
    }

    # "auto" cluster label distance scales with world diagonal; nova/adept don't.
    label_dist_cfg = render_cfg.get("label_distance", {})
    default_cluster_label_dist = diagonal * 0.45
    raw = label_dist_cfg.get("cluster", "auto")
    if raw is None or (isinstance(raw, str) and raw.strip().lower() == "auto"):
        cluster_dist = default_cluster_label_dist
        log.debug("label_distance.cluster (auto) = %.0f", default_cluster_label_dist)
    else:
        try:
            cluster_dist = float(raw)
        except (TypeError, ValueError):
            log.warning("label_distance.cluster invalid (%r) -> recomputed: %.0f", raw, default_cluster_label_dist)
            cluster_dist = default_cluster_label_dist
    label_distance = {
        "cluster": cluster_dist,
        "nova": cfg_get(label_dist_cfg, "nova", 80, float),
        "adept": cfg_get(label_dist_cfg, "adept", 80, float),
    }

    label_minfont_cfg = render_cfg.get("label_min_font_size", {})
    label_min_font_size = {
        "cluster": cfg_get(label_minfont_cfg, "cluster", 40, float),
        "nova": cfg_get_silent(label_minfont_cfg, "nova", 0, float),
        "adept": cfg_get_silent(label_minfont_cfg, "adept", 0, float),
    }

    label_screenpx_cfg = render_cfg.get("label_screen_px", {})
    label_screen_px = {
        "cluster": cfg_get(label_screenpx_cfg, "cluster", 10, float),
        "nova": cfg_get_silent(label_screenpx_cfg, "nova", 0, float),
        "adept": cfg_get_silent(label_screenpx_cfg, "adept", 0, float),
    }

    label_maxvisible_cfg = render_cfg.get("label_max_visible", {})
    label_max_visible = {"cluster": cfg_get(label_maxvisible_cfg, "cluster", 30, int)}

    label_nudge_cfg = render_cfg.get("label_nudge", {})

    return RenderConfig(
        dash_size=cfg_get(edge_dash_cfg, "dash_size", 20.0, float),
        gap_size=cfg_get(edge_dash_cfg, "gap_size", 14.0, float),
        flow_speed=cfg_get(edge_dash_cfg, "flow_speed", 1.2, float),
        tree_dash_size=cfg_get(edge_dash_cfg, "tree_dash_size", 20.0, float),
        tree_gap_size=cfg_get(edge_dash_cfg, "tree_gap_size", 14.0, float),
        tree_flow_speed=cfg_get(edge_dash_cfg, "tree_flow_speed", 0.6, float),
        nova_spline=nova_spline,
        label_font_size=label_font_size,
        label_offset=label_offset,
        label_distance=label_distance,
        label_min_font_size=label_min_font_size,
        label_screen_px=label_screen_px,
        label_collision_margin_px=cfg_get(render_cfg, "label_collision_margin_px", 6, float),
        label_backdrop_padx_ratio=cfg_get(render_cfg, "label_backdrop_pad_x_ratio", 0.4, float),
        label_guaranteed_radius_mult=cfg_get(render_cfg, "label_guaranteed_radius_mult", 1.2, float),
        label_max_visible=label_max_visible,
        label_nudge_enabled=cfg_get(label_nudge_cfg, "enabled", True, bool),
        label_nudge_max_px=cfg_get(label_nudge_cfg, "max_px", 140.0, float),
        label_nudge_iterations=cfg_get(label_nudge_cfg, "iterations", 4, int),
        label_nudge_ease_speed=cfg_get(label_nudge_cfg, "ease_speed", 0.2, float),
        label_fade_speed=cfg_get(render_cfg, "label_fade_speed", 0.25, float),
        cluster_min_radius=cfg_get(render_cfg, "cluster_min_radius", 150, float),
    )
