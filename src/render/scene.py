"""World bounds, nav limits and camera spawn — feeds the `UNIVERSE` JS const."""

from dataclasses import dataclass

import pandas as pd

from .cfgutil import cfg_get


@dataclass
class SceneBounds:
    x_min: float; x_max: float
    y_min: float; y_max: float
    z_min: float; z_max: float
    size_x: float; size_y: float; size_z: float
    max_size: float
    diagonal: float
    nav_min_x: float; nav_max_x: float
    nav_min_y: float; nav_max_y: float
    nav_min_z: float; nav_max_z: float
    camera_spawn_x: float; camera_spawn_y: float; camera_spawn_z: float
    fly_speed_base: float
    far_plane: int


def compute_scene_bounds(posts_df: pd.DataFrame, render_cfg: dict) -> SceneBounds:
    x_min, x_max = float(posts_df["x"].min()), float(posts_df["x"].max())
    y_min, y_max = float(posts_df["y"].min()), float(posts_df["y"].max())
    z_min, z_max = float(posts_df["z"].min()), float(posts_df["z"].max())

    size_x, size_y, size_z = x_max - x_min, y_max - y_min, z_max - z_min
    max_size = max(size_x, size_y, size_z)
    diagonal = (size_x**2 + size_y**2 + size_z**2) ** 0.5

    nav_margin_ratio = cfg_get(render_cfg, "nav_margin_ratio", 0.20, float)
    margin = diagonal * nav_margin_ratio

    nav_min_x, nav_max_x = x_min - margin, x_max + margin
    nav_min_y, nav_max_y = y_min - margin, y_max + margin
    nav_min_z, nav_max_z = z_min - margin, z_max + margin

    fly_speed_base = 100000.0
    far_plane = int(diagonal * 4)

    return SceneBounds(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
        size_x=size_x, size_y=size_y, size_z=size_z, max_size=max_size,
        diagonal=diagonal,
        nav_min_x=nav_min_x, nav_max_x=nav_max_x,
        nav_min_y=nav_min_y, nav_max_y=nav_max_y,
        nav_min_z=nav_min_z, nav_max_z=nav_max_z,
        camera_spawn_x=(nav_min_x + nav_max_x) / 2,
        camera_spawn_y=(nav_min_y + nav_max_y) / 2,
        camera_spawn_z=nav_max_z,
        fly_speed_base=fly_speed_base,
        far_plane=far_plane,
    )


def build_universe_js(b: SceneBounds) -> str:
    """The `UNIVERSE` JS const consumed by assets/app.js."""
    return (
        f"const UNIVERSE = {{\n"
        f"  bounds: {{ x: {{min: {b.x_min}, max: {b.x_max}}}, y: {{min: {b.y_min}, max: {b.y_max}}}, z: {{min: {b.z_min}, max: {b.z_max}}} }},\n"
        f"  size: {{ x: {b.size_x}, y: {b.size_y}, z: {b.size_z}, max: {b.max_size} }},\n"
        f"  navLimits: {{\n"
        f"    x: {{min: {b.nav_min_x}, max: {b.nav_max_x}}},\n"
        f"    y: {{min: {b.nav_min_y}, max: {b.nav_max_y}}},\n"
        f"    z: {{min: {b.nav_min_z}, max: {b.nav_max_z}}}\n"
        f"  }},\n"
        f"  cameraSpawn: {{ x: {b.camera_spawn_x}, y: {b.camera_spawn_y}, z: {b.camera_spawn_z} }},\n"
        f"  flySpeedBase: {b.fly_speed_base},\n"
        f"  farPlane: {b.far_plane}\n"
        f"}};"
    )
