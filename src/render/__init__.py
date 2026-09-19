"""Interactive 3D rendering (Three.js) — optional, post-pipeline.

Runs after the pipeline's checkpointed stages complete (see pipeline.py /
run_store.STAGE_ORDER); invoked via `--render threejs` or
`reload.py --show ... --render threejs`.

Pure Three.js renderer, no physics simulation: node positions (x/y/z) are
precomputed upstream and rendered passively.

    posts_df (x, y, z) + edges_df -> data.build_json() -> nodes[] + links[]
    -> build.render_threejs() -> render.html (BufferGeometry edges, Mesh
       nodes, Pointer Lock free-fly camera, Raycaster hover/click)
"""

from .build import render_threejs, run

__all__ = ["run", "render_threejs"]
