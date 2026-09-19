# 3D view

`render.py` writes a self-contained Three.js scene to `render.html` at the run root. Open it directly in a browser, or from the dashboard's **Generate 3D view** action.

The file is written from `output/clusters_render.parquet` and `output/edges.parquet`, so it only exists once a run is complete. `pipeline.py --render threejs` writes it at the end of a run without opening a tab; `reload.py show <run_id> --render threejs` regenerates it later.

## Layout

- **Canvas** — the scene itself, filling the window.
- **Top bar** — universe size, current speed, position, and a one-line control reminder.
- **Sidebar** (right, two tabs):
  - **Display** — toggle visibility by category: Nova, ADEPT, Edges (`semantic_inter`, `temporal`, `temporal_influence`), and View (labels, edge fade).
  - **Map** — every cluster, searchable. Expand a row to browse its posts, sorted by engagement, with "+N more" beyond 15.
- **Controls legend** (bottom-left) — always-visible reminder of the tables below.

Node colours and shapes follow the node's role (`pioneer`, `hub`, `amplifier`, `latest`, `member`, `offspring`, `orphan`) and node size follows engagement percentile. Both are configurable under `render.*` in [`CONFIGURATION.md`](CONFIGURATION.md#render).

## Moving around

| Input | Action |
|---|---|
| `W` / `A` / `S` / `D` or arrow keys | Move |
| `Q` / `E` | Up / down |
| `Shift` | ×5 speed |
| Scroll | Adjust base flight speed |
| Right-click (hold) | Camera look |
| `Space` | Free look (toggle; `Space` again, or click, to exit) |
| `F` | Overview (return to the spawn point) |

`W` / `A` / `S` / `D` are physical key positions, so on an AZERTY keyboard this is `Z` / `Q` / `S` / `D`. On touch devices, a single-finger drag does the same as camera look.

## Selecting

| Input | Action |
|---|---|
| Click a node or edge | Enter Tree View, rooted there |
| Click a cluster label | Fly to that cluster |
| Click empty space | Deselect |

## Tree View

A flattened, front-on layout of one narrative chain (a Nova subtopic or an ADEPT pool), reached by clicking any node or edge. Its sidebar shows node, cluster, and edge counts plus maximum depth.

| Input | Action |
|---|---|
| Right-click + drag | Pan |
| Scroll | Zoom |
| Double-click, `Esc`, or the **×** button | Back to the universe |

## Fullscreen

Top-right button, or `F11`. While fullscreen, `Esc` must be held rather than tapped, so a quick press does not drop you out by accident.

## See also

- [`CONFIGURATION.md`](CONFIGURATION.md#render) — colours, sizes, labels, and splines
- [`ALGORITHMS.md`](ALGORITHMS.md#layout--forceatlas2-in-3d) — how the positions are computed
- [`DASHBOARD.md`](DASHBOARD.md) — opening the view from the web UI
- [`CLI.md`](CLI.md#reloadpy) — regenerating `render.html` for an existing run
