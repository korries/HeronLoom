"""Step 6 — 3D hybrid layout: PaCMAP global + FA2 local.

PaCMAP positions clusters in 3D space (macro-structure), preserving both
local and global structure. FA2 then refines intra-cluster geometry
(micro-structure). XYZ coordinates are exported for the Three.js render
(see render.py).
"""

import logging

import numpy as np
import pandas as pd

from run_store import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

logging.getLogger("pacmap").setLevel(logging.ERROR)


def run_pacmap_3d(
    posts_df: pd.DataFrame, cfg: dict
) -> dict:
    """Project all posts to 3D via PaCMAP on LDA or raw embeddings.

    PaCMAP preserves both local (neighbourhood) and global (inter-cluster)
    structure, unlike UMAP which favours the local. This prevents hub nodes
    from appearing visually distant from their true neighbours.

    Returns:
        Dict mapping post_id → (x, y, z).
    """
    try:
        import pacmap
    except ImportError as exc:
        raise ImportError(
            "pacmap not found — install: pip install pacmap"
        ) from exc

    pcfg = cfg.get("pacmap_layout", {})

    # Embedding space for the 3D layout:
    #   "lda" → embedding_lda
    #   "raw" → embedding_raw
    layout_space = pcfg.get("layout_space", "lda")

    if layout_space == "lda":
        emb_col  = "embedding_lda"
        distance = "euclidean"   # LDA is not L2-normalised → Euclidean
    else:
        emb_col  = "embedding_raw"
        distance = pcfg.get("metric", "cosine")  # raw embeddings are L2-normalised → cosine

    embeddings = np.array(posts_df[emb_col].tolist())  # shape (N, D)

    logger.info(
        "PaCMAP 3D — %s posts x %dd  (space=%s, distance=%s)",
        f"{len(posts_df):,}", embeddings.shape[1], layout_space, distance,
    )

    n_components = pcfg.get("n_components", 3)
    n_neighbors  = pcfg.get("n_neighbors", 15)
    MN_ratio     = pcfg.get("MN_ratio", 0.5)
    FP_ratio     = pcfg.get("FP_ratio", 2.0)
    random_state = pcfg.get("random_state", 42)

    reducer = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        distance=distance,
        apply_pca=False,       # LDA/raw already in the right space
        random_state=random_state,
        verbose=False,
    )
    coords_3d = reducer.fit_transform(embeddings)  # shape (N, 3)

    # Centre (pure translation, no distortion)
    for axis in range(3):
        coords_3d[:, axis] -= coords_3d[:, axis].mean()

    # Scale XYZ — two modes via pacmap_layout.scale_mode:
    #   "per_axis" : each axis normalised independently → spherical universe.
    #                Compensates for Z naturally compressed by PaCMAP (3rd
    #                component captures less variance → flat if uniform scale).
    #   "uniform"  : single global max_abs → PaCMAP proportions preserved,
    #                geometry faithful but Z often shorter.
    output_scale = pcfg.get("output_scale", 500)
    scale_mode = pcfg.get("scale_mode", "per_axis")
    logger.info("PaCMAP scale_mode='%s'", scale_mode)

    if scale_mode == "uniform":
        max_abs = np.abs(coords_3d).max()
        if max_abs > 0:
            coords_3d = coords_3d / max_abs * output_scale
    else:  # per_axis (default)
        for axis in range(3):
            max_abs_axis = np.abs(coords_3d[:, axis]).max()
            if max_abs_axis > 0:
                coords_3d[:, axis] = coords_3d[:, axis] / max_abs_axis * output_scale

    positions = {
        row["id"]: (
            float(coords_3d[i, 0]),
            float(coords_3d[i, 1]),
            float(coords_3d[i, 2]),
        )
        for i, (_, row) in enumerate(posts_df.iterrows())
    }

    logger.info(
        "PaCMAP OK — X: [%.1f, %.1f]  Y: [%.1f, %.1f]  Z: [%.1f, %.1f]",
        coords_3d[:,0].min(), coords_3d[:,0].max(),
        coords_3d[:,1].min(), coords_3d[:,1].max(),
        coords_3d[:,2].min(), coords_3d[:,2].max(),
    )

    return positions


def _straighten_nova_chains(
    posts_df: pd.DataFrame,
    final_positions: dict,
    cfg: dict,
) -> dict:
    """Re-project each Nova narrative chain onto its own principal axis.

    FA2 only models pairwise attraction (parent<->child) plus global
    repulsion/gravity — it has no notion that a whole subtopic_id is meant
    to read as a single sequence. Once a chain has more than a handful of
    nodes and shares cluster space with other chains, nothing stops a
    middle node from drifting toward a spot already claimed by another part
    of the same chain, producing loops/squares/knots in the render instead
    of a legible line. Short chains (2-3 nodes) rarely show this because
    there simply aren't enough attractions to fold back on themselves.

    Fits each chain's own principal axis via SVD, then re-sorts the FA2
    positions along that axis in chain order — guaranteeing a monotonic,
    non-crossing sequence. The perpendicular residual from FA2 is kept,
    attenuated by ``residual_factor``, so some organic wobble survives
    (0 = perfectly straight rod, 1 = FA2's result untouched, loops included).

    Runs after the per-cluster FA2 pass, on final world-space positions, so
    it works regardless of per-cluster scaling/centring.

    Args:
        posts_df: must contain ``id``, ``subtopic_id``, ``nova_depth``.
        final_positions: post_id -> (x, y, z), mutated in place and returned.
        cfg: pipeline config; reads ``fa2.nova_straighten``.

    Returns:
        The same ``final_positions`` dict, with chain member positions
        straightened where applicable.
    """
    scfg = cfg.get("fa2", {}).get("nova_straighten", {})
    if not scfg.get("enabled", True):
        return final_positions

    if "subtopic_id" not in posts_df.columns or "nova_depth" not in posts_df.columns:
        logger.debug("nova straighten skipped — subtopic_id/nova_depth not in posts_df")
        return final_positions

    residual_factor = float(scfg.get("residual_factor", 0.35))
    min_chain_len    = int(scfg.get("min_chain_len", 3))

    chains = posts_df.dropna(subset=["subtopic_id"]).groupby("subtopic_id")

    n_straightened = 0
    for _subtopic_id, group in chains:
        ordered = group.sort_values("nova_depth")
        pids    = [pid for pid in ordered["id"].tolist() if pid in final_positions]
        if len(pids) < min_chain_len:
            continue

        pts = np.array([final_positions[pid] for pid in pids], dtype=np.float64)

        centroid = pts.mean(axis=0)
        centered = pts - centroid

        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        axis = vt[0]

        # Orient the axis pioneer → latest so chain order and axis direction agree
        # (cosmetic only — the monotonic re-sort below is what kills loops).
        if np.dot(pts[-1] - pts[0], axis) < 0:
            axis = -axis

        proj        = centered @ axis                    # signed position along the axis
        proj_sorted = np.sort(proj)                       # same spread, now monotonic in chain order
        perp        = centered - np.outer(proj, axis)     # FA2's off-axis residual

        new_pts = centroid + np.outer(proj_sorted, axis) + perp * residual_factor

        for pid, new_pt in zip(pids, new_pts, strict=False):
            final_positions[pid] = (float(new_pt[0]), float(new_pt[1]), float(new_pt[2]))
        n_straightened += 1

    logger.info(
        "nova straighten — %d/%d chains re-projected (residual_factor=%.2f, min_chain_len=%d)",
        n_straightened, chains.ngroups, residual_factor, min_chain_len,
    )
    return final_positions

SIZE_MULT = {"mega": 2.5, "big": 1.5, "normal": 1.0}
_BASE_GEOM_SCALE = 1.5  # BASE_GEOMS[...] = unit-radius geometry, scale = size * 1.5


def _sf(v, d=0.0):
    """Safe float — same behaviour as render.py::_sf."""
    try:
        if v is None:
            return d
        f = float(v)
        return d if np.isnan(f) else f
    except (TypeError, ValueError):
        return d


def _cfg_get(d, key, default, cast=None):
    v = d.get(key, default) if isinstance(d, dict) else default
    if v is None:
        return default
    if cast is not None:
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default
    return v


def _compute_node_radius(posts_df: pd.DataFrame, cfg: dict) -> dict:
    """Real radius of each node in the 3D scene — identical to what
    render.py computes for display (same percentiles, same size
    formula, same geometry-to-world-radius conversion). This radius
    (not an arbitrary constant) is what all spacing below is based on:
    if ``render.node_size_mult`` goes from 2 to 10 in config.yaml, the
    margins grow accordingly, automatically.

    Args:
        posts_df: must contain 'id'; 'engagement' ideally (0 otherwise).
        cfg: reads render.big_pct / render.mega_pct / render.node_size_mult.

    Returns:
        dict post_id -> radius (float), same units as final_positions x/y/z.
    """
    render_cfg = cfg.get("render", {})
    big_pct = _cfg_get(render_cfg, "big_pct", 80, float)
    mega_pct = _cfg_get(render_cfg, "mega_pct", 95, float)
    if big_pct <= 1.0:
        big_pct *= 100
    if mega_pct <= 1.0:
        mega_pct *= 100
    node_size_mult = _cfg_get(render_cfg, "node_size_mult", 3.0, float)

    eng = posts_df["engagement"].apply(_sf) if "engagement" in posts_df.columns else pd.Series(0.0, index=posts_df.index)
    max_eng = max(1.0, float(eng.max()))
    p_big = float(np.percentile(eng.values, big_pct))
    p_mega = float(np.percentile(eng.values, mega_pct))

    cats = pd.Series("normal", index=posts_df.index)
    cats[eng >= p_mega] = "mega"
    cats[(eng >= p_big) & (eng < p_mega)] = "big"

    size = 2.5 * (0.6 + 0.4 * np.sqrt(1 + eng / max_eng)) * cats.map(SIZE_MULT).astype(float) * node_size_mult
    radius = size * _BASE_GEOM_SCALE

    return dict(zip(posts_df["id"], radius, strict=False))


def _min_cluster_radius(cfg: dict) -> float:
    """Minimum radius a cluster occupies for inter-cluster spacing —
    same value as ``CLUSTER_MIN_RADIUS`` in render.py
    (``render.cluster_min_radius``). Recomputed here independently, from
    the same config.yaml, to stay in sync with the render without a
    cross-dependency between the two files.

    Used ONLY as the "effective" radius for spacing BETWEEN clusters
    (_resolve_cluster_overlaps / _declump_clusters below): it never
    changes a cluster's actual internal geometry (its nodes stay where
    FA2 placed them) — only the minimum distance kept from its
    neighbours, so that an enlarged render-side halo (a small cluster
    below the floor — see render.py) never ends up overlapping its
    neighbour's.

    Args:
        cfg: reads render.cluster_min_radius.

    Returns:
        Minimum radius (same units as final_positions x/y/z), >= 0.
    """
    render_cfg = cfg.get("render", {})
    return max(0.0, _cfg_get(render_cfg, "cluster_min_radius", 150, float))


def _space_out_chain_nodes(
    posts_df: pd.DataFrame,
    final_positions: dict,
    node_radius: dict,
    cfg: dict,
) -> dict:
    """Guarantees a minimum spacing between CONSECUTIVE nodes of the same
    nova chain, based on their real render radius — not a point.

    Only STRETCHES spacing along the axis already computed by
    ``_straighten_nova_chains`` when needed (never compresses); the
    perpendicular component (the slight curvature residual — see
    fa2.nova_straighten.residual_factor) is left untouched, so the
    chain's shape (straight or slightly curved) is preserved. The
    stretch is recentred on the chain's original centre so it doesn't
    drift.

    The margin is MULTIPLICATIVE on (radius_i + radius_j) — same as at
    the block level (``_pack_cluster_blocks``) — not additive: 1.0 = the
    spheres are tangent, 1.3 = 30% extra empty space, etc. An additive
    value doesn't work here since a real radius can be tens of units
    (mega + high node_size_mult) — adding 1 to that would be invisible
    on screen.

    Runs after ``_straighten_nova_chains``, before ``_reorganize_cluster_blocks``
    (so the bubble size used by block packing already reflects the
    correct internal spacing).

    Args:
        posts_df: must contain 'id', 'subtopic_id' (and ideally 'nova_depth').
        final_positions: post_id -> (x, y, z). Mutated and returned.
        node_radius: post_id -> real radius, see ``_compute_node_radius``.
        cfg: reads fa2.node_spacing.margin / .enabled.
    """
    scfg = cfg.get("fa2", {}).get("node_spacing", {})
    if not scfg.get("enabled", True) or "subtopic_id" not in posts_df.columns:
        return final_positions

    margin = float(scfg.get("margin", 1.4))
    depth_col = "nova_depth" if "nova_depth" in posts_df.columns else None
    chains = posts_df.dropna(subset=["subtopic_id"]).groupby("subtopic_id")
    n_adjusted = 0

    for _subtopic_id, grp in chains:
        pids = [pid for pid in grp["id"] if pid in final_positions]
        if len(pids) < 2:
            continue

        if depth_col:
            depths = grp.set_index("id")[depth_col].reindex(pids)
            pids = [p for p, _ in sorted(
                zip(pids, depths, strict=False), key=lambda t: (np.inf if pd.isna(t[1]) else t[1])
            )]

        pts = np.array([final_positions[p] for p in pids], dtype=np.float64)
        centroid = pts.mean(axis=0)
        direction = pts[-1] - pts[0]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction = direction / norm

        rel = pts - centroid
        t = rel @ direction                      # position along the axis
        perp = rel - np.outer(t, direction)       # perpendicular residual — untouched

        radii = np.array([node_radius.get(p, 1.0) for p in pids])
        t_new = t.copy()
        for i in range(1, len(t_new)):
            min_gap = (radii[i - 1] + radii[i]) * margin
            if t_new[i] - t_new[i - 1] < min_gap:
                t_new[i] = t_new[i - 1] + min_gap

        t_new -= (t_new.mean() - t.mean())         # recentre — doesn't drift the block
        if not np.allclose(t_new, t, atol=1e-6):
            n_adjusted += 1

        new_pts = centroid + perp + np.outer(t_new, direction)
        for p, pt in zip(pids, new_pts, strict=False):
            final_positions[p] = (float(pt[0]), float(pt[1]), float(pt[2]))

    logger.info(
        "chain node spacing — %d/%d chains stretched (margin=%.2f)",
        n_adjusted, chains.ngroups, margin,
    )
    return final_positions


def _space_out_pool_nodes(
    posts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    final_positions: dict,
    node_radius: dict,
    cfg: dict,
) -> dict:
    """Guarantees a minimum spacing between an ADEPT hub and each of its
    spokes, based on their real render radius: any spoke that's too close
    is pushed out radially, along its own hub -> spoke direction, until
    (hub_radius + spoke_radius) * margin. Only ever pushes apart, never
    pulls together.

    MULTIPLICATIVE margin, same convention as ``_space_out_chain_nodes`` /
    ``_pack_cluster_blocks``: 1.0 = tangent, 1.3 = 30% extra space.

    Only handles the radial distance to the hub — two spokes of the same
    pool still too close to each other after this (rare, FA2's star
    layout already separates them angularly) are caught by the global
    safety net ``_resolve_node_collisions`` that runs right after.

    Args:
        posts_df, final_positions, node_radius: see ``_space_out_chain_nodes``.
        edges_df: read for 'adept_spoke' edges (hub -> member).
        cfg: reads fa2.node_spacing.margin / .enabled (same setting as chains).
    """
    scfg = cfg.get("fa2", {}).get("node_spacing", {})
    if not scfg.get("enabled", True):
        return final_positions
    if edges_df is None or not len(edges_df) or "type" not in edges_df.columns:
        return final_positions

    margin = float(scfg.get("margin", 1.4))
    spokes_df = edges_df[edges_df["type"] == "adept_spoke"]
    if not len(spokes_df):
        return final_positions

    n_adjusted = 0
    for hub, grp in spokes_df.groupby("source"):
        if hub not in final_positions:
            continue
        hub_pt = np.array(final_positions[hub], dtype=np.float64)
        hub_r = node_radius.get(hub, 1.0)

        for member in grp["target"]:
            if member not in final_positions or member == hub:
                continue
            pt = np.array(final_positions[member], dtype=np.float64)
            direction = pt - hub_pt
            dist = float(np.linalg.norm(direction))
            min_dist = (hub_r + node_radius.get(member, 1.0)) * margin
            if dist >= min_dist:
                continue

            if dist > 1e-6:
                direction = direction / dist
            else:
                rng = np.random.default_rng(abs(hash(member)) % (2**32))
                direction = rng.normal(size=3)
                direction = direction / (np.linalg.norm(direction) + 1e-9)

            new_pt = hub_pt + direction * min_dist
            final_positions[member] = (float(new_pt[0]), float(new_pt[1]), float(new_pt[2]))
            n_adjusted += 1

    logger.info(
        "pool node spacing — %d spokes pushed out from their hub (margin=%.2f)",
        n_adjusted, margin,
    )
    return final_positions


def _resolve_node_collisions(
    posts_df: pd.DataFrame,
    final_positions: dict,
    node_radius: dict,
    cfg: dict,
) -> dict:
    """Final safety net: guarantees that NO pair of nodes in the same
    cluster overlaps — chain, pool, or isolated node, whatever the
    block — based on their real render radius. This is what specifically
    covers isolated nodes (no other pass handles them individually).

    MULTIPLICATIVE margin on (radius_i + radius_j), same convention as
    ``_space_out_chain_nodes`` / ``_pack_cluster_blocks``: 1.0 = tangent,
    1.4 = 40% extra space. This is the setting to raise if clumps of
    nodes still look stuck together visually — an additive margin isn't
    enough once the real radius exceeds a few units.

    Done directly, without simulation: per cluster, nodes are sorted
    along one axis (so only nearby pairs are compared, not O(n²) over the
    whole cluster), and every detected conflict is resolved by pushing
    the two nodes apart equally along the line joining them. Repeats
    until there's no conflict left or ``max_passes`` is reached (a
    safety guard, not a setting to tune case by case).

    Args:
        posts_df: must contain 'id', 'cluster_id'.
        final_positions: post_id -> (x, y, z). Mutated and returned.
        node_radius: post_id -> real radius, see ``_compute_node_radius``.
        cfg: reads fa2.node_spacing.margin / .max_passes / .enabled.
    """
    scfg = cfg.get("fa2", {}).get("node_spacing", {})
    if not scfg.get("enabled", True) or "cluster_id" not in posts_df.columns:
        return final_positions

    margin = float(scfg.get("margin", 1.4))
    max_passes = int(scfg.get("max_passes", 6))
    total_fixed = 0

    for _cluster_id, grp in posts_df.dropna(subset=["cluster_id"]).groupby("cluster_id"):
        pids = [pid for pid in grp["id"] if pid in final_positions]
        n = len(pids)
        if n < 2:
            continue

        radii = np.array([node_radius.get(p, 1.0) for p in pids])

        for _ in range(max_passes):
            pos = np.array([final_positions[p] for p in pids], dtype=np.float64)
            any_conflict = False

            order = np.argsort(pos[:, 0])
            max_gap_x = float(radii.max()) * 2 * margin

            for oi in range(n):
                i = order[oi]
                for oj in range(oi + 1, n):
                    j = order[oj]
                    if pos[j, 0] - pos[i, 0] > max_gap_x:
                        break  # sorted by x -> nothing closer past this point
                    d = pos[j] - pos[i]
                    dist = float(np.linalg.norm(d))
                    min_dist = (radii[i] + radii[j]) * margin
                    if dist < min_dist:
                        any_conflict = True
                        total_fixed += 1
                        push = (min_dist - dist) / 2 + 1e-3
                        dirv = d / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
                        pos[i] -= dirv * push
                        pos[j] += dirv * push

            for p, pt in zip(pids, pos, strict=False):
                final_positions[p] = (float(pt[0]), float(pt[1]), float(pt[2]))

            if not any_conflict:
                break

    logger.info("node collisions — %d conflicts resolved (margin=%.2f)", total_fixed, margin)
    return final_positions


def _assign_block_id(posts_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.Series:
    """Group every post into exactly one "block": a Nova narrative chain, an
    ADEPT pool (hub + spokes), or — failing both — a singleton block made of
    just that post.

    Priority: a post already in a Nova chain (subtopic_id) keeps that
    membership even if it's also referenced by an adept_spoke edge (a
    grafted pool bridges to a Nova node without changing that node's own
    block).

    Args:
        posts_df: needs 'id', ideally 'subtopic_id'.
        edges_df: needs 'source', 'target', 'type' — reads 'adept_spoke'
            edges (hub -> member) to reconstruct ADEPT pools.

    Returns:
        pd.Series aligned with posts_df.index — block id (str) per post.
    """
    block_id = posts_df["id"].astype(str).copy()  # default: singleton = its own id
    block_id.index = posts_df.index

    has_chain = pd.Series(False, index=posts_df.index)
    if "subtopic_id" in posts_df.columns:
        has_chain = posts_df["subtopic_id"].notna()
        block_id[has_chain] = "nova:" + posts_df.loc[has_chain, "subtopic_id"].astype(str)

    if edges_df is not None and len(edges_df) and "type" in edges_df.columns:
        spokes = edges_df[edges_df["type"] == "adept_spoke"]
        pool_of: dict = {}
        for _, e in spokes.iterrows():
            pool_of.setdefault(e["source"], e["source"])  # the hub is its own pool
            pool_of[e["target"]] = e["source"]

        id_to_idx = dict(zip(posts_df["id"], posts_df.index, strict=False))
        for pid, hub in pool_of.items():
            idx = id_to_idx.get(pid)
            if idx is None or has_chain.get(idx, False):
                continue
            block_id.loc[idx] = "adept:" + str(hub)

    return block_id


def _pack_cluster_blocks(cluster_center: np.ndarray, bubbles: list, margin: float, rng) -> list:
    """Places each bubble (centroid + radius) without overlap, via direct
    search — no simulation, no iterations to tune.

    For each block, largest first: start from its original direction
    relative to the cluster center (this preserves the semantic layout
    already computed by PaCMAP/FA2 — no arbitrary rearranging), and if
    the spot is taken, move away from the center in small steps until a
    free spot is found, deviating the direction slightly if several
    bubbles block the same line.

    Args:
        cluster_center: (3,) ndarray, cluster center.
        bubbles: list of dicts with "centroid" (3,) and "radius" (float).
            Mutated: each dict receives "new_centroid" (3,) as output.
        margin: safety factor on (r_i + r_j) — 1.0 = bubbles tangent.
        rng: np.random.Generator, for reproducible deviations.

    Returns:
        The same ``bubbles`` list, with "new_centroid" filled in.
    """
    bubbles = sorted(bubbles, key=lambda b: -b["radius"])
    placed: list = []

    for b in bubbles:
        direction = b["centroid"] - cluster_center
        base_dist = float(np.linalg.norm(direction))
        if base_dist < 1e-6:
            direction = rng.normal(size=3)
            base_dist = 0.0
        direction = direction / (np.linalg.norm(direction) + 1e-9)

        step = max(b["radius"], 1.0) * 0.6
        candidate = cluster_center + direction * base_dist
        for i in range(60):
            dist = base_dist + i * step
            candidate = cluster_center + direction * dist
            if all(
                np.linalg.norm(candidate - pc) >= (b["radius"] + pr) * margin
                for pc, pr in placed
            ):
                break
            if i % 4 == 3:  # stuck in this direction -> deviate a bit
                direction = direction + rng.normal(size=3) * 0.3
                direction = direction / (np.linalg.norm(direction) + 1e-9)

        placed.append((candidate, b["radius"]))
        b["new_centroid"] = candidate

    return bubbles


def _reorganize_cluster_blocks(
    posts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    final_positions: dict,
    cfg: dict,
    node_radius: dict = None,
) -> dict:
    """Repositions each Nova chain and each ADEPT pool as a whole block
    inside its cluster, so they no longer visually overlap.

    Unlike ``_declump_clusters`` / ``_resolve_cluster_overlaps`` (which
    separate entire *clusters* in world space), this pass works *inside*
    a single cluster at a time — a block's internal shape (a straightened
    Nova chain's line, an ADEPT pool's star) is never altered, only its
    overall position moves.

    A bubble's radius includes the REAL render radius of each member
    node (not just the spread between points), so that a solo block — an
    isolated node — gets a bubble at its true visual size rather than a
    point of arbitrary radius.

    Runs after ``_space_out_chain_nodes`` / ``_space_out_pool_nodes``
    (blocks already have their final internal spacing) and before the
    macro passes ``_declump_clusters`` / ``_resolve_cluster_overlaps``,
    which only move whole clusters and are therefore unaffected by the
    order.

    Args:
        posts_df: must contain 'id' and 'cluster_id' (and ideally
            'subtopic_id' for Nova chains).
        edges_df: read for 'adept_spoke' edges (hub -> member), to
            reconstruct ADEPT pools — see ``_assign_block_id``.
        final_positions: post_id -> (x, y, z). Mutated and returned.
        cfg: reads fa2.block_reorganize.
        node_radius: post_id -> real radius (see ``_compute_node_radius``).
            Falls back to a radius of 1.0 if absent (degraded behaviour).
    """
    bcfg = cfg.get("fa2", {}).get("block_reorganize", {})
    if not bcfg.get("enabled", True):
        return final_positions

    if "cluster_id" not in posts_df.columns:
        logger.debug("block reorganize skipped — no cluster_id column")
        return final_positions

    margin = float(bcfg.get("margin", 1.15))
    min_blocks = int(bcfg.get("min_blocks_to_run", 2))
    node_radius = node_radius or {}

    block_id = _assign_block_id(posts_df, edges_df)
    rng = np.random.default_rng(0)

    n_clusters_touched = 0
    n_blocks_moved = 0

    for _cluster_id, cluster_group in posts_df.dropna(subset=["cluster_id"]).groupby("cluster_id"):
        cluster_pids = [pid for pid in cluster_group["id"] if pid in final_positions]
        if len(cluster_pids) < 4:
            continue

        pid_to_block = dict(zip(cluster_group["id"], block_id.loc[cluster_group.index], strict=False))

        blocks: dict = {}
        for pid in cluster_pids:
            blocks.setdefault(pid_to_block.get(pid, pid), []).append(pid)

        if len(blocks) < min_blocks:
            continue  # a single block (or fewer) in this cluster — nothing to separate

        cluster_center = np.array(
            [final_positions[pid] for pid in cluster_pids], dtype=np.float64
        ).mean(axis=0)

        bubbles = []
        for bid, pids in blocks.items():
            pts = np.array([final_positions[pid] for pid in pids], dtype=np.float64)
            centroid = pts.mean(axis=0)
            own_r = np.array([node_radius.get(pid, 1.0) for pid in pids])
            # bubble radius = farthest centroid->point distance + that
            # point's own render radius, maximized over all members —
            # this captures each node's true visual size, not just its position
            radius = float(np.max(np.linalg.norm(pts - centroid, axis=1) + own_r))
            bubbles.append({"id": bid, "pids": pids, "centroid": centroid, "radius": max(radius, 0.5)})

        bubbles = _pack_cluster_blocks(cluster_center, bubbles, margin, rng)

        n_clusters_touched += 1
        for b in bubbles:
            delta = b["new_centroid"] - b["centroid"]
            if np.allclose(delta, 0, atol=1e-6):
                continue
            n_blocks_moved += 1
            for pid in b["pids"]:
                x, y, z = final_positions[pid]
                final_positions[pid] = (x + delta[0], y + delta[1], z + delta[2])

    logger.info(
        "block reorganize — %d clusters touched, %d blocks translated (margin=%.2f)",
        n_clusters_touched, n_blocks_moved, margin,
    )
    return final_positions


def _resolve_cluster_overlaps(
    posts_df: pd.DataFrame,
    final_positions: dict,
    cfg: dict,
) -> dict:
    """Push apart cluster bounding spheres that overlap, without touching
    internal FA2 geometry or rescaling non-conflicting clusters.

    Same idea as PRISM (Gansner & Hu 2010) / d3-force forceCollide:
    iterative pairwise push on centroid+radius spheres, since separating
    one pair can create a new conflict elsewhere. anchor_strength cools
    to 0 across iterations so it doesn't leave residual overlap.

    Runs after _straighten_nova_chains — rigid translation preserves
    internal chain shape.

    Args:
        posts_df: needs 'id' and 'cluster_id'.
        final_positions: post_id -> (x, y, z). Mutated in place.
        cfg: reads fa2.cluster_overlap_removal.
    """
    ocfg = cfg.get("fa2", {}).get("cluster_overlap_removal", {})
    if not ocfg.get("enabled", True):
        return final_positions

    if "cluster_id" not in posts_df.columns:
        logger.debug("cluster overlap removal skipped — no cluster_id column")
        return final_positions

    margin          = float(ocfg.get("margin", 1.05))
    max_iter        = int(ocfg.get("max_iter", 200))
    damping         = float(ocfg.get("damping", 0.5))
    mass_weighting  = bool(ocfg.get("mass_weighting", True))
    anchor_strength = float(ocfg.get("anchor_strength", 0.02))

    # Minimum "effective" radius — see _min_cluster_radius: a small
    # cluster is treated here as at least this big, otherwise two small
    # clusters close to each other can end up with halos (enlarged on
    # the render side) that overlap each other, even if this pass found
    # no conflict at the time.
    min_radius = _min_cluster_radius(cfg)

    # One bounding sphere per cluster
    cluster_ids, centroids, radii, sizes, members = [], [], [], [], []
    for cid, group in posts_df.dropna(subset=["cluster_id"]).groupby("cluster_id"):
        pids = [pid for pid in group["id"].tolist() if pid in final_positions]
        if not pids:
            continue
        pts = np.array([final_positions[pid] for pid in pids], dtype=np.float64)
        centroid = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - centroid, axis=1).max()) if len(pts) > 1 else 0.0
        radius = max(radius, min_radius)

        cluster_ids.append(cid)
        centroids.append(centroid)
        radii.append(radius)
        sizes.append(len(pids))
        members.append(pids)

    n = len(cluster_ids)
    if n < 2:
        return final_positions

    centroids = np.array(centroids, dtype=np.float64)  # (n, 3)
    radii     = np.array(radii, dtype=np.float64)       # (n,)
    sizes     = np.array(sizes, dtype=np.float64)       # (n,)
    origin    = centroids.copy()                         # anchor target = pre-resolution position

    # bigger clusters move less; normalised so avg inv_mass = 1
    if mass_weighting:
        inv_mass = 1.0 / sizes
        inv_mass = inv_mass / inv_mass.mean()
    else:
        inv_mass = np.ones(n)

    rng = np.random.default_rng(0)
    n_iters_used = max_iter
    for iteration in range(max_iter):
        diff = centroids[:, None, :] - centroids[None, :, :]       # (n, n, 3)
        dist = np.linalg.norm(diff, axis=2)                        # (n, n)
        np.fill_diagonal(dist, np.inf)

        min_dist = (radii[:, None] + radii[None, :]) * margin      # (n, n)
        overlap  = np.clip(min_dist - dist, 0, None)                # 0 = no conflict

        n_conflicts = int((overlap > 0).sum() / 2)
        if n_conflicts == 0:
            n_iters_used = iteration
            break

        # unit direction i->away from j; random fallback if centroids coincide
        safe_dist = np.where(dist > 0, dist, 1.0)
        direction = diff / safe_dist[..., None]
        coincident = (dist == 0)
        if coincident.any():
            rand_dirs = rng.normal(size=(n, n, 3))
            rand_dirs /= np.linalg.norm(rand_dirs, axis=2, keepdims=True) + 1e-12
            direction = np.where(coincident[..., None], rand_dirs, direction)

        push = direction * overlap[..., None] * 0.5 * damping        # (n, n, 3)
        displacement = push.sum(axis=1) * inv_mass[:, None]          # (n, 3)

        # anchor cools to 0, otherwise it settles at an equilibrium with residual overlap
        anchor_now = anchor_strength * (1.0 - iteration / max_iter)
        displacement += (origin - centroids) * anchor_now

        centroids = centroids + displacement

    logger.info(
        "cluster overlap removal — %d clusters, converged in %d/%d iterations "
        "(margin=%.2f, damping=%.2f, mass_weighting=%s)",
        n, n_iters_used, max_iter, margin, damping, mass_weighting,
    )

    n_moved = 0
    for pids, centroid, origin_c in zip(members, centroids, origin, strict=False):
        delta = centroid - origin_c
        if np.allclose(delta, 0, atol=1e-9):
            continue
        n_moved += 1
        for pid in pids:
            x, y, z = final_positions[pid]
            final_positions[pid] = (x + delta[0], y + delta[1], z + delta[2])

    logger.info("cluster overlap removal — %d/%d clusters translated", n_moved, n)
    return final_positions


def _declump_clusters(
    posts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    final_positions: dict,
    cfg: dict,
) -> dict:
    """Spread cluster centroids apart with a continuous n-body repulsion,
    proportional to local crowding — an organic alternative to a uniform
    ``fa2.universe_scale`` multiply.

    ``universe_scale`` alone can't do this: it's one linear factor applied
    identically to every centroid, so it can only zoom — it never changes
    the *ratio* between a tight group's internal spacing and the spacing
    between groups. Raising ``cluster_overlap_removal.margin`` isn't a fix
    either: that pass only pushes a pair apart past a hard distance
    threshold, which flattens density contrast instead of preserving it.

    This pass treats every cluster as a point mass and repels every pair
    continuously, with a force that decays with distance — same spirit as
    ForceAtlas2's own repulsion, one level up (centroids instead of posts).
    Tightly packed groups get pushed apart proportionally more than
    already-isolated ones, so the layout gains breathing room without
    losing its neighbourhood structure.

    An optional spring attraction along real inter-cluster edges
    (semantic_inter / adept_graft / temporal*) can keep connected
    cluster-groups together; off by default (attraction=0).

    Each cluster is anchored back toward its *own* pre-declump centroid
    (``anchor_strength``), not a shared moving center — anchoring to a
    global center over-corrects isolated clusters, dragging them toward
    the crowd. The two forces reach a stable equilibrium, so ``iterations``
    mainly needs to be enough to converge, not precisely tuned.

    Runs after _straighten_nova_chains (rigid per-cluster translation
    preserves internal/chain shape) and before _resolve_cluster_overlaps,
    which then only has to mop up any residual tight pairs.

    Args:
        posts_df: needs 'id' and 'cluster_id'.
        edges_df: only read if fa2.cluster_declump.attraction > 0, to build
            the inter-cluster attraction graph. Safe to pass None/empty
            otherwise.
        final_positions: post_id -> (x, y, z). Mutated in place.
        cfg: reads fa2.cluster_declump.
    """
    dcfg = cfg.get("fa2", {}).get("cluster_declump", {})
    if not dcfg.get("enabled", False):
        return final_positions

    if "cluster_id" not in posts_df.columns:
        logger.debug("cluster declump skipped — no cluster_id column")
        return final_positions

    strength        = float(dcfg.get("strength", 1.0))
    iterations      = int(dcfg.get("iterations", 200))
    damping         = float(dcfg.get("damping", 0.3))
    anchor_strength = float(dcfg.get("anchor_strength", 0.05))
    attraction      = float(dcfg.get("attraction", 0.0))
    mass_mode       = dcfg.get("mass_mode", "size")  # "size" | "radius" | "uniform"

    if strength <= 0 or iterations <= 0:
        return final_positions

    # Same floor as in _resolve_cluster_overlaps — see
    # _min_cluster_radius. Kept identical between the two passes so
    # organic declumping and the hard cleanup pass after it agree on
    # the same effective size per cluster.
    min_radius = _min_cluster_radius(cfg)

    # One point-mass per cluster
    cluster_ids, centroids, radii, sizes, members = [], [], [], [], []
    id_to_cluster = {}
    for cid, group in posts_df.dropna(subset=["cluster_id"]).groupby("cluster_id"):
        pids = [pid for pid in group["id"].tolist() if pid in final_positions]
        if not pids:
            continue
        pts = np.array([final_positions[pid] for pid in pids], dtype=np.float64)
        centroid = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - centroid, axis=1).max()) if len(pts) > 1 else 0.0
        radius = max(radius, min_radius)

        cluster_ids.append(cid)
        centroids.append(centroid)
        radii.append(radius)
        sizes.append(len(pids))
        members.append(pids)
        for pid in pids:
            id_to_cluster[pid] = cid

    n = len(cluster_ids)
    if n < 2:
        return final_positions

    centroids = np.array(centroids, dtype=np.float64)  # (n, 3)
    radii     = np.array(radii, dtype=np.float64)        # (n,)
    sizes     = np.array(sizes, dtype=np.float64)         # (n,)
    original_centroids = centroids.copy()
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}

    if mass_mode == "radius":
        mass = np.maximum(radii, 1e-6)
    elif mass_mode == "uniform":
        mass = np.ones(n)
    else:  # "size" (default) — bigger clusters push harder, move less
        mass = np.maximum(sizes, 1.0)
    mass = mass / mass.mean()          # avg mass = 1 → `strength` stays comparable across datasets
    inv_mass = 1.0 / mass

    # Natural per-pair size scale for the repulsion law. Falls back to the
    # dataset's mean radius for near-zero-radius clusters (e.g. tight,
    # barely-spread pools) instead of an absolute magic-number floor, so
    # this stays meaningful regardless of local_scale/universe_scale.
    radius_floor = max(float(radii.mean()), 1e-6)

    # Optional inter-cluster attraction matrix (off by default)
    W = np.zeros((n, n), dtype=np.float64)
    if attraction > 0 and edges_df is not None and len(edges_df):
        src_cid = edges_df["source"].map(id_to_cluster)
        tgt_cid = edges_df["target"].map(id_to_cluster)
        inter_mask = src_cid.notna() & tgt_cid.notna() & (src_cid != tgt_cid)
        if inter_mask.any():
            idx_i = src_cid[inter_mask].map(cid_to_idx).to_numpy()
            idx_j = tgt_cid[inter_mask].map(cid_to_idx).to_numpy()
            np.add.at(W, (idx_i, idx_j), 1.0)
            np.add.at(W, (idx_j, idx_i), 1.0)

    rng = np.random.default_rng(0)
    for _ in range(iterations):
        diff = centroids[:, None, :] - centroids[None, :, :]   # (n, n, 3), i - j
        dist = np.linalg.norm(diff, axis=2)                     # (n, n)
        np.fill_diagonal(dist, np.inf)

        safe_dist = np.where(np.isfinite(dist) & (dist > 0), dist, 1e-6)
        direction = diff / safe_dist[..., None]
        coincident = dist == 0
        if coincident.any():
            rand_dirs = rng.normal(size=(n, n, 3))
            rand_dirs /= np.linalg.norm(rand_dirs, axis=2, keepdims=True) + 1e-12
            direction = np.where(coincident[..., None], rand_dirs, direction)

        contact = np.maximum(radii[:, None] + radii[None, :], radius_floor)  # (n, n)
        rep_mag = strength * mass[:, None] * mass[None, :] * contact / safe_dist
        repulsion = (direction * rep_mag[..., None]).sum(axis=1)             # (n, 3)

        if attraction > 0:
            attr_mag = attraction * W * safe_dist
            attraction_force = -(direction * attr_mag[..., None]).sum(axis=1)
        else:
            attraction_force = 0.0

        # Restoring spring toward each cluster's OWN starting position (not
        # a shared moving center — see docstring for why that matters).
        anchor = anchor_strength * mass[:, None] * (original_centroids - centroids)

        net = repulsion + attraction_force + anchor
        centroids = centroids + net * inv_mass[:, None] * damping

    n_moved   = 0
    max_shift = 0.0
    for pids, centroid, origin_c in zip(members, centroids, original_centroids, strict=False):
        delta = centroid - origin_c
        shift = float(np.linalg.norm(delta))
        max_shift = max(max_shift, shift)
        if shift < 1e-9:
            continue
        n_moved += 1
        for pid in pids:
            x, y, z = final_positions[pid]
            final_positions[pid] = (x + delta[0], y + delta[1], z + delta[2])

    logger.info(
        "cluster declump — %d clusters, %d iterations (strength=%.2f, anchor_strength=%.3f, "
        "attraction=%.4f, mass_mode=%s) — %d/%d clusters moved, max shift %.1f",
        n, iterations, strength, anchor_strength, attraction, mass_mode, n_moved, n, max_shift,
    )
    return final_positions


def run_hybrid_layout(posts_df, edges_df, cfg, n_iter=None):
    """Hybrid PaCMAP + FA2 layout strategy.

    1. PaCMAP 3D on LDA/raw embeddings — macro-structure (cluster positions).
    2. FA2 3D (backend=vectorized) per cluster — micro-structure (intra-cluster).
       External nodes act as pinned ghosts: their positions are reset to their
       PaCMAP values every 10 FA2 iterations.
    3. Final XYZ = PaCMAP centroid + FA2 local displacement.
       No per-role Z post-processing — FA2 handles Z coherently for the whole
       cluster. A separate per-role Z override would conflict with this: NOVA
       edges connect to a parent's actual FA2-computed Z, so shifting only
       some roles' Z produces visible vertical streaks between connected
       nodes.
    """
    # Step 1: PaCMAP 3D macro positions
    pacmap_positions = run_pacmap_3d(posts_df, cfg)

    # Step 2: FA2 3D per cluster
    try:
        import scipy.sparse
        from fa2 import ForceAtlas2

        fa2cfg = cfg["fa2"]
        iterations = n_iter if n_iter is not None else min(200, fa2cfg["max_iter"])

        final_positions = dict(pacmap_positions)  # fallback to PaCMAP positions

        # Edge weights by type (6 active types)
        weight_cfg = cfg.get("fa2", {}).get("edge_weights", {})
        DEFAULT_WEIGHTS = {
            # NOVA — chronological chain intra-subtopic;
            #        row["force"] = force_nova_edge, fixed value on every
            #        offspring edge, no rank-based decay (computed in nova.py)
            "nova": weight_cfg.get("nova", 0.6),
            # ADEPT — 2 types. force fixed at edge-creation
            # (adept.force_spoke / edges.adept_graft.force_graft); this
            # multiplier tunes the FA2 weight without rerunning ADEPT.
            "adept_spoke": weight_cfg.get("adept_spoke", 1.0),
            "adept_graft": weight_cfg.get("adept_graft", 1.0),
            # TEMPORAL_INFLUENCE — directed cascade (pass 1: timestamp + engagement, Rule B)
            "temporal_influence": weight_cfg.get("temporal_influence", 0.0),
            # TEMPORAL — chronological link (pass 2: timestamp only, no engagement condition)
            "temporal": weight_cfg.get("temporal", 0.0),
            # SEMANTIC_INTER — global inter-cluster greedy 1-to-1 (cyan in render)
            "semantic_inter": weight_cfg.get("semantic_inter", 0.0),
        }

        def _edge_weight(row) -> float:
            """Return the FA2 weight for an edge row.

            NOVA edges use row["force"] * nova_weight; all others use a fixed
            weight from config. Edge types not present in DEFAULT_WEIGHTS
            (e.g. "pad", "adept_bridge") fall through to 0.0 and are dropped.
            """
            etype = row.get("type", "")
            w = DEFAULT_WEIGHTS.get(etype, 0.0)
            return float(row["force"]) * w

        if "cluster_id" in posts_df.columns:
            cluster_groups = [
                (cid, g)
                for cid, g in posts_df.groupby("cluster_id")
                if cid is not None and not pd.isna(cid) and len(g) >= 3
            ]
            n_clusters_total = len(cluster_groups)
            logger.info("FA2 3D pinned-ghosts — %d clusters", n_clusters_total)
            for cluster_idx, (cluster_id, group) in enumerate(cluster_groups):
                if (cluster_idx + 1) % 10 == 0 or cluster_idx == n_clusters_total - 1:
                    logger.debug(
                        "%d/%d clusters done", cluster_idx + 1, n_clusters_total
                    )

                cluster_post_ids = group["id"].tolist()
                cluster_set = set(cluster_post_ids)

                # All edges where at least one endpoint is in this cluster
                all_cluster_edges = edges_df[
                    edges_df["source"].isin(cluster_set)
                    | edges_df["target"].isin(cluster_set)
                ].copy()

                # Apply config weights and drop zero-weight edges
                all_cluster_edges["weight"] = all_cluster_edges.apply(
                    _edge_weight, axis=1
                )
                all_cluster_edges = all_cluster_edges[all_cluster_edges["weight"] > 0]

                # Cluster centroid in PaCMAP space (XYZ)
                cluster_pacmap = np.array(
                    [
                        pacmap_positions[pid]
                        for pid in cluster_post_ids
                        if pid in pacmap_positions
                    ]
                )
                centroid = cluster_pacmap.mean(axis=0)  # (cx, cy, cz)

                # Ghost nodes: external nodes connected to this cluster via inter-cluster edges
                external_pids = set()
                for _, edge in all_cluster_edges.iterrows():
                    if (
                        edge["source"] not in cluster_set
                        and edge["source"] in pacmap_positions
                    ):
                        external_pids.add(edge["source"])
                    if (
                        edge["target"] not in cluster_set
                        and edge["target"] in pacmap_positions
                    ):
                        external_pids.add(edge["target"])
                external_pids = list(external_pids)

                # Index: cluster nodes first, ghosts after
                all_node_ids = cluster_post_ids + external_pids
                pid_to_idx = {pid: i for i, pid in enumerate(all_node_ids)}
                n_total = len(all_node_ids)
                n_cluster = len(cluster_post_ids)

                # Initial positions relative to cluster centroid (cluster nodes + ghosts)
                pos_current = np.array(
                    [
                        [
                            pacmap_positions[pid][0] - centroid[0],
                            pacmap_positions[pid][1] - centroid[1],
                            pacmap_positions[pid][2] - centroid[2],
                        ]
                        for pid in all_node_ids
                    ],
                    dtype=np.float64,
                )
                # Ghost positions are fixed — reset after every FA2 chunk
                ghost_positions = pos_current[n_cluster:].copy()

                # Build FA2 sparse adjacency (cluster nodes + ghosts)
                rows, cols, data = [], [], []
                for _, edge in all_cluster_edges.iterrows():
                    s = pid_to_idx.get(edge["source"])
                    t = pid_to_idx.get(edge["target"])
                    if s is None or t is None:
                        continue
                    w = float(edge["weight"])
                    rows += [s, t]
                    cols += [t, s]
                    data += [w, w]

                # Edge-less clusters (all weights 0.0): csr_matrix(([], ([], [])), shape=...)
                # raises ValueError on some scipy versions. Build an empty matrix explicitly
                # so FA2 still runs repulsion + gravity (which produces a sphere).
                if data:
                    A = scipy.sparse.csr_matrix(
                        (data, (rows, cols)), shape=(n_total, n_total), dtype=np.float64
                    )
                else:
                    A = scipy.sparse.csr_matrix((n_total, n_total), dtype=np.float64)

                try:
                    fa2_instance = ForceAtlas2(
                        scalingRatio=fa2cfg["scaling_ratio"] * 0.1,
                        gravity=fa2cfg["gravity"],
                        outboundAttractionDistribution=False,
                        linLogMode=fa2cfg.get("lin_log_mode", False),
                        edgeWeightInfluence=fa2cfg.get("edge_weight_influence", 1.0),
                        jitterTolerance=fa2cfg.get("jitter_tolerance", 1.0),
                        strongGravityMode=fa2cfg.get("strong_gravity_mode", False),
                        verbose=False,
                        dim=3,
                        backend="vectorized",
                    )

                    CHUNK = 10
                    n_iter_total = iterations
                    for _chunk_start in range(0, n_iter_total, CHUNK):
                        raw = fa2_instance.forceatlas2(
                            A, pos=pos_current, iterations=CHUNK
                        )
                        pos_current = np.array(raw, dtype=np.float64)
                        # Pin ghosts back to their PaCMAP positions each chunk
                        pos_current[n_cluster:] = ghost_positions

                    local_coords = pos_current  # (n_total, 3)

                    # Scaling
                    # local_scale   : expands intra-cluster space (avoids compact blobs)
                    # universe_scale: pushes centroids apart (avoids inter-cluster overlap)
                    # Both tunable via config.yaml fa2.local_scale / fa2.universe_scale.
                    local_scale = float(fa2cfg.get("local_scale", 50.0))
                    universe_scale = float(fa2cfg.get("universe_scale", 3.0))

                    # Re-centre before scaling: FA2 gravity drifts the centre of mass
                    # unevenly by node degree. Without this, * local_scale amplifies
                    # the drift and ejects nodes far from their cluster.
                    local_coords[:n_cluster] -= local_coords[:n_cluster].mean(axis=0)

                    local_coords[:n_cluster] *= local_scale
                    centroid_scaled = centroid * universe_scale

                    # Update cluster nodes only — ghosts are not written back
                    for i, pid in enumerate(cluster_post_ids):
                        if pid not in final_positions:
                            continue
                        final_positions[pid] = (
                            centroid_scaled[0] + float(local_coords[i, 0]),
                            centroid_scaled[1] + float(local_coords[i, 1]),
                            centroid_scaled[2] + float(local_coords[i, 2]),
                        )

                except Exception as e:
                    logger.warning("FA2 3D failed for cluster %s: %s", cluster_id, e)

        # Post-processing: straighten Nova chains that FA2 folded into
        # loops/squares (see _straighten_nova_chains docstring). Runs once,
        # globally, after every cluster's FA2 pass — a subtopic_id never
        # spans clusters, so per-chain work here is independent of the
        # per-cluster loop above. Toggle/tune via config.yaml fa2.nova_straighten.
        final_positions = _straighten_nova_chains(posts_df, final_positions, cfg)

        # real render-size radius per node — everything below uses this,
        # not an arbitrary constant, so margins scale automatically with
        # render.node_size_mult (see _compute_node_radius docstring)
        node_radius = _compute_node_radius(posts_df, cfg)

        # minimum spacing INSIDE each block, based on real node size —
        # nova chain nodes along the straightened axis, then ADEPT spokes
        # radially from their hub. See fa2.node_spacing.
        final_positions = _space_out_chain_nodes(posts_df, final_positions, node_radius, cfg)
        final_positions = _space_out_pool_nodes(posts_df, edges_df, final_positions, node_radius, cfg)

        # intra-cluster reorganization: each Nova chain / ADEPT pool is
        # moved as a rigid block so it no longer overlaps other blocks in
        # the same cluster — see fa2.block_reorganize / _reorganize_cluster_blocks
        final_positions = _reorganize_cluster_blocks(
            posts_df, edges_df, final_positions, cfg, node_radius=node_radius
        )

        # final safety net: no pair of nodes in a cluster overlaps, whatever
        # their block (chain, pool, or isolated singleton) — this is what
        # specifically covers isolated nodes. See fa2.node_spacing.
        final_positions = _resolve_node_collisions(posts_df, final_positions, node_radius, cfg)

        # organic macro-scale spread, proportional to local crowding —
        # see fa2.cluster_declump / _declump_clusters docstring
        final_positions = _declump_clusters(posts_df, edges_df, final_positions, cfg)

        # final safety net — only moves clusters still in conflict after
        # declumping — see fa2.cluster_overlap_removal
        final_positions = _resolve_cluster_overlaps(posts_df, final_positions, cfg)

        return final_positions

    except ImportError:
        logger.warning("fa2 not available — using PaCMAP only")
        return pacmap_positions


def run(
    posts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """Compute 3D positions for all posts and attach x/y/z columns.

    Returns:
        posts_df with columns x, y, z, position_xyz added.
    """
    cfg = load_config(config_path)
    n_iter = cfg["fa2"]["max_iter"]

    positions_dict = run_hybrid_layout(posts_df, edges_df, cfg, n_iter=n_iter)

    posts_df = posts_df.copy()
    posts_df["x"] = posts_df["id"].map(
        lambda i: positions_dict.get(i, (0.0, 0.0, 0.0))[0]
    )
    posts_df["y"] = posts_df["id"].map(
        lambda i: positions_dict.get(i, (0.0, 0.0, 0.0))[1]
    )
    posts_df["z"] = posts_df["id"].map(
        lambda i: positions_dict.get(i, (0.0, 0.0, 0.0))[2]
    )
    posts_df["position_xyz"] = posts_df[["x", "y", "z"]].apply(
        lambda r: [r["x"], r["y"], r["z"]], axis=1
    )

    logger.info(
        "XYZ spread — X: [%.1f, %.1f]  Y: [%.1f, %.1f]  Z: [%.1f, %.1f]",
        posts_df['x'].min(), posts_df['x'].max(),
        posts_df['y'].min(), posts_df['y'].max(),
        posts_df['z'].min(), posts_df['z'].max(),
    )

    return posts_df