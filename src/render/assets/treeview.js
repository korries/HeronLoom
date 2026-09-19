// Tree-view radial mode. Concatenated after app.js into the same module
// scope (not imported) -- reads/writes globals defined there: RAW, NM,
// ADJ_ALL, THREE, camera, scene, instancedMeshes, instanceMap, _dummy,
// pickableEdges, groupLines, vis, fps, hideSkeletonHighlight, EC,
// LineSegmentsGeometry, LineMaterial, LineSegments2, fatLineMaterials,
// SKELETON_TYPES, swTab, sbLastTab, _forceClearClusterSelection,
// DIRECTIONAL_DASH_TYPES.

const TREE_TITLE_FONT_SIZE = 150; // title text size
const TREE_TITLE_Z_OFFSET  = 1000; // common depth offset in front of the graph
const TREE_TITLE_LABEL_GAP = 50;  // minimum margin between titles

// "Collapse / snap / expand": a node/edge shrinks to scale 0 at its start
// position, snaps instantly to its end position while invisible, then grows
// back to scale 1 — avoids interpolating across the screen, which would
// stretch edges between the universe and tree layouts. Edge geometry
// rebuilds once per phase; only opacity changes per frame afterward.
const TREE_COLLAPSE_MS = 150;
const TREE_EXPAND_MS   = 240;
const TREE_ANIM_MS     = 720;  // camera flight duration between universe and tree view

// Below this node count, reveal work is faster than a spinner show/hide cycle.
const TREE_INSTANT_REVEAL_THRESHOLD = 160;

function _easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
function _easeInCubic(t)  { return t * t * t; }
// startPos is a Map<id,{x,y,z}>, targetPos (layout) is a plain object.
function _treePosOf(store, id) { return store instanceof Map ? store.get(id) : store[id]; }

// Branch length calibrated on node size in the isolated subgraph, not on
// world edge length (which can be arbitrarily large for a given dataset).
const TREE_BRANCH_LEN = (function() {
  const radii = [];
  Object.values(NM).forEach(n => {
    if (!n || typeof n.size !== 'number' || !Number.isFinite(n.size)) return;
    radii.push(n.size * 1.5 * _geomScaleMult(n.geometry));
  });
  radii.sort((a,b) => a-b);
  const med = radii.length ? radii[Math.floor(radii.length / 2)] : 20;
  return THREE.MathUtils.clamp(med * 5.0, 70, 240);  // center-to-center, ~2 diameters
})();;

// Bridge edge dash/gap, world units; render.edge_dash.tree_dash_size / tree_gap_size.
const TREE_DASH_SIZE = __TREE_DASH_SIZE__;
const TREE_GAP_SIZE  = __TREE_GAP_SIZE__;

// Decoupled from the universe mode's FLOW_SPEED; render.edge_dash.tree_flow_speed.
const TREE_FLOW_SPEED = __TREE_FLOW_SPEED__;

const treeView = {
  active: false,
  animating: false,
  animDir: 1,              // 1 = entering, -1 = exiting
  rootId: null,
  keepSet: null,            // Set<id> of nodes in the isolated subgraph
  startPos: null,           // Map id -> {x,y,z} original universe position
  targetPos: null,          // {} id -> {x,y,z} computed tree layout position
  edgesByType: null,        // Map type -> [[sId,tId], ...]
  overlayObjs: {},          // type -> LineSegments2 for the tree view
  dashOffset: 0,            // independent from the universe mode's dash flow
  animStart: 0,
  preCam: null,             // camera pose before entry, for exit restore
  camFrom: null,
  camTo: null,
  camCenter: null,
  camFromQuat: null,
  camToQuat: null,
  canonicalRoot: null,
  treeLabelObjs: [],
  treeLeaderObjs: [],
  legendSpecs: [],
  summary: null,
  hoverIds: null,
  hoverSpecs: new Set(),
  hoverEdge: null,
  hoverObjs: [],
  hoverMats: [],
  _overlayPhase: null,      // 'collapse' | 'expand' | null — see updateTreeViewFrame
  _revealPending: false,    // true while the post-flight reveal is deferred — see updateTreeViewFrame

  // Locked pan/zoom camera (replaces free 6DOF flight once active and the
  // entry animation has settled -- see applyTreePanZoom / treePan / treeZoom).
  camBasis: null,           // {right, up, fwd} unit Vector3, fixed for the session
  panZoom: null,            // {x, y, dist} current offset along right/up + distance along -fwd
  panLimits: null,          // {x, y} max |panZoom.x|/|panZoom.y|
  zoomLimits: null,         // {min, max} clamp for panZoom.dist
};

// Swapped into #nav-hint while tree view is active/inactive, so the topbar
// hint always names the controls that are actually live.
const NAV_HINT_DEFAULT = 'WASD·Q·E·Shift | right-click=look · Space=free look (click to exit) · F=overview';
const NAV_HINT_TREE    = 'Right-click drag = pan · Scroll = zoom · double-click/Esc = exit tree';

// Swapped into #leg-rows (the bottom-left "Controls" legend) for the same
// reason: WASD/Q/E/Shift/Space do nothing while the tree view owns the
// camera, and right-click/scroll no longer mean look/speed but pan/zoom.
// Kept as a literal copy of the template's default rows (mirrors
// NAV_HINT_DEFAULT above) rather than reading #leg-rows.innerHTML on entry,
// so exit always restores the exact original markup even across repeated
// enter/exit cycles.
const LEG_ROWS_DEFAULT =
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">W</span><span class="kbd">A</span><span class="kbd">S</span><span class="kbd">D</span></span><span class="ctl-label">Move <b>(or ↑←↓→)</b></span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Q</span><span class="kbd">E</span></span><span class="ctl-label">Up / down</span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Shift</span></span><span class="ctl-label">Speed ×5</span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Scroll</span></span><span class="ctl-label">Adjust speed</span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Right-click</span></span><span class="ctl-label">Camera look</span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Space</span></span><span class="ctl-label">Free look <b>(toggle)</b></span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">F</span></span><span class="ctl-label">Overview</span></div>';
const LEG_ROWS_TREE =
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Click node</span></span><span class="ctl-label">Show its details</span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Click edge</span></span><span class="ctl-label">Show relation <b>(both ends)</b></span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Right-click</span></span><span class="ctl-label">Pan <b>(drag)</b></span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Scroll</span></span><span class="ctl-label">Zoom</span></div>' +
  '<div class="ctl-row"><span class="ctl-keys"><span class="kbd">Double-click</span><span class="kbd">Esc</span></span><span class="ctl-label">Exit tree</span></div>';

// The parent map is a placement skeleton only; rendered edges are always
// the original RAW.links, never added to or removed from.
function treeLinksFor(ids) {
  const keep = ids instanceof Set ? ids : new Set(ids.map(String));
  return RAW.links
    .filter(l => keep.has(String(l.source)) && keep.has(String(l.target)))
    .map(l => ({
      source: String(l.source),
      target: String(l.target),
      group: String(l.group || '')
    }));
}

function bfsWithParent(rootId) {
  const root = String(rootId);
  const depth  = { [root]: 0 };
  const parent = { [root]: null };
  const order  = [root];
  const queue  = [root];
  let qi = 0;

  // Skeleton priority: 1) actual Nova direction, 2) ADEPT hub -> spoke,
  // 3) other links. Does not affect which edges are rendered.
  // Read from cur's own incident edges (ADJ_ALL[cur]) instead of scanning
  // every link in the graph for each (cur, other) pair.
  const edgePriorityFor = (cur, e) => {
    if (e.type === 'nova') return String(e.link.source) === cur ? 0 : 8;
    if (e.type === 'adept_spoke') {
      const n = NM[cur], o = NM[String(e.other)];
      if (n?.adept_role === 'hub' || n?.adept_pool_label) return 12;
      if (o?.adept_role === 'hub' || o?.adept_pool_label) return 18;
      return 22;
    }
    return 30;
  };

  while (qi < queue.length) {
    const cur = queue[qi++];
    const rawAdj = ADJ_ALL[cur] || [];

    // Best priority per distinct neighbor, computed once per node instead
    // of rescanned on every pairwise sort comparison (a multi-edge between
    // the same pair takes the min, same as before).
    const priorityByOther = new Map();
    rawAdj.forEach(e => {
      const other = String(e.other);
      const p = edgePriorityFor(cur, e);
      const prev = priorityByOther.get(other);
      if (prev === undefined || p < prev) priorityByOther.set(other, p);
    });

    const neighbors = [...priorityByOther.keys()]
      .sort((a,b) => {
        const pa = priorityByOther.get(a), pb = priorityByOther.get(b);
        return pa - pb ||
          ((ADJ_ALL[b] || []).length - (ADJ_ALL[a] || []).length) ||
          a.localeCompare(b);
      });

    neighbors.forEach(other => {
      if (other in depth) return;
      depth[other]  = depth[cur] + 1;
      parent[other] = cur;
      order.push(other);
      queue.push(other);
    });
  }
  return { order, depth, parent };
}

// Grid-based repulsion with spring pull-back to each node's hierarchical
// position; optional `links` also keeps nodes clear of unrelated edges.
function resolveTreeCollisions(pos, ids, nodeRadius, edgeClearance, links) {
  if (ids.length < 2) return;
  const anchor = new Map(ids.map(id => [id, { x: pos[id].x, y: pos[id].y }]));
  const idx    = new Map(ids.map((id, i) => [id, i]));
  const n = ids.length;
  // Radius depends only on id: precomputed once, not per tested pair.
  const radiusArr = new Float64Array(n);
  for (let i = 0; i < n; i++) radiusArr[i] = nodeRadius(ids[i]);
  const radiusOf = id => radiusArr[idx.get(id)];
  const iterations = n > 800 ? 8 : n > 300 ? 16 : n > 100 ? 28 : 70;
  const springX  = 0.035;  // loose X pull-back: lets repulsion widen branches
  const springY  = 0.18;   // firmer Y pull-back: preserves per-row tree layout
  const vertDamp = 0.7;
  const padding  = TREE_BRANCH_LEN * 0.12;
  const cellSize = Math.max(30, TREE_BRANCH_LEN * 0.6);

  // Deduplicated (source,target) segments, used by the node<->edge pass below.
  const segments = [];
  if (links && links.length) {
    const seenSeg = new Set();
    links.forEach(l => {
      const s = String(l.source), t = String(l.target);
      if (s === t || !idx.has(s) || !idx.has(t)) return;
      const key = idx.get(s) < idx.get(t) ? s + '|' + t : t + '|' + s;
      if (seenSeg.has(key)) return;
      seenSeg.add(key);
      segments.push([s, t]);
    });
  }

  // Two rounds let both constraints settle together; capped to one above
  // a node-count threshold to bound per-click cost.
  const rounds = segments.length ? (n <= 500 ? 2 : 1) : 1;

  for (let round = 0; round < rounds; round++) {
    for (let iter = 0; iter < iterations; iter++) {
      // Uniform grid: only neighboring cells (3x3) are compared, not all pairs.
      const grid = new Map();
      const cellKey = p => Math.floor(p.x / cellSize) + ',' + Math.floor(p.y / cellSize);
      ids.forEach(id => {
        const k = cellKey(pos[id]);
        if (!grid.has(k)) grid.set(k, []);
        grid.get(k).push(id);
      });

      const push = new Map(ids.map(id => [id, { x: 0, y: 0 }]));
      const seen = new Set();
      let worstOverlap = 0;

      ids.forEach(a => {
        const pa = pos[a];
        const cx = Math.floor(pa.x / cellSize), cy = Math.floor(pa.y / cellSize);
        for (let gx = -1; gx <= 1; gx++) {
          for (let gy = -1; gy <= 1; gy++) {
            const bucket = grid.get((cx + gx) + ',' + (cy + gy));
            if (!bucket) continue;
            bucket.forEach(b => {
              if (b === a) return;
              const pairKey = idx.get(a) < idx.get(b) ? (a + '|' + b) : (b + '|' + a);
              if (seen.has(pairKey)) return;
              seen.add(pairKey);

              const pb = pos[b];
              let dx = pb.x - pa.x, dy = pb.y - pa.y;
              let dist = Math.sqrt(dx * dx + dy * dy);
              const minDist = radiusOf(a) + radiusOf(b) + edgeClearance + padding;
              if (dist >= minDist) return;
              if (dist < 0.001) { dx = (Math.random() - 0.5) * 0.1; dy = 0.001; dist = 0.1; }

              const overlap = minDist - dist;
              worstOverlap = Math.max(worstOverlap, overlap);
              const ux = dx / dist, uy = dy / dist;
              const mag = overlap * 0.5;
              push.get(a).x -= ux * mag;
              push.get(a).y -= uy * mag * vertDamp;
              push.get(b).x += ux * mag;
              push.get(b).y += uy * mag * vertDamp;
            });
          }
        }
      });

      ids.forEach(id => {
        const p = pos[id], k = anchor.get(id), f = push.get(id);
        p.x += f.x - (p.x - k.x) * springX;
        p.y += f.y - (p.y - k.y) * springY;
      });

      if (worstOverlap < 0.5) break;   // converged
    }

    // No spring pull-back here: a node pushed off an edge must stay off it,
    // or the spring would pull it back and the overlap would never resolve.
    if (segments.length) resolveNodeEdgeOverlaps(pos, ids, idx, radiusArr, segments, edgeClearance);
  }
}

// `idx`/`radiusArr` come from resolveTreeCollisions to avoid recomputing them.
function resolveNodeEdgeOverlaps(pos, ids, idx, radiusArr, segments, edgeClearance) {
  const relax    = 0.8;
  const n        = ids.length;
  const maxIter  = n > 1500 ? 12 : n > 800 ? 18 : n > 300 ? 24 : 50;
  const cellSize = Math.max(30, TREE_BRANCH_LEN * 0.5);
  const pushX    = new Float64Array(n);
  const pushY    = new Float64Array(n);
  // Resolved to numeric indices once, instead of comparing id strings below.
  const segIdx = segments.map(([s, t]) => [idx.get(s), idx.get(t)]);

  for (let iter = 0; iter < maxIter; iter++) {
    // Rebuilt each iteration since positions move.
    const grid = new Map();
    for (let i = 0; i < n; i++) {
      const p = pos[ids[i]];
      const k = Math.floor(p.x / cellSize) * 1000003 + Math.floor(p.y / cellSize);
      let bucket = grid.get(k);
      if (!bucket) { bucket = []; grid.set(k, bucket); }
      bucket.push(i);
    }

    pushX.fill(0);
    pushY.fill(0);
    let worst = 0;

    for (let si = 0; si < segIdx.length; si++) {
      const si0 = segIdx[si][0], ti0 = segIdx[si][1];
      const A = pos[ids[si0]], B = pos[ids[ti0]];
      if (!A || !B) continue;
      const dx = B.x - A.x, dy = B.y - A.y;
      const len2 = dx * dx + dy * dy;

      const cx0 = Math.floor(Math.min(A.x, B.x) / cellSize) - 1;
      const cx1 = Math.floor(Math.max(A.x, B.x) / cellSize) + 1;
      const cy0 = Math.floor(Math.min(A.y, B.y) / cellSize) - 1;
      const cy1 = Math.floor(Math.max(A.y, B.y) / cellSize) + 1;

      for (let gx = cx0; gx <= cx1; gx++) {
        for (let gy = cy0; gy <= cy1; gy++) {
          const bucket = grid.get(gx * 1000003 + gy);
          if (!bucket) continue;
          for (let bi = 0; bi < bucket.length; bi++) {
            const ni = bucket[bi];
            if (ni === si0 || ni === ti0) continue;  // endpoints define the edge, skip

            const p = pos[ids[ni]];
            // Point-segment distance: project onto [A,B], clamped to the endpoints.
            let tt = len2 > 1e-6 ? ((p.x - A.x) * dx + (p.y - A.y) * dy) / len2 : 0;
            tt = tt < 0 ? 0 : (tt > 1 ? 1 : tt);
            const px = A.x + dx * tt, py = A.y + dy * tt;
            let ex = p.x - px, ey = p.y - py;
            let dist = Math.sqrt(ex * ex + ey * ey);
            const minDist = radiusArr[ni] + edgeClearance;
            if (dist >= minDist) continue;
            if (dist < 0.001) { ex = (Math.random() - 0.5) * 0.1; ey = 0.1; dist = 0.1; }

            const overlap = minDist - dist;
            if (overlap > worst) worst = overlap;
            const ux = ex / dist, uy = ey / dist;
            pushX[ni] += ux * overlap * relax;
            pushY[ni] += uy * overlap * relax;
          }
        }
      }
    }

    for (let i = 0; i < n; i++) {
      if (pushX[i] || pushY[i]) {
        const p = pos[ids[i]];
        p.x += pushX[i];
        p.y += pushY[i];
      }
    }

    if (worst < 0.5) break;   // converged: no node touches a foreign edge
  }
}

// Hierarchical layout; branch order comes from all real connections in the subgraph.
function computeTreeLayout(rootId, order, parent) {
  const ids = order.map(String);
  const keep = new Set(ids);
  const links = treeLinksFor(keep);
  const depth = { [String(rootId)]: 0 };
  ids.forEach(id => {
    if (id !== String(rootId)) depth[id] = depth[String(parent[id])] + 1;
  });

  const childrenOf = {};
  ids.forEach(id => {
    const p = parent[id];
    if (p !== null && p !== undefined) (childrenOf[p] ??= []).push(id);
  });

  const branch = TREE_BRANCH_LEN;
  const xGap = branch * 0.95;
  const yGap = branch * 1.05;
  const edgeClearance = Math.max(18, branch * 0.28);

  // Reuses treeNodeRadiusForLabel (defined below, hoisted).
  const nodeRadius = treeNodeRadiusForLabel;

  const pairGroup = new Map();
  const incident = new Map(ids.map(id => [id, []]));
  links.forEach(l => {
    const k = l.source + '|' + l.target;
    if (!pairGroup.has(k)) pairGroup.set(k, []);
    pairGroup.get(k).push(l.group);
    incident.get(l.source)?.push(l);
    incident.get(l.target)?.push(l);
  });
  const hasGroup = (a,b,g) => {
    const ab = pairGroup.get(String(a) + '|' + String(b));
    if (ab && ab.includes(g)) return true;
    const ba = pairGroup.get(String(b) + '|' + String(a));
    return !!(ba && ba.includes(g));
  };

  // Skeleton subtrees; cross-branch RAW.links place communicating branches closer together.
  const subtree = new Map();
  function collect(id) {
    const set = new Set([id]);
    (childrenOf[id] || []).forEach(k => collectInto(k, set));
    subtree.set(id, set);
    return set;
  }
  function collectInto(id, out) {
    out.add(id);
    (childrenOf[id] || []).forEach(k => collectInto(k, out));
  }
  collect(rootId);

  // Sibling subtrees never overlap (each node has exactly one parent), so a
  // crossing edge always has exactly one endpoint in the smaller subtree —
  // walking that subtree's own incident edges (already indexed above) finds
  // every crossing edge without rescanning the full link list each time.
  const crossWeight = (a,b) => {
    const A = subtree.get(a) || new Set([a]);
    const B = subtree.get(b) || new Set([b]);
    const [small, big] = A.size <= B.size ? [A, B] : [B, A];
    let w = 0;
    small.forEach(id => {
      (incident.get(id) || []).forEach(l => {
        const other = l.source === id ? l.target : l.source;
        if (!big.has(other)) return;
        w += l.group === 'nova' ? 3 : l.group === 'adept_spoke' ? 2 : 1;
      });
    });
    return w;
  };

  const groupRank = (p,k) => {
    if (hasGroup(p,k,'nova')) return 0;
    if (hasGroup(p,k,'adept_spoke')) return 1;
    return 2;
  };

  // Child order: real branch families first, then a greedy pass over cross-subtree connections.
  Object.entries(childrenOf).forEach(([p, rawKids]) => {
    const kids = [...rawKids];
    const rank = k => groupRank(p,k);
    kids.sort((a,b) => rank(a)-rank(b) || String(a).localeCompare(String(b)));

    const ordered = [];
    while (kids.length) {
      if (!ordered.length) {
        ordered.push(kids.shift());
        continue;
      }
      const last = ordered[ordered.length - 1];
      let bestI = 0, bestScore = -Infinity;
      kids.forEach((k,i) => {
        const score = crossWeight(last,k) * 20 - rank(k) * 3 - i * 0.001;
        if (score > bestScore) { bestScore = score; bestI = i; }
      });
      ordered.push(kids.splice(bestI,1)[0]);
    }
    childrenOf[p] = ordered;
  });

  const pos = {};
  let cursor = 0;

  function place(id) {
    const kids = childrenOf[id] || [];
    const d = depth[id];
    if (!kids.length) {
      pos[id] = { x: cursor * xGap, y: -d * yGap, z: 0 };
      cursor += 1;
      return;
    }

    kids.forEach(place);
    const first = pos[kids[0]], last = pos[kids[kids.length - 1]];
    pos[id] = {
      x: (first.x + last.x) * 0.5,
      y: -d * yGap,
      z: 0
    };
  }
  place(String(rootId));

  // ADEPT pools stay arranged as bunches, only for children linked via adept_spoke.
  order.forEach(hubId => {
    const kids = (childrenOf[hubId] || []).filter(k => hasGroup(hubId,k,'adept_spoke'));
    if (kids.length < 2) return;
    const h = pos[hubId];
    if (!h) return;
    const radius = Math.max(nodeRadius(hubId) * 3.0, branch * 1.35);
    const start = -Math.PI * 0.78;
    const span = Math.PI * 1.56;
    kids.forEach((id,i) => {
      const t = kids.length === 1 ? 0.5 : i / (kids.length - 1);
      const ang = start + span * t;
      pos[id] = {
        x: h.x + Math.cos(ang) * radius,
        y: h.y + Math.sin(ang) * radius,
        z: 0
      };
    });
  });

  resolveTreeCollisions(pos, ids, nodeRadius, edgeClearance, links);

  const rx = pos[String(rootId)]?.x || 0;
  const ry = pos[String(rootId)]?.y || 0;
  ids.forEach(id => {
    if (!pos[id]) pos[id] = {x:0,y:0,z:0};
    pos[id].x -= rx;
    pos[id].y -= ry;
    pos[id].z = 0;
  });
  return pos;
}

// Per-geometry size multiplier, shared by baseScaleFor, nodeRadius, and
// TREE_BRANCH_LEN. "sphere_large"/"sphere_small" aren't real Three.js
// geometries — _geomKey() maps both to the 'sphere' InstancedMesh bucket.
function _geomScaleMult(geomName) {
  return {
    sphere: 1, sphere_large: 1.4, sphere_small: 0.65,
    octahedron: 1.2, icosahedron: 1.1, diamond: 1.15,
    cone: 1, tetrahedron: 1.1,
  }[geomName] ?? 1;
}

// Base node scale, shared by buildInstancedMeshes and the tree view's hide/restore logic.
function baseScaleFor(n, k) {
  return n.size * 1.5 * _geomScaleMult(n.geometry ?? k);
}

// id -> {geomKey, idx} in the InstancedMesh buckets. Rebuilt on every
// populateScene() since its contents change with showOrp.
let nodeInstanceIndex = {};
function rebuildNodeInstanceIndex() {
  nodeInstanceIndex = {};
  Object.entries(instanceMap).forEach(([k, arr]) => {
    arr.forEach((n, idx) => { nodeInstanceIndex[n.id] = { geomKey: k, idx }; });
  });
}

function setInstanceTransform(id, x, y, z, scale) {
  const entry = nodeInstanceIndex[id];
  if (!entry) return;
  const im = instancedMeshes[entry.geomKey];
  if (!im) return;
  _dummy.position.set(x, y, z);
  _dummy.scale.setScalar(scale);
  _dummy.updateMatrix();
  im.setMatrixAt(entry.idx, _dummy.matrix);
}
function flagInstancedDirty() {
  Object.values(instancedMeshes).forEach(im => { im.instanceMatrix.needsUpdate = true; });
}

// Instantly scales down (to 0) all nodes outside the subgraph, in a single pass.
function hideNonKeptNodes(keepSet) {
  Object.keys(nodeInstanceIndex).forEach(id => {
    if (keepSet.has(id)) return;
    const n = NM[id];
    setInstanceTransform(id, n.x, n.y, n.z, 0);
  });
  flagInstancedDirty();
}
function restoreAllNodeScales() {
  Object.keys(nodeInstanceIndex).forEach(id => {
    const entry = nodeInstanceIndex[id];
    const n = NM[id];
    setInstanceTransform(id, n.x, n.y, n.z, baseScaleFor(n, entry.geomKey));
  });
  flagInstancedDirty();
}
// One LineSegments2 per edge type in the subgraph, repositioned each frame
// during the tween. Matches buildEdges()/showSkeletonHighlight() in
// render.py: same color (EC), width (EW), opacity (EO), and dashes.
function ensureTreeOverlay(type) {
  if (treeView.overlayObjs[type]) return treeView.overlayObjs[type];
  const dashedType = SKELETON_TYPES.includes(type);
  const geo = new LineSegmentsGeometry();
  geo.setPositions([0, 0, 0, 0, 0, 0]);
  const matParams = {
    color: new THREE.Color(EC[type] || '#ffffff'),
    transparent: true,
    opacity: EO[type] ?? 0.9,
    linewidth: EW[type] || 1.5,
    resolution: new THREE.Vector2(window.innerWidth - 280, window.innerHeight - 36),
  };
  if (dashedType) { matParams.dashed = true; matParams.dashSize = TREE_DASH_SIZE; matParams.gapSize = TREE_GAP_SIZE; }
  const mat = new LineMaterial(matParams);
  const obj = new LineSegments2(geo, mat);
  obj.renderOrder = 999;
  obj.visible = false;
  // Full opacity target for the collapse/expand fade, captured once here.
  obj.userData.baseOpacity = matParams.opacity;
  scene.add(obj);
  fatLineMaterials.push(mat);
  treeView.overlayObjs[type] = obj;
  return obj;
}
function updateTreeOverlayGeometry(getPos) {
  treeView.edgesByType.forEach((pairs, type) => {
    const arr = [];
    pairs.forEach(([s, t]) => {
      const a = getPos(s), b = getPos(t);
      if (!a || !b) return;
      arr.push(a.x, a.y, a.z, b.x, b.y, b.z);
    });
    const obj = ensureTreeOverlay(type);
    obj.userData.treePairs = pairs;
    obj.userData.treeType = type;
    if (!arr.length) { obj.visible = false; return; }
    obj.geometry.dispose();
    obj.geometry = new LineSegmentsGeometry();
    obj.geometry.setPositions(arr);
    if (SKELETON_TYPES.includes(type)) obj.computeLineDistances();  // required for the dash pattern
    obj.visible = true;
  });
}
function hideTreeOverlays() {
  Object.values(treeView.overlayObjs).forEach(o => { o.visible = false; });
}

// Normal edges hide during the tree view; restored on exit per the checkbox state.
function setNormalEdgesVisible(show) {
  Object.entries(pickableEdges).forEach(([g, e]) => { e.obj.visible = show ? vis.has(g) : false; });
  Object.entries(groupLines).forEach(([g, arr]) => {
    arr.forEach(l => { l.visible = show ? vis.has(g) : false; });
  });
}

function clearTreeLabels() {
  hideTreeHoverOverlay();
  treeView.treeLabelObjs.forEach(o => {
    if (o && o.parent) o.parent.remove(o);
    if (o && o.dispose) o.dispose();
  });
  treeView.treeLeaderObjs.forEach(o => {
    if (o && o.parent) o.parent.remove(o);
    if (o && o.geometry) o.geometry.dispose();
    if (o && o.material) o.material.dispose();
  });
  treeView.treeLabelObjs = [];
  treeView.treeLeaderObjs = [];
  treeView.legendSpecs = [];
  treeView.hoverIds = null;
  treeView.hoverSpecs = new Set();
  treeView.hoverEdge = null;
  treeView.labelsPlaced = false;
}

function addTreeLegendLabel(text, color, spec) {
  if (typeof TroikaText === 'undefined') return null;
  const t = new TroikaText();
  t.text = text;
  t.fontSize = TREE_TITLE_FONT_SIZE;
  t.color = color;
  t.anchorX = 'center';
  t.anchorY = 'middle';
  t.outlineWidth = '3%';
  t.outlineColor = '#05060b';
  t.outlineOpacity = 0.95;
  t.renderOrder = 1002;
  t.frustumCulled = false;
  // Actual 3D offset toward the camera is handled by positionTreeTitleInFront().
  t.depthOffset = 0;
  t.userData.legendSpec = spec;
  t.userData.treeTitle = true;
  t.userData.treeTitleIds = spec.ids;
  t.sync();
  scene.add(t);
  treeView.treeLabelObjs.push(t);
  return t;
}

// Dedicated sidebar for the tree view (#tt), replacing Edges/Map while
// active; purely informational, nothing here triggers a scene action.
function _treeStatsFor(keepSet, edgesByType) {
  const pools = new Set();
  const branches = new Set();
  let maxDepth = 0;
  keepSet.forEach(id => {
    const n = NM[id];
    if (!n) return;
    if (n.adept_pool_hub_id) pools.add(n.adept_pool_hub_id);
    if (n.nova_id) branches.add(n.nova_id);
    if (typeof n.depth === 'number' && n.depth > maxDepth) maxDepth = n.depth;
  });
  const edgeCounts = [...(edgesByType || new Map()).entries()]
    .map(([type, arr]) => ({ type, count: arr.length }))
    .sort((a, b) => b.count - a.count);
  return {
    nodeCount: keepSet.size,
    poolCount: pools.size,
    branchCount: branches.size,
    maxDepth,
    edgeCounts,
  };
}

function showTreeLoading() {
  const el = document.getElementById('tree-loading');
  if (el) el.classList.add('visible');
}
function hideTreeLoading() {
  const el = document.getElementById('tree-loading');
  if (el) el.classList.remove('visible');
}

// Tabs are hidden, not just their content, so they aren't clickable during the tree view.
function showTreeSidebar() {
  const tabs = document.getElementById('sb-tabs');
  const te = document.getElementById('te');
  const tm = document.getElementById('tm');
  const tt = document.getElementById('tt');
  if (tabs) tabs.style.display = 'none';
  if (te) te.classList.remove('s');
  if (tm) tm.classList.remove('s');
  if (tt) tt.classList.add('s');
}

// Restores the sidebar tab active before entering the tree view; called
// from finishExitTreeView() once treeView.active is already false.
function hideTreeSidebar() {
  const tabs = document.getElementById('sb-tabs');
  const tt = document.getElementById('tt');
  if (tt) tt.classList.remove('s');
  if (tabs) tabs.style.display = '';
  swTab(sbLastTab || 'e');
}

function buildTreeSidebarContent(keepSet, edgesByType) {
  const tree = document.getElementById('tree-content');
  if (!tree) return;

  const clusters = new Map();
  keepSet.forEach(id => {
    const n = NM[id];
    if (!n || !n.cluster_id) return;
    const key = String(n.cluster_id);
    if (!clusters.has(key)) {
      clusters.set(key, { id: key, label: n.cluster_label || key, color: n.color || '#9fb3ff', count: 0 });
    }
    clusters.get(key).count++;
  });
  const clusterArr = [...clusters.values()].sort((a, b) =>
    String(a.label).localeCompare(String(b.label), 'fr')
  );

  const stats = _treeStatsFor(keepSet, edgesByType);
  treeView.summary = stats;

  const clusterHtml = clusterArr.length
    ? clusterArr.map(c =>
        '<div class="gx-item static" title="' + c.id + '">' +
          '<div class="gx-dot" style="background:' + c.color + '"></div>' +
          '<div class="gx-name">' + c.label + '</div>' +
          '<div class="gx-count">' + c.count + '</div>' +
        '</div>'
      ).join('')
    : '<div style="color:#444;font-size:11px;padding:6px 2px">No cluster</div>';

  const edgeHtml = stats.edgeCounts.length
    ? stats.edgeCounts.map(({ type, count }) => {
        const col = EC[type] || '#888';
        return '<div class="gx-item static">' +
          '<div class="gx-dot" style="background:' + col + '"></div>' +
          '<div class="gx-name">' + type.replace(/_/g, ' ') + '</div>' +
          '<div class="gx-count">' + count + '</div>' +
        '</div>';
      }).join('')
    : '';

  tree.innerHTML =
    '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">' +
      '<div style="font-size:13px;font-weight:700;color:#eaf0ff;flex:1">Network View</div>' +
      '<button id="tt-close" aria-label="Back" style="width:24px;height:24px;border:0;border-radius:50%;' +
        'background:rgba(255,255,255,.08);color:#fff;font-size:15px;line-height:22px;cursor:pointer">×</button>' +
    '</div>' +
    '<div class="gx-section">' +
      '<div class="ir"><div class="il">Nodes</div><div class="iv">' + stats.nodeCount + '</div></div>' +
      '<div class="ir"><div class="il">Clusters</div><div class="iv">' + clusterArr.length + '</div></div>' +
      '<div class="ir"><div class="il">ADEPT pools</div><div class="iv">' + stats.poolCount + '</div></div>' +
      '<div class="ir"><div class="il">Nova branches</div><div class="iv">' + stats.branchCount + '</div></div>' +
      '<div class="ir"><div class="il">Max depth</div><div class="iv">' + stats.maxDepth + '</div></div>' +
    '</div>' +
    '<div class="gx-section"><div class="gx-title">Clusters</div>' + clusterHtml + '</div>' +
    (edgeHtml ? '<div class="gx-section"><div class="gx-title">Edges</div>' + edgeHtml + '</div>' : '') +
    '<div style="font-size:10px;opacity:.5;text-align:center;margin-top:8px">Esc · double-click to go back</div>';

  const closeBtn = document.getElementById('tt-close');
  if (closeBtn) closeBtn.onclick = () => exitTreeView(false);
}

function updateTreeSummary(rootId, keepSet) {
  showTreeSidebar();
  buildTreeSidebarContent(keepSet, treeView.edgesByType);
}

// Tree titles are annotations, not edges — no leader lines. Tested against
// nodes, edges, and other titles in the tree's XY plane; the first free
// side (left/right, then up/down) is picked with a progressive margin.

function treeTextMetrics(text, tier) {
  // Collision box must be at the same scale as the Troika fontSize.
  const fs = TREE_TITLE_FONT_SIZE;
  const raw = String(text || '');
  // Conservative estimate, independent of an async Troika render.
  const width = Math.max(fs * 1.5, raw.length * fs * 0.60);
  const height = fs * 1.20;
  return { width, height, fs };
}

function treeNodeRadiusForLabel(id) {
  const n = NM[id] || {};
  const size = Number(n.size) || 10;
  return Math.max(6, size * 1.5 * _geomScaleMult(n.geometry));
}

function treeRectOverlap(a, b, pad = 0) {
  return !(a.x2 + pad < b.x1 || a.x1 - pad > b.x2 ||
           a.y2 + pad < b.y1 || a.y1 - pad > b.y2);
}

function treePointSegDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const l2 = dx * dx + dy * dy;
  let t = l2 > 1e-9 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
  t = Math.max(0, Math.min(1, t));
  const qx = ax + t * dx, qy = ay + t * dy;
  return (px - qx) ** 2 + (py - qy) ** 2;
}

function treeSegmentHitsRect(a, b, r, pad = 0) {
  const x1 = r.x1 - pad, x2 = r.x2 + pad;
  const y1 = r.y1 - pad, y2 = r.y2 + pad;
  // Liang-Barsky segment/rectangle intersection.
  const dx = b.x - a.x, dy = b.y - a.y;
  let t0 = 0, t1 = 1;
  const clip = (p, q) => {
    if (Math.abs(p) < 1e-12) return q >= 0;
    const t = q / p;
    if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; }
    else       { if (t < t0) return false; if (t < t1) t1 = t; }
    return true;
  };
  return clip(-dx, a.x - x1) &&
         clip( dx, x2 - a.x) &&
         clip(-dy, a.y - y1) &&
         clip( dy, y2 - a.y);
}

function placeTreeLabels(specs, layout, links, rootId) {
  // Strictly 2D (X/Y); Z is set once for all labels in positionTreeTitleInFront().
  const placed = [];
  const ids = Object.keys(layout);

  const directions = [
    [ 1, 0], [-1, 0], [0, 1], [0,-1],
    [ 0.707, 0.707], [-0.707, 0.707],
    [ 0.707,-0.707], [-0.707,-0.707]
  ];

  const sorted = [...specs].sort((a,b) => {
    const ma = treeTextMetrics(a.text,a.tier), mb = treeTextMetrics(b.text,b.tier);
    return (mb.width * mb.height) - (ma.width * ma.height);
  });

  // Purely planar collision test: title box expanded by GAP on X/Y.
  function collides(rect) {
    for (const r of placed) {
      if (treeRectOverlap(rect, r, TREE_TITLE_LABEL_GAP)) return true;
    }

    for (const id of ids) {
      const p = layout[id];
      if (!p) continue;
      const rr = treeNodeRadiusForLabel(id) + TREE_TITLE_LABEL_GAP;
      const cx = Math.max(rect.x1, Math.min(p.x, rect.x2));
      const cy = Math.max(rect.y1, Math.min(p.y, rect.y2));
      if ((p.x-cx)*(p.x-cx) + (p.y-cy)*(p.y-cy) < rr*rr) return true;
    }

    for (const l of links) {
      const A = layout[String(l.source)], B = layout[String(l.target)];
      if (!A || !B) continue;
      if (treeSegmentHitsRect(A, B, rect, TREE_TITLE_LABEL_GAP)) return true;
    }
    return false;
  }

  function makeRect(cx, cy, m) {
    return {
      x1: cx - m.width / 2,
      x2: cx + m.width / 2,
      y1: cy - m.height / 2,
      y2: cy + m.height / 2
    };
  }

  sorted.forEach(spec => {
    const m = treeTextMetrics(spec.text, spec.tier);
    const anchors = [...new Set((spec.ids || []).map(String))].filter(id => layout[id]);
    if (!anchors.length) return;

    let best = null;

    // 1) Natural placement: GAP is folded into the distance from the node.
    for (const anchorId of anchors) {
      const a = layout[anchorId];
      const base = treeNodeRadiusForLabel(anchorId);
      const gaps = [
        base + TREE_TITLE_LABEL_GAP + 8,
        base + TREE_TITLE_LABEL_GAP + Math.max(24, TREE_BRANCH_LEN * 0.10),
        base + TREE_TITLE_LABEL_GAP + Math.max(40, TREE_BRANCH_LEN * 0.20),
        base + TREE_TITLE_LABEL_GAP + Math.max(60, TREE_BRANCH_LEN * 0.35)
      ];

      for (const gap of gaps) {
        for (const d of directions) {
          const cx = a.x + d[0] * gap;
          const cy = a.y + d[1] * gap;
          const rect = makeRect(cx, cy, m);
          if (collides(rect)) continue;
          const score = gap + (Math.abs(d[1]) < 0.1 ? -0.01 : 0);
          if (!best || score < best.score) best = {cx, cy, score, anchorId};
        }
      }
    }

    // 2) Fallback: expanding ring search on a 2D grid.
    if (!best) {
      const anchorId = anchors[0];
      const a = layout[anchorId];
      const step = Math.max(12, TREE_TITLE_LABEL_GAP * 0.75, TREE_TITLE_FONT_SIZE * 0.18);
      const maxRadius = Math.max(TREE_BRANCH_LEN * 5, 1200);

      outer:
      for (let radius = step; radius <= maxRadius; radius += step) {
        for (let i = 0; i < 32; i++) {
          const angle = (Math.PI * 2 * i) / 32;
          const cx = a.x + Math.cos(angle) * radius;
          const cy = a.y + Math.sin(angle) * radius;
          const rect = makeRect(cx, cy, m);
          if (!collides(rect)) {
            best = {cx, cy, score: radius, anchorId};
            break outer;
          }
        }
      }
    }

    // 3) Last resort: progressively widening 2D grid search.
    if (!best) {
      const anchorId = anchors[0];
      const a = layout[anchorId];
      const step = Math.max(8, TREE_TITLE_LABEL_GAP * 0.5);
      const maxCells = 180;
      outerGrid:
      for (let ring = 1; ring <= maxCells && !best; ring++) {
        for (let ix = -ring; ix <= ring; ix++) {
          for (const iy of [-ring, ring]) {
            const cx = a.x + ix * step;
            const cy = a.y + iy * step;
            const rect = makeRect(cx, cy, m);
            if (!collides(rect)) { best = {cx, cy, score: Math.abs(ix)+Math.abs(iy), anchorId}; break outerGrid; }
          }
        }
        for (let iy = -ring + 1; iy <= ring - 1; iy++) {
          for (const ix of [-ring, ring]) {
            const cx = a.x + ix * step;
            const cy = a.y + iy * step;
            const rect = makeRect(cx, cy, m);
            if (!collides(rect)) { best = {cx, cy, score: Math.abs(ix)+Math.abs(iy), anchorId}; break outerGrid; }
          }
        }
      }
    }

    // Saturated: don't force a colliding position, place it outside the tree area.
    if (!best) {
      const a = layout[anchors[0]];
      const far = Math.max(TREE_BRANCH_LEN * 20, 5000);
      best = { cx: a.x + far, cy: a.y + far, score: Infinity, anchorId: anchors[0] };
    }

    const anchor = layout[best.anchorId];
    spec.x = best.cx;
    spec.y = best.cy;
    spec.z = TREE_TITLE_Z_OFFSET; // informational only: same Z for all titles
    spec.anchorId = best.anchorId;
    spec.anchorX = anchor.x;
    spec.anchorY = anchor.y;
    spec.offsetX = best.cx - anchor.x;
    spec.offsetY = best.cy - anchor.y;
    spec.side = best.cx < anchor.x ? 'left' : 'right';
    spec.width = m.width;
    spec.height = m.height;
    spec.textMetrics = m;

    placed.push(makeRect(best.cx, best.cy, m));
  });

  // Final pass: enforces TREE_TITLE_LABEL_GAP between any two boxes, X/Y only.
  const labelItems = specs.filter(s =>
    Number.isFinite(s.x) && Number.isFinite(s.y) &&
    Number.isFinite(s.width) && Number.isFinite(s.height)
  );

  const makeExpandedRect = spec => ({
    x1: spec.x - spec.width / 2 - TREE_TITLE_LABEL_GAP,
    x2: spec.x + spec.width / 2 + TREE_TITLE_LABEL_GAP,
    y1: spec.y - spec.height / 2 - TREE_TITLE_LABEL_GAP,
    y2: spec.y + spec.height / 2 + TREE_TITLE_LABEL_GAP
  });

  // Separates overlapping rectangles along their axis of minimal penetration.
  for (let pass = 0; pass < 160; pass++) {
    let changed = false;

    for (let i = 0; i < labelItems.length; i++) {
      for (let j = i + 1; j < labelItems.length; j++) {
        const a = labelItems[i], b = labelItems[j];
        const ra = makeExpandedRect(a), rb = makeExpandedRect(b);

        const overlapX = Math.min(ra.x2, rb.x2) - Math.max(ra.x1, rb.x1);
        const overlapY = Math.min(ra.y2, rb.y2) - Math.max(ra.y1, rb.y1);
        if (overlapX <= 0 || overlapY <= 0) continue;

        // Push along the axis that needs the smaller move.
        if (overlapX <= overlapY) {
          const dir = a.x <= b.x ? -1 : 1;
          const move = overlapX * 0.5 + 0.01;
          a.x += dir * move;
          b.x -= dir * move;
        } else {
          const dir = a.y <= b.y ? -1 : 1;
          const move = overlapY * 0.5 + 0.01;
          a.y += dir * move;
          b.y -= dir * move;
        }
        changed = true;
      }
    }

    if (!changed) break;
  }

  // Recompute offsets after the final pass; Z stays identical for all titles.
  labelItems.forEach(spec => {
    const anchor = layout[spec.anchorId];
    if (!anchor) return;
    spec.offsetX = spec.x - anchor.x;
    spec.offsetY = spec.y - anchor.y;
    spec.z = TREE_TITLE_Z_OFFSET;
  });
}

function makeTreeNodeLabels(keepSet, layout) {
  clearTreeLabels();
  const specs = [];

  // NOVA: any node in the Nova can anchor; placement picks the nearest free one.
  const novaGroups = new Map();
  keepSet.forEach(id => {
    const n = NM[id];
    if (!n || !n.subtopic_label) return;
    const key = String(n.subtopic_label).trim();
    if (!novaGroups.has(key)) novaGroups.set(key, []);
    novaGroups.get(key).push(String(id));
  });
  novaGroups.forEach((ids, key) => {
    const unique = [...new Set(ids)];
    specs.push({
      tier:'nova',
      ids:unique,
      anchorId:unique[0],
      text:'✦ ' + key,
      color:'#d7a7ff',
      anchorMode:'node'
    });
  });

  // ADEPT: the anchor is always the hub carrying the title.
  const adeptGroups = new Map();
  RAW.links.forEach(l => {
    if (l.group !== 'adept_spoke') return;
    const s = String(l.source), t = String(l.target);
    const sn = NM[s], tn = NM[t];
    const hubId = sn?.adept_pool_label ? s :
                  (tn?.adept_pool_label ? t : null);
    if (!hubId || !keepSet.has(s) || !keepSet.has(t)) return;
    if (!adeptGroups.has(hubId)) adeptGroups.set(hubId, new Set([hubId]));
    adeptGroups.get(hubId).add(s === hubId ? t : s);
  });
  adeptGroups.forEach((set, hubId) => {
    const hub = NM[hubId];
    if (!hub?.adept_pool_label) return;
    specs.push({
      tier:'adept',
      ids:[...set],
      anchorId:String(hubId),
      text:'⬢ ' + hub.adept_pool_label,
      color:'#ffb15c',
      anchorMode:'node'
    });
  });

  // Created invisible; positioned once the tree has settled (see finalizeTreeLabels).
  specs.forEach(spec => {
    const t = addTreeLegendLabel(spec.text, spec.color, spec);
    if (t) t.visible = false;
  });
  treeView.legendSpecs = specs;
  treeView.labelsPlaced = false;
}

// One shared Z for every title (never per-title, or they lose coplanarity);
// billboarded to the camera every frame while the tree view is active.
function positionTreeTitleInFront(t, spec) {
  if (!t || !spec) return;
  t.position.set(spec.x, spec.y, TREE_TITLE_Z_OFFSET);
  t.quaternion.copy(camera.quaternion);
}

function finalizeTreeLabels() {
  if (!treeView.active || treeView.labelsPlaced) return;
  const layout = treeView.targetPos || {};
  const keepSet = treeView.keepSet || new Set();
  const links = treeLinksFor(keepSet);

  placeTreeLabels(treeView.legendSpecs || [], layout, links, treeView.canonicalRoot);

  (treeView.legendSpecs || []).forEach(spec => {
    const t = treeView.treeLabelObjs.find(o => o.userData.legendSpec === spec);
    if (!t) return;
    positionTreeTitleInFront(t, spec);
    t.visible = true;
  });
  treeView.labelsPlaced = true;
}

// Titles never animate with the tree: hidden until fully placed, then held
// fixed at their anchor — no interpolation during tree-view entry.
function updateTreeLabels(getPos) {
  if (!treeView.labelsPlaced) {
    (treeView.treeLabelObjs || []).forEach(t => { t.visible = false; });
    return;
  }

  (treeView.legendSpecs || []).forEach(spec => {
    const t = treeView.treeLabelObjs.find(o => o.userData.legendSpec === spec);
    if (!t) return;
    positionTreeTitleInFront(t, spec);
    t.visible = true;
  });
}

// Full ADEPT pool a node belongs to: adept_role/adept_pool_label first,
// falling back to the adept_spoke connected component if metadata is missing.
function treeAdeptPoolIdsForNode(id) {
  const sid = String(id);
  const keep = treeView.keepSet || new Set();
  if (!keep.has(sid)) return new Set([sid]);

  const adj = new Map();
  keep.forEach(x => adj.set(String(x), []));
  RAW.links.forEach(l => {
    if (l.group !== 'adept_spoke') return;
    const a = String(l.source), b = String(l.target);
    if (!keep.has(a) || !keep.has(b)) return;
    adj.get(a)?.push(b);
    adj.get(b)?.push(a);
  });

  // ADEPT connected component containing the node.
  const component = new Set([sid]);
  const q = [sid];
  for (let i = 0; i < q.length; i++) {
    const cur = q[i];
    (adj.get(cur) || []).forEach(n => {
      if (!component.has(n)) { component.add(n); q.push(n); }
    });
  }
  if (component.size <= 1) return new Set([sid]);

  // Prefer the hub explicitly declared in the data.
  let hub = [...component].find(x => NM[x]?.adept_role === 'hub' || NM[x]?.adept_pool_label);

  // Structural fallback: hub = node with the most spokes (stays identifiable without a title).
  if (!hub) {
    hub = [...component].sort((a,b) =>
      (adj.get(b)?.length || 0) - (adj.get(a)?.length || 0) || a.localeCompare(b)
    )[0];
  }

  const pool = new Set([hub]);
  const qq = [hub];
  for (let i = 0; i < qq.length; i++) {
    const cur = qq[i];
    (adj.get(cur) || []).forEach(n => {
      if (!pool.has(n)) { pool.add(n); qq.push(n); }
    });
  }
  return pool;
}

function treeRelatedIdsForNode(id) {
  const sid = String(id);
  const out = new Set([sid]);

  // ADEPT: the pool is a first-class entity even without a title.
  treeAdeptPoolIdsForNode(sid).forEach(x => out.add(String(x)));

  (treeView.legendSpecs || []).forEach(spec => {
    if (spec.ids.includes(sid)) spec.ids.forEach(x => out.add(String(x)));
  });
  return out;
}

function treeRelatedIdsForEdge(edge) {
  if (!edge) return new Set();
  const source = String(edge.source), target = String(edge.target);
  const out = new Set([source, target]);

  // A pool is first-class regardless of edge type: resolve each endpoint to its pool.
  treeAdeptPoolIdsForNode(source).forEach(x => out.add(String(x)));
  treeAdeptPoolIdsForNode(target).forEach(x => out.add(String(x)));

  (treeView.legendSpecs || []).forEach(spec => {
    if (spec.ids.includes(source) || spec.ids.includes(target))
      spec.ids.forEach(x => out.add(String(x)));
  });
  return out;
}

function clearTreeHover() {
  treeView.hoverIds = null;
  treeView.hoverSpecs = new Set();
  treeView.hoverEdge = null;
  (treeView.treeLabelObjs || []).forEach(t => {
    const spec = t.userData.legendSpec;
    t.color = spec?.color || '#ffffff';
    t.outlineColor = '#05060b';
    t.outlineOpacity = 0.95;
    t.scale.setScalar(1);
    t.opacity = 1;
  });
  hideTreeHoverOverlay();
}

function setTreeHover(ids, edge = null, spec = null) {
  const set = ids ? new Set([...ids].map(String)) : null;
  treeView.hoverIds = set;
  treeView.hoverEdge = edge;
  treeView.hoverSpecs = new Set(
    (treeView.legendSpecs || []).filter(s =>
      spec ? s === spec : set && s.ids.some(id => set.has(String(id)))
    )
  );

  (treeView.treeLabelObjs || []).forEach(t => {
    const sp = t.userData.legendSpec;
    const on = treeView.hoverSpecs.has(sp);
    t.color = sp?.color || '#ffffff';
    t.outlineColor = on ? '#ffffff' : '#05060b';
    t.outlineOpacity = on ? 1 : 0.95;
    t.scale.setScalar(on ? 1.10 : 1);
    t.opacity = on ? 1 : 0.78;
  });

  updateTreeHoverOverlay();
}

function updateTreeHoverOverlay() {
  // Overlay only: highlighted edges get a brighter copy, nodes get a ring,
  // without touching the InstancedMesh or the real edge geometry.
  hideTreeHoverOverlay();
  const ids = treeView.hoverIds;
  if (!ids || !ids.size) return;

  const pts = [];
  treeView.edgesByType?.forEach((pairs, type) => {
    const arr = [];
    pairs.forEach(([s,t]) => {
      if (!ids.has(String(s)) || !ids.has(String(t))) return;
      const a = treeView.targetPos[String(s)], b = treeView.targetPos[String(t)];
      if (!a || !b) return;
      arr.push(a.x,a.y,a.z+5,b.x,b.y,b.z+5);
    });
    if (!arr.length) return;
    const geo = new LineSegmentsGeometry();
    geo.setPositions(arr);
    const mat = new LineMaterial({
      color: new THREE.Color('#ffffff'),
      transparent:true, opacity:0.98,
      linewidth: type === 'nova' ? 5 : 4,
      resolution:new THREE.Vector2(window.innerWidth-280,window.innerHeight-36)
    });
    const obj = new LineSegments2(geo,mat);
    obj.renderOrder = 1005;
    scene.add(obj);
    treeView.hoverObjs.push(obj);
    treeView.hoverMats.push(mat);
  });

  ids.forEach(id => {
    const p = treeView.targetPos[String(id)];
    const n = NM[String(id)];
    if (!p || !n) return;
    const r = treeNodeRadiusForLabel(String(id)) * 1.35;
    const ring = new THREE.Mesh(
      new THREE.SphereGeometry(r, 12, 8),
      new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0.16,wireframe:true,depthTest:false})
    );
    ring.position.set(p.x,p.y,p.z+6);
    ring.renderOrder = 1006;
    scene.add(ring);
    treeView.hoverObjs.push(ring);
  });
}

function hideTreeHoverOverlay() {
  (treeView.hoverObjs || []).forEach(o => {
    if (o.parent) o.parent.remove(o);
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
  (treeView.hoverMats || []).forEach(m => { if (m.dispose) m.dispose(); });
  treeView.hoverObjs = [];
  treeView.hoverMats = [];
}


// Deterministic canonical root for a component, so the rendered tree is
// identical regardless of which node/edge in that component was clicked.
function canonicalRootFor(order) {
  return [...order].sort((a, b) => {
    const da = (ADJ_ALL[a] || []).length, db = (ADJ_ALL[b] || []).length;
    if (db !== da) return db - da;
    const ca = String(NM[a]?.cluster_label || NM[a]?.cluster_id || '');
    const cb = String(NM[b]?.cluster_label || NM[b]?.cluster_id || '');
    const cc = ca.localeCompare(cb, 'fr');
    if (cc) return cc;
    return String(a).localeCompare(String(b));
  })[0];
}

function enterTreeView(rootId) {
  if (treeView.animating) { hideTreeLoading(); return; }
  if (treeView.active) { exitTreeView(true); }  // re-root: instant restore then re-enter

  // Clears any Map-mode cluster selection/camera flight first, so a
  // lingering glow can't interfere with the entry animation.
  _forceClearClusterSelection();

  const first = bfsWithParent(rootId);
  const canonicalRoot = canonicalRootFor(first.order);
  const { order, parent } = bfsWithParent(canonicalRoot);
  const keepSet = new Set(order);
  const layout  = computeTreeLayout(canonicalRoot, order, parent);

  const edgesByType = new Map();
  RAW.links.forEach(l => {
    if (keepSet.has(l.source) && keepSet.has(l.target)) {
      if (!edgesByType.has(l.group)) edgesByType.set(l.group, []);
      edgesByType.get(l.group).push([l.source, l.target]);
    }
  });

  const startPos = new Map();
  order.forEach(id => { const n = NM[id]; startPos.set(id, { x: n.x, y: n.y, z: n.z }); });

  treeView.rootId      = canonicalRoot;
  treeView.canonicalRoot = canonicalRoot;
  treeView.keepSet     = keepSet;
  treeView.startPos    = startPos;
  treeView.targetPos   = layout;
  treeView.edgesByType = edgesByType;
  treeView.animDir     = 1;
  treeView.animating   = true;
  treeView.active      = true;
  treeView.animStart   = performance.now();
  treeView._overlayPhase = null; // force geometry rebuild on the first tick
  treeView._revealPending = false;

  // Camera: frame the whole computed tree.
  let cx = 0, cy = 0, cz = 0, maxR = 1;
  order.forEach(id => { const p = layout[id]; cx += p.x; cy += p.y; cz += p.z; });
  cx /= order.length; cy /= order.length; cz /= order.length;
  order.forEach(id => {
    const p = layout[id];
    const r = Math.hypot(p.x - cx, p.y - cy, p.z - cz);
    if (r > maxR) maxR = r;
  });
  const xs = order.map(id => layout[id].x);
  const ys = order.map(id => layout[id].y);
  const spanX = Math.max(1, Math.max(...xs) - Math.min(...xs));
  const spanY = Math.max(1, Math.max(...ys) - Math.min(...ys));
  const vHalf = THREE.MathUtils.degToRad(camera.fov * 0.5);
  // Floor only, so titles stay readable on small trees; a large tree keeps
  // its full computed distance.
  const TREE_MIN_CAMERA_DISTANCE = 5000;
  const dist = Math.max(
    spanY * 0.5 / Math.tan(vHalf) * 1.35,
    spanX * 0.5 / (Math.max(0.5, camera.aspect) * Math.tan(vHalf)) * 1.35,
    TREE_MIN_CAMERA_DISTANCE
  );
  treeView.preCam = {
    pos: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
    yaw: fps.yaw, pitch: fps.pitch,
    quat: camera.quaternion.clone(),
  };
  treeView.camFrom   = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
  // Front-on view: keeps branches readable instead of oblique diagonals;
  // the user can still orbit freely afterward.
  treeView.camTo     = { x: cx, y: cy, z: cz + dist };
  treeView.camCenter = { x: cx, y: cy, z: cz };

  // Build the final tree-facing orientation once. During the transition the
  // camera slerps toward it rather than recomputing lookAt() every frame, so
  // orientation eases in smoothly instead of snapping between yaw/pitch values.
  const _treeLookM = new THREE.Matrix4();
  const _treeLookQ = new THREE.Quaternion();
  _treeLookM.lookAt(
    new THREE.Vector3(treeView.camTo.x, treeView.camTo.y, treeView.camTo.z),
    new THREE.Vector3(cx, cy, cz),
    camera.up
  );
  _treeLookQ.setFromRotationMatrix(_treeLookM);
  treeView.camFromQuat = treeView.preCam.quat.clone();
  treeView.camToQuat   = _treeLookQ.clone();

  // Right/up/forward derived from the exact same quaternion used to frame
  // the tree, so panning tracks the diagram's own plane regardless of any
  // camera.up quirk -- fwd points from the camera toward the tree center.
  treeView.camBasis = {
    right: new THREE.Vector3(1, 0, 0).applyQuaternion(treeView.camToQuat),
    up:    new THREE.Vector3(0, 1, 0).applyQuaternion(treeView.camToQuat),
    fwd:   new THREE.Vector3(0, 0, -1).applyQuaternion(treeView.camToQuat),
  };
  treeView.panZoom = { x: 0, y: 0, dist };
  // Pan a bit past the strict bounding box so edge nodes aren't glued to
  // the screen border; zoom stays anchored around the initial framing
  // distance rather than the universe's own navigation limits.
  treeView.panLimits = {
    x: spanX * 0.5 + Math.max(TREE_BRANCH_LEN, spanX * 0.25),
    y: spanY * 0.5 + Math.max(TREE_BRANCH_LEN, spanY * 0.25),
  };
  treeView.zoomLimits = {
    min: Math.max(200, TREE_BRANCH_LEN * 1.2),
    max: dist * 3,
  };

  const navHint = document.getElementById('nav-hint');
  if (navHint) navHint.textContent = NAV_HINT_TREE;
  const legRows = document.getElementById('leg-rows');
  if (legRows) legRows.innerHTML = LEG_ROWS_TREE;
  // Speed/scroll-speed readout means nothing once WASD/scroll stop driving
  // flight -- hide rather than show a frozen, now-misleading number.
  const spdBlock = document.getElementById('spd-block');
  const spdSep   = document.getElementById('spd-sep');
  const spdEl    = document.getElementById('speed');
  if (spdBlock) spdBlock.style.display = 'none';
  if (spdSep)   spdSep.style.display   = 'none';
  if (spdEl)    spdEl.style.display    = 'none';

  makeTreeNodeLabels(keepSet, layout);
  updateTreeSummary(canonicalRoot, keepSet);

  // Rest of the universe is already collapsed; the subgraph starts at
  // scale 0 so no tree overlay appears before the camera flight ends.
  hideNonKeptNodes(keepSet);
  keepSet.forEach(id => {
    const n = NM[id];
    const entry = nodeInstanceIndex[id];
    if (!n || !entry) return;
    setInstanceTransform(id, n.x, n.y, n.z, 0);
  });
  flagInstancedDirty();
  hideTreeOverlays();
  hideSkeletonHighlight();
  setNormalEdgesVisible(false);
}

function exitTreeView(instant) {
  if (!treeView.active) return;

  // The selection panel belongs to tree view mode; close it on exit,
  // including during the return animation or on double-click.
  closeSelectionPanel();

  // Tree titles must disappear immediately on exit. They must never travel
  // backward with the graph during the exit camera/node animation.
  clearTreeLabels();

  if (instant) {
    treeView.animating = false;
    finishExitTreeView();
    return;
  }
  // Keep the tree-view graph hidden during the return flight; the actual
  // universe is only restored once the transition ends, in finishExitTreeView().
  if (treeView.keepSet) {
    treeView.keepSet.forEach(id => {
      const n = NM[id];
      const entry = nodeInstanceIndex[id];
      if (!n || !entry) return;
      setInstanceTransform(id, n.x, n.y, n.z, 0);
    });
    flagInstancedDirty();
  }
  hideTreeOverlays();

  treeView.animDir   = -1;
  treeView.animating = true;
  treeView.animStart  = performance.now();
  treeView._overlayPhase = null;
}

function finishExitTreeView() {
  restoreAllNodeScales();
  hideTreeOverlays();
  clearTreeHover();
  clearTreeLabels();
  setNormalEdgesVisible(true);
  treeView.active  = false;
  treeView.rootId  = null;
  treeView.canonicalRoot = null;
  treeView.summary = null;
  treeView.legendSpecs = [];
  treeView.labelsPlaced = false;
  treeView.keepSet = null;
  treeView.camBasis = null;
  treeView.panZoom = null;
  treeView.panLimits = null;
  treeView.zoomLimits = null;
  hideTreeSidebar();
  const navHint = document.getElementById('nav-hint');
  if (navHint) navHint.textContent = NAV_HINT_DEFAULT;
  const legRows = document.getElementById('leg-rows');
  if (legRows) legRows.innerHTML = LEG_ROWS_DEFAULT;
  const spdBlock = document.getElementById('spd-block');
  const spdSep   = document.getElementById('spd-sep');
  const spdEl    = document.getElementById('speed');
  if (spdBlock) spdBlock.style.display = '';
  if (spdSep)   spdSep.style.display   = '';
  if (spdEl)    spdEl.style.display    = '';
  if (treeView.preCam) {
    camera.position.set(treeView.preCam.pos.x, treeView.preCam.pos.y, treeView.preCam.pos.z);
    if (treeView.preCam.quat) camera.quaternion.copy(treeView.preCam.quat);
    fps.yaw   = treeView.preCam.yaw;
    fps.pitch = treeView.preCam.pitch;
    camera.rotation.y = fps.yaw;
    camera.rotation.x = fps.pitch;
  }
}

// Locked pan/zoom camera for tree view -- see treeView.camBasis/panZoom above.
// Pan and zoom are called from render.py's RMB-drag/wheel/touch handlers
// once treeView.active; applyTreePanZoom() (called every frame from
// updateTreeViewFrame below) is the single place that actually writes
// camera.position/quaternion, so the two can never fight each other.
function treePan(dxWorld, dyWorld) {
  if (!treeView.active || treeView.animating || !treeView.panZoom) return;
  const pz = treeView.panZoom, lim = treeView.panLimits;
  pz.x = THREE.MathUtils.clamp(pz.x + dxWorld, -lim.x, lim.x);
  pz.y = THREE.MathUtils.clamp(pz.y + dyWorld, -lim.y, lim.y);
}

function treeZoom(factor) {
  if (!treeView.active || treeView.animating || !treeView.panZoom) return;
  const pz = treeView.panZoom, lim = treeView.zoomLimits;
  pz.dist = THREE.MathUtils.clamp(pz.dist * factor, lim.min, lim.max);
}

// Screen-pixel-to-world-unit conversion at the current zoom distance, so a
// drag tracks the cursor 1:1 regardless of how close/far the camera is --
// same perspective-projection formula used elsewhere for label sizing.
function treePanWorldPerPixel() {
  if (!treeView.panZoom) return 1;
  const vHalf = THREE.MathUtils.degToRad(camera.fov * 0.5);
  const h = Math.max(1, window.innerHeight - 36);
  return (2 * treeView.panZoom.dist * Math.tan(vHalf)) / h;
}

function applyTreePanZoom() {
  const pz = treeView.panZoom, b = treeView.camBasis, c = treeView.camCenter;
  if (!pz || !b || !c) return;
  camera.position.set(
    c.x + b.right.x * pz.x + b.up.x * pz.y - b.fwd.x * pz.dist,
    c.y + b.right.y * pz.x + b.up.y * pz.y - b.fwd.y * pz.dist,
    c.z + b.right.z * pz.x + b.up.z * pz.y - b.fwd.z * pz.dist,
  );
  camera.quaternion.copy(treeView.camToQuat);
}

// Called from animate(); only touches the isolated subgraph, never the rest of the universe.
function updateTreeViewFrame(now) {
  if (!treeView.active) return;

  // Graph stays invisible during a transition — camera can keep flying,
  // but subgraph nodes/edges must never cross the screen mid-flight.
  if (treeView.animating) {
    const dir = treeView.animDir;
    const elapsed = now - treeView.animStart;
    const camT = Math.min(1, elapsed / TREE_ANIM_MS);
    // Quintic smoothstep gives a soft start and finish without adding any
    // visual effect: only the camera motion itself is eased.
    const camFactor = camT * camT * camT * (camT * (camT * 6 - 15) + 10);

    // Keeps the subgraph at scale 0 even if another hook touched transforms this frame.
    if (treeView.keepSet) {
      treeView.keepSet.forEach(id => {
        const n = NM[id];
        const entry = nodeInstanceIndex[id];
        if (!n || !entry) return;
        setInstanceTransform(id, n.x, n.y, n.z, 0);
      });
      flagInstancedDirty();
    }
    hideTreeOverlays();
    updateTreeLabels(null);

    // Entry: universe camera -> tree. Exit: tree camera -> original position.
    const camFrom = dir === 1 ? treeView.camFrom : treeView.camTo;
    const camTo   = dir === 1 ? treeView.camTo   : treeView.preCam.pos;

    camera.position.set(
      camFrom.x + (camTo.x - camFrom.x) * camFactor,
      camFrom.y + (camTo.y - camFrom.y) * camFactor,
      camFrom.z + (camTo.z - camFrom.z) * camFactor,
    );

    // Interpolate orientation independently from position. This keeps the
    // user's initial view stable for the first instant, then turns smoothly
    // toward the tree without the abrupt lookAt() correction.
    const qFrom = dir === 1 ? treeView.camFromQuat : treeView.camToQuat;
    const qTo   = dir === 1 ? treeView.camToQuat   : treeView.camFromQuat;
    if (qFrom && qTo) {
      camera.quaternion.copy(qFrom).slerp(qTo, camFactor);
      fps.yaw = camera.rotation.y;
      fps.pitch = camera.rotation.x;
    }

    if (elapsed >= TREE_ANIM_MS) {
      if (dir === 1) {
        if (!treeView._revealPending) {
          treeView._revealPending = true;

          // The tree becomes visible only at this point, already in its final layout.
          const reveal = () => {
            treeView.keepSet.forEach(id => {
              const p = treeView.targetPos[String(id)];
              const entry = nodeInstanceIndex[id];
              if (!entry || !p) return;
              setInstanceTransform(
                id, p.x, p.y, p.z,
                baseScaleFor(NM[id], entry.geomKey)
              );
            });
            flagInstancedDirty();

            // Built directly at their final position, never at universe positions.
            updateTreeOverlayGeometry(id => treeView.targetPos[String(id)]);
            treeView.edgesByType.forEach((_, type) => {
              const obj = treeView.overlayObjs[type];
              if (obj && obj.material) {
                obj.material.opacity = obj.userData.baseOpacity ?? 0.9;
                obj.visible = true;
              }
            });

            finalizeTreeLabels();
            treeView.animating = false;
            treeView._revealPending = false;
          };

          if (treeView.keepSet.size <= TREE_INSTANT_REVEAL_THRESHOLD) {
            // Small subgraph: fast enough to reveal instantly, no spinner.
            reveal();
          } else {
            // Larger subgraph: spinner gets two frames to paint before the sync work.
            showTreeLoading();
            requestAnimationFrame(() => requestAnimationFrame(() => {
              reveal();
              hideTreeLoading();
            }));
          }
        }
      } else {
        treeView.animating = false;
        // Stays hidden until this restores the full universe in one step.
        finishExitTreeView();
      }
    }
    return;
  }

  applyTreePanZoom();

  treeView.dashOffset -= TREE_FLOW_SPEED;
  DIRECTIONAL_DASH_TYPES.forEach(t => {
    const obj = treeView.overlayObjs[t];
    if (obj && obj.visible && obj.material) {
      obj.material.dashOffset = treeView.dashOffset;
    }
  });

  updateTreeLabels(null);
}

// A Nova or ADEPT pool is a first-class entity; clicking any member node or
// its internal edge resolves to that entity. An edge between two distinct
// entities opens a two-entity comparison instead.
function selectionText(v, fallback = 'No description available.') {
  const s = String(v ?? '').trim();
  return s || fallback;
}
function selectionEsc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function selectionDateValid(v) {
  if (!v) return null;
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? null : d;
}
function selectionNode(id) { return NM[String(id)] || null; }
function selectionEntityForNode(id, preferredKind = null) {
  const n = selectionNode(id);
  if (!n) return null;

  const sid = String(id);
  const novaId = String(n.nova_id || '').trim();
  const hasNova = Boolean(novaId || n.subtopic_label || n.nova_title || n.nova_role);
  const poolIds = treeAdeptPoolIdsForNode(sid);
  const hasPool = Boolean(n.adept_pool_hub_id || n.adept_pool_label || n.adept_role || poolIds.size > 1);

  // A graph node belongs to exactly one structural category:
  // Nova, ADEPT pool, or standalone. No node is treated as belonging to both.
  const kind = preferredKind || (hasNova ? 'nova' : hasPool ? 'pool' : 'standalone');

  if (kind === 'nova' && hasNova) {
    const ids = novaId
      ? RAW.nodes.filter(x => String(x.nova_id || '') === novaId).map(x => String(x.id))
      : RAW.nodes.filter(x => {
          const key = String(n.subtopic_label || n.nova_title || '').trim();
          return key && String(x.subtopic_label || x.nova_title || '').trim() === key;
        }).map(x => String(x.id));
    const members = new Set(ids.length ? ids : [sid]);
    const representative = ids.map(selectionNode).find(x => x) || n;
    return { kind:'nova', id:novaId || String(n.subtopic_label || n.nova_title || sid), ids:members, node:representative };
  }

  if (kind === 'pool' && hasPool) {
    const members = poolIds.size > 1 ? poolIds : new Set([sid]);
    const hub = [...members].map(selectionNode).find(x => x?.adept_pool_hub_id) ||
                [...members].map(selectionNode).find(x => x?.adept_role === 'hub') || n;
    const hubId = String(n.adept_pool_hub_id || hub?.adept_pool_hub_id || hub?.id || sid);
    return { kind:'pool', id:hubId, ids:members, node:hub };
  }

  return { kind:'standalone', id:sid, ids:new Set([sid]), node:n };
}
function selectionEntityBlock(entity) {
  if (!entity) return '<div class="sel-card"><div class="sel-muted">Entity information is unavailable.</div></div>';
  const n = entity.node || {};
  if (entity.kind === 'standalone') {
    return '<div class="sel-entity sel-standalone">' +
      '<div class="sel-entity-title">Standalone node</div>' +
      '<div class="sel-label">Content</div><div class="sel-text">' + selectionEsc(selectionText(n.fullContent || n.content)) + '</div>' +
      '</div>';
  }
  const kindLabel = entity.kind === 'nova' ? 'Nova' : 'ADEPT Pool';
  const title = entity.kind === 'nova'
    ? (n.nova_title || n.subtopic_label || 'Nova')
    : (n.adept_pool_label || 'ADEPT Pool');
  const reasoning = entity.kind === 'nova' ? n.nova_reasoning : n.pool_reasoning;
  let html = '<div class="sel-entity ' + (entity.kind === 'nova' ? 'sel-nova' : 'sel-pool') + '">' +
    '<div class="sel-entity-title">' + selectionEsc(title) + '</div>' +
    '<div class="sel-label">Entity type</div><div class="sel-text">' + kindLabel + '</div>' +
    '<div class="sel-label">Explanation</div><div class="sel-text">' + selectionEsc(selectionText(reasoning)) + '</div>';
  if (entity.kind === 'nova') {
    html += '<div class="sel-label">Narrative evolution</div><div class="sel-text">' + selectionEsc(selectionText(n.nova_sentiment_arc)) + '</div>';
    if (!String(n.nova_reasoning || '').trim()) html += '<div class="sel-note">No Nova explanation was found in <code>nova_metadata.parquet</code>.</div>';
  } else if (!String(n.pool_reasoning || '').trim()) {
    html += '<div class="sel-note">No pool explanation was found in <code>pool_explanations.parquet</code>.</div>';
  }
  return html + '</div>';
}
function selectionEntityForEdge(edge) {
  if (!edge) return { a:null, b:null };
  const type = String(edge.group || '');

  if (type === 'nova') {
    return { a:selectionEntityForNode(edge.source, 'nova') || selectionEntityForNode(edge.target, 'nova'), b:null };
  }
  if (type === 'adept_spoke') {
    return { a:selectionEntityForNode(edge.source, 'pool') || selectionEntityForNode(edge.target, 'pool'), b:null };
  }
  if (type === 'adept_graft') {
    const a = selectionEntityForNode(edge.source);
    const b = selectionEntityForNode(edge.target);
    return { a, b };
  }

  // semantic_inter / temporal / temporal_influence are inter-entity bridges.
  const a = selectionEntityForNode(edge.source);
  const b = selectionEntityForNode(edge.target);
  if (!a && !b) return { a:null, b:null };
  if (!a) return { a:b, b:null };
  if (!b) return { a:a, b:null };
  if (a.kind === b.kind && a.id === b.id) return { a:a, b:null };
  return { a:a, b:b };
}
function selectionEntityName(entity) {
  if (!entity) return 'Unknown entity';
  if (entity.kind === 'standalone') return 'Standalone node';
  const n = entity.node || {};
  return entity.kind === 'nova'
    ? String(n.nova_title || n.subtopic_label || 'Nova')
    : String(n.adept_pool_label || 'ADEPT Pool');
}
function selectionEntityTimestamp(entity) {
  if (!entity) return null;
  const values = [...(entity.ids || [])].map(id => selectionNode(id)?.timestamp).filter(Boolean);
  const dates = values.map(selectionDateValid).filter(Boolean).sort((a,b) => a-b);
  return dates.length ? dates[0] : null;
}
function orderSemanticEntities(a, b) {
  const da = selectionEntityTimestamp(a), db = selectionEntityTimestamp(b);
  if (da && db && da.getTime() !== db.getTime()) return da < db ? [a,b] : [b,a];
  const ka = selectionEntityName(a).toLocaleLowerCase('en');
  const kb = selectionEntityName(b).toLocaleLowerCase('en');
  if (ka !== kb) return ka < kb ? [a,b] : [b,a];
  return String(a?.id || '').localeCompare(String(b?.id || '')) <= 0 ? [a,b] : [b,a];
}
function selectionEdgeExplanation(edge, a, b) {
  const type = String(edge?.group || '');
  const force = Number(edge?.force);
  let html = '<div class="sel-card"><div class="sel-label">Edge type</div><div class="sel-edge-type">' + selectionEsc(type || 'Unknown') + '</div>';
  if (Number.isFinite(force) && force !== 0) html += '<div class="sel-label">Force</div><div class="sel-text">' + force.toLocaleString('en-US', {maximumFractionDigits: 4}) + '</div>';

  if (type === 'semantic_inter') {
    html += '<div class="sel-label">Direction</div><div class="sel-text">Non-directional semantic relation.</div>';
    const da = selectionEntityTimestamp(a), db = selectionEntityTimestamp(b);
    if (da && db && da.getTime() !== db.getTime()) {
      const [older, newer] = orderSemanticEntities(a, b);
      html += '<div class="sel-label">Display order</div><div class="sel-text"><b>' + selectionEsc(selectionEntityName(older)) + '</b> comes before <b>' + selectionEsc(selectionEntityName(newer)) + '</b> by timestamp.</div>';
    } else {
      html += '<div class="sel-label">Display order</div><div class="sel-text">No complete timestamp ordering is available; entities are shown in a deterministic order. This does not imply direction.</div>';
    }
  } else if (type === 'temporal' || type === 'temporal_influence') {
    html += '<div class="sel-label">Direction</div><div class="sel-text"><b>' + selectionEsc(selectionEntityName(a)) + '</b> → <b>' + selectionEsc(selectionEntityName(b)) + '</b>.</div>';
    const da = selectionDateValid(selectionNode(edge.source)?.timestamp), db = selectionDateValid(selectionNode(edge.target)?.timestamp);
    if (da && db) {
      html += '<div class="sel-label">Temporal order</div><div class="sel-text">The source is before the target in time.</div>';
    } else {
      html += '<div class="sel-label">Temporal order</div><div class="sel-text">Timestamp data is incomplete; the stored edge direction is used.</div>';
    }
    if (type === 'temporal_influence') {
      const na = selectionNode(edge.source), nb = selectionNode(edge.target);
      html += '<div class="sel-label">What defines the influence</div>';
      if (na && nb && Number.isFinite(Number(na.engagement)) && Number.isFinite(Number(nb.engagement))) {
        html += '<div class="sel-text">Influence is evaluated using the stored engagement relationship together with semantic similarity and temporal decay. Source engagement: <b>' + Number(na.engagement).toLocaleString('en-US') + '</b>; target engagement: <b>' + Number(nb.engagement).toLocaleString('en-US') + '</b>.</div>';
      } else {
        html += '<div class="sel-text">Engagement data is unavailable for one or both endpoints.</div>';
      }
    }
  } else if (type === 'adept_graft') {
    html += '<div class="sel-label">Interpretation</div><div class="sel-text">Bridge between an ADEPT pool and a Nova.</div>';
  } else if (type === 'adept_spoke') {
    html += '<div class="sel-label">Interpretation</div><div class="sel-text">Internal ADEPT pool connection.</div>';
  } else if (type === 'nova') {
    html += '<div class="sel-label">Interpretation</div><div class="sel-text">Internal Nova chain connection.</div>';
  }
  return html + '</div>';
}
function openSelectionPanelForNode(n) {
  if (!n) return;
  const entity = selectionEntityForNode(n.id);
  const panel = document.getElementById('selection-panel'), body = document.getElementById('selection-body'), title = document.getElementById('selection-title');
  if (!panel || !body || !title) return;
  if (entity) {
    title.textContent = entity.kind === 'nova' ? 'Nova' : 'ADEPT Pool';
    body.innerHTML = selectionEntityBlock(entity);
  } else {
    title.textContent = 'Node';
    body.innerHTML = '<div class="sel-card"><div class="sel-label">Node</div><div class="sel-text">This node is not associated with a Nova or ADEPT pool.</div></div>';
  }
  panel.style.display = 'flex';
}
function openSelectionPanelForTitle(spec) {
  if (!spec) return;
  const ids = (spec.ids || []).map(String);
  const entity = ids.map(selectionEntityForNode).find(Boolean) || null;
  const panel = document.getElementById('selection-panel'), body = document.getElementById('selection-body'), title = document.getElementById('selection-title');
  if (!panel || !body || !title) return;
  title.textContent = entity?.kind === 'nova' ? 'Nova' : 'ADEPT Pool';
  body.innerHTML = selectionEntityBlock(entity);
  panel.style.display = 'flex';
}
function openSelectionPanelForEdge(edge) {
  if (!edge) return;
  const raw = RAW.links.find(l => String(l.source) === String(edge.source) && String(l.target) === String(edge.target) && String(l.group) === String(edge.group)) || edge;
  const detailEdge = { ...edge, force: raw.force };
  const { a, b } = selectionEntityForEdge(detailEdge);
  const panel = document.getElementById('selection-panel'), body = document.getElementById('selection-body'), title = document.getElementById('selection-title');
  if (!panel || !body || !title) return;
  title.textContent = b ? 'Edge' : (a?.kind === 'nova' ? 'Nova' : 'ADEPT Pool');
  if (!a) {
    body.innerHTML = '<div class="sel-card"><div class="sel-muted">Entity information is unavailable.</div></div>';
  } else if (!b) {
    body.innerHTML = selectionEntityBlock(a);
  } else {
    body.innerHTML = selectionEdgeExplanation(detailEdge, a, b) + selectionEntityBlock(a, 'First entity') + selectionEntityBlock(b, 'Second entity');
  }
  panel.style.display = 'flex';
}
function closeSelectionPanel() {
  const panel = document.getElementById('selection-panel');
  if (panel) panel.style.display = 'none';
}
(function installSelectionPanel() {
  if (document.getElementById('selection-panel')) return;
  const panel = document.createElement('div');
  panel.id = 'selection-panel';
  panel.style.cssText = 'display:none;position:fixed;top:52px;right:292px;width:min(460px,calc(100vw - 320px));max-height:calc(100vh - 72px);background:rgba(7,9,18,.98);border:1px solid rgba(180,195,255,.22);border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,.55);z-index:70;flex-direction:column;overflow:hidden;backdrop-filter:blur(10px);';
  panel.innerHTML = '<div style="display:flex;align-items:center;gap:8px;padding:12px 14px;border-bottom:1px solid rgba(255,255,255,.08);flex-shrink:0"><div id="selection-title" style="font-size:13px;font-weight:700;flex:1;color:#eaf0ff">Selection</div><button id="selection-close" style="width:28px;height:28px;border:0;border-radius:50%;background:rgba(255,255,255,.08);color:#fff;font-size:18px;cursor:pointer">×</button></div><div id="selection-body" style="padding:12px 14px;overflow-y:auto;font-size:11px;color:#c8cce0;line-height:1.45"></div>';
  document.body.appendChild(panel);
  document.getElementById('selection-close').onclick = closeSelectionPanel;
  const style = document.createElement('style');
  style.textContent = '.sel-card{padding:10px 0;border-bottom:1px solid rgba(255,255,255,.07)}.sel-entity{margin-top:10px;padding:9px 10px;border:1px solid rgba(255,255,255,.08);border-radius:8px;background:rgba(255,255,255,.025)}.sel-nova{border-color:rgba(204,136,255,.25)}.sel-pool{border-color:rgba(255,140,0,.25)}.sel-standalone{border-color:rgba(255,255,255,.16)}.sel-entity-title{font-size:12px;font-weight:700;color:#eef2ff;margin-bottom:8px}.sel-label{font-size:9px;color:#707790;text-transform:uppercase;letter-spacing:.08em;margin-top:8px;margin-bottom:3px}.sel-text{white-space:pre-wrap;overflow-wrap:anywhere;color:#cfd4e6}.sel-muted,.sel-note{margin-top:7px;color:#777f96;font-size:10px}.sel-note code{color:#aab3d1}.sel-edge-type{font-size:12px;font-weight:700;color:#a9b7ff;padding:2px 0}.sel-card:last-child{border-bottom:0}';
  document.head.appendChild(style);
})();

document.addEventListener('keydown', e => { if (e.code === 'Escape' && treeView.active) exitTreeView(false); });
