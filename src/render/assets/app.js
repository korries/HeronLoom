// Main Three.js scene: camera/controls, node/edge instancing, labels,
// picking, sidebar UI. Placeholder tokens below are substituted by build.py.
// treeview.js (radial mode) is concatenated in further down.

import * as THREE from 'three';
import { LineSegments2 }     from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial }      from 'three/addons/lines/LineMaterial.js';
import { ConvexGeometry }    from 'three/addons/geometries/ConvexGeometry.js';

let TroikaText = null;
import('https://esm.sh/troika-three-text?external=three')
  .then(mod => { TroikaText = mod.Text; if (window.__initLabels) window.__initLabels(); })
  .catch(err => console.warn('[Labels] troika-three-text unavailable, labels disabled:', err));

const RAW      = __RENDER_DATA_JSON__;
const EC       = __EDGE_COLORS_JSON__;
const EW       = __EDGE_WIDTH_JSON__;  // edge width by type, config.yaml render.edge_width
const EO       = __EDGE_OPACITY_JSON__;  // edge opacity by type, config.yaml render.edge_opacity
const DASH_SIZE = __DASH_SIZE__;
const GAP_SIZE  = __GAP_SIZE__;
const FLOW_SPEED = __FLOW_SPEED__;  // animated dash flow speed (temporal edges)
const NOVA_SPLINE = __NOVA_SPLINE_JSON__;  // enabled, samples, tension
const LABEL_FONT_SIZE = __LABEL_FONT_SIZE_JSON__;  // base font size per tier
const LABEL_OFFSET = __LABEL_OFFSET_JSON__;  // vertical offset above anchor
const LABEL_DISTANCE = __LABEL_DISTANCE_JSON__;  // visibility distance per tier
const LABEL_MIN_FONT_SIZE = __LABEL_MIN_FONT_SIZE_JSON__;  // floor for rendered size
const CLUSTER_MIN_RADIUS = __CLUSTER_MIN_RADIUS__;  // cluster radius never reported smaller than this (world units) — see clusterCentroids below
const LABEL_SCREEN_PX = __LABEL_SCREEN_PX_JSON__;  // fixed on-screen size in px; 0 = perspective size
const LABEL_COLLISION_MARGIN_PX = __LABEL_COLLISION_MARGIN_PX__;  // px margin before labels collide
const LABEL_BACKDROP_PAD_X_RATIO = __LABEL_BACKDROP_PADX_RATIO__;  // horizontal panel margin, x text size
const LABEL_GUARANTEED_RADIUS_MULT = __LABEL_GUARANTEED_RADIUS_MULT__;  // always-shown radius around a cluster
const LABEL_FADE_SPEED = __LABEL_FADE_SPEED__;  // opacity transition speed
const LABEL_MAX_VISIBLE = __LABEL_MAX_VISIBLE_JSON__;  // visible-label budget, 0 = unlimited
const CLUSTER_LABEL_ANCHOR_OPACITY = 0.7;
const CLUSTER_LABEL_FADE_OUTER_MULT = 4.0;
const CLUSTER_LABEL_FADE_INNER_MULT = 1.3;
const CLUSTER_OCCLUSION_INNER_MARGIN_DEG = 4;
const CLUSTER_OCCLUSION_OUTER_MARGIN_DEG = 10;
const CLUSTER_OCCLUSION_DIST_OUTER_MULT = 2.0;
const CLUSTER_OCCLUSION_DIST_INNER_MULT = 2.0;
const CLUSTER_LABEL_SELECT_FADE_MS = 900;  // min fade-in time for Nova/Adept titles on a newly (re)selected cluster,
                                            // independent of distance — prevents an instant pop-in when the auto-
                                            // proximity switch lands on a cluster that is already close (e.g. two
                                            // clusters sitting near each other), without touching the existing
                                            // distance-based fade used for a normal approach from far away.
const LABEL_NUDGE_ENABLED = __LABEL_NUDGE_ENABLED__;  // declutter by nudging instead of hiding
const LABEL_HOVER_EASE_SPEED = 0.18;      // ramp speed for the Nova/Adept hover halo
const LABEL_HOVER_OUTLINE_PCT = 2;       // outline width (% of fontSize) at full hover, vs 7% at rest
const LABEL_NUDGE_MAX_PX = __LABEL_NUDGE_MAX_PX__;  // max nudge offset in px
const LABEL_NUDGE_ITERATIONS = __LABEL_NUDGE_ITERATIONS__;  // resolution passes per frame
const LABEL_NUDGE_EASE_SPEED = __LABEL_NUDGE_EASE_SPEED__;  // ease-back speed toward anchor
const GROUPS   = __EDGE_GROUPS_JSON__;
__UNIVERSE_JS__

const NM = {};
RAW.nodes.forEach(n => NM[n.id] = n);

const SKELETON_TYPES     = ['semantic_inter', 'temporal', 'temporal_influence', 'adept_graft'];
const SKELETON_TYPES_SET = new Set(SKELETON_TYPES);
const DIRECTIONAL_DASH_TYPES = new Set(['temporal', 'temporal_influence']);
const ADJ_ALL = {};  // nodeId -> [other, type, link], all edge types
RAW.links.forEach(l => {
  (ADJ_ALL[l.source] ??= []).push({ other: l.target, type: l.group, link: l });
  (ADJ_ALL[l.target] ??= []).push({ other: l.source, type: l.group, link: l });
});

const SKELETON_MAX_NODES = 3000;
function computeSkeleton(seedIds) {
  const nodes = new Set(seedIds);
  const edgeKeys = new Set();
  const edges = [];
  const queue = [...seedIds];
  let qi = 0;  // read index (avoids Array.shift cost)
  while (qi < queue.length && nodes.size < SKELETON_MAX_NODES) {
    const cur = queue[qi++];
    (ADJ_ALL[cur] || []).forEach(({other, type, link}) => {
      const ek = type + '|' + link.source + '|' + link.target;
      if (!edgeKeys.has(ek)) {
        edgeKeys.add(ek);
        edges.push({ source: link.source, target: link.target, type });
      }
      if (!nodes.has(other) && nodes.size < SKELETON_MAX_NODES) {
        nodes.add(other);
        queue.push(other);
      }
    });
  }
  return { nodes, edges };
}

let rotating  = false;
let rotTimer  = null;
let rotAngle  = 0;
const vis        = new Set(GROUPS.filter(g => !SKELETON_TYPES.includes(g)));
const groupLines = {};  // group -> [THREE.Line | LineSegments2]
let selectedNode = null;

const canvas   = document.getElementById('gc');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);

const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x080810);

const camera   = new THREE.PerspectiveCamera(75, 1, 1, UNIVERSE.farPlane);

const GLOBAL_NODE_LIGHT_INTENSITY = 1.0;

scene.add(new THREE.AmbientLight(0x404060, 1.5 * GLOBAL_NODE_LIGHT_INTENSITY));
const dl = new THREE.DirectionalLight(0xffffff, 1.0 * GLOBAL_NODE_LIGHT_INTENSITY);
dl.position.set(600, 600, 600);
scene.add(dl);
const dl2 = new THREE.DirectionalLight(0x8080ff, 0.3 * GLOBAL_NODE_LIGHT_INTENSITY);
dl2.position.set(-400, -400, 200);
scene.add(dl2);

// Declared here (not near pickableEdges below) because the frame IIFE
// runs immediately and pushes into it before buildEdges() is ever called.
const fatLineMaterials = [];  // active LineMaterials, resolution updated on resize

// Corner brackets and a single floor grid outline the navigable volume.
(function() {
  const nl = UNIVERSE.navLimits;
  if (!nl) return;

  const width  = nl.x.max - nl.x.min;
  const height = nl.y.max - nl.y.min;
  const depth  = nl.z.max - nl.z.min;
  const x0 = nl.x.min, x1 = nl.x.max;
  const y0 = nl.y.min, y1 = nl.y.max;
  const z0 = nl.z.min, z1 = nl.z.max;
  // Keep the floor grid on the exact same physical plane as the frame floor edge.
  // A tiny epsilon is only used to avoid depth fighting; the grid is still bounded
  // by the exact x/z extents of the frame.
  const zLift = Math.min(Math.max(height * 0.00015, 0.001), 0.005);
  const resW = window.innerWidth - 280, resH = window.innerHeight - 36;

  const bracketSize = Math.min(width, height, depth) * 0.055;
  const bracketPts = [];
  const addBracket = (cx, cy, cz, sx, sy, sz) => {
    // Three orthogonal "L" accents per corner.
    bracketPts.push(
      cx, cy, cz, cx + sx * bracketSize, cy, cz,
      cx, cy, cz, cx, cy + sy * bracketSize, cz,
      cx, cy, cz, cx, cy, cz + sz * bracketSize
    );
  };
  addBracket(x0, y0, z0, +1, +1, +1);
  addBracket(x1, y0, z0, -1, +1, +1);
  addBracket(x1, y0, z1, -1, +1, -1);
  addBracket(x0, y0, z1, +1, +1, -1);
  addBracket(x0, y1, z0, +1, -1, +1);
  addBracket(x1, y1, z0, -1, -1, +1);
  addBracket(x1, y1, z1, -1, -1, -1);
  addBracket(x0, y1, z1, +1, -1, -1);

  const bracketGeo = new LineSegmentsGeometry();
  bracketGeo.setPositions(bracketPts);
  const bracketMat = new LineMaterial({
    color: 0x86a3dc, transparent: true, depthWrite: false,
    opacity: 0.52, linewidth: 1.9, resolution: new THREE.Vector2(resW, resH),
  });
  const brackets = new LineSegments2(bracketGeo, bracketMat);
  brackets.renderOrder = 3;
  scene.add(brackets);
  fatLineMaterials.push(bracketMat);

  const connectorPts = [
    x0,y0,z0, x1,y0,z0,  x1,y0,z0, x1,y0,z1,
    x1,y0,z1, x0,y0,z1,  x0,y0,z1, x0,y0,z0,
    x0,y1,z0, x1,y1,z0,  x1,y1,z0, x1,y1,z1,
    x1,y1,z1, x0,y1,z1,  x0,y1,z1, x0,y1,z0,
    x0,y0,z0, x0,y1,z0,  x1,y0,z0, x1,y1,z0,
    x1,y0,z1, x1,y1,z1,  x0,y0,z1, x0,y1,z1,
  ];
  const connectorGeo = new THREE.BufferGeometry();
  connectorGeo.setAttribute('position', new THREE.Float32BufferAttribute(connectorPts, 3));
  const connectorMat = new THREE.LineBasicMaterial({
    color: 0x16243A, transparent: true, opacity: 0.30,
    depthWrite: false, depthTest: true,
  });
  const connectors = new THREE.LineSegments(connectorGeo, connectorMat);
  connectors.renderOrder = 2;
  scene.add(connectors);

  const gridGeo = new THREE.BufferGeometry();
  const gridPts = [];
  const gridStep = Math.max(Math.min(width, depth) / 64, 0.35);
  for (let x = x0 + gridStep; x < x1 - gridStep * 0.5; x += gridStep) {
    gridPts.push(x, y0 + zLift, z0, x, y0 + zLift, z1);
  }
  for (let z = z0 + gridStep; z < z1 - gridStep * 0.5; z += gridStep) {
    gridPts.push(x0, y0 + zLift, z, x1, y0 + zLift, z);
  }
  gridGeo.setAttribute('position', new THREE.Float32BufferAttribute(gridPts, 3));
  const gridMat = new THREE.LineBasicMaterial({
    color: 0x1a2a44, transparent: true, opacity: 0.24,
    depthWrite: false, depthTest: true,
  });
  const grid = new THREE.LineSegments(gridGeo, gridMat);
  grid.renderOrder = 1;
  scene.add(grid);
})();

const SPEED_MIN          = 50;      // scroll floor, world units/s
const SPEED_MAX          = 200000;  // scroll ceiling, world units/s
const SPEED_SHIFT_MULT   = 2;
const SPEED_ABSOLUTE_MAX = 400000;  // hard ceiling with Shift held, world units/s

const fps = {
  yaw:     0.0,  // left/right
  pitch:  -0.05,  // up/down
  speed:  UNIVERSE.flySpeedBase,
  keys:   {},
  prevTime: performance.now()
};

camera.rotation.order = 'YXZ';
camera.position.set(
  UNIVERSE.cameraSpawn.x,
  UNIVERSE.cameraSpawn.y,
  UNIVERSE.cameraSpawn.z
);
camera.rotation.y = fps.yaw;
camera.rotation.x = fps.pitch;

// Spawn orientation, captured once at load time. This is the target that
// F/overview also restores the camera to, in addition to position.
// Independent clone, never mutated afterward.
const SPAWN_QUATERNION = camera.quaternion.clone();

// Camera look: right-click hold or Space toggle, both via the real Pointer
// Lock API. This is the standard approach for FPS-style controls on the
// web (Unity WebGL, browser games, etc): only Pointer Lock gives genuine OS
// cursor capture with unbounded movementX/Y deltas, so looking around never
// hits an invisible ceiling at the screen edge the way a CSS cursor:none
// hack does.
//
// F11 (browser-level fullscreen) does NOT interact with Pointer Lock in
// current engines: F11 never fires the API-level 'fullscreenchange' event
// and is handled entirely outside the Fullscreen/Pointer Lock APIs, so the
// two are independent in practice. Escape is the browser's own built-in
// exit gesture for Pointer Lock and needs no handling here -- the
// 'pointerlockchange' listener below picks it up automatically.
let cameraLookMode = null; // null | 'hold' | 'toggle'

// True whenever the camera is actually translating this frame -- WASD/arrow
// flight or an automatic flyToCluster(). Updated once per frame in animate(),
// read by onMouseMove() to suppress hover picking entirely while moving (see
// there for why). Deliberately excludes tree view, which has its own much
// lighter pan/zoom controls.
let cameraIsMoving = false;
function _movementKeysActive() {
  return !!(
    fps.keys['KeyW'] || fps.keys['ArrowUp']    || fps.keys['KeyS'] || fps.keys['ArrowDown'] ||
    fps.keys['KeyA'] || fps.keys['ArrowLeft']  || fps.keys['KeyD'] || fps.keys['ArrowRight'] ||
    fps.keys['KeyQ'] || fps.keys['KeyE']       || fps.keys['ControlLeft']
  );
}
let dragMoved = false;
let lastMouseClient = {x: window.innerWidth / 2, y: window.innerHeight / 2};

// Tree view uses a locked-orientation pan/zoom camera instead of free 6DOF
// flight (see treeview.py): right-click-drag pans, scroll zooms, no
// pointer lock involved -- plain OS cursor throughout.
let treePanDragActive = false;

// rmbHeld tracks the *physical* right-mouse-button state, independently of
// cameraLookMode. requestPointerLock() is async -- the lock can take one or
// more frames (or fail entirely) to actually engage, so a fast click
// (mousedown+mouseup within that window, e.g. a spammed double right-click)
// can resolve before the lock does. cameraLookMode and the cursor state are
// therefore never set from the click itself: both are only ever set inside
// the pointerlockchange/pointerlockerror events below, from the two real
// sources of truth at that moment -- rmbHeld (is the button still physically
// down?) and document.pointerLockElement (did the lock actually engage?).
let rmbHeld = false;
let lockRequestPending = false;

function isCameraLooking() {
  return cameraLookMode !== null;
}

function updateLookCursor() {
  canvas.style.cursor = isCameraLooking() ? 'none' : 'default';
}

function requestCameraLock() {
  // Guard against firing overlapping requestPointerLock() calls (e.g. a
  // spammed double right-click): a second call while one is already in
  // flight would produce mismatched accept/deny races.
  if (lockRequestPending || document.pointerLockElement === canvas) return;

  // If a text field (e.g. the map search box) still has focus, the keydown
  // handler's "typing in an input" guard swallows WASD/Space, so free-look
  // would engage the mouse-look but movement would silently do nothing.
  // Right-click (or Space) requesting free look is an explicit "I want to
  // fly around now" gesture, so drop focus from whatever was selected --
  // e.preventDefault() on the triggering mousedown blocks the browser's own
  // focus-shift, so we do it explicitly instead of relying on that.
  const focused = document.activeElement;
  if (focused && focused !== document.body && typeof focused.blur === 'function') {
    focused.blur();
  }

  lockRequestPending = true;
  canvas.requestPointerLock();
}

function enterFreeLook() {
  cameraLookMode = 'toggle';
  dragMoved = false;
  clOverlay.style.display = 'none';
  requestCameraLock();
  updateLookCursor();
}

function exitFreeLook() {
  cameraLookMode = null;
  if (document.pointerLockElement === canvas) {
    document.exitPointerLock();
  }
  updateLookCursor();
}

// Single source of truth for "did the lock actually (dis)engage". Runs on
// every real transition, including ones neither mousedown nor mouseup
// caused directly (Escape, alt-tab, devtools, browser-forced release...).
document.addEventListener('pointerlockchange', () => {
  lockRequestPending = false;

  if (document.pointerLockElement === canvas) {
    // Lock just engaged. Reconcile with what's *actually* true right now
    // rather than whatever mousedown optimistically assumed: if the RMB
    // was already released before the lock caught up, back out of it
    // immediately instead of getting stuck holding a lock nothing wants.
    if (rmbHeld) {
      cameraLookMode = 'hold';
    } else if (cameraLookMode !== 'toggle') {
      document.exitPointerLock();
      cameraLookMode = null;
    }
  } else if (cameraLookMode !== null) {
    // Lock was released (by us or by the browser) -- always drop back to
    // the neutral state so the cursor/UI can never stay stuck looking.
    cameraLookMode = null;
  }
  updateLookCursor();
});

// A request can also be flat-out denied (e.g. the browser's own cooldown
// after rapid lock/unlock cycles) without ever firing pointerlockchange.
// Without this, a denied request left cameraLookMode/cursor stuck in the
// "looking" state forever since nothing ever told them it failed.
document.addEventListener('pointerlockerror', () => {
  lockRequestPending = false;
  cameraLookMode = null;
  updateLookCursor();
});

// While in fullscreen, the fullscreen-button setup below locks the Escape
// key (Keyboard Lock API) so a quick tap reaches us here instead of
// exiting fullscreen on its own. This handler is what actually closes
// toggle mode on that tap. Outside fullscreen (or on browsers without
// Keyboard Lock), the browser's own Escape-exits-pointer-lock behavior
// still fires and the pointerlockchange listener above already covers it,
// so calling exitFreeLook() here again is harmless (it no-ops if already
// unlocked).
window.addEventListener('keydown', e => {
  if (e.key === 'Escape' && cameraLookMode === 'toggle') {
    exitFreeLook();
  }
});

canvas.addEventListener('contextmenu', e => {
  e.preventDefault();
});

canvas.addEventListener('mousedown', e => {
  if (document.pointerLockElement === canvas) {
    // Locked: nothing under the hidden cursor is clickable, full stop.
    // What the mousedown *means* depends on which mode holds the lock:
    //  - toggle mode: left OR right click is its documented exit gesture.
    //  - hold mode: only releasing the right button (see mouseup) ends it;
    //    an incidental mousedown here has nothing to do and is swallowed.
    e.preventDefault();
    if (cameraLookMode === 'toggle') exitFreeLook();
    return;
  }

  if (e.button === 0) {
    // Normal visible-cursor left click: selection remains fully functional.
    dragMoved = false;
    return;
  }

  if (e.button === 2) {
    if (treeView.active) {
      // Tree view: the layout is flat and pre-framed front-on, so RMB pans
      // across it with the plain OS cursor instead of rotating a free-look
      // camera -- no pointer lock, orientation stays locked throughout.
      treePanDragActive = true;
      lastMouseClient = {x: e.clientX, y: e.clientY};
      canvas.style.cursor = 'grabbing';
      e.preventDefault();
      return;
    }

    // RMB hold starts camera look.
    // IMPORTANT: update the physical-button state BEFORE requesting Pointer Lock.
    // requestPointerLock() is asynchronous, so pointerlockchange may arrive
    // after mousedown (and potentially after a very fast mouseup).
    rmbHeld = true;
    cameraLookMode = 'hold';
    dragMoved = false;
    clOverlay.style.display = 'none';
    requestCameraLock();
    updateLookCursor();
    e.preventDefault();
  }
});

window.addEventListener('mouseup', e => {
  if (e.button === 2) {
    if (treePanDragActive) {
      treePanDragActive = false;
      canvas.style.cursor = 'default';
      return;
    }

    // Always clear the physical RMB state, even if the Pointer Lock has not
    // engaged yet. This is what makes rapid click/release sequences safe.
    rmbHeld = false;
    if (cameraLookMode === 'hold') {
      e.preventDefault();
      cameraLookMode = null;
      if (document.pointerLockElement === canvas) document.exitPointerLock();
      updateLookCursor();
    }
  }
});

window.addEventListener('blur', () => {
  // Losing window focus (alt-tab, devtools...) also releases Pointer Lock
  // on its own, which the pointerlockchange listener above already
  // handles -- this just covers the moment before that event fires.
  if (cameraLookMode === 'hold') {
    cameraLookMode = null;
  }
  if (treePanDragActive) {
    treePanDragActive = false;
    canvas.style.cursor = 'default';
  }
  updateLookCursor();
});

let _pendingMouseMoveEvent = null;  // rAF-throttled raycasting, see animate()

window.addEventListener('mousemove', e => {
  if (document.pointerLockElement === canvas) {
    // Locked (hold or toggle, doesn't matter which): read the raw device
    // delta. clientX/clientY freeze at the lock point while locked, so
    // they can't be used here -- movementX/Y is what removes the
    // "cursor stuck at the screen edge" ceiling entirely.
    const dx = e.movementX || 0;
    const dy = e.movementY || 0;
    if (Math.abs(dx) + Math.abs(dy) > 2) dragMoved = true;
    fps.yaw   -= dx * 0.003;
    fps.pitch -= dy * 0.003;
    fps.pitch  = Math.max(-Math.PI/2 + 0.01, Math.min(Math.PI/2 - 0.01, fps.pitch));
    camera.rotation.y = fps.yaw;
    camera.rotation.x = fps.pitch;
    clOverlay.style.display = 'none';
    return;
  }

  if (treePanDragActive) {
    const dx = e.clientX - lastMouseClient.x;
    const dy = e.clientY - lastMouseClient.y;
    lastMouseClient = {x: e.clientX, y: e.clientY};
    const worldPerPx = treePanWorldPerPixel();
    treePan(-dx * worldPerPx, dy * worldPerPx);
    return;
  }

  lastMouseClient = {x: e.clientX, y: e.clientY};
  // Raycasting against a large graph is too costly to run on every raw
  // mousemove -- that event can fire far more often than the screen
  // refreshes. Stash the event and let animate() consume at most one per
  // rendered frame; the native MouseEvent object stays valid for this
  // (no pooling like React's synthetic events).
  _pendingMouseMoveEvent = e;
});

// Capture phase: while locked, no click can ever select a node/edge --
// the lock guarantees no click reaches anything underneath it anyway, but
// this is a defensive backstop in case a click still slips through.
canvas.addEventListener('click', e => {
  if (isCameraLooking()) {
    e.preventDefault();
    e.stopImmediatePropagation();
  }
}, true);

canvas.addEventListener('wheel', e => {
  if (treeView.active) {
    treeZoom(e.deltaY > 0 ? 1.25 : 0.8);
    e.preventDefault();
    return;
  }
  fps.speed = Math.max(SPEED_MIN, Math.min(SPEED_MAX, fps.speed * (e.deltaY > 0 ? 1.25 : 0.8)));
  e.preventDefault();
}, {passive: false});

document.addEventListener('keydown', e => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable)) return;
  fps.keys[e.code] = true;

  if (e.code === 'Space') {
    // Always suppress the browser default (page scroll, or activating
    // whatever element currently has focus -- e.g. the fullscreen button,
    // which would otherwise toggle fullscreen off). Tree view just skips
    // the free-look toggle itself; Space is inert there, not unhandled.
    e.preventDefault();
    if (!treeView.active && !e.repeat && cameraLookMode !== 'hold') {
      // Toggle mode exits with either Space or RMB.
      if (cameraLookMode === 'toggle') exitFreeLook();
      else enterFreeLook();
    }
  }

  if (e.code === 'KeyF' && !e.repeat) {
    flyToSpawn();
  }
});

document.addEventListener('keyup', e => {
  fps.keys[e.code] = false;
});


let touchPrev = null;
canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 1) touchPrev = {x: e.touches[0].clientX, y: e.touches[0].clientY};
});
canvas.addEventListener('touchmove', e => {
  if (e.touches.length === 1 && touchPrev) {
    const dx = e.touches[0].clientX - touchPrev.x;
    const dy = e.touches[0].clientY - touchPrev.y;
    if (treeView.active) {
      const worldPerPx = treePanWorldPerPixel();
      treePan(-dx * worldPerPx, dy * worldPerPx);
    } else {
      fps.yaw   -= dx * 0.004;
      fps.pitch -= dy * 0.004;
      fps.pitch  = Math.max(-Math.PI/2 + 0.01, Math.min(Math.PI/2 - 0.01, fps.pitch));
      camera.rotation.y = fps.yaw;
      camera.rotation.x = fps.pitch;
    }
    touchPrev = {x: e.touches[0].clientX, y: e.touches[0].clientY};
  }
  e.preventDefault();
}, {passive: false});


const BASE_GEOMS = {
  sphere:      new THREE.SphereGeometry(1, 8, 8),
  octahedron:  new THREE.OctahedronGeometry(1, 0),
  icosahedron: new THREE.IcosahedronGeometry(1, 0),
  cone:        new THREE.ConeGeometry(0.8, 1.6, 6),
  tetrahedron: new THREE.TetrahedronGeometry(1, 0),
  diamond: (() => {
    const g = new THREE.OctahedronGeometry(1, 0);
    g.scale(0.72, 1.4, 0.72);
    return g;
  })(),
};

const instanceMap     = {};  // geomKey -> [nodeData, ...]
const instancedMeshes = {};  // geomKey -> InstancedMesh
const NODE_INSTANCE   = {};  // node.id -> k/idx, for recoloring a single node

const _dummy = new THREE.Object3D();

const GLOBAL_NODE_SATURATION_BOOST = 0.9;

const GLOBAL_NODE_DIM_LEVEL        = 0.35;  // 0 = unchanged, 1 = black
const GLOBAL_NODE_DIM_DESATURATION = 0.35;  // 0 = unchanged, 1 = grayscale

const _hslBase     = { h: 0, s: 0, l: 0 };
const _boostedBase = new THREE.Color();
const _dimHSL       = { h: 0, s: 0, l: 0 };

function _baseColorFor(n) {
  _boostedBase.set(n.color);
  if (GLOBAL_NODE_SATURATION_BOOST > 0) {
    _boostedBase.getHSL(_hslBase);
    const s = _hslBase.s + (1 - _hslBase.s) * GLOBAL_NODE_SATURATION_BOOST;
    _boostedBase.setHSL(_hslBase.h, s, _hslBase.l);
  }
  if (!treeView.active && (GLOBAL_NODE_DIM_LEVEL > 0 || GLOBAL_NODE_DIM_DESATURATION > 0)) {
    _boostedBase.getHSL(_dimHSL);
    const s2 = _dimHSL.s * (1 - GLOBAL_NODE_DIM_DESATURATION);
    const l2 = _dimHSL.l * (1 - GLOBAL_NODE_DIM_LEVEL);
    _boostedBase.setHSL(_dimHSL.h, s2, l2);
  }
  return _boostedBase;
}

function _geomKey(n) {
  return BASE_GEOMS[n.geometry] ? n.geometry : 'sphere';
}

const GLOBAL_NODE_EMISSIVE_INTENSITY = 0.1;

function _patchNodeMaterialEmissive(mat) {
  mat.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <color_fragment>',
      '#include <color_fragment>\n' +
      '#ifdef USE_COLOR\n' +
      '  totalEmissiveRadiance += diffuseColor.rgb * ' + GLOBAL_NODE_EMISSIVE_INTENSITY + ';\n' +
      '#endif\n'
    );
  };
}

const GLOBAL_NODE_MIN_SCREEN_PX = 1.0;

function updateNodeMinScreenSize() {
  if (GLOBAL_NODE_MIN_SCREEN_PX <= 0) return;
  if (treeView.active) return;

  const viewportH   = window.innerHeight - 36;
  const tanHalfFov  = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  const camX = camera.position.x, camY = camera.position.y, camZ = camera.position.z;

  Object.keys(instanceMap).forEach(k => {
    const im  = instancedMeshes[k];
    const arr = instanceMap[k];
    if (!im || !arr || !arr.length) return;

    for (let idx = 0; idx < arr.length; idx++) {
      const n    = arr[idx];
      const base = baseScaleFor(n, k);
      if (base <= 0) continue;  // node collapsed (e.g. leaving tree view)

      const dx = n.x - camX, dy = n.y - camY, dz = n.z - camZ;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;

      const minWorldRadius = (GLOBAL_NODE_MIN_SCREEN_PX * dist * tanHalfFov * 2) / viewportH;
      const finalScale = Math.max(base, minWorldRadius);

      _dummy.position.set(n.x, n.y, n.z);
      _dummy.scale.setScalar(finalScale);
      _dummy.updateMatrix();
      im.setMatrixAt(idx, _dummy.matrix);
    }
    im.instanceMatrix.needsUpdate = true;
  });
}

function buildInstancedMeshes(nodes) {
  const counts = {};
  nodes.forEach(n => {
    const k = _geomKey(n);
    counts[k] = (counts[k] || 0) + 1;
  });

  Object.keys(counts).forEach(k => {
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.1 });
    _patchNodeMaterialEmissive(mat);
    const im  = new THREE.InstancedMesh(BASE_GEOMS[k], mat, counts[k]);
    im.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    im.userData.geomKey = k;
    instancedMeshes[k] = im;
    instanceMap[k]     = [];
    scene.add(im);
  });

  nodes.forEach(n => {
    const k   = _geomKey(n);
    const im  = instancedMeshes[k];
    const idx = instanceMap[k].length;
    instanceMap[k].push(n);
    NODE_INSTANCE[n.id] = { k, idx };

    const scale = baseScaleFor(n, k);

    _dummy.position.set(n.x, n.y, n.z);
    _dummy.scale.setScalar(scale);
    _dummy.updateMatrix();
    im.setMatrixAt(idx, _dummy.matrix);
    im.setColorAt(idx, _baseColorFor(n));
    n._curHighlightFactor = 1.0;

  });

  Object.values(instancedMeshes).forEach(im => {
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
  });
}

function removeInstancedMeshes() {
  Object.values(instancedMeshes).forEach(im => scene.remove(im));
  Object.keys(instancedMeshes).forEach(k => { delete instancedMeshes[k]; delete instanceMap[k]; });
  Object.keys(NODE_INSTANCE).forEach(id => delete NODE_INSTANCE[id]);
}

const PICKABLE_GROUPS = new Set(['nova', 'adept_spoke']);  // controls line style only (solid vs dashed) -- see buildEdges()
const pickableEdges   = {};  // group -> obj/lks/segToEdge, ALL groups (raycast targets, click + hover)
const edgeColorBuffers = {};

function _fillFlatColor(colorsArr, c) {
  for (let i = 0; i < colorsArr.length; i += 3) {
    colorsArr[i] = c.r; colorsArr[i + 1] = c.g; colorsArr[i + 2] = c.b;
  }
}

function buildEdges(links) {
  const byGroup = {};
  links.forEach(lk => {
    if (!byGroup[lk.group]) byGroup[lk.group] = [];
    byGroup[lk.group].push(lk);
  });

  const novaParentOf = {};  // target -> source
  const novaChildOf  = {};  // source -> first child
  (byGroup['nova'] || []).forEach(lk => {
    novaParentOf[lk.target] = lk.source;
    if (!(lk.source in novaChildOf)) novaChildOf[lk.source] = lk.target;
  });
  const NOVA_SAMPLES = NOVA_SPLINE.samples;
  const NOVA_TENSION = NOVA_SPLINE.tension;

  const resW = window.innerWidth - 280, resH = window.innerHeight - 36;

  Object.entries(byGroup).forEach(([grp, lks]) => {
    const verts = [];
    const segToEdge = [];  // segment index -> index in lks

    if (grp === 'nova' && NOVA_SPLINE.enabled) {
      lks.forEach((lk, edgeIdx) => {
        const p = NM[lk.source];
        const c = NM[lk.target];
        if (!p || !c) return;

        const gpId = novaParentOf[lk.source];
        const gcId = novaChildOf[lk.target];
        const gp = (gpId && NM[gpId]) ? NM[gpId] : p;
        const gc = (gcId && NM[gcId]) ? NM[gcId] : c;

        const curve = new THREE.CatmullRomCurve3(
          [
            new THREE.Vector3(gp.x, gp.y, gp.z),
            new THREE.Vector3(p.x, p.y, p.z),
            new THREE.Vector3(c.x, c.y, c.z),
            new THREE.Vector3(gc.x, gc.y, gc.z),
          ],
          false,
          'catmullrom',
          NOVA_TENSION  // config.yaml render.nova_spline.tension
        );

        const pts = [];
        for (let i = 0; i <= NOVA_SAMPLES; i++) {
          const u = 1 / 3 + (i / NOVA_SAMPLES) * (1 / 3);
          pts.push(curve.getPoint(u));
        }

        for (let i = 0; i < pts.length - 1; i++) {
          verts.push(pts[i].x, pts[i].y, pts[i].z,
                     pts[i + 1].x, pts[i + 1].y, pts[i + 1].z);
          segToEdge.push(edgeIdx);
        }
      });
    } else {
      lks.forEach((lk, edgeIdx) => {
        const s = NM[lk.source];
        const t = NM[lk.target];
        if (!s || !t) return;
        verts.push(s.x, s.y, s.z,
                   t.x, t.y, t.z);
        segToEdge.push(edgeIdx);
      });
    }

    if (verts.length === 0) return;

    const col = new THREE.Color(EC[grp] || '#888');
    const opacity = lks[0].opacity || 0.6;

    const edgeKeyToSegs = new Map();
    segToEdge.forEach((edgeIdx, segIdx) => {
      const lk = lks[edgeIdx];
      const key = lk.source + '|' + lk.target;
      if (!edgeKeyToSegs.has(key)) edgeKeyToSegs.set(key, []);
      edgeKeyToSegs.get(key).push(segIdx);
    });

    const segMidpoints = new Float32Array((verts.length / 6) * 3);
    for (let segIdx = 0; segIdx < verts.length / 6; segIdx++) {
      const o = segIdx * 6;
      segMidpoints[segIdx * 3]     = (verts[o] + verts[o + 3]) / 2;
      segMidpoints[segIdx * 3 + 1] = (verts[o + 1] + verts[o + 4]) / 2;
      segMidpoints[segIdx * 3 + 2] = (verts[o + 2] + verts[o + 5]) / 2;
    }

    if (PICKABLE_GROUPS.has(grp)) {
      const geo = new LineSegmentsGeometry();
      geo.setPositions(verts);
      const colorsArr = new Float32Array(verts.length);
      _fillFlatColor(colorsArr, col);
      geo.setColors(colorsArr);
      const mat = new LineMaterial({
        color: 0xffffff,
        vertexColors: true,
        transparent: true,
        depthWrite: false,
        opacity,
        linewidth: EW[grp] || 1.4,
        resolution: new THREE.Vector2(resW, resH),
      });
      const obj = new LineSegments2(geo, mat);
      obj.userData.group = grp;
      obj.visible = vis.has(grp);
      scene.add(obj);
      fatLineMaterials.push(mat);

      pickableEdges[grp]   = { obj, lks, segToEdge };
      edgeColorBuffers[grp] = { geo, colorsArr, baseColor: col, edgeKeyToSegs, segMidpoints };
      if (!groupLines[grp]) groupLines[grp] = [];
      groupLines[grp].push(obj);
    } else {
      const geo = new LineSegmentsGeometry();
      geo.setPositions(verts);
      const colorsArr = new Float32Array(verts.length);
      _fillFlatColor(colorsArr, col);
      geo.setColors(colorsArr);
      const mat = new LineMaterial({
        color: 0xffffff,
        vertexColors: true,
        transparent: true,
        depthWrite: false,
        opacity,
        linewidth: EW[grp] || 1.5,
        dashed: true,
        dashSize: DASH_SIZE,
        gapSize: GAP_SIZE,
        resolution: new THREE.Vector2(resW, resH),
      });
      const obj = new LineSegments2(geo, mat);
      obj.computeLineDistances();  // required for dash pattern to render
      obj.userData.group = grp;
      obj.visible = vis.has(grp);
      scene.add(obj);
      fatLineMaterials.push(mat);
      pickableEdges[grp]   = { obj, lks, segToEdge };
      edgeColorBuffers[grp] = { geo, colorsArr, baseColor: col, edgeKeyToSegs, segMidpoints };
      if (!groupLines[grp]) groupLines[grp] = [];
      groupLines[grp].push(obj);
    }
  });
}


const HOVER_EDGE_LIGHTEN = 0.45;  // fraction of remaining distance to L=1
const _hiEdgeColor = new THREE.Color();
const _hsl = { h: 0, s: 0, l: 0 };

function _lightenColor(c, amount) {
  c.getHSL(_hsl);
  const l = _hsl.l + (1 - _hsl.l) * amount;
  return _hiEdgeColor.setHSL(_hsl.h, _hsl.s, l);
}

const GLOBAL_EDGE_FADE_START       = 300;
const GLOBAL_EDGE_FADE_END         = 35000;
const GLOBAL_EDGE_FADE_MIN_OPACITY = 0;
const _edgeFadeBg = new THREE.Color(0x080810);  // matches scene.background
const _edgeFadeColor = new THREE.Color();  // shared scratch, never allocated per frame

const EDGE_FADE_UPDATE_INTERVAL_MS = 100;
const EDGE_FADE_MIN_CAM_MOVE_SQ = 25;  // 5 world units, squared
let _lastEdgeFadeTime = 0;
const _lastEdgeFadeCamPos = new THREE.Vector3(Infinity, Infinity, Infinity);

function _fadeMixForDist(dist) {
  if (!edgeFadeEnabled) return 0;
  const span = Math.max(1, GLOBAL_EDGE_FADE_END - GLOBAL_EDGE_FADE_START);
  const tLinear = Math.min(1, Math.max(0, (dist - GLOBAL_EDGE_FADE_START) / span));
  const t = tLinear * tLinear * (3 - 2 * tLinear);  // smoothstep
  return t * (1 - GLOBAL_EDGE_FADE_MIN_OPACITY);
}

// geo.setColors() (three.js LineSegmentsGeometry) does NOT update an existing
// GPU buffer -- it allocates a BRAND NEW InstancedInterleavedBuffer every
// single call and swaps it in, never disposing the one it replaces. Since
// colorsArr IS the same Float32Array backing that buffer (no copy happens
// for Float32Array input), we can write into it in place and just bump the
// buffer's version to trigger a re-upload of the SAME buffer -- no
// allocation, no leak. Used by the distance-fade update below.
function _pushColorUpdate(geo) {
  const attr = geo.attributes.instanceColorStart;
  if (attr && attr.data) attr.data.needsUpdate = true;
}

function updateEdgeDistanceFade(now) {
  if (treeView.active) return;  // tree view renders its own edges
  if (now - _lastEdgeFadeTime < EDGE_FADE_UPDATE_INTERVAL_MS) return;
  if (camera.position.distanceToSquared(_lastEdgeFadeCamPos) < EDGE_FADE_MIN_CAM_MOVE_SQ) return;
  _lastEdgeFadeTime = now;
  _lastEdgeFadeCamPos.copy(camera.position);

  const camX = camera.position.x, camY = camera.position.y, camZ = camera.position.z;

  Object.keys(edgeColorBuffers).forEach(grp => {
    if (!vis.has(grp)) return;  // hidden group: nobody sees it, don't touch its buffer
    const pe = edgeColorBuffers[grp];
    const { geo, colorsArr, baseColor, segMidpoints } = pe;
    if (!segMidpoints || !segMidpoints.length) return;

    const segCount = segMidpoints.length / 3;
    for (let segIdx = 0; segIdx < segCount; segIdx++) {
      const mo = segIdx * 3;
      const dx = segMidpoints[mo] - camX, dy = segMidpoints[mo + 1] - camY, dz = segMidpoints[mo + 2] - camZ;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      _edgeFadeColor.copy(baseColor).lerp(_edgeFadeBg, _fadeMixForDist(dist));
      const o = segIdx * 6;
      colorsArr[o]     = _edgeFadeColor.r; colorsArr[o + 1] = _edgeFadeColor.g; colorsArr[o + 2] = _edgeFadeColor.b;
      colorsArr[o + 3] = _edgeFadeColor.r; colorsArr[o + 4] = _edgeFadeColor.g; colorsArr[o + 5] = _edgeFadeColor.b;
    }

    _pushColorUpdate(geo);
  });
}

const skeletonHL = {};  // type -> LineSegments2, one layer per bridge type
SKELETON_TYPES.forEach(type => {
  const isGraft     = type === 'adept_graft';
  const baseOpacity = EO[type] ?? 0.7;
  const baseWidth   = EW[type] || 1.5;
  const baseColor = new THREE.Color(EC[type] || '#ffffff');
  const mat = new LineMaterial({
    color: isGraft ? _lightenColor(baseColor, HOVER_EDGE_LIGHTEN).clone() : baseColor,
    transparent: true,
    depthWrite: false,
    opacity:   baseOpacity,
    linewidth: baseWidth,
    dashed: true,
    dashSize: DASH_SIZE,
    gapSize: GAP_SIZE,
    resolution: new THREE.Vector2(window.innerWidth - 280, window.innerHeight - 36),
  });
  const geo = new LineSegmentsGeometry();
  geo.setPositions([0, 0, 0, 0, 0, 0]);  // placeholder, replaced on use
  const obj = new LineSegments2(geo, mat);
  obj.visible = false;
  obj.renderOrder = 999;  // draw on top
  scene.add(obj);
  fatLineMaterials.push(mat);
  skeletonHL[type] = obj;
});

function hideSkeletonHighlight() {
  SKELETON_TYPES.forEach(t => { skeletonHL[t].visible = false; });
}
let skeletonDashOffset = 0;  // animated in animate() for a drawing effect

function showSkeletonHighlight(edges) {
  const byType = {};
  GROUPS.forEach(t => byType[t] = []);
  edges.forEach(e => { if (byType[e.type]) byType[e.type].push(e); });

  SKELETON_TYPES.forEach(type => {
    if (vis.has(type)) {
      // Already rendered as part of the normal graph -- nothing more to do,
      // no color/brightness change on hover.
      skeletonHL[type].visible = false;
      return;
    }
    const obj = skeletonHL[type];
    const verts = [];
    byType[type].forEach(e => {
      const s = NM[e.source], t = NM[e.target];
      if (s && t) verts.push(s.x, s.y, s.z, t.x, t.y, t.z);
    });
    if (!verts.length) { obj.visible = false; return; }
    obj.geometry.dispose();
    obj.geometry = new LineSegmentsGeometry();
    obj.geometry.setPositions(verts);
    obj.computeLineDistances();
    obj.userData.skeletonType = type;
    obj.userData.skeletonPairs = byType[type].map(e => [e.source, e.target]);
    obj.visible = true;
  });
}


__TREEVIEW_JS__

function populateScene() {
  removeInstancedMeshes();
  buildInstancedMeshes(RAW.nodes);
  rebuildNodeInstanceIndex();
}

populateScene();
buildEdges(RAW.links);

// Scene is populated and ready to render -- dismiss the boot loading overlay.
// Without this, #app-loading (opaque, full-viewport, z-index:900) stays on
// top of the canvas AND swallows all clicks forever, which looks exactly
// like "nothing works / nothing displays" even though the scene underneath
// is rendering and interactive fine.
document.getElementById('app-loading')?.classList.add('hidden');

const raycaster  = new THREE.Raycaster();
raycaster.camera = camera;  // required by LineSegments2 in screen-space mode
raycaster.params.Line2 = { threshold: 6 };  // hit tolerance in px
const mouseNDC   = new THREE.Vector2();
const clOverlay  = document.getElementById('cluster-overlay');
let hoveredNode  = null;
let hoveredLabelEntry = null;  // Nova/Adept title currently under the mouse, for the hover glow
let htimer       = null;

let pinnedIds      = null;  // ids pinned by click, or null
let lastSkeletonKey = undefined;
function idsKey(ids) { return ids ? [...ids].sort().join(',') : null; }
function revealSkeleton(ids) {
  const key = idsKey(ids);
  if (key === lastSkeletonKey) return;  // already up to date
  lastSkeletonKey = key;
  if (!ids || !ids.length) {
    hideSkeletonHighlight();
    showHoverGlow(null);  // fade out any in-progress hover glow
    return;
  }
  const { nodes, edges } = computeSkeleton(ids);
  if (!edges.length) { hideSkeletonHighlight(); } else { showSkeletonHighlight(edges); }
  showHoverGlow(nodes);
}


const clusterCentroids = (function() {
  const acc = {}, cnt = {};
  RAW.nodes.forEach(n => {
    if (!n.cluster_id) return;
    if (!acc[n.cluster_id]) { acc[n.cluster_id] = {x:0,y:0,z:0}; cnt[n.cluster_id] = 0; }
    acc[n.cluster_id].x += n.x; acc[n.cluster_id].y += n.y; acc[n.cluster_id].z += n.z;
    cnt[n.cluster_id]++;
  });
  const result = {};
  Object.keys(acc).forEach(cid => {
    result[cid] = {
      x: acc[cid].x / cnt[cid],
      y: acc[cid].y / cnt[cid],
      z: acc[cid].z / cnt[cid],
      label: null, color: null, count: cnt[cid], radius: 0
    };
  });
  RAW.nodes.forEach(n => {
    if (!n.cluster_id || !result[n.cluster_id]) return;
    const c = result[n.cluster_id];
    if (n.cluster_label) { c.label = n.cluster_label; c.color = n.color; }
    const dx = n.x - c.x, dy = n.y - c.y, dz = n.z - c.z;
    const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
    if (d > c.radius) c.radius = d;
  });

  // Never report a cluster smaller than CLUSTER_MIN_RADIUS (see above).
  // This is the single source clusterCentroids[cid].radius flows out to —
  // flyToCluster()'s camera distance, the aura/halo sphere, label
  // occlusion/anchor fade, proximity auto-select — so fixing it here fixes
  // all of them at once, with zero changes to any of that downstream
  // logic. A cluster whose real radius already exceeds this floor (the
  // normal/large case) is untouched.
  Object.values(result).forEach(c => {
    if (c.radius < CLUSTER_MIN_RADIUS) c.radius = CLUSTER_MIN_RADIUS;
  });

  return result;
})();

const CLUSTER_AURA_MARGIN         = 0.90;
const CLUSTER_AURA_GLOBAL_OPACITY = 0.08;
const CLUSTER_AURA_FRESNEL_POWER  = 3.4;
const CLUSTER_AURA_FRESNEL_BASE   = 0.0;
const CLUSTER_AURA_ARC_WIDTH      = 0.50;
const CLUSTER_AURA_ARC_FADE       = 0.75;
const CLUSTER_AURA_BOTTOM_FACTOR  = 0.55;
const CLUSTER_AURA_VERT_SPREAD    = 1.4;  // vertical fade width
const CLUSTER_AURA_NEAR_FACTOR    = 1.6;
const CLUSTER_AURA_FAR_FACTOR     = 5.0;
const CLUSTER_AURA_MIN_STRENGTH   = 0.08;
const CLUSTER_AURA_DEFAULT_TINT   = '#DCE1EF';

const CLUSTER_AURA_OCCLUSION_MAX_FADE = 0.82;  // 82% max, never fully hidden
const CLUSTER_AURA_OCCLUSION_RADIUS_PAD = 1.08; // tolerance for apparent overlap
const CLUSTER_AURA_OCCLUSION_SOFTNESS = 0.34;
const CLUSTER_AURA_OCCLUSION_MIN_DEPTH_GAP = 0.01;
const CLUSTER_AURA_OCCLUSION_UPDATE_MS = 45;
const CLUSTER_AURA_SMOOTH_SPEED = 0.10;


const _clusterAuraVertexShader = `
varying vec3 vNormal;
varying vec3 vViewPosition;
void main() {
  vNormal = normalize(normalMatrix * normal);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const _clusterAuraFragmentShader = `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uFresnelPower;
uniform float uArcWidth;
uniform float uArcFade;
uniform float uBottomFactor;
uniform float uVertSpread;
varying vec3 vNormal;
varying vec3 vViewPosition;

float smootherstep(float e0, float e1, float x) {
  float t = clamp((x - e0) / max(e1 - e0, 0.0001), 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float ditherNoise(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec3 n = normalize(vNormal);
  vec3 viewDir = normalize(vViewPosition);

  float rim = pow(1.0 - abs(dot(n, viewDir)), uFresnelPower);

  float vert = max(n.y, -n.y * uBottomFactor);
  float vertMask = smootherstep(-0.15, uVertSpread, vert);

  float sideMask = 1.0 - smootherstep(uArcWidth, uArcWidth + uArcFade, abs(n.x));

  float arcMask = vertMask * sideMask;

  float alpha = rim * arcMask * uOpacity;
  alpha += (ditherNoise(gl_FragCoord.xy) - 0.5) * (1.0 / 255.0);
  alpha = max(alpha, 0.0);

  if (alpha < 0.0015) discard;
  gl_FragColor = vec4(uColor, alpha);
}
`;

const _clusterAuraGeom = new THREE.SphereGeometry(1, 32, 20);

const clusterAurasGroup = new THREE.Group();
scene.add(clusterAurasGroup);
const clusterAuras = [];

(function buildClusterAuras() {
  Object.values(clusterCentroids).forEach(c => {
    if (!c.radius) return;
    const tint = new THREE.Color(CLUSTER_AURA_DEFAULT_TINT);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uColor:        { value: tint },
        uOpacity:      { value: CLUSTER_AURA_GLOBAL_OPACITY },
        uFresnelPower: { value: CLUSTER_AURA_FRESNEL_POWER },
        uArcWidth:     { value: CLUSTER_AURA_ARC_WIDTH },
        uArcFade:      { value: CLUSTER_AURA_ARC_FADE },
        uBottomFactor: { value: CLUSTER_AURA_BOTTOM_FACTOR },
        uVertSpread:   { value: CLUSTER_AURA_VERT_SPREAD },
      },
      vertexShader: _clusterAuraVertexShader,
      fragmentShader: _clusterAuraFragmentShader,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
    });
    const mesh = new THREE.Mesh(_clusterAuraGeom, material);
    mesh.position.set(c.x, c.y, c.z);
    mesh.scale.setScalar(c.radius * CLUSTER_AURA_MARGIN);
    mesh.renderOrder = 500;
    clusterAurasGroup.add(mesh);
    clusterAuras.push({
      material,
      cx: c.x, cy: c.y, cz: c.z,
      radius: c.radius,
      targetOpacity: CLUSTER_AURA_GLOBAL_OPACITY,
      currentOpacity: CLUSTER_AURA_GLOBAL_OPACITY,
    });
  });
  console.log('[ClusterAuras]', clusterAuras.length, 'fragmented aura(s)');
})();

let _lastAuraFadeTime = 0;
const _lastAuraFadeCamPos = new THREE.Vector3(Infinity, Infinity, Infinity);
const _auraProj = new THREE.Vector3();
const _auraCam = new THREE.Vector3();

function _smoothstep01(x) {
  x = Math.max(0, Math.min(1, x));
  return x * x * (3 - 2 * x);
}

function _clusterScreenRadiusPx(depth, radius, tanHalfFov, viewportH) {
  if (depth <= camera.near || depth >= camera.far) return 0;
  return (radius * CLUSTER_AURA_MARGIN / (depth * tanHalfFov)) * (viewportH * 0.5);
}

function _clusterAuraOcclusionFactor(target, allAuras) {
  if (!target.scrValid || target.scrRadius <= 0) return 0;

  let strongest = 0;
  for (const blocker of allAuras) {
    if (blocker === target) continue;
    if (!blocker.scrValid || blocker.scrRadius <= 0) continue;

    if (blocker.scrDepth >= target.scrDepth - Math.max(CLUSTER_AURA_OCCLUSION_MIN_DEPTH_GAP, target.radius * 0.02)) continue;

    const dx = blocker.scrX - target.scrX;
    const dy = blocker.scrY - target.scrY;
    const overlap = (blocker.scrRadius + target.scrRadius * CLUSTER_AURA_OCCLUSION_RADIUS_PAD) - Math.hypot(dx, dy);
    if (overlap <= 0) continue;

    const normalized = overlap / Math.max(1, Math.min(blocker.scrRadius, target.scrRadius));
    const coverage = _smoothstep01(normalized / Math.max(0.001, CLUSTER_AURA_OCCLUSION_SOFTNESS));

    const depthRatio = Math.max(0, Math.min(1, (target.scrDepth - blocker.scrDepth) / Math.max(target.scrDepth, 1)));
    const depthWeight = 0.55 + 0.45 * _smoothstep01(depthRatio * 6.0);
    strongest = Math.max(strongest, coverage * depthWeight);
  }
  return Math.min(1, strongest);
}

function updateClusterAuraFade(now, dt) {
  if (treeView.active) return;

  const camMoved = camera.position.distanceToSquared(_lastAuraFadeCamPos) >= EDGE_FADE_MIN_CAM_MOVE_SQ;
  const shouldRecompute = camMoved && (now - _lastAuraFadeTime >= CLUSTER_AURA_OCCLUSION_UPDATE_MS);
  const viewportW = Math.max(1, canvas.clientWidth || window.innerWidth - 280);
  const viewportH = Math.max(1, canvas.clientHeight || window.innerHeight - 36);

  if (shouldRecompute) {
    _lastAuraFadeTime = now;
    _lastAuraFadeCamPos.copy(camera.position);

    const camX = camera.position.x, camY = camera.position.y, camZ = camera.position.z;
    const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));

    clusterAuras.forEach(a => {
      _auraCam.set(a.cx, a.cy, a.cz).applyMatrix4(camera.matrixWorldInverse);
      const depth = -_auraCam.z;
      _auraProj.set(a.cx, a.cy, a.cz).project(camera);
      a.scrDepth = depth;
      a.scrX = (_auraProj.x * 0.5 + 0.5) * viewportW;
      a.scrY = (-_auraProj.y * 0.5 + 0.5) * viewportH;
      a.scrValid = depth > camera.near && depth < camera.far && _auraProj.z >= -1 && _auraProj.z <= 1;
      a.scrRadius = a.scrValid ? _clusterScreenRadiusPx(depth, a.radius, tanHalfFov, viewportH) : 0;
    });

    clusterAuras.forEach(a => {
      const dx = camX - a.cx, dy = camY - a.cy, dz = camZ - a.cz;
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      const near = a.radius * CLUSTER_AURA_NEAR_FACTOR;
      const far  = a.radius * CLUSTER_AURA_FAR_FACTOR;
      const t = Math.min(1, Math.max(0, (dist - near) / Math.max(1, far - near)));
      const distanceStrength = CLUSTER_AURA_MIN_STRENGTH + (1 - CLUSTER_AURA_MIN_STRENGTH) * t;

      const occ = _clusterAuraOcclusionFactor(a, clusterAuras);

      const occlusionStrength = 1 - CLUSTER_AURA_OCCLUSION_MAX_FADE * occ;
      a.targetOpacity = CLUSTER_AURA_GLOBAL_OPACITY * distanceStrength * occlusionStrength;
    });
  }

  const smooth = 1 - Math.pow(1 - CLUSTER_AURA_SMOOTH_SPEED, Math.max(1, (dt || 1/60) * 60));
  clusterAuras.forEach(a => {
    a.currentOpacity += (a.targetOpacity - a.currentOpacity) * smooth;
    a.material.uniforms.uOpacity.value = a.currentOpacity;
  });
}

const labelObjs = [];
const pickableLabelMeshes = [];  // Nova/Adept/cluster title meshes -> raycast targets so CLICKING a title
                                  // behaves like clicking its node (Nova/Adept) or like the Map-panel entry (cluster).
                                  // Hover must NOT behave like hovering the node (no node info panel) — see fromLabel in pickAt()/onMouseMove().
const _labelNudgeRight = new THREE.Vector3();
const _labelNudgeUp = new THREE.Vector3();
const LABEL_PREFIX = { cluster: '▣ ', nova: '✦ ', adept: '⬢ ' };
// Single source of truth for nova/adept title colors: read by initLabels()
// below (the actual 3D titles) AND by the sidebar legend that documents them,
// so the two can never drift apart.
const TITLE_TIER_COLOR = { nova: '#9859c9', adept: '#ac7e46' };

function makeClusterLabelBackdrop(textWidth, fontSize, tintHex) {
  const padX = fontSize * LABEL_BACKDROP_PAD_X_RATIO, padY = fontSize * 0.5;
  const worldW = Math.max(fontSize * 1.4, textWidth) + padX * 2;
  const worldH = fontSize * 1.6 + padY * 2;

  const cw = 512;
  const ch = Math.max(64, Math.round(cw * (worldH / worldW)));
  const canvas = document.createElement('canvas');
  canvas.width = cw; canvas.height = ch;
  const ctx = canvas.getContext('2d');
  const r = Math.min(ch * 0.32, 34);
  ctx.beginPath();
  ctx.moveTo(r, 0);
  ctx.arcTo(cw, 0, cw, ch, r);
  ctx.arcTo(cw, ch, 0, ch, r);
  ctx.arcTo(0, ch, 0, 0, r);
  ctx.arcTo(0, 0, cw, 0, r);
  ctx.closePath();
  ctx.fillStyle = 'rgba(5,7,16,0.7)';
  ctx.fill();
  ctx.lineWidth = Math.max(2, ch * 0.035);
  ctx.strokeStyle = tintHex || '#88aaff';
  ctx.globalAlpha = 0.6;
  ctx.stroke();

  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(worldW, worldH, 1);
  spr.userData.baseW = worldW;
  spr.userData.baseH = worldH;
  spr.renderOrder = 997;  // just below text layer
  return spr;
}

function makeLabel(text, tier, x, y, z, color, far, tint, priority, fontSizeOverride, radius, clusterId, nodeId) {
  const t = new TroikaText();
  t.text = LABEL_PREFIX[tier] + text;
  t.fontSize = fontSizeOverride || LABEL_FONT_SIZE[tier];
  t.sdfGlyphSize = 128;
  t.color = color;
  t.anchorX = 'center';
  t.anchorY = 'middle';
  const isCluster = tier === 'cluster';
  t.outlineWidth = isCluster ? '16%' : '7%';
  t.outlineColor = '#000000';
  const baseOutlineOpacity = isCluster ? 0.9 : 0.7;
  t.outlineOpacity = 0;
  const posY = y + LABEL_OFFSET[tier];
  t.position.set(x, posY, z);
  t.renderOrder = 998;
  t.visible = false;
  t.fillOpacity = 0;
  if (nodeId != null) t.userData.pickNodeId = nodeId;
  scene.add(t);

  const entry = {
    obj: t, backdrop: null, tier, far, baseFont: t.fontSize, baseOutlineOpacity,
    hoverT: 0,
    priority: priority || 0, radius: radius || 0,
    curOpacity: 0, wantShow: false,
    clusterId: clusterId != null ? String(clusterId) : undefined,
    nodeId: nodeId != null ? nodeId : undefined,
    anchorPos: t.position.clone(),
    nudge: { x: 0, y: 0 },  // current on-screen offset, eased per frame
    nudgeTarget: { x: 0, y: 0 },  // target on-screen offset, recomputed each frame
  };
  t.userData.labelEntry = entry;
  labelObjs.push(entry);
  if (nodeId != null || tier === 'cluster') pickableLabelMeshes.push(t);

  t.sync(() => {
    // Measured once, from the real rendered text bounds — used by the (very
    // light) Nova/Adept overlap check below so it never has to guess a width
    // from character count.
    const bounds = t.textRenderInfo && t.textRenderInfo.blockBounds;
    const textWidth = bounds ? bounds[2] - bounds[0] : t.text.length * t.fontSize * 0.62;
    entry.textWidthWorld = textWidth;
    if (!isCluster) return;
    const backdrop = makeClusterLabelBackdrop(textWidth, t.fontSize, tint);
    backdrop.position.copy(t.position);
    backdrop.visible = false;
    backdrop.material.opacity = 0;
    scene.add(backdrop);
    entry.backdrop = backdrop;
  });
}

function initLabels() {
  Object.entries(clusterCentroids).forEach(([cid, c]) => {
    if (!c.label) return;
    makeLabel(c.label, 'cluster', c.x, c.y, c.z, '#dcecff', LABEL_DISTANCE.cluster, c.color, c.count, LABEL_FONT_SIZE.cluster, c.radius, cid);
  });
  RAW.nodes.forEach(n => {
    if (n.nova_role !== 'pioneer' || !n.subtopic_label) return;
    makeLabel(n.subtopic_label, 'nova', n.x, n.y, n.z, TITLE_TIER_COLOR.nova, LABEL_DISTANCE.nova, undefined, undefined, undefined, undefined, n.cluster_id, n.id);
  });
  RAW.nodes.forEach(n => {
    if (n.adept_role !== 'hub' || !n.adept_pool_label) return;
    makeLabel(n.adept_pool_label, 'adept', n.x, n.y, n.z, TITLE_TIER_COLOR.adept, LABEL_DISTANCE.adept, undefined, undefined, undefined, undefined, n.cluster_id, n.id);
  });
  console.log(
    '[Labels]',
    labelObjs.filter(l => l.tier === 'cluster').length, 'cluster —',
    labelObjs.filter(l => l.tier === 'nova').length,    'nova —',
    labelObjs.filter(l => l.tier === 'adept').length,   'adept'
  );
}
let labelsInitialized = false;
function tryInitLabels() {
  if (labelsInitialized || !TroikaText) return;
  labelsInitialized = true;
  initLabels();
}
window.__initLabels = tryInitLabels;
tryInitLabels();  // in case the CDN already responded

function updateLabelScale(viewportH) {
  if (treeView.active) return;  // tree view manages its own scales
  const camPos = camera.position;
  const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  labelObjs.forEach(entry => {
    const { obj, backdrop, tier, baseFont, anchorPos } = entry;
    const fixedPx = LABEL_SCREEN_PX[tier] || 0;
    const d = camPos.distanceTo(anchorPos);
    const worldPerPx = (Math.max(1e-6, d) * tanHalfFov * 2) / viewportH;  // epsilon guards d=0 only

    const targetPx = fixedPx > 0 ? fixedPx : (baseFont / worldPerPx);

    let worldSize = targetPx * worldPerPx;
    const floor = LABEL_MIN_FONT_SIZE[tier] || 0;
    if (floor > 0) worldSize = Math.max(worldSize, floor);

    const scale = worldSize / baseFont;
    entry.screenScale = scale;
    obj.scale.setScalar(scale);
    if (backdrop) {
      backdrop.scale.set(backdrop.userData.baseW * scale, backdrop.userData.baseH * scale, 1);
    }
  });
}

function resolveClusterLabelOverlap() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const halfW = rect.width / 2, halfH = rect.height / 2;
  const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  const camPos = camera.position;
  const v = new THREE.Vector3();

  const boxes = [];
  labelObjs.forEach(entry => {
    if (entry.tier !== 'cluster') return;
    const d = camPos.distanceTo(entry.anchorPos);
    if (d >= entry.far) { entry.wantShow = false; entry.nudge.x = entry.nudge.y = entry.nudgeTarget.x = entry.nudgeTarget.y = 0; return; }

    v.copy(entry.anchorPos).project(camera);
    if (v.z < -1 || v.z > 1 || v.x < -1 || v.x > 1 || v.y < -1 || v.y > 1) { entry.wantShow = false; entry.nudge.x = entry.nudge.y = entry.nudgeTarget.x = entry.nudgeTarget.y = 0; return; }  // out of view frustum

    const sx = (v.x + 1) * halfW, sy = (-v.y + 1) * halfH;
    const dSafe = Math.max(1, d);
    const pxPerWorld = rect.height / (2 * dSafe * tanHalfFov);
    const scale = entry.screenScale || 1;
    const worldW = entry.backdrop ? entry.backdrop.userData.baseW * scale : entry.baseFont * 6 * scale;
    const worldH = entry.backdrop ? entry.backdrop.userData.baseH * scale : entry.baseFont * 2 * scale;
    const halfPxW = (worldW * pxPerWorld) / 2 + LABEL_COLLISION_MARGIN_PX;
    const halfPxH = (worldH * pxPerWorld) / 2 + LABEL_COLLISION_MARGIN_PX;

    const guardRadius = Math.max(entry.radius, entry.baseFont) * LABEL_GUARANTEED_RADIUS_MULT;
    const guaranteed = d < (entry.wasGuaranteed ? guardRadius * 1.15 : guardRadius);
    entry.wasGuaranteed = guaranteed;

    boxes.push({ entry, d, guaranteed, sx, sy, halfPxW, halfPxH, nudgeX: entry.nudge.x, nudgeY: entry.nudge.y });
  });

  const sortKey = b => b.d * (b.entry.wantShow ? 0.85 : 1);
  boxes.sort((a, b) => (b.guaranteed - a.guaranteed) || (sortKey(a) - sortKey(b)));

  const budget = (LABEL_MAX_VISIBLE.cluster) || 0;
  const guaranteedCount = boxes.reduce((n, b) => n + (b.guaranteed ? 1 : 0), 0);
  const showCount = budget > 0 ? Math.min(boxes.length, Math.max(budget, guaranteedCount)) : boxes.length;
  const mustShow = boxes.slice(0, showCount);
  const hidden = boxes.slice(showCount);

  hidden.forEach(b => {
    b.entry.wantShow = false;
    b.entry.nudgeTarget.x = 0; b.entry.nudgeTarget.y = 0;
  });
  mustShow.forEach(b => { b.entry.wantShow = true; });

  if (LABEL_NUDGE_ENABLED && mustShow.length > 1) {
    for (let iter = 0; iter < LABEL_NUDGE_ITERATIONS; iter++) {
      for (let i = 0; i < mustShow.length; i++) {
        for (let j = i + 1; j < mustShow.length; j++) {
          const a = mustShow[i], b = mustShow[j];
          const ax = a.sx + a.nudgeX, ay = a.sy + a.nudgeY;
          const bx = b.sx + b.nudgeX, by = b.sy + b.nudgeY;
          const overlapX = (a.halfPxW + b.halfPxW) - Math.abs(ax - bx);
          const overlapY = (a.halfPxH + b.halfPxH) - Math.abs(ay - by);
          if (overlapX <= 0 || overlapY <= 0) continue;  // no overlap

          if (overlapX < overlapY) {
            const sign = (ax <= bx) ? -1 : 1;
            a.nudgeX += sign * overlapX * 0.15;
            b.nudgeX -= sign * overlapX * 0.85;
          } else {
            const sign = (ay <= by) ? -1 : 1;
            a.nudgeY += sign * overlapY * 0.15;
            b.nudgeY -= sign * overlapY * 0.85;
          }

          [a, b].forEach(o => {
            const mag = Math.hypot(o.nudgeX, o.nudgeY);
            if (mag > LABEL_NUDGE_MAX_PX) {
              const k = LABEL_NUDGE_MAX_PX / mag;
              o.nudgeX *= k; o.nudgeY *= k;
            }
          });
        }
      }
    }
    mustShow.forEach(b => { b.entry.nudgeTarget.x = b.nudgeX; b.entry.nudgeTarget.y = b.nudgeY; });
  } else {
    mustShow.forEach(b => { b.entry.nudgeTarget.x = 0; b.entry.nudgeTarget.y = 0; });
  }
}

// Nova/Adept titles: no backdrop, no priority, no budget — just a plain
// binary rule. For each pair of currently-visible titles belonging to the
// active cluster: if their screen boxes overlap, push them apart by exactly
// the overlap amount (split evenly) so they stop touching edge-to-edge.
// If they don't overlap, nothing happens — they sit exactly at their anchor.
// The overlap test always starts from the true, un-nudged anchor position,
// never from a previous frame's own offset, so this can't drift or compound
// across frames: it's a fresh, deterministic answer every time.
function resolveNovaAdeptLabelOverlap() {
  if (!_activeHighlightedClusterId) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const halfW = rect.width / 2, halfH = rect.height / 2;
  const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  const camPos = camera.position;
  const v = new THREE.Vector3();

  const boxes = [];
  labelObjs.forEach(entry => {
    if (entry.tier !== 'nova' && entry.tier !== 'adept') return;
    if (entry.clusterId !== _activeHighlightedClusterId || entry.curOpacity <= 0.01) {
      entry.nudgeTarget.x = 0; entry.nudgeTarget.y = 0;
      return;
    }

    v.copy(entry.anchorPos).project(camera);
    if (v.z < -1 || v.z > 1) { entry.nudgeTarget.x = 0; entry.nudgeTarget.y = 0; return; }

    const d = Math.max(1, camPos.distanceTo(entry.anchorPos));
    const pxPerWorld = rect.height / (2 * d * tanHalfFov);
    const scale = entry.screenScale || 1;
    const textWidthWorld = entry.textWidthWorld || (entry.obj.text.length * entry.baseFont * 0.62);
    const halfPxW = (textWidthWorld * scale * pxPerWorld) / 2 + LABEL_COLLISION_MARGIN_PX;
    const halfPxH = (entry.baseFont * 1.1 * scale * pxPerWorld) / 2 + LABEL_COLLISION_MARGIN_PX;

    boxes.push({ entry, sx: (v.x + 1) * halfW, sy: (-v.y + 1) * halfH, halfPxW, halfPxH, nx: 0, ny: 0 });
  });

  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      const overlapX = (a.halfPxW + b.halfPxW) - Math.abs(a.sx - b.sx);
      const overlapY = (a.halfPxH + b.halfPxH) - Math.abs(a.sy - b.sy);
      if (overlapX <= 0 || overlapY <= 0) continue;  // not touching: leave both exactly where they are

      if (overlapX < overlapY) {
        const sign = (a.sx <= b.sx) ? -1 : 1;
        a.nx += sign * overlapX * 0.5;
        b.nx -= sign * overlapX * 0.5;
      } else {
        const sign = (a.sy <= b.sy) ? -1 : 1;
        a.ny += sign * overlapY * 0.5;
        b.ny -= sign * overlapY * 0.5;
      }
    }
  }

  boxes.forEach(b => {
    b.entry.nudgeTarget.x = b.nx;
    b.entry.nudgeTarget.y = b.ny;
  });
}


function fmtDate(v) {
  if (!v) return '—';
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

const ROLE_BADGE_CLASS = { pioneer: 'bp', nova: 'bn', hub: 'bh' };

// Single source of truth for the role shown on a node: any node belonging to
// a Nova (subtopic) is tagged 'nova' regardless of its ADEPT role; otherwise
// falls back to the ADEPT role. Used by both the badge and the overlay row.
function displayRole(n) {
  if (!n) return null;
  const role = (n.nova_id ? 'nova' : (n.adept_role || '')).toLowerCase();
  if (!role) return null;
  return { key: role, label: role.charAt(0).toUpperCase() + role.slice(1) };
}

function badges(n) {
  if (!n) return '';
  let h = '';
  const dr = displayRole(n);
  if (dr) {
    const cls = ROLE_BADGE_CLASS[dr.key] || 'bd';
    h += '<span class="bx ' + cls + '">' + dr.label + '</span>';
  }
  // A node can be the pool's hub AND its chronological pioneer at once — the
  // backend only shows one role bucket at a time (hub wins), so add the
  // Pioneer badge separately whenever it's not already the displayed role.
  if (n.adept_is_pioneer && (!dr || dr.key !== 'pioneer')) {
    h += '<span class="bx bp">Pioneer</span>';
  }
  if (n.eng_cat === 'mega') h += '<span class="bx bm">MEGA</span>';
  else if (n.eng_cat === 'big') h += '<span class="bx bb">BIG</span>';
  return h;
}

function updateClusterOverlay(hovNode, mouseX, mouseY) {
  if (!hovNode) { clOverlay.style.display = 'none'; return; }

  const clabel = hovNode.cluster_label || hovNode.cluster_id || '—';

  let html = '<div class="ov-badges">' + badges(hovNode) + '</div>';
  html += '<div class="ov-row"><div class="ov-label">Content</div><div class="ov-content">' + (hovNode.content || '—') + '</div></div>';
  html += '<div class="ov-row"><div class="ov-label">Date</div><div class="ov-val">' + fmtDate(hovNode.timestamp) + '</div></div>';
  html += '<div class="ov-row"><div class="ov-label">Engagement</div><div class="ov-val">' + (hovNode.engagement || 0).toLocaleString('en-US') + '</div></div>';
  html += '<div class="ov-row"><div class="ov-label">Cluster</div><div class="ov-val" style="color:#aac">' + clabel + '</div></div>';
  if (hovNode.subtopic_label) html += '<div class="ov-row"><div class="ov-label">Subtopic (Nova)</div><div class="ov-val" style="color:#88ddcc">' + hovNode.subtopic_label + '</div></div>';
  if (hovNode.adept_pool_label) html += '<div class="ov-row"><div class="ov-label">Pool (ADEPT)</div><div class="ov-val" style="color:#FF8C00">' + hovNode.adept_pool_label + '</div></div>';
  const dr = displayRole(hovNode);
  // nova_depth only carries meaning inside a Nova chain (pioneer=0, offspring=rank);
  // ADEPT pools have no notion of rank, so no number is appended for those roles.
  let roleVal = dr ? (dr.key === 'nova' ? (dr.label + ' ' + hovNode.depth) : dr.label) : '—';
  if (hovNode.adept_is_pioneer && (!dr || dr.key !== 'pioneer')) roleVal += ' + Pioneer';
  html += '<div class="ov-row"><div class="ov-label">Role</div><div class="ov-val">' + roleVal + '</div></div>';

  clOverlay.innerHTML = html;
  clOverlay.style.display = 'block';

  const W = window.innerWidth;
  const ow = 280;
  const oh = clOverlay.offsetHeight || 180;
  let lx = mouseX + 18;
  let ly = mouseY - oh - 10;
  if (lx + ow > W - 10) lx = mouseX - ow - 12;
  if (ly < 40) ly = mouseY + 14;
  clOverlay.style.left = lx + 'px';
  clOverlay.style.top  = ly + 'px';
}

function _hitNode(hits) {
  if (!hits.length) return null;
  const h = hits[0];
  const k = h.object.userData.geomKey;
  if (k && instanceMap[k] && h.instanceId !== undefined)
    return instanceMap[k][h.instanceId] || null;
  return null;
}

function _hitEdge(hits) {
  if (!hits.length) return null;
  const h = hits[0];
  const grp = h.object.userData.group;
  const entry = pickableEdges[grp];
  if (!entry || typeof h.faceIndex !== 'number') return null;
  const edgeIdx = entry.segToEdge[h.faceIndex];
  const lk = entry.lks[edgeIdx];
  return lk ? { source: lk.source, target: lk.target, group: grp } : null;
}

function _hitSkeleton(hits) {
  if (!hits.length) return null;
  const h = hits[0];
  const pairs = h.object.userData.skeletonPairs || [];
  if (!pairs.length || !h.point) return null;
  let best = null, bestD = Infinity;
  const p = h.point;
  pairs.forEach(([source, target]) => {
    const a = NM[source], b = NM[target];
    if (!a || !b) return;
    const dx = b.x-a.x, dy = b.y-a.y, dz = b.z-a.z;
    const len2 = dx*dx + dy*dy + dz*dz || 1;
    const u = Math.max(0, Math.min(1, ((p.x-a.x)*dx + (p.y-a.y)*dy + (p.z-a.z)*dz) / len2));
    const qx = a.x + u*dx, qy = a.y + u*dy, qz = a.z + u*dz;
    const d2 = (p.x-qx)**2 + (p.y-qy)**2 + (p.z-qz)**2;
    if (d2 < bestD) {
      bestD = d2;
      best = { source, target, group: h.object.userData.skeletonType };
    }
  });
  return best;
}

function _hitTreeOverlay(hits) {
  if (!hits.length) return null;
  const h = hits[0];
  const pairs = h.object.userData.treePairs || [];
  if (!pairs.length) return null;

  let pair = (typeof h.faceIndex === 'number') ? pairs[h.faceIndex] : null;
  if (!pair && h.point && treeView.targetPos) {
    let best = null, bestD = Infinity;
    pairs.forEach(([s,t]) => {
      const a = treeView.targetPos[String(s)], b = treeView.targetPos[String(t)];
      if (!a || !b) return;
      const d2 = treePointSegDist2(h.point.x, h.point.y, a.x, a.y, b.x, b.y);
      if (d2 < bestD) { bestD = d2; best = [s,t]; }
    });
    pair = best;
  }
  if (!pair) return null;
  return { source: String(pair[0]), target: String(pair[1]),
            group: h.object.userData.treeType || '' };
}

function _hitTreeTitle(hits) {
  if (!hits.length) return null;
  const h = hits[0];
  const spec = h.object?.userData?.legendSpec;
  if (!spec || !h.object.userData.treeTitle) return null;
  return spec;
}

function pickAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  mouseNDC.x =  ((clientX - rect.left) / rect.width)  * 2 - 1;
  mouseNDC.y = -((clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouseNDC, camera);

  const nodeTargets = [...Object.values(instancedMeshes)];
  const nodeHits = raycaster.intersectObjects(nodeTargets, false);
  const topNode = nodeHits[0] || null;

  if (treeView.active) {
    if (topNode) {
      const node = _hitNode([topNode]);
      if (node && treeView.keepSet.has(node.id))
        return { node, edge:null, title:null };
      return { node:null, edge:null, title:null };
    }

    const treeEdgeTargets = Object.values(treeView.overlayObjs || {})
      .filter(o => o && o.visible);
    const titleTargets = (treeView.treeLabelObjs || [])
      .filter(o => o && o.visible);

    const hits = raycaster.intersectObjects(
      [...treeEdgeTargets, ...titleTargets], false
    );
    const top = hits[0] || null;
    if (!top) return { node:null, edge:null, title:null };

    if (top.object.userData?.treeTitle) {
      return { node:null, edge:null, title:_hitTreeTitle([top]) };
    }
    if (top.object.isLineSegments2) {
      return { node:null, edge:_hitTreeOverlay([top]), title:null };
    }
    return { node:null, edge:null, title:null };
  }

  if (topNode) {
    return { node:_hitNode([topNode]), edge:null, title:null, fromLabel:false };
  }

  if (pickableLabelMeshes.length) {
    const visibleLabelMeshes = pickableLabelMeshes.filter(m => m.visible);
    if (visibleLabelMeshes.length) {
      const labelHits = raycaster.intersectObjects(visibleLabelMeshes, false);
      const hitObj = labelHits[0] && labelHits[0].object;
      if (hitObj) {
        const entry = hitObj.userData.labelEntry || null;
        const nid = hitObj.userData.pickNodeId;
        const labelNode = nid != null ? NM[nid] : null;
        if (labelNode) {
          return { node:labelNode, edge:null, title:null, fromLabel:true, labelEntry: entry };
        }
        if (entry && entry.tier === 'cluster' && entry.clusterId != null) {
          return { node:null, edge:null, title:null, fromLabel:true, labelEntry: entry, clusterId: entry.clusterId };
        }
      }
    }
  }

  // Skeleton-type edges (temporal, semantic_inter, adept_graft, ...) are
  // invisible by default and are only ever meant to be shown via hover
  // proximity (see revealSkeleton()/showSkeletonHighlight()), so they must
  // stay raycast targets even while obj.visible is false -- otherwise the
  // mouse can never "land" on one to reveal it. Non-skeleton groups the user
  // toggled off in the sidebar are excluded as before.
  const edgeTargets = Object.values(pickableEdges)
    .map(p => p.obj)
    .filter(o => o.visible || SKELETON_TYPES_SET.has(o.userData.group));
  const skeletonTargets = Object.values(skeletonHL)
    .filter(o => o.visible && SKELETON_TYPES_SET.has(o.userData.skeletonType));
  const hits = raycaster.intersectObjects(
    [...edgeTargets, ...skeletonTargets], false
  );
  const top = hits[0] || null;
  if (!top) return { node:null, edge:null, title:null, fromLabel:false };

  const eg = top.object.userData.skeletonType ? _hitSkeleton([top]) : _hitEdge([top]);
  return { node:null, edge:eg, title:null, fromLabel:false };
}

function onMouseMove(e) {
  if (cameraLookMode === 'hold') { clOverlay.style.display = 'none'; return; }

  if (cameraIsMoving) {
    // Flying (WASD/arrows) or auto-flying to a cluster: skip picking
    // entirely, exactly like camera-look-hold above -- no raycast at all
    // this frame. Raycasting the graph while the camera is also moving is
    // what was still costly even with hover recoloring gone; this removes
    // it outright while in motion and interaction resumes the instant
    // movement stops (next mousemove after keys are released).
    if (hoveredNode !== null) clearTimeout(htimer);
    hoveredNode = null;
    hoveredLabelEntry = null;
    clOverlay.style.display = 'none';
    canvas.style.cursor = 'default';
    revealSkeleton(pinnedIds);
    return;
  }

  if (e.target !== canvas) {
    // The pointer is over some UI chrome (side panel, topbar, legend...)
    // that visually covers the scene at this screen position. That chrome
    // is opaque to the eye, so it must be opaque to picking too -- don't
    // raycast "through" it and reveal info for a node hidden underneath.
    if (hoveredNode !== null) clearTimeout(htimer);
    hoveredNode = null;
    hoveredLabelEntry = null;
    clOverlay.style.display = 'none';
    canvas.style.cursor = 'default';
    if (treeView.active) {
      clearTreeHover();
    } else {
      revealSkeleton(pinnedIds);
    }
    return;
  }

  const { node: nodeHit, edge: edgeHit, title: titleHit, fromLabel, labelEntry, clusterId: clusterHit } = pickAt(e.clientX, e.clientY);

  const newHoveredLabel = (fromLabel && labelEntry
      && (labelEntry.tier === 'nova' || labelEntry.tier === 'adept'))
    ? labelEntry : null;
  hoveredLabelEntry = newHoveredLabel;

  // Cluster titles are clickable (same action as the Map-panel entry): show a
  // pointer cursor whenever one is under the mouse, exactly like any other
  // clickable UI affordance.
  canvas.style.cursor = clusterHit ? 'pointer' : 'default';

  if (nodeHit !== hoveredNode) {
    hoveredNode = nodeHit;
    clearTimeout(htimer);
    if (!nodeHit || fromLabel) clOverlay.style.display = 'none';
  }
  if (nodeHit && !fromLabel) updateClusterOverlay(hoveredNode, e.clientX, e.clientY);
  else if (!nodeHit && treeView.active && !titleHit) clOverlay.style.display = 'none';

  if (treeView.active) {
    if (titleHit) {
      setTreeHover(new Set(titleHit.ids || []), null, titleHit);
    } else if (nodeHit) {
      setTreeHover(treeRelatedIdsForNode(nodeHit.id));
    } else if (edgeHit) {
      setTreeHover(treeRelatedIdsForEdge(edgeHit), edgeHit);
    } else {
      clearTreeHover();
    }
    return;
  }

  const hoverIds = nodeHit ? [nodeHit.id] : (edgeHit ? [edgeHit.source, edgeHit.target] : null);
  revealSkeleton(pinnedIds || hoverIds);
}



canvas.addEventListener('click', e => {
  if (dragMoved) return;
  const { node: nd, edge: eg, title: th, clusterId: cid } = pickAt(e.clientX, e.clientY);

  if (treeView.active) {
    if (th) {
      setTreeHover(new Set(th.ids || []), null, th);
      openSelectionPanelForTitle(th);
    } else if (nd) {
      setTreeHover(treeRelatedIdsForNode(nd.id));
      openSelectionPanelForNode(nd);
    } else if (eg) {
      setTreeHover(treeRelatedIdsForEdge(eg), eg);
      openSelectionPanelForEdge(eg);
    }
    return;
  }

  if (nd) {
    enterTreeView(nd.id);
    return;
  }
  if (eg) {
    enterTreeView(eg.source);
    return;
  }
  if (cid) {
    // Clicking a cluster title in the scene = clicking that cluster's row in
    // the Map panel: identical flight-to-cluster behaviour.
    flyToCluster(cid);
    return;
  }

  pinnedIds = null;
  revealSkeleton(null);
});

canvas.addEventListener('dblclick', e => {
  if (!treeView.active || dragMoved) return;
  e.preventDefault();
  exitTreeView(false);
});

function buildToggles() {
  GROUPS.forEach(g => {
    const col  = EC[g] || '#888';
    const dash = g === 'adept_bridge';
    const div  = document.createElement('div');
    div.className = 'eg'; div.id = 'eg_' + g;
    const dotStyle = dash
      ? 'width:18px;height:0;border-top:2px dashed ' + col + ';margin-top:1px;flex-shrink:0'
      : 'width:18px;height:3px;border-radius:1px;background:' + col + ';flex-shrink:0';
    const on = vis.has(g);  // reflects actual state
    div.innerHTML =
      '<div style="' + dotStyle + '"></div>' +
      '<div class="el">' + g.replace(/_/g, ' ') + '</div>' +
      '<div class="eck' + (on ? ' on' : '') + '" id="ck_' + g + '">' + (on ? '✓' : '') + '</div>';
    div.onclick = () => togGroup(g);
    if      (g === 'nova')           document.getElementById('ep').appendChild(div);
    else if (g.startsWith('adept_')) document.getElementById('ea').appendChild(div);
    else                              document.getElementById('eo').appendChild(div);
  });
}
function togGroup(g) {
  if (treeView.active) return;
  if (vis.has(g)) vis.delete(g); else vis.add(g);
  const ck = document.getElementById('ck_' + g);
  if (ck) { ck.classList.toggle('on', vis.has(g)); ck.textContent = vis.has(g) ? '✓' : ''; }
  const show = vis.has(g);
  (groupLines[g] || []).forEach(l => l.visible = show);
}
buildToggles();

// Legend for the floating Nova/ADEPT titles in the scene: same glyph
// (LABEL_PREFIX) and same color (TITLE_TIER_COLOR) as makeLabel() actually
// draws, so this can't drift out of sync with what's on screen. Static —
// only shown for a tier if the data actually produces at least one such
// title (mirrors initLabels()'s own nova/adept conditions).
function buildTitleLegend() {
  const container = document.getElementById('etl');
  if (!container) return;
  const rows = [];
  if (RAW.nodes.some(n => n.nova_role === 'pioneer' && n.subtopic_label))
    rows.push({ tier: 'nova', label: 'Nova subtopic' });
  if (RAW.nodes.some(n => n.adept_role === 'hub' && n.adept_pool_label))
    rows.push({ tier: 'adept', label: 'ADEPT pool' });
  rows.forEach(r => {
    const div = document.createElement('div');
    div.className = 'eg';
    div.style.cursor = 'default';
    div.innerHTML =
      '<div class="el" style="flex:none;color:' + TITLE_TIER_COLOR[r.tier] + '">' + LABEL_PREFIX[r.tier] + '</div>' +
      '<div class="el">' + r.label + '</div>';
    container.appendChild(div);
  });
}
buildTitleLegend();

let labelsEnabled = true;
let edgeFadeEnabled = true;
function buildViewOptions() {
  const opts = [
    { key: 'labels',   label: 'Labels',    get: () => labelsEnabled,   set: v => { labelsEnabled = v; } },
    { key: 'edgeFade', label: 'Edge fade', get: () => edgeFadeEnabled, set: v => {
      edgeFadeEnabled = v;
      _lastEdgeFadeTime = 0;
      _lastEdgeFadeCamPos.set(Infinity, Infinity, Infinity);
    } },
  ];
  const container = document.getElementById('ev');
  opts.forEach(o => {
    const div = document.createElement('div');
    div.className = 'eg';
    div.innerHTML = '<div class="el">' + o.label + '</div><div class="eck' + (o.get() ? ' on' : '') + '" id="ck_' + o.key + '">' + (o.get() ? '✓' : '') + '</div>';
    div.onclick = () => {
      const v = !o.get();
      o.set(v);
      const ck = document.getElementById('ck_' + o.key);
      ck.classList.toggle('on', v);
      ck.textContent = v ? '✓' : '';
    };
    container.appendChild(div);
  });
}
buildViewOptions();

let sbLastTab = 'e';  // last active Edges/Map tab, restored on tree view exit
function swTab(t) {
  if (treeView.active) return;
  if (t !== 'e' && t !== 'm') t = 'e';
  sbLastTab = t;
  const tabs = document.querySelectorAll('#sb-tabs .sbt');
  tabs.forEach(el => el.classList.remove('active'));
  const active = t === 'e' ? tabs[0] : tabs[1];
  if (active) active.classList.add('active');
  const edgesPanel = document.getElementById('te');
  const mapPanel = document.getElementById('tm');
  if (edgesPanel) edgesPanel.classList.toggle('s', t === 'e');
  if (mapPanel) mapPanel.classList.toggle('s', t === 'm');
  if (t === 'm') buildMapPanel();
}

document.querySelectorAll('#sb-tabs .sbt').forEach((el, i) => {
  el.onclick = null;
  el.addEventListener('click', ev => {
    ev.preventDefault();
    ev.stopPropagation();
    swTab(i === 0 ? 'e' : 'm');
  });
});

let mapBuilt = false;

function buildMapPanel() {
  if (mapBuilt) return;
  mapBuilt = true;

  const clusters = {};

  RAW.nodes.forEach(n => {
    const cid = n.cluster_id;
    if (!cid) return;
    if (!clusters[cid]) {
      clusters[cid] = {
        label: n.cluster_label || cid,
        color: n.color,
        count: 0
      };
    }
    clusters[cid].count++;
  });

  let html = '<div class="gx-section"><div class="gx-title">Clusters</div>';
  const cidsByLabel = Object.keys(clusters).sort((a, b) =>
    clusters[a].label.localeCompare(clusters[b].label, undefined, { sensitivity: 'base', numeric: true })
  );
  cidsByLabel.forEach(cid => {
    const c = clusters[cid];
    html += `<div class="gx-row" data-row-cluster-id="${encodeURIComponent(cid)}">
      <div class="gx-item" data-cluster-id="${encodeURIComponent(cid)}" data-label="${_escHtml(c.label)}" title="${cid}">
        <div class="gx-toggle" data-toggle-cluster-id="${encodeURIComponent(cid)}" title="Toggle ideas">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>
        </div>
        <div class="gx-dot" style="background:${c.color}"></div>
        <div class="gx-name">${_escHtml(c.label)}</div>
        <div class="gx-count">${c.count}</div>
      </div>
      <div class="gx-sub" id="gx-sub-${encodeURIComponent(cid)}"></div>
    </div>`;
  });
  html += '</div>';

  const mapContent = document.getElementById('map-content');
  mapContent.innerHTML = Object.keys(clusters).length
    ? html
    : '<div style="color:#444;font-size:11px;padding:20px;text-align:center">No clusters</div>';

  mapContent.querySelectorAll('.gx-item[data-cluster-id]').forEach(el => {
    el.addEventListener('click', ev => {
      ev.preventDefault();
      ev.stopPropagation();
      const cid = decodeURIComponent(el.dataset.clusterId);
      flyToCluster(cid);
    });
  });

  // Chevron: expand/collapse this cluster's idea list WITHOUT flying the
  // camera there — a lighter way to browse ideas cluster by cluster.
  // stopPropagation keeps this from also bubbling into the .gx-item click
  // above (which would trigger flyToCluster on every peek).
  mapContent.querySelectorAll('.gx-toggle[data-toggle-cluster-id]').forEach(el => {
    el.addEventListener('click', ev => {
      ev.preventDefault();
      ev.stopPropagation();
      toggleClusterIdeas(decodeURIComponent(el.dataset.toggleClusterId));
    });
  });

  filterMapClusters(mapSearchInput ? mapSearchInput.value : '');
  updateMapStickyOffset();
}

// How many ideas (posts) a freshly-opened cluster shows before needing
// "+N more" — keeps a big cluster from dumping hundreds of rows into the
// DOM the moment you glance at it.
const IDEA_PREVIEW_COUNT = 15;

function _truncate(s, n) {
  s = String(s || '').trim();
  return s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s;
}

// Builds the idea rows for one cluster, straight from RAW.nodes (already in
// memory client-side — no extra fetch/data structure needed). Sorted by
// engagement so the most representative ideas surface first, same spirit as
// how BERTopic/Nomic surface "representative docs" for a topic.
//
// Nodes without a date or without an engagement value are not skipped or
// treated as errors: fmtDate() falls back to '—' for any missing/invalid
// timestamp, and `n.engagement || 0` falls back to 0 for any missing
// engagement (the pipeline itself already fills missing engagement with a
// per-file/global median before it ever reaches here, so this is a second,
// defensive layer, not the only one). A cluster where every node shares the
// same engagement value — including an all-missing cluster, where every
// node falls back to 0 — simply keeps a stable, deterministic order (the
// order nodes were built in) rather than erroring or reshuffling.
function _clusterIdeasHtml(cid, limit) {
  const nodes = RAW.nodes
    .filter(n => String(n.cluster_id) === String(cid))
    .sort((a, b) => (b.engagement || 0) - (a.engagement || 0));

  if (!nodes.length) return '<div class="gx-sub-empty">No ideas</div>';

  const shown = limit ? nodes.slice(0, limit) : nodes;
  let html = shown.map(n => `<div class="gx-idea">
      <div class="gx-idea-content">${_escHtml(_truncate(n.content, 160))}</div>
      <div class="gx-idea-meta"><span>${fmtDate(n.timestamp)}</span><span>${(n.engagement || 0).toLocaleString('en-US')}</span></div>
    </div>`).join('');

  if (limit && nodes.length > limit) {
    html += `<button type="button" class="gx-more" data-more-cluster-id="${encodeURIComponent(cid)}">+ ${nodes.length - limit} more</button>`;
  }
  return html;
}

// Accordion state: at most one cluster's idea list is open at a time, same
// convention as a standard accordion widget (Bootstrap/MUI/Radix) — opening
// one closes whichever other one was open, instead of letting them stack up.
let openClusterId = null;

function closeClusterIdeas(cid) {
  const sub = document.getElementById('gx-sub-' + encodeURIComponent(cid));
  const toggle = document.querySelector('.gx-toggle[data-toggle-cluster-id="' + encodeURIComponent(cid) + '"]');
  if (sub) sub.classList.remove('open');
  if (toggle) toggle.classList.remove('open');
  if (openClusterId === cid) openClusterId = null;
}

function toggleClusterIdeas(cid) {
  const sub = document.getElementById('gx-sub-' + encodeURIComponent(cid));
  const toggle = document.querySelector('.gx-toggle[data-toggle-cluster-id="' + encodeURIComponent(cid) + '"]');
  if (!sub || !toggle) return;

  const willOpen = !sub.classList.contains('open');

  // Accordion: close whatever other cluster is currently open before
  // opening this one, so only one idea list is ever expanded at once.
  if (willOpen && openClusterId && openClusterId !== cid) {
    closeClusterIdeas(openClusterId);
  }

  if (willOpen && !sub.dataset.built) {
    sub.innerHTML = _clusterIdeasHtml(cid, IDEA_PREVIEW_COUNT);
    sub.dataset.built = '1';
    const moreBtn = sub.querySelector('.gx-more[data-more-cluster-id]');
    if (moreBtn) {
      moreBtn.addEventListener('click', ev => {
        ev.preventDefault();
        ev.stopPropagation();
        sub.innerHTML = _clusterIdeasHtml(cid, null);  // reveal all, uncapped
      });
    }
  }
  sub.classList.toggle('open', willOpen);
  toggle.classList.toggle('open', willOpen);
  openClusterId = willOpen ? cid : null;
}

// Called from flyToCluster(): clicking a cluster title in the 3D scene goes
// through the exact same function as clicking its row in the Map panel (see
// the canvas click handler above), so this one line makes BOTH entry points
// also expand+scroll to that cluster's idea list — no separate code path to
// keep in sync between "click in scene" and "click in menu".
function revealClusterInMapPanel(cid) {
  if (sbLastTab !== 'm') swTab('m');
  if (!mapBuilt) buildMapPanel();

  const sub = document.getElementById('gx-sub-' + encodeURIComponent(cid));
  if (!sub) return;
  if (!sub.classList.contains('open')) toggleClusterIdeas(cid);

  const row = document.querySelector('.gx-row[data-row-cluster-id="' + encodeURIComponent(cid) + '"]');
  if (row) row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Map search: live filter + highlight over the already-built cluster list.
function _escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const mapSearchInput = document.getElementById('map-search');
const mapSearchCount = document.getElementById('map-search-count');
const mapSearchEmpty = document.getElementById('map-search-empty');
const mapSearchClear = document.getElementById('map-search-clear');

// The search bar is sticky (position:sticky, top:0) and sits on top of
// everything that scrolls past it, so a cluster row scrolled to y:0 would
// land right behind it — same class of problem as a sticky nav hiding an
// anchor target. Fix: give every .gx-row a scroll-margin-top matching the
// sticky bar's real rendered height (+ the search-count line right below
// it, which also scrolls under it), read via getBoundingClientRect() so it
// stays correct even if fonts/padding change, instead of a guessed number.
function updateMapStickyOffset() {
  const wrap = document.getElementById('map-search-wrap');
  const tm = document.getElementById('tm');
  if (!wrap || !tm) return;
  const h = wrap.getBoundingClientRect().height + (mapSearchCount ? mapSearchCount.getBoundingClientRect().height : 0);
  if (h > 0) tm.style.setProperty('--map-sticky-offset', (h + 6) + 'px');
}

function updateMapSearchClear() {
  if (!mapSearchClear) return;
  mapSearchClear.classList.toggle('visible', !!(mapSearchInput && mapSearchInput.value));
}

// Strips accents one character at a time (not on the whole string) so the
// output stays exactly the same length as the input, with each accented
// letter replaced by its plain base letter at the same index — e.g. 'é'
// (1 char) becomes 'e' (1 char), never 'e' + a separate combining mark.
// That's what lets the fold be used purely for matching while indices from
// it (idx, query.length) still line up correctly when slicing the
// *original* label for display/highlighting below.
function _foldAccents(s) {
  let out = '';
  for (const ch of String(s)) {
    out += ch.normalize('NFD').replace(/[̀-ͯ]/g, '') || ch;
  }
  return out;
}

function filterMapClusters(rawQuery) {
  const items = document.querySelectorAll('#map-content .gx-item[data-cluster-id]');
  const query = (rawQuery || '').trim();
  // Fold + lowercase both sides so "é"/"e" and "É"/"É" all match each
  // other — search shouldn't care about accents any more than it cares
  // about case.
  const qFold = _foldAccents(query).toLowerCase();

  let total = 0, visible = 0;
  items.forEach(el => {
    total++;
    const label = el.dataset.label || '';
    const labelFold = _foldAccents(label).toLowerCase();
    const idx = qFold ? labelFold.indexOf(qFold) : 0;
    const match = !qFold || idx !== -1;
    const row = el.closest('.gx-row');
    if (row) row.style.display = match ? '' : 'none';
    if (match) visible++;

    const nameEl = el.querySelector('.gx-name');
    if (!nameEl) return;
    if (qFold && idx !== -1) {
      // Slicing the ORIGINAL label (accents intact) at the folded match's
      // position — safe because _foldAccents never changes string length.
      const before = label.slice(0, idx);
      const hit = label.slice(idx, idx + query.length);
      const after = label.slice(idx + query.length);
      nameEl.innerHTML = _escHtml(before) + '<mark>' + _escHtml(hit) + '</mark>' + _escHtml(after);
    } else {
      nameEl.textContent = label;
    }
  });

  if (mapSearchCount) mapSearchCount.textContent = query ? `${visible} / ${total}` : '';
  if (mapSearchEmpty) mapSearchEmpty.style.display = (query && visible === 0) ? '' : 'none';
}

if (mapSearchInput) {
  mapSearchInput.addEventListener('input', () => {
    filterMapClusters(mapSearchInput.value);
    updateMapSearchClear();
  });
  updateMapSearchClear();
}

if (mapSearchClear) {
  // type="button" already keeps this out of any form submit path; stop the
  // mousedown too so focus never leaves the input before the click lands.
  mapSearchClear.addEventListener('mousedown', e => e.preventDefault());
  mapSearchClear.addEventListener('click', () => {
    if (!mapSearchInput) return;
    mapSearchInput.value = '';
    filterMapClusters('');
    updateMapSearchClear();
    mapSearchInput.focus();
  });
}

let clusterFlight = null;

const NODE_HIGHLIGHT_BRIGHTEN = 5.0;  // >1 = brighter, 1.0 = no effect
const _hiNodeColor = new THREE.Color();

function _vividColor(c, factor) {
  return _hiNodeColor.setRGB(c.r * factor, c.g * factor, c.b * factor);
}

function _setNodeColorRaw(n, colorObj) {
  const loc = NODE_INSTANCE[n.id];
  if (!loc) return;
  const im = instancedMeshes[loc.k];
  if (!im) return;
  im.setColorAt(loc.idx, colorObj);
  im.instanceColor.needsUpdate = true;
}

const CLUSTER_HIGHLIGHT_TRANSITION_MS = 220;  // 0 = instant
const _colorTransitions = new Map();  // node -> fromFactor/toFactor/start

function _easeInOutQuad(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function _startColorTransition(n, toFactor) {
  const fromFactor = n._curHighlightFactor !== undefined ? n._curHighlightFactor : 1.0;
  if (!CLUSTER_HIGHLIGHT_TRANSITION_MS || Math.abs(fromFactor - toFactor) < 0.001) {
    n._curHighlightFactor = toFactor;
    _colorTransitions.delete(n);
    _setNodeColorRaw(n, _vividColor(_baseColorFor(n), toFactor));
    return;
  }
  _colorTransitions.set(n, { fromFactor, toFactor, start: performance.now() });
}

function updateNodeColorTransitions(now) {
  if (!_colorTransitions.size) return;
  _colorTransitions.forEach((t, n) => {
    const e = Math.min(1, (now - t.start) / CLUSTER_HIGHLIGHT_TRANSITION_MS);
    const factor = t.fromFactor + (t.toFactor - t.fromFactor) * _easeInOutQuad(e);
    n._curHighlightFactor = factor;
    _setNodeColorRaw(n, _vividColor(_baseColorFor(n), factor));
    if (e >= 1) _colorTransitions.delete(n);
  });
}

function _highlightNodeColor(n) {
  _startColorTransition(n, NODE_HIGHLIGHT_BRIGHTEN);
}

function _baseNodeColor(n) {
  _startColorTransition(n, 1.0);
}

const HOVER_BLINK_MIN_FACTOR = 0.6;  // blink trough = normal (unhighlighted) render
const HOVER_BLINK_PERIOD_MS  = 1400;  // full cycle duration, ms
let _hoverBlinkStart = 0;  // reset on every new hover target

function updateHoverBlink(now) {
  if (!highlightedHoverNodes || !highlightedHoverNodes.size) return;
  const phase = ((now - _hoverBlinkStart) % HOVER_BLINK_PERIOD_MS) / HOVER_BLINK_PERIOD_MS;  // 0..1
  const wave = 0.5 + 0.5 * Math.cos(phase * Math.PI * 2);  // 1 at start, trough at mid-cycle
  const factor = HOVER_BLINK_MIN_FACTOR + (NODE_HIGHLIGHT_BRIGHTEN - HOVER_BLINK_MIN_FACTOR) * wave;
  highlightedHoverNodes.forEach(n => {
    n._curHighlightFactor = factor;
    _colorTransitions.delete(n);  // blink takes over; drop any in-flight transition
    _setNodeColorRaw(n, _vividColor(_baseColorFor(n), factor));
  });
}

let highlightedClusterNodes = null;  // currently selected cluster
let highlightedHoverNodes   = null;  // currently hovered subgraph

let _activeHighlightedClusterId = null;
let _clusterSelectionTime = 0;  // performance.now() when _activeHighlightedClusterId last changed

function _restoreNodeColor(n) {
  const stillCluster = highlightedClusterNodes && highlightedClusterNodes.has(n);
  const stillHover    = highlightedHoverNodes && highlightedHoverNodes.has(n);
  if (stillCluster || stillHover) {
    _highlightNodeColor(n);
  } else {
    _baseNodeColor(n);
  }
}

function selectCluster(cid) {
  if (treeView.active) return;
  const key = String(cid);
  const nodes = RAW.nodes.filter(n => String(n.cluster_id) === key);
  if (!nodes.length) return;

  const prev = highlightedClusterNodes;
  highlightedClusterNodes = new Set(nodes);
  if (prev) prev.forEach(n => { if (!highlightedClusterNodes.has(n)) _restoreNodeColor(n); });
  nodes.forEach(_highlightNodeColor);
  if (_activeHighlightedClusterId !== key) _clusterSelectionTime = performance.now();
  _activeHighlightedClusterId = key;
}

function deselectCluster() {
  if (treeView.active) return;
  if (!highlightedClusterNodes) return;
  const prev = highlightedClusterNodes;
  highlightedClusterNodes = null;
  prev.forEach(_restoreNodeColor);
  _activeHighlightedClusterId = null;
}

function _forceClearClusterSelection() {
  clusterFlight = null;
  deselectCluster();
  _forceClearHoverGlow();
  lastSkeletonKey = undefined;
  autoHighlightedClusterId = null;
  _lastClusterProximityCamPos.set(Infinity, Infinity, Infinity);
}

function showHoverGlow(ids) {
  const nodes = (ids && ids.size) ? [...ids].map(id => NM[id]).filter(Boolean) : [];
  const prev = highlightedHoverNodes;
  highlightedHoverNodes = nodes.length ? new Set(nodes) : null;
  if (prev) prev.forEach(n => { if (!highlightedHoverNodes || !highlightedHoverNodes.has(n)) _restoreNodeColor(n); });
  if (nodes.length) {
    _hoverBlinkStart = performance.now();
    nodes.forEach(n => {
      n._curHighlightFactor = NODE_HIGHLIGHT_BRIGHTEN;
      _colorTransitions.delete(n);
      _setNodeColorRaw(n, _vividColor(_baseColorFor(n), NODE_HIGHLIGHT_BRIGHTEN));
    });
  }
}

function _forceClearHoverGlow() {
  showHoverGlow(null);
}

const CLUSTER_PROXIMITY_FACTOR      = 5;
const CLUSTER_PROXIMITY_MIN_RADIUS  = 20;

const CLUSTER_PROXIMITY_UPDATE_INTERVAL_MS = 70;
const CLUSTER_PROXIMITY_MIN_CAM_MOVE_SQ    = 9;
const CLUSTER_PROXIMITY_MIN_ROTATION_DEG   = 2;  // min camera rotation to re-trigger
let _lastClusterProximityTime = 0;
const _lastClusterProximityCamPos = new THREE.Vector3(Infinity, Infinity, Infinity);
const _lastClusterProximityForward = new THREE.Vector3(0, 0, -1);
const _minForwardDot = Math.cos(THREE.MathUtils.degToRad(CLUSTER_PROXIMITY_MIN_ROTATION_DEG));

let autoHighlightedClusterId = null;

const CLUSTER_PROXIMITY_FOV_MARGIN_DEG = 10;

const _camForward = new THREE.Vector3();
const _toCluster   = new THREE.Vector3();
const _toLabel     = new THREE.Vector3();  // reused by label occlusion below
const _hoverColor  = new THREE.Color();    // reused by the Nova/Adept hover brighten
const LABEL_HOVER_OUTLINE_COLOR   = new THREE.Color(0xffffff);
const LABEL_HOVER_OUTLINE_OPACITY = 1;

function updateClusterProximityHighlight(now) {
  if (treeView.active) return;  // isolated scene, no cluster concept
  if (now - _lastClusterProximityTime < CLUSTER_PROXIMITY_UPDATE_INTERVAL_MS) return;

  camera.getWorldDirection(_camForward);
  const posMoved = camera.position.distanceToSquared(_lastClusterProximityCamPos) >= CLUSTER_PROXIMITY_MIN_CAM_MOVE_SQ;
  const rotated  = _camForward.dot(_lastClusterProximityForward) < _minForwardDot;
  if (!posMoved && !rotated) return;  // no significant move or rotation

  _lastClusterProximityTime = now;
  _lastClusterProximityCamPos.copy(camera.position);
  _lastClusterProximityForward.copy(_camForward);

  // Sticky selection: as long as the camera is still physically inside the
  // currently selected cluster's volume (within its bounding radius), keep
  // it selected — even if the view direction has turned away from its
  // centroid, e.g. looking at one of the cluster's own Nova/Adept notes near
  // the edge on the way out. Without this, a camera rotation alone could
  // drop the cluster (and with it, the Nova/Adept titles) while the camera
  // hadn't actually left. The FOV-based search below only runs once the
  // camera has genuinely exited the cluster's volume.
  if (autoHighlightedClusterId != null) {
    const cur = clusterCentroids[autoHighlightedClusterId];
    if (cur) {
      const curRadius = Math.max(Number(cur.radius) || 0, CLUSTER_PROXIMITY_MIN_RADIUS);
      _toCluster.set(cur.x - camera.position.x, cur.y - camera.position.y, cur.z - camera.position.z);
      if (_toCluster.length() < curRadius) return;  // still inside: stay selected, skip the FOV search
    }
  }

  const maxAngleRad = THREE.MathUtils.degToRad(camera.fov * 0.5 + CLUSTER_PROXIMITY_FOV_MARGIN_DEG);
  const minDot = Math.cos(maxAngleRad);  // avoids per-cluster acos

  let bestCid = null, bestDist = Infinity;
  Object.keys(clusterCentroids).forEach(cid => {
    const c = clusterCentroids[cid];
    const radius = Math.max(Number(c.radius) || 0, CLUSTER_PROXIMITY_MIN_RADIUS);
    _toCluster.set(c.x - camera.position.x, c.y - camera.position.y, c.z - camera.position.z);
    const dist = _toCluster.length();
    if (dist > 0.001) {
      const dot = _toCluster.dot(_camForward) / dist;
      if (dot < minDot) return;
    }
    if (dist < radius * CLUSTER_PROXIMITY_FACTOR && dist < bestDist) {
      bestDist = dist;
      bestCid = cid;
    }
  });

  if (bestCid === autoHighlightedClusterId) return;  // no change
  autoHighlightedClusterId = bestCid;
  if (bestCid) selectCluster(bestCid); else deselectCluster();
}

function _clusterLabelAnchorFactor(entry) {
  if (entry.tier !== 'cluster' || _activeHighlightedClusterId == null) return null;
  if (entry.clusterId !== _activeHighlightedClusterId) return null;
  const effRadius = Math.max(entry.radius || 0, CLUSTER_PROXIMITY_MIN_RADIUS);
  const d = camera.position.distanceTo(entry.anchorPos);
  const outer = effRadius * CLUSTER_LABEL_FADE_OUTER_MULT;
  const inner = effRadius * CLUSTER_LABEL_FADE_INNER_MULT;
  const t = THREE.MathUtils.clamp((d - inner) / Math.max(outer - inner, 1e-6), 0, 1);
  const smooth = t * t * (3 - 2 * t);  // smoothstep: 0 near (full fade), 1 far (full opacity)
  return CLUSTER_LABEL_ANCHOR_OPACITY + (1 - CLUSTER_LABEL_ANCHOR_OPACITY) * smooth;
}

function flyToCluster(cid) {
  if (treeView.active) return;
  const c = clusterCentroids[String(cid)];
  if (!c) {
    console.warn('[MAP] Cluster introuvable:', cid);
    return;
  }

  const radius = Math.max(Number(c.radius) || 0, 1);
  const center = new THREE.Vector3(c.x, c.y, c.z);

  const forward = new THREE.Vector3(0, 0, -1)
    .applyQuaternion(camera.quaternion).normalize();
  if (forward.lengthSq() < 1e-8) forward.set(0, 0, -1);

  const halfFov = THREE.MathUtils.degToRad(camera.fov * 0.5);
  const fitDistance = radius / Math.max(Math.tan(halfFov), 0.01);
  const distance = Math.max(fitDistance * 1.10, radius * 1.65);

  const targetPos = center.clone().sub(forward.clone().multiplyScalar(distance));
  targetPos.x = Math.max(UNIVERSE.navLimits.x.min, Math.min(UNIVERSE.navLimits.x.max, targetPos.x));
  targetPos.y = Math.max(UNIVERSE.navLimits.y.min, Math.min(UNIVERSE.navLimits.y.max, targetPos.y));
  targetPos.z = Math.max(UNIVERSE.navLimits.z.min, Math.min(UNIVERSE.navLimits.z.max, targetPos.z));

  clusterFlight = {
    start: performance.now(),
    duration: 1700,
    startPos: camera.position.clone(),
    targetPos,
    startQuat: camera.quaternion.clone(),
    cid: String(cid)
  };

  revealClusterInMapPanel(String(cid));

  console.log('[MAP] Flying to cluster:', cid,
              'nodes:', c.count, 'radius:', radius, 'distance:', distance);
}

// Overview shortcut (F key): reuses the clusterFlight animation to fly back to spawn.
function flyToSpawn() {
  if (treeView.active) return;

  clusterFlight = {
    start: performance.now(),
    duration: 1700,
    startPos: camera.position.clone(),
    targetPos: new THREE.Vector3(UNIVERSE.cameraSpawn.x, UNIVERSE.cameraSpawn.y, UNIVERSE.cameraSpawn.z),
    startQuat: camera.quaternion.clone(),
    // Only flyToSpawn sets a targetQuat different from startQuat: this is
    // what triggers the orientation slerp in animate() below, so the
    // original view direction is restored together with the position.
    // flyToCluster does not set one: it keeps whatever direction the
    // camera is currently facing for the entire flight.
    targetQuat: SPAWN_QUATERNION.clone(),
    cid: 'spawn'
  };

  console.log('[NAV] Flying to spawn (overview)');
}

function onResize() {
  const w = window.innerWidth - 280;
  const h = window.innerHeight - 36;
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  fatLineMaterials.forEach(m => m.resolution.set(w, h));
}
window.addEventListener('resize', onResize);
onResize();

// Fullscreen toggle button: uses the standard Fullscreen API on <html>,
// which is the same end result as pressing F11 (native fullscreen, no
// browser chrome). Icon and title stay in sync with the real state so it
// also reflects the user exiting via Escape or F11 itself.
//
// Keyboard Lock (navigator.keyboard.lock(['Escape'])) is engaged while in
// fullscreen so a quick Escape tap no longer exits fullscreen on its own --
// it reaches our keydown handler below instead, which only closes free-look
// toggle mode (see the pointer-lock section). Escape still exits fullscreen
// if held down for ~1-2s: that fallback is native browser behavior, nothing
// to implement here. Chromium only (Chrome/Edge); unsupported browsers just
// keep the old single-tap-Escape-exits-fullscreen behavior via feature
// detection below.
(function setupFullscreenButton() {
  const btn = document.getElementById('fullscreen-btn');
  if (!btn) return;
  const iconExpand = document.getElementById('fs-icon-expand');
  const iconCollapse = document.getElementById('fs-icon-collapse');
  const supportsKeyboardLock = ('keyboard' in navigator) && ('lock' in navigator.keyboard);

  function isFullscreen() {
    return !!document.fullscreenElement;
  }

  function syncIcon() {
    const fs = isFullscreen();
    if (iconExpand) iconExpand.style.display = fs ? 'none' : '';
    if (iconCollapse) iconCollapse.style.display = fs ? '' : 'none';
    btn.classList.toggle('on', fs);
    btn.title = fs
      ? (supportsKeyboardLock
          ? 'Exit fullscreen (hold Esc)'
          : 'Exit fullscreen (Esc)')
      : 'Fullscreen (F11)';
    btn.setAttribute('aria-label', btn.title);
  }

  btn.addEventListener('click', () => {
    if (!isFullscreen()) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else if (document.exitFullscreen) {
      document.exitFullscreen().catch(() => {});
    }
  });

  document.addEventListener('fullscreenchange', () => {
    syncIcon();
    if (!supportsKeyboardLock) return;
    if (isFullscreen()) {
      navigator.keyboard.lock(['Escape']).catch(() => {});
    } else {
      navigator.keyboard.unlock();
    }
  });
  syncIcon();
})();

function fmtUnits(v) {
  const r = Math.round(v);
  return (r === 0 ? 0 : r).toLocaleString('en-US');
}

document.getElementById('univ-size').textContent =
  `World: ${fmtUnits(UNIVERSE.size.x)} × ${fmtUnits(UNIVERSE.size.y)} × ${fmtUnits(UNIVERSE.size.z)} units`;

function animate() {
  requestAnimationFrame(animate);

  cameraIsMoving = !treeView.active && (!!clusterFlight || _movementKeysActive());

  if (_pendingMouseMoveEvent) {
    const ev = _pendingMouseMoveEvent;
    _pendingMouseMoveEvent = null;
    onMouseMove(ev);
  }

  const now  = performance.now();
  const dt   = Math.min((now - fps.prevTime) / 1000, 0.1);
  fps.prevTime = now;

  const spd  = Math.min(SPEED_ABSOLUTE_MAX, fps.speed * (fps.keys['ShiftLeft'] || fps.keys['ShiftRight'] ? SPEED_SHIFT_MULT : 1));
  const dist = spd * dt;

  const fwd = new THREE.Vector3(0, 0, -1);
  fwd.applyEuler(camera.rotation);

  const right = new THREE.Vector3(1, 0, 0);
  right.applyEuler(camera.rotation);

  if (clusterFlight) {
    const t = Math.min(1, (now - clusterFlight.start) / clusterFlight.duration);
    const e = t * t * (3 - 2 * t); // smoothstep
    camera.position.lerpVectors(clusterFlight.startPos, clusterFlight.targetPos, e);
    if (clusterFlight.targetQuat) {
      camera.quaternion.slerpQuaternions(clusterFlight.startQuat, clusterFlight.targetQuat, e);
    } else {
      camera.quaternion.copy(clusterFlight.startQuat);
    }
    camera.updateMatrixWorld(true);

    if (t >= 1) {
      camera.position.copy(clusterFlight.targetPos);
      camera.quaternion.copy(clusterFlight.targetQuat || clusterFlight.startQuat);
      camera.updateMatrixWorld(true);
      fps.yaw = camera.rotation.y;
      fps.pitch = camera.rotation.x;
      clusterFlight = null;
    }
  } else if (!treeView.active) {
    if (fps.keys['KeyW'] || fps.keys['ArrowUp'])     camera.position.addScaledVector(fwd,    dist);
    if (fps.keys['KeyS'] || fps.keys['ArrowDown'])   camera.position.addScaledVector(fwd,   -dist);
    if (fps.keys['KeyA'] || fps.keys['ArrowLeft'])   camera.position.addScaledVector(right, -dist);
    if (fps.keys['KeyD'] || fps.keys['ArrowRight'])  camera.position.addScaledVector(right,  dist);
    if (fps.keys['KeyQ'])                            camera.position.y += dist;
    if (fps.keys['KeyE'] || fps.keys['ControlLeft']) camera.position.y -= dist;
  }
  // else: tree view is active -- position/orientation are owned by
  // applyTreePanZoom(), called from updateTreeViewFrame() below.

  document.getElementById('speed').textContent =
    `Speed: ${fmtUnits(spd)} u/s`;

  document.getElementById('position').textContent =
    `Pos: X:${fmtUnits(camera.position.x)} Y:${fmtUnits(camera.position.y)} Z:${fmtUnits(camera.position.z)}`;

  camera.position.x = Math.max(UNIVERSE.navLimits.x.min, Math.min(UNIVERSE.navLimits.x.max, camera.position.x));
  camera.position.y = Math.max(UNIVERSE.navLimits.y.min, Math.min(UNIVERSE.navLimits.y.max, camera.position.y));
  camera.position.z = Math.max(UNIVERSE.navLimits.z.min, Math.min(UNIVERSE.navLimits.z.max, camera.position.z));

  updateTreeViewFrame(now);

  clusterAurasGroup.visible = !treeView.active;
  updateClusterAuraFade(now, dt);

  updateNodeMinScreenSize();

  updateEdgeDistanceFade(now);

  skeletonDashOffset -= FLOW_SPEED;
  DIRECTIONAL_DASH_TYPES.forEach(t => {
    const o = skeletonHL[t];
    if (o.visible) o.material.dashOffset = skeletonDashOffset;
    (groupLines[t] || []).forEach(l => {
      if (l.visible && l.material) l.material.dashOffset = skeletonDashOffset;
    });
  });

  updateClusterProximityHighlight(now);
  updateNodeColorTransitions(now);
  updateHoverBlink(now);

  if (treeView.active || !labelsEnabled) {
    labelObjs.forEach(entry => {
      entry.wantShow = false;
      entry.curOpacity = 0;
      entry.obj.visible = false;
      if (entry.backdrop) entry.backdrop.visible = false;
    });
  } else {
    updateLabelScale(window.innerHeight - 36);

    labelObjs.forEach(entry => {
      if (entry.tier === 'cluster') return;  // handled by resolveClusterLabelOverlap()
      // Nova/Adept titles have no flat distance cutoff of their own: they are
      // hidden by default and only become visible through the cluster-fade
      // reveal below (_novaAdeptRevealFactor), so no conflicting gate here.
      entry.wantShow = true;
    });
    resolveClusterLabelOverlap();
    resolveNovaAdeptLabelOverlap();

    // Reused below: how "into" the highlighted cluster the camera currently is
    // (0 at CLUSTER_OCCLUSION_DIST_OUTER_MULT*radius, 1 at ...INNER_MULT*radius).
    // Nova/Adept titles for that cluster fade in on this exact same curve
    // instead of popping in at a flat, cluster-agnostic distance cutoff.
    let _novaAdeptRevealFactor = 0;

    if (_activeHighlightedClusterId) {
      const hc = clusterCentroids[_activeHighlightedClusterId];
      if (hc) {
        _toCluster.set(hc.x - camera.position.x, hc.y - camera.position.y, hc.z - camera.position.z);
        const distToCluster = _toCluster.length();
        if (distToCluster > 0.001) {
          const hcRadius = Math.max(Number(hc.radius) || 0, 1);
          const distOuter = hcRadius * CLUSTER_OCCLUSION_DIST_OUTER_MULT;
          const distInner = hcRadius * CLUSTER_OCCLUSION_DIST_INNER_MULT;
          const distT = THREE.MathUtils.clamp((distToCluster - distInner) / Math.max(distOuter - distInner, 1e-6), 0, 1);
          const distGate = 1 - distT * distT * (3 - 2 * distT);  // 0 far from cluster (no occlusion), 1 right up against it

          // Time-based floor on top of the distance-based curve: guarantees the reveal
          // always ramps in over CLUSTER_LABEL_SELECT_FADE_MS, even if this cluster was
          // already well inside its near zone the instant it got selected (e.g. the
          // auto-proximity switch jumping straight to a neighboring cluster). On a normal
          // approach from far away this has already reached 1 long before distGate becomes
          // meaningful, so it changes nothing there.
          const timeT = THREE.MathUtils.clamp((now - _clusterSelectionTime) / CLUSTER_LABEL_SELECT_FADE_MS, 0, 1);
          const timeGate = timeT * timeT * (3 - 2 * timeT);  // smoothstep
          _novaAdeptRevealFactor = distGate * timeGate;

          if (distGate > 0) {
            _toCluster.multiplyScalar(1 / distToCluster);  // normalized camera->cluster direction
            const nearEdge   = Math.max(0, distToCluster - hcRadius);  // cluster's near face
            const innerAngle = Math.atan2(hcRadius, distToCluster)
                                + THREE.MathUtils.degToRad(CLUSTER_OCCLUSION_INNER_MARGIN_DEG);
            const outerAngle = innerAngle + THREE.MathUtils.degToRad(CLUSTER_OCCLUSION_OUTER_MARGIN_DEG);
            const cosInner = Math.cos(innerAngle), cosOuter = Math.cos(outerAngle);

            labelObjs.forEach(entry => {
              if (!entry.wantShow) return;  // already hidden
              if (entry.clusterId === _activeHighlightedClusterId) return;  // never occlude the highlighted cluster's own title
              const dLabel = camera.position.distanceTo(entry.obj.position);
              if (dLabel <= nearEdge) return;  // in front of (or on) the cluster: never occlude
              _toLabel.set(entry.obj.position.x - camera.position.x,
                           entry.obj.position.y - camera.position.y,
                           entry.obj.position.z - camera.position.z).multiplyScalar(1 / dLabel);
              const dot = _toLabel.dot(_toCluster);
              const t = THREE.MathUtils.clamp((dot - cosOuter) / Math.max(cosInner - cosOuter, 1e-6), 0, 1);
              const angularMul = 1 - (t * t * (3 - 2 * t));  // inverted smoothstep
              entry.occlusionMul = 1 - distGate * (1 - angularMul);
            });
          }
        }
      }
    }

    labelObjs.forEach(entry => {
      const anchorFactor = _clusterLabelAnchorFactor(entry);
      const isAnchored = anchorFactor !== null;

      // Nova/Adept titles are invisible by default. They only fade in for the
      // actively highlighted cluster, on the same timing as that cluster's own
      // occlusion zone; every other cluster's Nova/Adept titles stay hidden.
      const revealFactor = (entry.tier === 'nova' || entry.tier === 'adept')
        ? (entry.clusterId === _activeHighlightedClusterId ? _novaAdeptRevealFactor : 0)
        : 1;

      const target = (entry.wantShow ? 1 : 0)
        * (entry.occlusionMul !== undefined ? entry.occlusionMul : 1)
        * (isAnchored ? anchorFactor : 1)
        * revealFactor;
      entry.occlusionMul = 1;  // reset for next frame
      entry.curOpacity += (target - entry.curOpacity) * LABEL_FADE_SPEED;
      if (Math.abs(target - entry.curOpacity) < 0.01) entry.curOpacity = target;
      const shown = entry.curOpacity > 0.01;
      entry.obj.visible = shown;
      if (entry.backdrop) entry.backdrop.visible = shown;
      if (shown) {
        entry.obj.fillOpacity = entry.curOpacity;
        entry.obj.outlineOpacity = entry.baseOutlineOpacity * entry.curOpacity;

        if (entry.tier === 'nova' || entry.tier === 'adept') {
          const hoverTarget = (entry === hoveredLabelEntry) ? 1 : 0;
          entry.hoverT += (hoverTarget - entry.hoverT) * LABEL_HOVER_EASE_SPEED;
          if (Math.abs(hoverTarget - entry.hoverT) < 0.01) entry.hoverT = hoverTarget;
          // Same technique cartography/map labels use for a hover/active state
          // (e.g. Mapbox's text-halo): fill color never changes, so legibility
          // never depends on the color scheme. Only the outline the text already
          // has at rest brightens toward a muted tone and widens a bit.
          entry.obj.outlineWidth = (7 + entry.hoverT * (LABEL_HOVER_OUTLINE_PCT - 7)) + '%';
          _hoverColor.set(0x000000).lerp(LABEL_HOVER_OUTLINE_COLOR, entry.hoverT);
          entry.obj.outlineColor = '#' + _hoverColor.getHexString();
          entry.obj.outlineOpacity = THREE.MathUtils.lerp(entry.baseOutlineOpacity, LABEL_HOVER_OUTLINE_OPACITY, entry.hoverT) * entry.curOpacity;
        }

        entry.obj.quaternion.copy(camera.quaternion);

        if (isAnchored) {
          entry.nudge.x = 0; entry.nudge.y = 0;
          entry.nudgeTarget.x = 0; entry.nudgeTarget.y = 0;
          if (!entry.obj.position.equals(entry.anchorPos)) {
            entry.obj.position.copy(entry.anchorPos);
            if (entry.backdrop) entry.backdrop.position.copy(entry.anchorPos);
          }
        } else if (LABEL_NUDGE_ENABLED && (entry.tier === 'cluster' || entry.tier === 'nova' || entry.tier === 'adept')) {
          entry.nudge.x += (entry.nudgeTarget.x - entry.nudge.x) * LABEL_NUDGE_EASE_SPEED;
          entry.nudge.y += (entry.nudgeTarget.y - entry.nudge.y) * LABEL_NUDGE_EASE_SPEED;
          if (Math.abs(entry.nudgeTarget.x - entry.nudge.x) < 0.05) entry.nudge.x = entry.nudgeTarget.x;
          if (Math.abs(entry.nudgeTarget.y - entry.nudge.y) < 0.05) entry.nudge.y = entry.nudgeTarget.y;

          if (Math.abs(entry.nudge.x) > 0.01 || Math.abs(entry.nudge.y) > 0.01) {
            const d = camera.position.distanceTo(entry.anchorPos);
            const dSafe = Math.max(1, d);
            const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
            const worldPerPx = (2 * dSafe * tanHalfFov) / (window.innerHeight - 36);
            _labelNudgeRight.set(1, 0, 0).applyQuaternion(camera.quaternion);
            _labelNudgeUp.set(0, 1, 0).applyQuaternion(camera.quaternion);
            entry.obj.position.copy(entry.anchorPos)
              .addScaledVector(_labelNudgeRight, entry.nudge.x * worldPerPx)
              .addScaledVector(_labelNudgeUp, -entry.nudge.y * worldPerPx);  // screen y grows downward, invert for world "up"
            if (entry.backdrop) entry.backdrop.position.copy(entry.obj.position);
          } else if (!entry.obj.position.equals(entry.anchorPos)) {
            entry.obj.position.copy(entry.anchorPos);
            if (entry.backdrop) entry.backdrop.position.copy(entry.anchorPos);
          }
        }

        if (entry.backdrop) entry.backdrop.material.opacity = entry.curOpacity;
      }
    });
  }

  renderer.render(scene, camera);
}
animate();
