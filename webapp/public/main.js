// main.js — Three.js scene + UI logic
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  simulateGBMPaths, simulateMJDPaths,
  mcOptionPrice, bsPrice, mertonPrice,
  bsGreeks, mcGreeksRange,
} from './finance.js';

// ── Constants ─────────────────────────────────────────────────────────────────
const N_DISPLAY   = 30;    // max paths rendered per model (pricing uses full nPaths)
const Z_SPACING   = 0.8;   // Z-axis spread between paths
const MJD_Z_BASE  = -28;   // MJD fan: z = -28 … -28+(N_DISPLAY-1)*Z_SPACING ≈ -4.8
const ANIM_SPEED  = 2;     // steps revealed per animation frame
// TIME_SCALE is computed dynamically in startAnimation based on actual price range

const COLOR_GBM  = 0xFFD700;
const COLOR_MJD  = 0xFF6B35;
const COLOR_AXIS = 0x555555;

// ── Three.js setup ────────────────────────────────────────────────────────────
const canvas = document.getElementById('three-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 0); // transparent — CSS grid shows through

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 10000);
camera.position.set(-8, 145, 55);
camera.lookAt(5, 100, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(5, 100, 0);

function onResize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(onResize).observe(canvas);
onResize();

// ── Text sprite helper ────────────────────────────────────────────────────────
function makeTextLabel(text, x, y, z) {
  const cvs = document.createElement('canvas');
  cvs.width  = 256;
  cvs.height = 48;
  const c = cvs.getContext('2d');
  c.fillStyle = '#888888';
  c.font = 'bold 20px Consolas, monospace';
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.fillText(text, 128, 24);
  const tex = new THREE.CanvasTexture(cvs);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(10, 2, 1);
  spr.position.set(x, y, z);
  return spr;
}

// ── Axis helper ───────────────────────────────────────────────────────────────
function buildAxes(S0, timeScale, yMin, yMax) {
  const group = new THREE.Group();
  const mat = new THREE.LineBasicMaterial({ color: COLOR_AXIS });

  // X axis — time
  const xGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, yMin, 0),
    new THREE.Vector3(timeScale, yMin, 0),
  ]);
  group.add(new THREE.Line(xGeo, mat));

  // Y axis — price
  const yGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, yMin, 0),
    new THREE.Vector3(0, yMax, 0),
  ]);
  group.add(new THREE.Line(yGeo, mat));

  // Origin dot
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(S0 * 0.004, 8, 8),
    new THREE.MeshBasicMaterial({ color: COLOR_AXIS }),
  );
  dot.position.set(0, S0, 0);
  group.add(dot);

  // Axis labels
  const labelOffset = (yMax - yMin) * 0.06;
  group.add(makeTextLabel('Time (days)',     timeScale / 2,   yMin - labelOffset * 1.5, 0));
  group.add(makeTextLabel('Stock Price ($)', -timeScale * 0.06, (yMin + yMax) / 2,      0));

  return group;
}

// ── Build path geometries ─────────────────────────────────────────────────────
function buildPathObjects(paths, t, color, zBase, T, timeScale) {
  const group = new THREE.Group();
  const nShow = Math.min(paths.length, N_DISPLAY);
  const tScale = timeScale / T;

  for (let i = 0; i < nShow; i++) {
    const path = paths[i];
    const steps = path.length;
    const positions = new Float32Array(steps * 3);

    for (let s = 0; s < steps; s++) {
      positions[s * 3]     = t[s] * tScale;
      positions[s * 3 + 1] = path[s];
      positions[s * 3 + 2] = zBase + i * Z_SPACING;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setDrawRange(0, 0); // animation reveals it

    const mat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.45,
    });

    group.add(new THREE.Line(geo, mat));
  }
  return group;
}

// ── Floor grid plane ──────────────────────────────────────────────────────────
function buildFloor(timeScale, yMin) {
  const zMax    = (N_DISPLAY - 1) * Z_SPACING;   // top of GBM fan
  const zMin    = MJD_Z_BASE;                      // bottom of MJD fan
  const zCenter = (zMin + zMax) / 2;
  const zSpan   = zMax - zMin + 4;

  const geo = new THREE.PlaneGeometry(timeScale + 2, zSpan, 10, 10);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x2a2a2a,
    transparent: true,
    opacity: 0.35,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(timeScale / 2, yMin * 0.97, zCenter);
  return mesh;
}

// ── Animation state ───────────────────────────────────────────────────────────
let gbmGroup = null;
let mjdGroup = null;
let axisGroup = null;
let floorMesh = null;
let animStep = 0;
let totalSteps = 0;
let animating = false;

function clearScene() {
  [gbmGroup, mjdGroup, axisGroup, floorMesh].forEach(g => {
    if (g) scene.remove(g);
  });
  gbmGroup = mjdGroup = axisGroup = floorMesh = null;
}

function startAnimation(gbmPaths, mjdPaths, t, T, S0) {
  clearScene();
  totalSteps = t.length;
  animStep = 0;
  animating = true;

  // ── 1. Compute price range from actual path data ───────────────────────────
  let yMin = Infinity, yMax = -Infinity;
  for (const paths of [gbmPaths, mjdPaths]) {
    for (const path of paths) {
      for (const price of path) {
        if (price < yMin) yMin = price;
        if (price > yMax) yMax = price;
      }
    }
  }
  const yRange = yMax - yMin;

  // ── 2. Dynamic time scale: x spans same scene units as y (≥70% visible) ───
  const timeScale = yRange;

  // ── 3. Build scene objects ─────────────────────────────────────────────────
  floorMesh = buildFloor(timeScale, yMin);
  axisGroup = buildAxes(S0, timeScale, yMin, yMax);
  gbmGroup  = buildPathObjects(gbmPaths, t, COLOR_GBM, 0,          T, timeScale);
  mjdGroup  = buildPathObjects(mjdPaths, t, COLOR_MJD, MJD_Z_BASE, T, timeScale);
  scene.add(floorMesh, axisGroup, gbmGroup, mjdGroup);

  // ── 4. Bounding box of all rendered 3D points ──────────────────────────────
  let xMin3 =  Infinity, xMax3 = -Infinity;
  let yMin3 =  Infinity, yMax3 = -Infinity;
  let zMin3 =  Infinity, zMax3 = -Infinity;

  for (const group of [gbmGroup, mjdGroup]) {
    for (const line of group.children) {
      const pos = line.geometry.attributes.position.array;
      for (let i = 0; i < pos.length; i += 3) {
        if (pos[i]     < xMin3) xMin3 = pos[i];
        if (pos[i]     > xMax3) xMax3 = pos[i];
        if (pos[i + 1] < yMin3) yMin3 = pos[i + 1];
        if (pos[i + 1] > yMax3) yMax3 = pos[i + 1];
        if (pos[i + 2] < zMin3) zMin3 = pos[i + 2];
        if (pos[i + 2] > zMax3) zMax3 = pos[i + 2];
      }
    }
  }

  const cx = (xMin3 + xMax3) / 2;
  const cy = (yMin3 + yMax3) / 2;
  const cz = (zMin3 + zMax3) / 2;

  const dx = xMax3 - xMin3;
  const dy = yMax3 - yMin3;
  const dz = zMax3 - zMin3;
  const diagonal = Math.sqrt(dx * dx + dy * dy + dz * dz);

  // ── 5. Camera distance so paths fill ~65% of canvas ───────────────────────
  const fovRad = (50 * Math.PI) / 180;
  const dist   = (diagonal / 2) / (Math.tan(fovRad / 2) * 0.65);

  // ── 6. Position at 30° elevation, 20° azimuth for diagonal view ───────────
  const elev = (30 * Math.PI) / 180;
  const azim = (20 * Math.PI) / 180;
  camera.position.set(
    cx + dist * Math.sin(azim) * Math.cos(elev),
    cy + dist * Math.sin(elev),
    cz + dist * Math.cos(azim) * Math.cos(elev),
  );
  controls.target.set(cx, cy, cz);
  camera.lookAt(cx, cy, cz);
  controls.update();
}

function tickAnimation() {
  if (!animating) return;
  animStep = Math.min(animStep + ANIM_SPEED, totalSteps);
  const drawCount = animStep * 3; // 3 floats per vertex

  gbmGroup?.children.forEach(line => line.geometry.setDrawRange(0, drawCount));
  mjdGroup?.children.forEach(line => line.geometry.setDrawRange(0, drawCount));

  if (animStep >= totalSteps) animating = false;
}

// ── Render loop ───────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  tickAnimation();
  renderer.render(scene, camera);
}
animate();

// ── UI helpers ────────────────────────────────────────────────────────────────
function getParams() {
  const v = id => parseFloat(document.getElementById(id).value);
  return {
    S0:      v('S0'),
    K:       v('K'),
    r:       v('r'),
    sigma:   v('sigma'),
    T:       v('T'),
    nPaths:  parseInt(document.getElementById('nPaths').value, 10),
    lam:     v('lam'),
    muJ:     v('muJ'),
    sigmaJ:  v('sigmaJ'),
  };
}

// ── Table update ──────────────────────────────────────────────────────────────
function updateTable(rowKey, callVal, putVal, delta, sigma) {
  const fmt = x => x.toFixed(4);

  document.getElementById(`row-${rowKey}-call`).textContent  = fmt(callVal);
  document.getElementById(`row-${rowKey}-put`).textContent   = fmt(putVal);
  document.getElementById(`row-${rowKey}-sigma`).textContent = sigma.toFixed(4);

  const deltaEl = document.getElementById(`row-${rowKey}-delta`);
  if (delta === null) {
    deltaEl.textContent = '—';
    deltaEl.className = 'delta-zero';
  } else {
    const sign = delta >= 0 ? '+' : '';
    deltaEl.textContent = `${sign}${delta.toFixed(4)}`;
    deltaEl.className = delta > 0.0001 ? 'delta-pos' : delta < -0.0001 ? 'delta-neg' : 'delta-zero';
  }

  ['call', 'put', 'delta', 'sigma'].forEach(col => {
    document.getElementById(`row-${rowKey}-${col}`)?.classList.remove('placeholder');
  });
}

// ── Interpretation text ───────────────────────────────────────────────────────
function showInterpretation(mc, mj, bs, nPaths) {
  const panel = document.getElementById('interp-panel');
  const text  = document.getElementById('interp-text');

  const mjDiff  = mj.call - bs.call;
  const absDiff = Math.abs(mjDiff).toFixed(2);
  const dir     = mjDiff < 0 ? 'cheaper' : 'more expensive';

  const mcDiff = mc.call - bs.call;
  const mcAbs  = Math.abs(mcDiff).toFixed(2);
  const mcDir  = mcDiff < 0 ? 'below' : 'above';

  text.innerHTML =
    `MJD prices the call <strong>$${absDiff} ${dir}</strong> than Black-Scholes — ` +
    `downward jump bias shifts probability mass to the left tail, reducing upside and inflating put value. ` +
    `MC (GBM) call is $${mcAbs} ${mcDir} B-S (simulation noise with ${nPaths} paths).`;

  panel.style.display = 'block';
}

// ── Run button ────────────────────────────────────────────────────────────────
document.getElementById('btn-run').addEventListener('click', () => {
  const p = getParams();

  const { t, paths: gbmPaths } = simulateGBMPaths(p.S0, p.r, p.sigma, p.T, 252, p.nPaths);
  const { paths: mjdPaths }    = simulateMJDPaths(p.S0, p.r, p.sigma, p.T, 252, p.nPaths,
                                                   p.lam, p.muJ, p.sigmaJ);

  const mc = { call: mcOptionPrice(gbmPaths, p.K, p.r, p.T, 'call'),
               put:  mcOptionPrice(gbmPaths, p.K, p.r, p.T, 'put') };
  const mj = { call: mcOptionPrice(mjdPaths, p.K, p.r, p.T, 'call'),
               put:  mcOptionPrice(mjdPaths, p.K, p.r, p.T, 'put') };
  const bs = { call: bsPrice(p.S0, p.K, p.r, p.sigma, p.T, 'call'),
               put:  bsPrice(p.S0, p.K, p.r, p.sigma, p.T, 'put') };
  const mr = { call: mertonPrice(p.S0, p.K, p.r, p.sigma, p.T, p.lam, p.muJ, p.sigmaJ, 'call'),
               put:  mertonPrice(p.S0, p.K, p.r, p.sigma, p.T, p.lam, p.muJ, p.sigmaJ, 'put') };

  updateTable('mc',     mc.call, mc.put, mc.call - bs.call, p.sigma);
  updateTable('mjd',    mj.call, mj.put, mj.call - bs.call, p.sigma);
  updateTable('bs',     bs.call, bs.put, null,               p.sigma);
  updateTable('merton', mr.call, mr.put, mr.call - bs.call,  p.sigma);

  showInterpretation(mc, mj, bs, p.nPaths);

  startAnimation(gbmPaths, mjdPaths, t, p.T, p.S0);

  if (document.getElementById('greeks-panel').classList.contains('visible')) {
    requestAnimationFrame(() => drawGreeks(p));
  }
});

// ── Greeks button ─────────────────────────────────────────────────────────────
document.getElementById('btn-greeks').addEventListener('click', () => {
  const panel = document.getElementById('greeks-panel');
  const btn   = document.getElementById('btn-greeks');
  const open  = panel.classList.toggle('visible');
  btn.classList.toggle('active', open);
  btn.textContent = open ? 'Greeks ▲' : 'Greeks ▼';
  // rAF ensures the panel is rendered (display:block) before we measure canvas sizes
  if (open) requestAnimationFrame(() => drawGreeks(getParams()));
});

// ── Greeks canvas drawing ─────────────────────────────────────────────────────
function drawGreeks(p) {
  const N_SPOT = 40;
  const S0arr = Array.from({ length: N_SPOT }, (_, i) =>
    p.S0 * (0.70 + i * 0.60 / (N_SPOT - 1)));

  const bs = bsGreeks(S0arr, p.K, p.r, p.sigma, p.T, 'call');
  const mc = mcGreeksRange(S0arr, p.K, p.r, p.sigma, p.T, 2000, 'call');

  const charts = [
    { key: 'delta', id: 'canvas-delta' },
    { key: 'gamma', id: 'canvas-gamma' },
    { key: 'vega',  id: 'canvas-vega'  },
    { key: 'theta', id: 'canvas-theta' },
  ];

  for (const { key, id } of charts) {
    drawGreekChart(document.getElementById(id), S0arr, bs[key], mc[key], p.S0, p.K);
  }
}

function drawGreekChart(canvas, xArr, bsArr, mcArr, spot, strike) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth  * dpr;
  const H = canvas.offsetHeight * dpr;
  if (W === 0 || H === 0) return; // panel not yet laid out
  canvas.width  = W;
  canvas.height = H;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const PAD = { top: 8, right: 8, bottom: 18, left: 36 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  const allY = [...bsArr, ...mcArr].filter(isFinite);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const yRange = yMax - yMin || 1;

  const xMin = xArr[0], xRange = xArr[xArr.length - 1] - xArr[0];

  const toX = x => PAD.left + (x - xMin) / xRange * plotW;
  const toY = y => PAD.top  + (1 - (y - yMin) / yRange) * plotH;

  // Zero line
  if (yMin < 0 && yMax > 0) {
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 0.5 * dpr;
    ctx.beginPath();
    ctx.moveTo(PAD.left, toY(0));
    ctx.lineTo(PAD.left + plotW, toY(0));
    ctx.stroke();
  }

  // Spot & strike markers
  const vline = (x, color) => {
    if (x < xMin || x > xArr[xArr.length - 1]) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 0.8 * dpr;
    ctx.setLineDash([3 * dpr, 3 * dpr]);
    ctx.beginPath();
    ctx.moveTo(toX(x), PAD.top);
    ctx.lineTo(toX(x), PAD.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
  };
  vline(spot,   '#FF6B6B');
  vline(strike, '#98FB98');

  const drawLine = (arr, color, dash = []) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5 * dpr;
    ctx.setLineDash(dash.map(d => d * dpr));
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < xArr.length; i++) {
      if (!isFinite(arr[i])) { started = false; continue; }
      const px = toX(xArr[i]), py = toY(arr[i]);
      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  };

  drawLine(bsArr, '#FFD700');
  drawLine(mcArr, '#00BFFF', [4, 3]);

  // Y axis ticks
  ctx.fillStyle = '#666';
  ctx.font = `${9 * dpr}px Consolas, monospace`;
  ctx.textAlign = 'right';
  for (let i = 0; i <= 3; i++) {
    const val = yMin + yRange * (i / 3);
    ctx.fillText(val.toFixed(3), (PAD.left - 3) * dpr, toY(val) + 3 * dpr);
  }

  // X axis ticks
  ctx.textAlign = 'center';
  for (let i = 0; i <= 2; i++) {
    const xi = Math.floor(i * (xArr.length - 1) / 2);
    ctx.fillText('$' + xArr[xi].toFixed(0), toX(xArr[xi]), H - 3 * dpr);
  }
}

// ── Keyboard shortcut: R to run ───────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'r' || e.key === 'R') document.getElementById('btn-run').click();
});

// ── Sigma live-label tracking ─────────────────────────────────────────────────
let sigmaFromFetch = false;

function revertSigmaLabel() {
  sigmaFromFetch = false;
  document.getElementById('sigma-label').textContent = 'σ';
  document.getElementById('sigma-live-badge').style.display = 'none';
}

document.getElementById('sigma').addEventListener('input', () => {
  if (sigmaFromFetch) revertSigmaLabel();
});

// ── Stock ticker fetch ────────────────────────────────────────────────────────
document.getElementById('btn-fetch').addEventListener('click', async () => {
  const ticker = document.getElementById('ticker').value.trim().toUpperCase();
  const status = document.getElementById('ticker-status');
  if (!ticker) { status.textContent = 'Enter a ticker symbol.'; status.className = 'error'; return; }

  status.textContent = `Fetching ${ticker}…`;
  status.className = '';

  try {
    const res  = await fetch(`/api/stock/${ticker}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');

    // Populate S0
    document.getElementById('S0').value = data.price;

    // Auto-set K to nearest $5
    document.getElementById('K').value = Math.round(data.price / 5) * 5;

    // Populate sigma if IV available
    if (data.iv && data.iv > 0) {
      document.getElementById('sigma').value = data.iv.toFixed(4);
      document.getElementById('sigma-label').textContent = 'σ (implied)';
      document.getElementById('sigma-live-badge').style.display = 'inline';
      sigmaFromFetch = true;
      status.textContent = `${data.ticker}: $${data.price}  |  σ = ${(data.iv * 100).toFixed(1)}%`;
    } else {
      revertSigmaLabel();
      status.textContent = `${data.ticker}: $${data.price}  |  IV unavailable — set σ manually`;
    }
    status.className = 'ok';
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    status.className = 'error';
  }
});

// ── Number field +/− buttons ──────────────────────────────────────────────────
document.querySelectorAll('.nf-dec, .nf-inc').forEach(btn => {
  btn.addEventListener('click', () => {
    const input = document.getElementById(btn.dataset.target);
    const step  = parseFloat(input.step) || 1;
    const min   = input.min !== '' ? parseFloat(input.min) : -Infinity;
    const max   = input.max !== '' ? parseFloat(input.max) :  Infinity;
    const cur   = parseFloat(input.value) || 0;
    const dec   = Math.max(step.toString().split('.')[1]?.length ?? 0, 0);
    const next  = btn.classList.contains('nf-inc') ? cur + step : cur - step;
    input.value = Math.min(max, Math.max(min, next)).toFixed(dec);

    // Revert implied label if sigma stepper is used
    if (btn.dataset.target === 'sigma' && sigmaFromFetch) revertSigmaLabel();
  });
});
