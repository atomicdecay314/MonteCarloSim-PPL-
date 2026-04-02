// main.js — Three.js scene + UI logic
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  simulateGBMPaths, simulateMJDPaths,
  mcOptionPrice, bsPrice, mertonPrice,
  bsGreeks, mcGreeksRange,
} from './finance.js';

// ── Constants ─────────────────────────────────────────────────────────────────
const N_DISPLAY   = 6;     // paths shown per model
const Z_SPACING   = 1.0;   // Z-axis spread between paths
const MJD_Z_BASE  = -5;    // MJD fan starts here so it never overlaps GBM
const TIME_SCALE  = 10;    // scene units for full time horizon
const ANIM_SPEED  = 2;     // steps revealed per animation frame

const COLOR_GBM   = 0xFFD700;
const COLOR_MJD   = 0xFF6B35;
const COLOR_AXIS  = 0x555555;

// ── Three.js setup ────────────────────────────────────────────────────────────
const canvas = document.getElementById('three-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 0); // transparent — grid shows through from CSS

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
camera.position.set(-8, 145, 55);
camera.lookAt(5, 100, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(5, 100, 0);

// Resize handler
function onResize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(onResize).observe(canvas);
onResize();

// ── Axis helper ───────────────────────────────────────────────────────────────
function buildAxes(T, S0) {
  const group = new THREE.Group();
  const mat = new THREE.LineBasicMaterial({ color: COLOR_AXIS });

  // X axis — time
  const xGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, S0, 0),
    new THREE.Vector3(TIME_SCALE, S0, 0),
  ]);
  group.add(new THREE.Line(xGeo, mat));

  // Y axis — price
  const yGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, S0 * 0.4, 0),
    new THREE.Vector3(0, S0 * 1.7, 0),
  ]);
  group.add(new THREE.Line(yGeo, mat));

  // Origin dot
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.15, 8, 8),
    new THREE.MeshBasicMaterial({ color: COLOR_AXIS }),
  );
  dot.position.set(0, S0, 0);
  group.add(dot);

  return group;
}

// ── Build path geometries ─────────────────────────────────────────────────────
function buildPathObjects(paths, t, color, zBase, T) {
  const group = new THREE.Group();
  const nShow = Math.min(paths.length, N_DISPLAY);
  const tScale = TIME_SCALE / T;

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
    geo.setDrawRange(0, 0);   // start invisible; animation reveals it

    const mat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.50,
    });

    group.add(new THREE.Line(geo, mat));
  }
  return group;
}

// ── Floor grid plane ──────────────────────────────────────────────────────────
function buildFloor(S0) {
  const totalZ = (N_DISPLAY - 1) * Z_SPACING;
  const zCenter = (MJD_Z_BASE + totalZ) / 2;   // midpoint between both fans
  const zSpan   = Math.abs(MJD_Z_BASE) + totalZ + 4;

  const geo = new THREE.PlaneGeometry(TIME_SCALE + 2, zSpan, 10, 10);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x2a2a2a,
    transparent: true,
    opacity: 0.35,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;             // lay flat (XZ plane)
  mesh.position.set(TIME_SCALE / 2, S0 * 0.38, zCenter);
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

  // GBM fan: Z = 0 … +(N-1)*spacing
  // MJD fan: Z = MJD_Z_BASE … MJD_Z_BASE-(N-1)*spacing  (negative side, clear gap)
  const gbmZBase = 0;
  const mjdZBase = MJD_Z_BASE;

  // Camera: elevated, offset in +Z so time flows left-to-right diagonally
  const cx = TIME_SCALE * 0.3;
  const cy = S0 * 1.55;
  const cz = S0 * 0.30 + 30;
  camera.position.set(cx, cy, cz);
  controls.target.set(TIME_SCALE * 0.5, S0, 0);
  camera.lookAt(TIME_SCALE * 0.5, S0, 0);

  floorMesh = buildFloor(S0);
  axisGroup = buildAxes(T, S0);
  gbmGroup  = buildPathObjects(gbmPaths, t, COLOR_GBM, gbmZBase, T);
  mjdGroup  = buildPathObjects(mjdPaths, t, COLOR_MJD, mjdZBase, T);

  scene.add(floorMesh, axisGroup, gbmGroup, mjdGroup);
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

function setTableRow(id, callVal, putVal) {
  const fmt = x => x.toFixed(4);
  document.getElementById(id + '-call').textContent = fmt(callVal);
  document.getElementById(id + '-put').textContent  = fmt(putVal);
}

// ── Run button ────────────────────────────────────────────────────────────────
document.getElementById('btn-run').addEventListener('click', () => {
  const p = getParams();

  // Simulate paths
  const { t, paths: gbmPaths } = simulateGBMPaths(p.S0, p.r, p.sigma, p.T, 252, p.nPaths);
  const { paths: mjdPaths }    = simulateMJDPaths(p.S0, p.r, p.sigma, p.T, 252, p.nPaths,
                                                   p.lam, p.muJ, p.sigmaJ);

  // Prices
  const mc  = { call: mcOptionPrice(gbmPaths, p.K, p.r, p.T, 'call'),
                 put:  mcOptionPrice(gbmPaths, p.K, p.r, p.T, 'put') };
  const mj  = { call: mcOptionPrice(mjdPaths, p.K, p.r, p.T, 'call'),
                 put:  mcOptionPrice(mjdPaths, p.K, p.r, p.T, 'put') };
  const bs  = { call: bsPrice(p.S0, p.K, p.r, p.sigma, p.T, 'call'),
                 put:  bsPrice(p.S0, p.K, p.r, p.sigma, p.T, 'put') };
  const mr  = { call: mertonPrice(p.S0, p.K, p.r, p.sigma, p.T, p.lam, p.muJ, p.sigmaJ, 'call'),
                 put:  mertonPrice(p.S0, p.K, p.r, p.sigma, p.T, p.lam, p.muJ, p.sigmaJ, 'put') };

  setTableRow('row-mc',     mc.call,  mc.put);
  setTableRow('row-mjd',    mj.call,  mj.put);
  setTableRow('row-bs',     bs.call,  bs.put);
  setTableRow('row-merton', mr.call,  mr.put);

  // 3-D animation
  startAnimation(gbmPaths, mjdPaths, t, p.T, p.S0);

  // Re-render Greeks if panel is open
  if (document.getElementById('greeks-panel').classList.contains('visible')) {
    drawGreeks(p);
  }
});

// ── Greeks button ─────────────────────────────────────────────────────────────
document.getElementById('btn-greeks').addEventListener('click', () => {
  const panel = document.getElementById('greeks-panel');
  const btn   = document.getElementById('btn-greeks');
  const open  = panel.classList.toggle('visible');
  btn.classList.toggle('active', open);
  btn.textContent = open ? 'Greeks ▲' : 'Greeks ▼';
  if (open) drawGreeks(getParams());
});

// ── Greeks canvas drawing ─────────────────────────────────────────────────────
function drawGreeks(p) {
  const N_SPOT = 40;
  const S0arr = Array.from({ length: N_SPOT }, (_, i) =>
    p.S0 * (0.70 + i * 0.60 / (N_SPOT - 1)));

  const bs = bsGreeks(S0arr, p.K, p.r, p.sigma, p.T, 'call');
  const mc = mcGreeksRange(S0arr, p.K, p.r, p.sigma, p.T, 2000, 'call');

  const greeks = [
    { key: 'delta', label: 'Delta (Δ)',  id: 'canvas-delta' },
    { key: 'gamma', label: 'Gamma (Γ)',  id: 'canvas-gamma' },
    { key: 'vega',  label: 'Vega (ν)',   id: 'canvas-vega'  },
    { key: 'theta', label: 'Theta (Θ)',  id: 'canvas-theta' },
  ];

  for (const { key, id } of greeks) {
    const canvas = document.getElementById(id);
    drawGreekChart(canvas, S0arr, bs[key], mc[key], p.S0, p.K);
  }
}

function drawGreekChart(canvas, xArr, bsArr, mcArr, spot, strike) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth  * dpr;
  const H = canvas.offsetHeight * dpr;
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

  // Draw line helper
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
    const y = toY(val);
    ctx.fillText(val.toFixed(3), (PAD.left - 3) * dpr, y + 3 * dpr);
  }

  // X axis ticks (3 labels)
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

// ── Utility ───────────────────────────────────────────────────────────────────
function id(s) { return document.getElementById(s); }

// ── Stock ticker fetch ────────────────────────────────────────────────────────
document.getElementById('btn-fetch').addEventListener('click', async () => {
  const ticker = document.getElementById('ticker').value.trim().toUpperCase();
  const status = document.getElementById('ticker-status');
  if (!ticker) { status.textContent = 'Enter a ticker symbol.'; status.className = 'error'; return; }

  status.textContent = `Fetching ${ticker}…`;
  status.className = '';

  try {
    const res = await fetch(`/api/stock/${ticker}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');

    // Populate S0
    document.getElementById('S0').value = data.price;

    // Populate sigma if IV available
    if (data.iv && data.iv > 0) {
      document.getElementById('sigma').value = data.iv.toFixed(4);
      status.textContent = `${data.ticker}: $${data.price}  |  σ = ${(data.iv * 100).toFixed(1)}%`;
    } else {
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
  });
});
