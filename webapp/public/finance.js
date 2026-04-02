// finance.js — Financial math ported from utils.py
// Pure JS, no dependencies.

// ── Random number utilities ───────────────────────────────────────────────────

export function randn_bm() {
  // Box-Muller transform → N(0,1)
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function poissonSample(lambda) {
  // Knuth algorithm — exact for small lambda
  if (lambda === 0) return 0;
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= Math.random(); } while (p > L);
  return k - 1;
}

// Precomputed log-factorials for k = 0..59; Stirling approximation beyond.
const LOG_FACT = (() => {
  const t = [0];
  for (let i = 1; i <= 60; i++) t.push(t[i - 1] + Math.log(i));
  return t;
})();

function logFactorial(k) {
  if (k <= 60) return LOG_FACT[k];
  return k * Math.log(k) - k + 0.5 * Math.log(2 * Math.PI * k); // Stirling
}

export function poissonPMF(k, lambda) {
  if (lambda <= 0) return k === 0 ? 1 : 0;
  return Math.exp(-lambda + k * Math.log(lambda) - logFactorial(k));
}

// ── Normal distribution ───────────────────────────────────────────────────────

export function normalPDF(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

export function normalCDF(x) {
  // Abramowitz & Stegun 26.2.17 — accurate to ~5 decimal places
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = normalPDF(x);
  const p = d * t * (0.319381530
    + t * (-0.356563782
    + t * (1.781477937
    + t * (-1.821255978
    + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

// ── GBM path simulation ───────────────────────────────────────────────────────

export function simulateGBMPaths(S0, r, sigma, T, steps, nPaths) {
  const dt = T / steps;
  const t = Array.from({ length: steps + 1 }, (_, i) => i * dt);
  const paths = [];
  const drift = (r - 0.5 * sigma * sigma) * dt;
  const diffCoeff = sigma * Math.sqrt(dt);

  for (let p = 0; p < nPaths; p++) {
    const path = new Float64Array(steps + 1);
    path[0] = S0;
    for (let s = 0; s < steps; s++) {
      path[s + 1] = path[s] * Math.exp(drift + diffCoeff * randn_bm());
    }
    paths.push(path);
  }
  return { t, paths };
}

// ── Merton Jump Diffusion path simulation ─────────────────────────────────────

export function simulateMJDPaths(S0, r, sigma, T, steps, nPaths, lam, muJ, sigmaJ) {
  const dt = T / steps;
  const t = Array.from({ length: steps + 1 }, (_, i) => i * dt);
  const kBar = Math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1;
  const drift = (r - lam * kBar - 0.5 * sigma * sigma) * dt;
  const diffCoeff = sigma * Math.sqrt(dt);
  const lamDt = lam * dt;
  const paths = [];

  for (let p = 0; p < nPaths; p++) {
    const path = new Float64Array(steps + 1);
    path[0] = S0;
    for (let s = 0; s < steps; s++) {
      const N = poissonSample(lamDt);
      const jumpMean = N * muJ;
      const jumpStd = N > 0 ? Math.sqrt(N) * sigmaJ * randn_bm() : 0;
      path[s + 1] = path[s] * Math.exp(drift + diffCoeff * randn_bm() + jumpMean + jumpStd);
    }
    paths.push(path);
  }
  return { t, paths };
}

// ── Monte Carlo option price ──────────────────────────────────────────────────

export function mcOptionPrice(paths, K, r, T, type = 'call') {
  const disc = Math.exp(-r * T);
  let sum = 0;
  for (const path of paths) {
    const ST = path[path.length - 1];
    sum += type === 'call' ? Math.max(ST - K, 0) : Math.max(K - ST, 0);
  }
  return disc * sum / paths.length;
}

// ── Black-Scholes price ───────────────────────────────────────────────────────

export function bsPrice(S0, K, r, sigma, T, type = 'call') {
  const sqrtT = Math.sqrt(T);
  const d1 = (Math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
  const d2 = d1 - sigma * sqrtT;
  if (type === 'call') {
    return S0 * normalCDF(d1) - K * Math.exp(-r * T) * normalCDF(d2);
  } else {
    return K * Math.exp(-r * T) * normalCDF(-d2) - S0 * normalCDF(-d1);
  }
}

// ── Merton analytical price ───────────────────────────────────────────────────

export function mertonPrice(S0, K, r, sigma, T, lam, muJ, sigmaJ, type = 'call', nTerms = 50) {
  const kBar = Math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1;
  let price = 0;
  for (let n = 0; n < nTerms; n++) {
    const w = poissonPMF(n, lam * T);
    if (w < 1e-12) break;
    const sigmaN = Math.sqrt(sigma * sigma + n * sigmaJ * sigmaJ / T);
    const rN = r - lam * kBar + n * (muJ + 0.5 * sigmaJ * sigmaJ) / T;
    price += w * bsPrice(S0, K, rN, sigmaN, T, type);
  }
  return price;
}

// ── Black-Scholes Greeks (vectorised over S0arr) ──────────────────────────────

export function bsGreeks(S0arr, K, r, sigma, T, type = 'call') {
  const sqrtT = Math.sqrt(T);
  const delta = [], gamma = [], vega = [], theta = [];

  for (const S0 of S0arr) {
    const d1 = (Math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const d2 = d1 - sigma * sqrtT;
    const nd1 = normalPDF(d1);

    gamma.push(nd1 / (S0 * sigma * sqrtT));
    vega.push(S0 * nd1 * sqrtT / 100);

    if (type === 'call') {
      delta.push(normalCDF(d1));
      theta.push((-(S0 * nd1 * sigma) / (2 * sqrtT)
        - r * K * Math.exp(-r * T) * normalCDF(d2)) / 365);
    } else {
      delta.push(normalCDF(d1) - 1);
      theta.push((-(S0 * nd1 * sigma) / (2 * sqrtT)
        + r * K * Math.exp(-r * T) * normalCDF(-d2)) / 365);
    }
  }
  return { delta, gamma, vega, theta };
}

// ── MC Greeks via finite differences (common random numbers) ──────────────────

export function mcGreeksRange(S0arr, K, r, sigma, T, nPaths = 2000, type = 'call') {
  const dSig = 0.001;
  const dT = 1 / 365;
  const sqrtT = Math.sqrt(T);
  const sqrtTm = T > dT ? Math.sqrt(T - dT) : sqrtT;

  // Common random numbers — single shared Z array
  const Z = Array.from({ length: nPaths }, randn_bm);

  const disc = Math.exp(-r * T);
  const discTm = Math.exp(-r * (T - dT));

  const payoff = (ST, tau) => {
    const d = Math.exp(-r * tau);
    if (type === 'call') return d * Math.max(ST - K, 0);
    return d * Math.max(K - ST, 0);
  };

  const meanPayoff = (s, sig, sqT, d) => {
    let sum = 0;
    const drift = (r - 0.5 * sig * sig) * (sqT * sqT);
    for (let i = 0; i < nPaths; i++) {
      const ST = s * Math.exp(drift + sig * sqT * Z[i]);
      sum += type === 'call' ? Math.max(ST - K, 0) : Math.max(K - ST, 0);
    }
    return d * sum / nPaths;
  };

  const delta = [], gamma = [], vegaArr = [], thetaArr = [];

  for (const S0 of S0arr) {
    const dS = S0 * 0.01;

    const pMid  = meanPayoff(S0,      sigma,       sqrtT,  disc);
    const pUpS  = meanPayoff(S0 + dS, sigma,       sqrtT,  disc);
    const pDnS  = meanPayoff(S0 - dS, sigma,       sqrtT,  disc);
    const pUpSg = meanPayoff(S0,      sigma + dSig, sqrtT, disc);
    const pDnSg = meanPayoff(S0,      sigma - dSig, sqrtT, disc);

    delta.push((pUpS - pDnS) / (2 * dS));
    gamma.push((pUpS - 2 * pMid + pDnS) / (dS * dS));
    vegaArr.push((pUpSg - pDnSg) / (2 * dSig) / 100);

    if (T > dT) {
      const pTm = meanPayoff(S0, sigma, sqrtTm, discTm);
      thetaArr.push(pTm - pMid);
    } else {
      thetaArr.push(0);
    }
  }

  return { delta, gamma, vega: vegaArr, theta: thetaArr };
}
