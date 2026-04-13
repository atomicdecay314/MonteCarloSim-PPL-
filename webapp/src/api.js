/**
 * api.js — All API calls to the QuantSim Pro FastAPI backend at localhost:8000.
 * In dev mode, Vite proxies /api/* to http://localhost:8000.
 */

const BASE = '/api'

async function _post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

async function _get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

/**
 * Run a full simulation: GBM + MJD paths, BS + Merton pricing,
 * convergence series, sample paths for 3D, sensitivity table.
 *
 * @param {Object} params
 * @param {number} params.S0         Spot price
 * @param {number} params.K          Strike price
 * @param {number} params.r          Risk-free rate (decimal)
 * @param {number} params.sigma      Volatility (decimal)
 * @param {number} params.T          Time to expiry (years)
 * @param {number} params.nPaths     Number of MC paths
 * @param {number} params.lam        MJD jump intensity λ
 * @param {number} params.muJ        MJD jump mean μⱼ
 * @param {number} params.sigmaJ     MJD jump vol σⱼ
 * @param {string} params.optionType 'call' | 'put'
 * @param {number} [params.nSamplePaths=30] Paths to return for 3D viz
 * @returns {Promise<{pricing, convergence, sample_paths, sensitivity}>}
 */
export async function runSimulation(params) {
  return _post('/simulate', params)
}

/**
 * Compute Black-Scholes and MC Greeks over a spot price range.
 *
 * @param {Object} params  Same structure as SimParams (S0, K, r, sigma, T, optionType, nPaths)
 * @returns {Promise<{spot_range, bs, mc, current, S0, K}>}
 */
export async function fetchGreeks(params) {
  return _post('/greeks', params)
}

/**
 * Fetch live market data for a ticker: price + 21-day hist vol.
 *
 * @param {string} ticker  Yahoo Finance ticker symbol
 * @returns {Promise<{ticker, price, hist_vol}>}
 */
export async function fetchLiveMarket(ticker) {
  return _get(`/live-market?ticker=${encodeURIComponent(ticker.toUpperCase())}`)
}

/**
 * Run the Random Forest ML vol predictor for a ticker.
 *
 * @param {Object} params
 * @param {string} params.ticker
 * @param {number} [params.S0]
 * @param {number} [params.r]
 * @param {number} [params.T]
 * @param {string} [params.optionType]
 * @returns {Promise<{ticker, predicted_vol, hist_vol, implied_vol, r2, feat_importances}>}
 */
export async function predictVol(params) {
  return _post('/predict-vol', params)
}
