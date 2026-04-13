"""
server.py — FastAPI backend for QuantSim Pro.

Endpoints
---------
POST /api/simulate      Run MC (GBM+MJD) + BS + Merton pricing, sample paths, convergence
POST /api/greeks        Compute Greeks over a spot price range
GET  /api/live-market   Fetch live market data for a ticker
POST /api/predict-vol   Run ML volatility predictor (Random Forest)

Run with:
    cd Monte-Carlo-Option-Pricing-Simulator
    uvicorn server:app --reload --port 8000
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import (
    simulate_gbm_paths,
    simulate_mjd_paths,
    monte_carlo_option_price,
    black_scholes_price,
    merton_price,
    black_scholes_greeks,
    mc_greeks_range,
)
from vol_ml import train_vol_model, get_implied_vol

app = FastAPI(title="QuantSim Pro API", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────


class SimParams(BaseModel):
    S0: float = Field(100.0, gt=0)
    K: float = Field(100.0, gt=0)
    r: float = Field(0.05, ge=0)
    sigma: float = Field(0.20, gt=0)
    T: float = Field(1.0, gt=0)
    nPaths: int = Field(1000, ge=100, le=20000)
    lam: float = Field(1.0, ge=0)
    muJ: float = Field(-0.10)
    sigmaJ: float = Field(0.15, gt=0)
    optionType: str = Field("call")
    nSamplePaths: int = Field(30, ge=5, le=100)


class GreeksParams(BaseModel):
    S0: float = Field(100.0, gt=0)
    K: float = Field(100.0, gt=0)
    r: float = Field(0.05, ge=0)
    sigma: float = Field(0.20, gt=0)
    T: float = Field(1.0, gt=0)
    optionType: str = Field("call")
    nPaths: int = Field(5000, ge=500, le=20000)


class VolPredictParams(BaseModel):
    ticker: str = Field("SPY")
    S0: Optional[float] = None
    r: float = Field(0.05)
    T: float = Field(1.0)
    optionType: str = Field("call")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe(v) -> float:
    """Convert numpy scalar to Python float, coercing NaN/inf to 0."""
    try:
        f = float(v)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return 0.0


def _safe_list(arr) -> list:
    return [_safe(v) for v in arr]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/api/simulate")
def simulate(p: SimParams):
    ot = p.optionType.lower()
    ot2 = "put" if ot == "call" else "call"

    # Full MC simulations
    _, gbm_paths = simulate_gbm_paths(p.S0, p.r, p.sigma, p.T, 252, p.nPaths)
    t_arr, mjd_paths = simulate_mjd_paths(
        p.S0, p.r, p.sigma, p.T, 252, p.nPaths, p.lam, p.muJ, p.sigmaJ
    )

    # Pricing — primary option type
    mc_gbm_c = _safe(monte_carlo_option_price(gbm_paths, p.K, p.r, p.T, "call"))
    mc_gbm_p = _safe(monte_carlo_option_price(gbm_paths, p.K, p.r, p.T, "put"))
    mc_mjd_c = _safe(monte_carlo_option_price(mjd_paths, p.K, p.r, p.T, "call"))
    mc_mjd_p = _safe(monte_carlo_option_price(mjd_paths, p.K, p.r, p.T, "put"))
    bs_c = _safe(black_scholes_price(p.S0, p.K, p.r, p.sigma, p.T, "call"))
    bs_p = _safe(black_scholes_price(p.S0, p.K, p.r, p.sigma, p.T, "put"))
    me_c = _safe(merton_price(p.S0, p.K, p.r, p.sigma, p.T, p.lam, p.muJ, p.sigmaJ, "call"))
    me_p = _safe(merton_price(p.S0, p.K, p.r, p.sigma, p.T, p.lam, p.muJ, p.sigmaJ, "put"))

    # Sample paths for 3D viz (subsample)
    n_sample = min(p.nSamplePaths, p.nPaths)
    # Downsample time steps to 63 points for lighter payload
    step_idx = np.linspace(0, 252, 63, dtype=int)
    t_sampled = t_arr[step_idx].tolist()
    gbm_sample = gbm_paths[:n_sample][:, step_idx].tolist()
    mjd_sample = mjd_paths[:n_sample][:, step_idx].tolist()

    # Convergence data — rerun at increasing path counts with fixed seed
    path_counts = [100, 500, 1000, 2500, 5000]
    path_counts = sorted(set(c for c in path_counts if c <= p.nPaths))
    if p.nPaths not in path_counts:
        path_counts.append(p.nPaths)

    convergence = []
    for n_p in path_counts:
        _, g = simulate_gbm_paths(p.S0, p.r, p.sigma, p.T, 252, n_p)
        _, m = simulate_mjd_paths(
            p.S0, p.r, p.sigma, p.T, 252, n_p, p.lam, p.muJ, p.sigmaJ
        )
        convergence.append({
            "paths": n_p,
            "gbm": _safe(monte_carlo_option_price(g, p.K, p.r, p.T, ot)),
            "mjd": _safe(monte_carlo_option_price(m, p.K, p.r, p.T, ot)),
            "bs": bs_c if ot == "call" else bs_p,
        })

    # Sensitivity table — bump each param and measure BS price change
    def _bs_both(s0, k, r, sig, t):
        return (
            _safe(black_scholes_price(s0, k, r, sig, t, "call")),
            _safe(black_scholes_price(s0, k, r, sig, t, "put")),
        )

    sensitivity = []
    bumps = [
        ("Spot Price (S₀)", f"{p.S0:.2f}", p.S0 + 1, p.K, p.r, p.sigma, p.T),
        ("Strike (K)", f"{p.K:.2f}", p.S0, p.K + 1, p.r, p.sigma, p.T),
        ("Volatility (σ)", f"{p.sigma * 100:.2f}%", p.S0, p.K, p.r, p.sigma + 0.01, p.T),
        ("Risk-Free (r)", f"{p.r * 100:.2f}%", p.S0, p.K, p.r + 0.01, p.sigma, p.T),
        ("Tenor (T)", f"{p.T:.2f} Y", p.S0, p.K, p.r, p.sigma, p.T + 0.1),
    ]
    for label, val_str, s0, k, r, sig, t in bumps:
        bumped_c, bumped_p = _bs_both(s0, k, r, sig, t)
        sensitivity.append({
            "param": label,
            "value": val_str,
            "call": bs_c,
            "put": bs_p,
            "d_call": _safe(bumped_c - bs_c),
            "d_put": _safe(bumped_p - bs_p),
        })

    return {
        "pricing": {
            "mc_gbm": {"call": mc_gbm_c, "put": mc_gbm_p},
            "mc_mjd": {"call": mc_mjd_c, "put": mc_mjd_p},
            "bs": {"call": bs_c, "put": bs_p},
            "merton": {"call": me_c, "put": me_p},
        },
        "convergence": convergence,
        "sample_paths": {
            "t": t_sampled,
            "gbm": gbm_sample,
            "mjd": mjd_sample,
        },
        "sensitivity": sensitivity,
    }


@app.post("/api/greeks")
def greeks(p: GreeksParams):
    spot_range = np.linspace(p.S0 * 0.70, p.S0 * 1.30, 30)
    spot_list = spot_range.tolist()

    bs_g = black_scholes_greeks(spot_range, p.K, p.r, p.sigma, p.T, p.optionType)
    mc_g = mc_greeks_range(spot_list, p.K, p.r, p.sigma, p.T, p.nPaths, p.optionType)

    current_bs = black_scholes_greeks(p.S0, p.K, p.r, p.sigma, p.T, p.optionType)

    return {
        "spot_range": spot_list,
        "bs": {k: _safe_list(v) for k, v in bs_g.items()},
        "mc": {k: _safe_list(v) for k, v in mc_g.items()},
        "current": {k: _safe(v) for k, v in current_bs.items()},
        "S0": p.S0,
        "K": p.K,
    }


@app.get("/api/live-market")
def live_market(ticker: str = Query(..., min_length=1, max_length=10)):
    try:
        tk = yf.Ticker(ticker.upper())
        hist = tk.history(period="6mo")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        price = float(hist["Close"].iloc[-1])
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        hist_vol = float(log_ret.rolling(21).std().iloc[-1] * np.sqrt(252))
        return {
            "ticker": ticker.upper(),
            "price": round(price, 2),
            "hist_vol": round(hist_vol, 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-vol")
def predict_vol(p: VolPredictParams):
    try:
        predicted_vol, hist_vol, r2, feat_importances = train_vol_model(
            p.ticker.upper(), period="5y"
        )
        implied_vol = get_implied_vol(p.ticker.upper(), p.S0, p.T, p.optionType)
        return {
            "ticker": p.ticker.upper(),
            "predicted_vol": _safe(predicted_vol),
            "hist_vol": _safe(hist_vol),
            "implied_vol": _safe(implied_vol) if implied_vol else None,
            "r2": _safe(r2),
            "feat_importances": {k: _safe(v) for k, v in feat_importances.items()},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
