"""
ml_vol_predictor.py — Standalone ML volatility predictor with option pricing.

Pipeline
--------
1. Pull 2 years of daily OHLCV data via yfinance.
2. Engineer features: 21-day realized vol, log returns, RSI-14, MA-20/50,
   Bollinger Bands (20d ±2σ width and %B position).
3. Train a Random Forest with TimeSeriesSplit cross-validation to predict
   the next 21-day forward realized volatility.  Print CV scores + feature
   importances.
4. Feed predicted vol, 2y historical vol, and ATM implied vol into the
   existing black_scholes_price() and monte_carlo_option_price() from utils.py.
5. Render a single dark-themed figure:
     • top-left  : MC vs BS call prices for all three vol regimes
     • top-right : MC vs BS put  prices for all three vol regimes
     • bottom-left : predicted vol vs actual rolling vol over time
     • bottom-right: Random Forest feature importances

Usage
-----
    python ml_vol_predictor.py          # defaults to SPY
    python ml_vol_predictor.py AAPL
    python ml_vol_predictor.py TSLA put
"""

import datetime
import sys
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from utils import black_scholes_price, monte_carlo_option_price, simulate_gbm_paths
from vol_ml import train_vol_model, get_implied_vol

# ── Constants ─────────────────────────────────────────────────────────────────
R        = 0.05     # risk-free rate
T        = 1.0      # time-to-maturity (years)
N_PATHS  = 2_000    # MC paths per scenario
STEPS    = 252      # daily steps

COLOR_MC   = '#FFD700'   # gold
COLOR_BS   = '#00CED1'   # dark turquoise
COLOR_PRED = '#FF6B35'   # orange
COLOR_ACT  = '#00FA9A'   # spring green


# ── Dark theme ────────────────────────────────────────────────────────────────

def _set_dark_theme():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.labelcolor':    'white',
        'xtick.color':        'white',
        'ytick.color':        'white',
        'axes.edgecolor':     '#555555',
        'figure.facecolor':   '#1a1a1a',
        'axes.facecolor':     '#222222',
        'grid.color':         '#444444',
        'grid.alpha':         0.35,
        'savefig.facecolor':  '#1a1a1a',
    })


# ── Option pricing for a given vol ────────────────────────────────────────────

def price_scenario(S0: float, K: float, vol: float) -> dict:
    _, paths = simulate_gbm_paths(S0, R, vol, T, STEPS, N_PATHS)
    return {
        'mc_call': monte_carlo_option_price(paths, K, R, T, 'call'),
        'mc_put':  monte_carlo_option_price(paths, K, R, T, 'put'),
        'bs_call': black_scholes_price(S0, K, R, vol, T, 'call'),
        'bs_put':  black_scholes_price(S0, K, R, vol, T, 'put'),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _bar_subplot(ax, scenario_labels, mc_vals, bs_vals, title: str):
    """Render one set of MC vs BS bars for the given vol scenarios."""
    x     = np.arange(len(scenario_labels))
    width = 0.35

    bars_mc = ax.bar(x - width / 2, mc_vals, width,
                     label='Monte Carlo', color=COLOR_MC, alpha=0.85)
    bars_bs = ax.bar(x + width / 2, bs_vals, width,
                     label='Black-Scholes', color=COLOR_BS, alpha=0.85)

    for bar in list(bars_mc) + list(bars_bs):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8, color='white')

    ax.set_title(title, color='white', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=8)
    ax.set_ylabel('Option Price ($)', color='white')
    ax.legend(fontsize=8, framealpha=0.4)
    ax.grid(axis='y')


def build_figure(ticker: str, S0: float, K: float,
                 scenarios: dict, feat_df: pd.DataFrame,
                 feat_imp: dict, cv_r2: list,
                 pred_vol: float, hist_vol: float):
    """
    Four-panel figure:
      [0,0] Call option prices  (bar)
      [0,1] Put  option prices  (bar)
      [1,0] Predicted vs actual rolling vol  (line)
      [1,1] RF feature importances  (horizontal bar)
    """
    _set_dark_theme()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'{ticker}  ·  ML Volatility Predictor  ·  '
        f'RF Predicted σ={pred_vol:.3f}  ·  Historical σ={hist_vol:.3f}  ·  '
        f'Mean CV R²={np.mean(cv_r2):+.3f}',
        fontsize=12, color='white', y=0.995,
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.95, bottom=0.07)

    s_labels = list(scenarios.keys())

    # ── [0,0] Call prices ─────────────────────────────────────────────────────
    ax_call = fig.add_subplot(gs[0, 0])
    _bar_subplot(
        ax_call, s_labels,
        [scenarios[l]['mc_call'] for l in s_labels],
        [scenarios[l]['bs_call'] for l in s_labels],
        f'Call Option Prices  (S₀={S0:.0f}, K={K:.0f}, r={R:.0%}, T={T}y)',
    )

    # ── [0,1] Put prices ──────────────────────────────────────────────────────
    ax_put = fig.add_subplot(gs[0, 1])
    _bar_subplot(
        ax_put, s_labels,
        [scenarios[l]['mc_put'] for l in s_labels],
        [scenarios[l]['bs_put'] for l in s_labels],
        f'Put Option Prices  (S₀={S0:.0f}, K={K:.0f}, r={R:.0%}, T={T}y)',
    )

    # ── [1,0] Predicted vs actual rolling vol ─────────────────────────────────
    ax_vol = fig.add_subplot(gs[1, 0])
    if feat_df is not None:
        actual = feat_df['rv21'].dropna()
        ax_vol.plot(actual.index, actual.values,
                    color=COLOR_ACT, lw=1.2, alpha=0.8, label='Actual 21d RV')
        cv_pred = feat_df['fold_pred'].dropna()
        ax_vol.scatter(cv_pred.index, cv_pred.values,
                       color=COLOR_PRED, s=6, alpha=0.6, label='RF CV predictions')
    ax_vol.axhline(pred_vol, color=COLOR_PRED, lw=1.5, linestyle='--',
                   label=f'Latest RF forecast  σ={pred_vol:.3f}')
    ax_vol.axhline(hist_vol, color='#AAAAAA', lw=1.0, linestyle=':',
                   label=f'2y historical vol  σ={hist_vol:.3f}')
    ax_vol.set_title('Predicted vs Actual Rolling Volatility', color='white', fontsize=11)
    ax_vol.set_xlabel('Date')
    ax_vol.set_ylabel('Annualised Volatility')
    ax_vol.legend(fontsize=7.5, framealpha=0.4)
    ax_vol.grid(True)
    ax_vol.tick_params(axis='x', rotation=30)

    # ── [1,1] Feature importances ─────────────────────────────────────────────
    ax_fi = fig.add_subplot(gs[1, 1])

    sorted_feats = sorted(feat_imp.items(), key=lambda kv: kv[1])
    fi_names = [kv[0] for kv in sorted_feats]
    fi_vals  = [kv[1] for kv in sorted_feats]

    y_pos = list(range(len(fi_names)))
    ax_fi.barh(y_pos, fi_vals, color=COLOR_PRED, alpha=0.85)
    ax_fi.set_yticks(y_pos)
    ax_fi.set_yticklabels(fi_names, fontsize=9)
    ax_fi.set_title('RF Feature Importances', color='white', fontsize=11)
    ax_fi.set_xlabel('Importance')
    ax_fi.grid(axis='x')

    # Annotate CV R² per fold in a small text box
    cv_text = '\n'.join(
        [f'Fold {i+1}: {r2:+.3f}' for i, r2 in enumerate(cv_r2)]
        + [f'Mean:    {np.mean(cv_r2):+.3f}']
    )
    ax_fi.text(0.97, 0.03, cv_text, transform=ax_fi.transAxes,
               fontsize=7, color='#AAAAAA', va='bottom', ha='right',
               bbox=dict(facecolor='#333333', alpha=0.6, pad=3))

    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ticker      = sys.argv[1].upper() if len(sys.argv) > 1 else 'SPY'
    option_type = sys.argv[2].lower() if len(sys.argv) > 2 else 'call'
    if option_type not in ('call', 'put'):
        print(f"Unknown option type '{option_type}'; using 'call'.")
        option_type = 'call'

    # ── Download data ─────────────────────────────────────────────────────────
    print(f"\nDownloading 5y of {ticker} daily data …")
    raw = yf.download(ticker, period='5y', auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw['Close'].squeeze()
    S0    = float(close.iloc[-1])
    K     = round(S0 / 5) * 5
    print(f"  Current price: ${S0:.2f}  |  Using K=${K:.0f}")

    # ── Train model ───────────────────────────────────────────────────────────
    print(f"\nTraining Random Forest  (TimeSeriesSplit, 5 folds) …")
    pred_vol, hist_vol, r2, feat_imp = train_vol_model(ticker, period='5y')
    cv_r2   = [r2]
    feat_df = None
    print(f"\n  RF predicted vol (next 21d): {pred_vol:.4f}")
    print(f"  2y historical vol:           {hist_vol:.4f}")

    # ── Implied vol ───────────────────────────────────────────────────────────
    print(f"\nFetching ATM implied vol for {ticker} …")
    impl_vol = get_implied_vol(ticker, S0, T, option_type)
    if impl_vol is None:
        print("  Implied vol unavailable — substituting historical vol.")
        impl_vol = hist_vol

    # ── Feature importances table ─────────────────────────────────────────────
    print("\nFeature Importances:")
    for name, imp in sorted(feat_imp.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<12s}  {imp:.4f}")

    # ── Price all three vol scenarios ─────────────────────────────────────────
    print(f"\nPricing scenarios (S₀={S0:.2f}, K={K:.0f}, r={R:.0%}, T={T}y, "
          f"{N_PATHS:,} paths) …")
    scenarios = {}
    vol_map = {
        f'RF Predicted\n(σ={pred_vol:.3f})': pred_vol,
        f'Historical\n(σ={hist_vol:.3f})':   hist_vol,
        f'Implied\n(σ={impl_vol:.3f})':      impl_vol,
    }
    for label, vol in vol_map.items():
        res = price_scenario(S0, K, vol)
        scenarios[label] = res
        clean = label.replace('\n', ' ')
        print(f"  {clean:<32s}  "
              f"MC call={res['mc_call']:.4f}  BS call={res['bs_call']:.4f}  "
              f"MC put={res['mc_put']:.4f}  BS put={res['bs_put']:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nRendering figure …")
    build_figure(ticker, S0, K, scenarios, feat_df, feat_imp,
                 cv_r2, pred_vol, hist_vol)


if __name__ == '__main__':
    main()
