"""
greeks.py — Greeks visualization using real market data from yfinance.

Plots Delta, Gamma, Vega, and Theta vs stock price, comparing analytical
Black-Scholes values against Monte Carlo finite-difference estimates.

Usage:
    python greeks.py [TICKER]          # default: SPY
    python greeks.py AAPL call
    python greeks.py TSLA put
"""

import datetime
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yfinance as yf

from utils import black_scholes_greeks, mc_greeks_range

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_TICKER = 'SPY'
R = 0.045            # risk-free rate (approximate US short-term Treasury)
N_PATHS = 10_000     # MC paths per Greek calculation
N_SPOT_POINTS = 40   # number of spot levels to evaluate
SPOT_RANGE = 0.30    # ± 30 % around current spot


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_dark_theme():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.edgecolor': '#555555',
        'figure.facecolor': '#1a1a1a',
        'axes.facecolor': '#222222',
        'grid.color': '#444444',
        'grid.alpha': 0.4,
        'savefig.facecolor': '#1a1a1a',
    })


def fetch_market_params(ticker_symbol):
    """
    Fetch spot price, implied vol, ATM strike, and time-to-maturity from
    yfinance for the nearest ~30-day expiration.

    Falls back gracefully when data is unavailable.
    Returns (S0, sigma, T, K, expiration_str).
    """
    print(f"Fetching market data for {ticker_symbol} …")
    ticker = yf.Ticker(ticker_symbol)

    # ── Spot price ────────────────────────────────────────────────────────────
    hist = ticker.history(period='2d')
    if hist.empty:
        print(f"  Warning: no price data for {ticker_symbol}. Using synthetic defaults.")
        return 100.0, 0.20, 30 / 365, 100.0, 'N/A'

    S0 = float(hist['Close'].iloc[-1])

    # ── Nearest ~30-day expiration and ATM strike ─────────────────────────────
    T = 30 / 365
    K = round(S0)
    exp_str = 'N/A'
    chain_calls = None

    try:
        expirations = ticker.options
        if expirations:
            today = datetime.date.today()
            target = today + datetime.timedelta(days=30)
            exp_str = min(
                expirations,
                key=lambda x: abs((datetime.date.fromisoformat(x) - target).days)
            )
            days_out = (datetime.date.fromisoformat(exp_str) - today).days
            T = max(days_out, 1) / 365.0
            chain = ticker.option_chain(exp_str)
            chain_calls = chain.calls
            strikes = chain_calls['strike'].values
            K = float(strikes[np.argmin(np.abs(strikes - S0))])
    except Exception as exc:
        print(f"  Warning: could not fetch options chain ({exc}). Using K≈spot.")

    # ── Implied volatility ────────────────────────────────────────────────────
    sigma = None

    # 1) try options chain ATM IV
    if chain_calls is not None:
        try:
            atm_idx = (chain_calls['strike'] - S0).abs().idxmin()
            iv_candidate = float(chain_calls.loc[atm_idx, 'impliedVolatility'])
            if 0.01 < iv_candidate < 5.0:
                sigma = iv_candidate
        except Exception:
            pass

    # 2) try ticker.info
    if sigma is None:
        try:
            iv_candidate = ticker.info.get('impliedVolatility')
            if iv_candidate and 0.01 < iv_candidate < 5.0:
                sigma = iv_candidate
        except Exception:
            pass

    if sigma is None:
        print("  Warning: could not determine implied vol; defaulting to 20 %.")
        sigma = 0.20

    print(f"  Spot ${S0:.2f} | IV {sigma*100:.1f}% | K ${K:.2f} | "
          f"Expiry {exp_str} ({T*365:.0f}d) | r {R*100:.1f}%")
    return S0, sigma, T, K, exp_str


def plot_greeks(S0_range, bs_g, mc_g, spot, K, sigma, T, ticker, option_type, exp_str):
    """Render a 2×2 dark-themed plot of the four Greeks."""
    set_dark_theme()

    greeks_meta = [
        ('delta', 'Delta  (Δ)',  'Δ  (per $1 move in spot)'),
        ('gamma', 'Gamma  (Γ)',  'Γ  (per $1² move in spot)'),
        ('vega',  'Vega   (ν)',  'ν  (per 1 % move in vol)'),
        ('theta', 'Theta  (Θ)', 'Θ  ($ per calendar day)'),
    ]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"{ticker}  {option_type.capitalize()} Option Greeks  ·  "
        f"Spot ${spot:.2f}  ·  K ${K:.2f}  ·  "
        f"σ {sigma*100:.1f}%  ·  Expiry {exp_str}  ·  r {R*100:.1f}%",
        fontsize=12, color='white', y=0.995,
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)
    COLOR_BS = '#FFD700'    # gold  — analytical
    COLOR_MC = '#00BFFF'    # cyan  — MC

    for idx, (key, title, ylabel) in enumerate(greeks_meta):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        ax.plot(S0_range, bs_g[key], color=COLOR_BS, lw=2.0,
                label='Black-Scholes (analytical)')
        ax.plot(S0_range, mc_g[key], color=COLOR_MC, lw=1.5, linestyle='--',
                label='Monte Carlo (finite diff)')

        ax.axvline(spot, color='#FF6B6B', lw=1.0, linestyle=':',
                   label=f'Spot ${spot:.2f}')
        ax.axvline(K,    color='#98FB98', lw=1.0, linestyle=':',
                   label=f'Strike ${K:.2f}')
        ax.axhline(0,    color='#666666', lw=0.7)

        ax.set_title(title, fontsize=11, color='white', pad=6)
        ax.set_xlabel('Stock Price ($)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7.5, loc='best', framealpha=0.4)
        ax.grid(True)

    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ticker_symbol = sys.argv[1].upper() if len(sys.argv) > 1 else DEFAULT_TICKER
    option_type   = sys.argv[2].lower() if len(sys.argv) > 2 else 'call'
    if option_type not in ('call', 'put'):
        print(f"Unknown option type '{option_type}'; defaulting to 'call'.")
        option_type = 'call'

    S0, sigma, T, K, exp_str = fetch_market_params(ticker_symbol)

    S0_range = np.linspace((1 - SPOT_RANGE) * S0, (1 + SPOT_RANGE) * S0, N_SPOT_POINTS)

    print("Computing Black-Scholes Greeks …")
    bs_g = black_scholes_greeks(S0_range, K, R, sigma, T, option_type=option_type)

    print(f"Computing Monte Carlo Greeks  ({N_PATHS:,} paths × {N_SPOT_POINTS} spot levels) …")
    mc_g = mc_greeks_range(S0_range, K, R, sigma, T,
                           n_paths=N_PATHS, option_type=option_type)

    print("Rendering plot …")
    plot_greeks(S0_range, bs_g, mc_g, S0, K, sigma, T,
                ticker_symbol, option_type, exp_str)
