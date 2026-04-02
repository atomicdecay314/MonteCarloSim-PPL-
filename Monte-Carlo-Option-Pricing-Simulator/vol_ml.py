"""
vol_ml.py — ML-based volatility prediction using a Random Forest regressor.

Pipeline:
  1. Download 2 years of daily OHLCV data via yfinance.
  2. Engineer features: rolling realized vols, log returns, RSI, moving averages,
     Bollinger Band width/position.
  3. Train a Random Forest to predict the next 21-day realized volatility
     (temporal train/test split — no data leakage).
  4. Fetch near-ATM implied volatility from the live options chain.

Public API
----------
train_vol_model(ticker, period) -> (predicted_vol, hist_vol, r2, feat_importances)
get_implied_vol(ticker, S0, T, option_type) -> float | None
"""

import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# ── Feature Engineering ───────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _engineer_features(close: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of features aligned to the same index as *close*."""
    log_ret = np.log(close / close.shift(1))

    # Realized vol (annualized) at multiple lookback windows
    rv5  = log_ret.rolling(5).std()  * np.sqrt(252)
    rv10 = log_ret.rolling(10).std() * np.sqrt(252)
    rv21 = log_ret.rolling(21).std() * np.sqrt(252)
    rv63 = log_ret.rolling(63).std() * np.sqrt(252)

    # Moving averages and price ratios
    ma20       = close.rolling(20).mean()
    ma50       = close.rolling(50).mean()
    ma20_ratio = close / ma20
    ma50_ratio = close / ma50

    # Bollinger Bands (20-day, ±2σ)
    bb_std   = close.rolling(20).std()
    bb_upper = ma20 + 2.0 * bb_std
    bb_lower = ma20 - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / ma20
    band_range = (bb_upper - bb_lower).replace(0.0, np.nan)
    bb_pct   = (close - bb_lower) / band_range   # 0 = at lower, 1 = at upper band

    # RSI (14-day)
    rsi = _rsi(close, window=14)

    return pd.DataFrame({
        'log_ret':    log_ret,
        'rv5':        rv5,
        'rv10':       rv10,
        'rv21':       rv21,
        'rv63':       rv63,
        'ma20_ratio': ma20_ratio,
        'ma50_ratio': ma50_ratio,
        'bb_width':   bb_width,
        'bb_pct':     bb_pct,
        'rsi':        rsi,
    }, index=close.index)


# ── Model Training ────────────────────────────────────────────────────────────

def train_vol_model(ticker: str = 'SPY', period: str = '2y'):
    """
    Download historical data, build features, and train a Random Forest to
    predict the 21-day forward realized volatility.

    Parameters
    ----------
    ticker : str   Yahoo Finance ticker (default 'SPY').
    period : str   yfinance period string (default '2y').

    Returns
    -------
    predicted_vol    : float  — RF forecast for the next 21-day window
    hist_vol         : float  — trailing 21-day realized vol (simple baseline)
    r2               : float  — out-of-sample R² on the held-out 20 % test set
    feat_importances : dict   — feature name → importance score
    """
    print(f"[vol_ml] Downloading {period} of {ticker} price history…")
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns (newer yfinance versions may produce them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    close = raw['Close'].squeeze()

    # --- features and target ------------------------------------------------
    feats = _engineer_features(close)

    log_ret  = np.log(close / close.shift(1))
    rv21_fwd = log_ret.rolling(21).std() * np.sqrt(252)
    # target[t] = realized vol over the 21 trading days starting at t+1
    target = rv21_fwd.shift(-21)

    data           = feats.copy()
    data['target'] = target
    data           = data.dropna()

    feature_names = [c for c in data.columns if c != 'target']
    X = data[feature_names].values
    y = data['target'].values

    # --- temporal train / test split (no shuffle) ---------------------------
    split   = int(len(data) * 0.8)
    X_train = X[:split];  y_train = y[:split]
    X_test  = X[split:];  y_test  = y[split:]

    print(f"[vol_ml] Training Random Forest  "
          f"({split} train / {len(X_test)} test samples)…")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    oos_r2 = r2_score(y_test, model.predict(X_test))
    print(f"[vol_ml] Out-of-sample R²: {oos_r2:.4f}")

    # --- predict on the most-recent feature row -----------------------------
    predicted_vol = float(model.predict(X[[-1]])[0])

    # --- trailing 21-day realized vol (simple historical baseline) ----------
    hist_vol = float(log_ret.iloc[-21:].std() * np.sqrt(252))

    feat_importances = dict(zip(feature_names, model.feature_importances_))

    print(f"[vol_ml] RF predicted vol : {predicted_vol:.4f}"
          f"  |  Historical (21d): {hist_vol:.4f}")
    return predicted_vol, hist_vol, oos_r2, feat_importances


# ── Implied Volatility from Options Chain ─────────────────────────────────────

def get_implied_vol(ticker: str = 'SPY',
                    S0: Optional[float] = None,
                    T: float = 1.0,
                    option_type: str = 'call') -> Optional[float]:
    """
    Fetch near-ATM implied volatility from the live options chain.

    Selects the expiry closest to *T* years from today and the strike
    nearest to the current spot price (or *S0* if provided).

    Returns a float or None if the chain is unavailable / fetch fails.
    """
    try:
        tk   = yf.Ticker(ticker)
        exps = tk.options          # list of expiry date strings
        if not exps:
            print(f"[vol_ml] No options found for {ticker}.")
            return None

        target_date = datetime.date.today() + datetime.timedelta(days=int(T * 365))
        exp_dates   = [datetime.date.fromisoformat(e) for e in exps]
        closest_exp = min(exp_dates, key=lambda d: abs((d - target_date).days))
        exp_str     = closest_exp.isoformat()

        chain = tk.option_chain(exp_str)
        opts  = chain.calls if option_type == 'call' else chain.puts

        if S0 is None:
            hist = tk.history(period='1d')
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            S0 = float(hist['Close'].iloc[-1])

        # ATM strike
        idx = (opts['strike'] - S0).abs().idxmin()
        iv  = opts.loc[idx, 'impliedVolatility']

        result = float(iv) if iv and iv > 0 else None
        if result:
            print(f"[vol_ml] Market implied vol ({ticker}, exp {exp_str}, "
                  f"strike {opts.loc[idx, 'strike']:.1f}): {result:.4f}")
        return result

    except Exception as exc:
        print(f"[vol_ml] Could not fetch implied vol for {ticker}: {exc}")
        return None
