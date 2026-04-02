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
    """Return a DataFrame of features aligned to the same index as *close*.

    Feature groups
    --------------
    HAR-RV  : Corsi (2009) heterogeneous AR components — daily/weekly/monthly
               average realized vol.  These three features alone explain most
               of forecastable vol variation.
    RV lags : rolling realized vol at 5/21/63-day horizons plus explicit lags
               of rv21 (autocorrelation proxy).
    ARCH    : squared and absolute log returns, and 5-day max |return| (jump
               indicator).  Capture ARCH/GARCH vol-clustering effects.
    BB width: normalized Bollinger Band width — a pure vol-of-price signal,
              not a price-level feature.
    RSI     : momentum indicator; high/low RSI often precedes vol spikes.

    Deliberately excluded: price/MA ratios (ma50_ratio etc.) — these are
    price-trend features that dominate importance but add no vol-forecasting
    signal, causing out-of-sample degradation.
    """
    log_ret = np.log(close / close.shift(1))

    # ── HAR-RV components (Corsi 2009) ────────────────────────────────────────
    # Use |log_ret| * sqrt(252) as the daily annualised RV proxy, then average.
    rv1_proxy = log_ret.abs() * np.sqrt(252)
    rv_har_d  = rv1_proxy.shift(1)                     # yesterday's daily RV
    rv_har_w  = rv1_proxy.rolling(5).mean().shift(1)   # past 5-day avg RV
    rv_har_m  = rv1_proxy.rolling(22).mean().shift(1)  # past 22-day avg RV

    # ── Rolling realized vol at multiple horizons ─────────────────────────────
    rv5  = log_ret.rolling(5).std()  * np.sqrt(252)
    rv21 = log_ret.rolling(21).std() * np.sqrt(252)
    rv63 = log_ret.rolling(63).std() * np.sqrt(252)

    # Explicit lagged rv21 (autocorrelation of realized vol)
    rv21_lag1 = rv21.shift(1)
    rv21_lag5 = rv21.shift(5)

    # ── ARCH-effect features ──────────────────────────────────────────────────
    sq_ret  = log_ret ** 2
    abs_ret = log_ret.abs()
    jump_5d = log_ret.abs().rolling(5).max()   # largest |return| in 5 days

    # ── Bollinger Band width (vol-regime signal, not price-level) ─────────────
    ma20     = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_width = (4.0 * bb_std) / ma20.replace(0.0, np.nan)

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi = _rsi(close, window=14)

    return pd.DataFrame({
        'rv_har_d':  rv_har_d,
        'rv_har_w':  rv_har_w,
        'rv_har_m':  rv_har_m,
        'rv5':       rv5,
        'rv21':      rv21,
        'rv63':      rv63,
        'rv21_lag1': rv21_lag1,
        'rv21_lag5': rv21_lag5,
        'sq_ret':    sq_ret,
        'abs_ret':   abs_ret,
        'jump_5d':   jump_5d,
        'bb_width':  bb_width,
        'rsi':       rsi,
    }, index=close.index)


# ── Model Training ────────────────────────────────────────────────────────────

def train_vol_model(ticker: str = 'SPY', period: str = '5y'):
    """
    Download historical data, build features, and train a Random Forest to
    predict the 21-day forward realized volatility.

    Improvements over the naive version
    ------------------------------------
    * 5y of data (more independent vol regimes to learn from).
    * HAR-RV + ARCH features instead of price-ratio features.
    * Target log-transformed before fitting; predictions are exp()-back-
      transformed — vol is approximately log-normal so this reduces
      heteroscedastic residuals and stabilises the loss surface.
    * Rows sampled every 5 days (stride=5) to reduce the 95% overlap
      between consecutive 21-day rolling targets, giving the train/test
      split more independent observations to evaluate on.

    Parameters
    ----------
    ticker : str   Yahoo Finance ticker (default 'SPY').
    period : str   yfinance period string (default '5y').

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

    # Stride every 5 rows: cuts target overlap from ~95 % to ~76 %,
    # giving the CV split meaningfully more independent observations.
    data = data.iloc[::5].copy()

    feature_names = [c for c in data.columns if c != 'target']
    X = data[feature_names].values
    y = data['target'].values

    # Log-transform: vol is approximately log-normal; predicting log(vol)
    # stabilises variance and usually improves R².
    log_y = np.log(y.clip(min=1e-6))

    # --- temporal train / test split (no shuffle) ---------------------------
    split    = int(len(data) * 0.8)
    X_train  = X[:split];     log_y_train = log_y[:split]
    X_test   = X[split:];     y_test      = y[split:]

    print(f"[vol_ml] Training Random Forest  "
          f"({split} train / {len(X_test)} test samples, stride=5)…")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, log_y_train)

    # Evaluate in original vol space
    y_pred_test = np.exp(model.predict(X_test))
    oos_r2 = r2_score(y_test, y_pred_test)
    print(f"[vol_ml] Out-of-sample R²: {oos_r2:.4f}")

    # --- predict on the most-recent feature row (all data, then exp) --------
    # Refit on full dataset so prediction uses all available information.
    model.fit(X, log_y)
    predicted_vol = float(np.exp(model.predict(X[[-1]])[0]))

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
