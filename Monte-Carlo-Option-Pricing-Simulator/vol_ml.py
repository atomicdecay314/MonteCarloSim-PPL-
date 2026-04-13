"""
vol_ml.py — ML-based volatility prediction using Random Forest and XGBoost.

Pipeline:
  1. Download 5y of daily OHLCV data via yfinance.
  2. Engineer HAR-RV features (Corsi 2009) + ARCH + Bollinger Band + RSI.
  3. Target: 5-day forward realized volatility (log-transformed to stabilise variance).
  4. Sample every 5 rows (stride=5) so consecutive targets are non-overlapping.
  5. Evaluate RF and XGBoost with TimeSeriesSplit(n_splits=5).
  6. Compare against a naive 30-day rolling vol baseline.
  7. Refit winner on full dataset and return forecast.

Public API
----------
train_vol_model(ticker, period) -> dict with keys:
    predicted_vol, rf_predicted_vol, xgb_predicted_vol,
    hist_vol, baseline_vol,
    r2_rf, r2_xgb, r2_baseline,
    feat_importances, xgb_feat_importances
get_implied_vol(ticker, S0, T, option_type) -> float | None
"""

import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[vol_ml] xgboost not installed — only Random Forest will be used.")


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

def train_vol_model(ticker: str = 'SPY', period: str = '5y') -> dict:
    """
    Download historical data, build features, and train Random Forest + XGBoost
    to predict 5-day forward realized volatility.

    Key design choices
    ------------------
    * Target is 5-day forward RV (shift=-5).  With stride=5, consecutive
      targets are exactly non-overlapping — CV scores are unbiased.
    * TimeSeriesSplit(n_splits=5) for evaluation; models refit on full data
      for the final forecast (no future leakage).
    * Both models predict log(vol) to handle the approximately log-normal
      distribution; predictions are exp()-back-transformed.
    * Naive baseline: trailing 30-day rolling vol (what a practitioner would
      use without a model).  Its CV R² anchors interpretation of model R².

    Parameters
    ----------
    ticker : str   Yahoo Finance ticker (default 'SPY').
    period : str   yfinance period string (default '5y').

    Returns
    -------
    dict with keys:
        predicted_vol        float   best-model forecast (RF or XGB by CV R²)
        rf_predicted_vol     float   RF forecast
        xgb_predicted_vol    float|None
        hist_vol             float   trailing 21-day realized vol
        baseline_vol         float   trailing 30-day rolling vol (naive baseline)
        r2_rf                float   mean TimeSeriesSplit R² — RF
        r2_xgb               float|None
        r2_baseline          float   mean TimeSeriesSplit R² — naive baseline
        feat_importances     dict    RF importances (sum to 1)
        xgb_feat_importances dict|None
    """
    print(f"[vol_ml] Downloading {period} of {ticker} price history…")
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw['Close'].squeeze()

    log_ret = np.log(close / close.shift(1))

    # ── Features ──────────────────────────────────────────────────────────────
    feats = _engineer_features(close)

    # ── Target: 5-day forward realized volatility ─────────────────────────────
    # rv5_fwd[t] = std of log returns over the 5 trading days starting at t+1
    rv5_fwd = log_ret.rolling(5).std() * np.sqrt(252)
    target  = rv5_fwd.shift(-5)

    # ── Naive baseline: trailing 30-day rolling vol ───────────────────────────
    rv30 = log_ret.rolling(30).std() * np.sqrt(252)

    # ── Assemble and stride ───────────────────────────────────────────────────
    data           = feats.copy()
    data['target'] = target
    data['rv30']   = rv30          # baseline signal — excluded from model features
    data           = data.dropna()

    # stride=5: with a 5-day target and 5-row stride, no target overlap
    data = data.iloc[::5].copy()

    feature_names = [c for c in data.columns if c not in ('target', 'rv30')]
    X        = data[feature_names].values
    y        = data['target'].values
    baseline = data['rv30'].values
    log_y    = np.log(y.clip(min=1e-6))

    # ── TimeSeriesSplit cross-validation ──────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)

    rf_params = dict(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    xgb_params = dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ) if _HAS_XGB else None

    rf_scores       = []
    xgb_scores      = []
    baseline_scores = []

    print(f"[vol_ml] TimeSeriesSplit CV  "
          f"(5 folds, stride=5, target=5d forward RV)…")
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te     = X[tr_idx], X[te_idx]
        log_y_tr       = log_y[tr_idx]
        y_te           = y[te_idx]
        baseline_te    = baseline[te_idx]

        # RF
        rf_fold = RandomForestRegressor(**rf_params)
        rf_fold.fit(X_tr, log_y_tr)
        rf_scores.append(r2_score(y_te, np.exp(rf_fold.predict(X_te))))

        # XGBoost
        if _HAS_XGB:
            xgb_fold = XGBRegressor(**xgb_params)
            xgb_fold.fit(X_tr, log_y_tr, verbose=False)
            xgb_scores.append(r2_score(y_te, np.exp(xgb_fold.predict(X_te))))

        # Naive baseline (30-day trailing vol)
        baseline_scores.append(r2_score(y_te, baseline_te))

        fold_line = (f"[vol_ml]   Fold {fold}:  "
                     f"RF={rf_scores[-1]:+.3f}")
        if _HAS_XGB:
            fold_line += f"  XGB={xgb_scores[-1]:+.3f}"
        fold_line += f"  Baseline={baseline_scores[-1]:+.3f}"
        print(fold_line)

    r2_rf       = float(np.mean(rf_scores))
    r2_xgb      = float(np.mean(xgb_scores)) if _HAS_XGB else None
    r2_baseline = float(np.mean(baseline_scores))

    summary = (f"[vol_ml] Mean CV R²:  RF={r2_rf:+.4f}")
    if r2_xgb is not None:
        summary += f"  XGB={r2_xgb:+.4f}"
    summary += f"  Baseline={r2_baseline:+.4f}"
    print(summary)

    # ── Refit on full dataset ─────────────────────────────────────────────────
    rf_final = RandomForestRegressor(**rf_params)
    rf_final.fit(X, log_y)
    rf_predicted_vol  = float(np.exp(rf_final.predict(X[[-1]])[0]))
    feat_importances  = dict(zip(feature_names, rf_final.feature_importances_))

    if _HAS_XGB:
        xgb_final = XGBRegressor(**xgb_params)
        xgb_final.fit(X, log_y, verbose=False)
        xgb_predicted_vol    = float(np.exp(xgb_final.predict(X[[-1]])[0]))
        xgb_feat_importances = dict(zip(feature_names,
                                        xgb_final.feature_importances_))
    else:
        xgb_predicted_vol    = None
        xgb_feat_importances = None

    # Best model by mean CV R²
    if r2_xgb is not None and r2_xgb > r2_rf:
        predicted_vol = xgb_predicted_vol
    else:
        predicted_vol = rf_predicted_vol

    # ── Trailing baselines ────────────────────────────────────────────────────
    hist_vol     = float(log_ret.iloc[-21:].std() * np.sqrt(252))
    baseline_vol = float(log_ret.iloc[-30:].std() * np.sqrt(252))

    final_line = (f"[vol_ml] Final forecast:  RF={rf_predicted_vol:.4f}")
    if xgb_predicted_vol is not None:
        final_line += f"  XGB={xgb_predicted_vol:.4f}"
    final_line += f"  30d baseline={baseline_vol:.4f}"
    print(final_line)

    return {
        'predicted_vol':        predicted_vol,
        'rf_predicted_vol':     rf_predicted_vol,
        'xgb_predicted_vol':    xgb_predicted_vol,
        'hist_vol':             hist_vol,
        'baseline_vol':         baseline_vol,
        'r2_rf':                r2_rf,
        'r2_xgb':               r2_xgb,
        'r2_baseline':          r2_baseline,
        'feat_importances':     feat_importances,
        'xgb_feat_importances': xgb_feat_importances,
    }


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
        exps = tk.options
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
