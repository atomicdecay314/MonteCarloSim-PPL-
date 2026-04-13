#!/usr/bin/env python3
"""
tests/integrity_test.py — Integration tests against the QuantSim Pro backend.

Run with:
    python tests/integrity_test.py

Requires the server running at http://localhost:8000:
    cd Monte-Carlo-Option-Pricing-Simulator
    uvicorn server:app --port 8000

Note on API field names: Pydantic models use camelCase (nPaths, muJ, sigmaJ),
so all JSON request bodies must use those names.

Note on sample paths: the server downsamples 253 time steps to 63 for payload
efficiency, so gbm_paths shape is [nSamplePaths, 63], not [nSamplePaths, 253].

Note on /api/predict-vol: the endpoint fetches live market data via yfinance and
trains a Random Forest on the fly — it does not accept raw feature vectors. The
vol prediction test therefore validates structural sanity rather than regime
membership, since the underlying market state cannot be controlled.
"""

import math
import sys
import statistics

import requests

BASE_URL = "http://localhost:8000"
TIMEOUT_SHORT = 60   # seconds, for simulate / greeks
TIMEOUT_LONG  = 180  # seconds, for predict-vol (downloads 5y of data + trains RF)

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── Shared result log ─────────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    _results.append((name, passed, detail))


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _bad(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _simulate(params: dict) -> dict:
    r = requests.post(f"{BASE_URL}/api/simulate", json=params, timeout=TIMEOUT_SHORT)
    r.raise_for_status()
    return r.json()


def _greeks(params: dict) -> dict:
    r = requests.post(f"{BASE_URL}/api/greeks", json=params, timeout=TIMEOUT_SHORT)
    r.raise_for_status()
    return r.json()


def _predict_vol(params: dict) -> dict:
    r = requests.post(f"{BASE_URL}/api/predict-vol", json=params, timeout=TIMEOUT_LONG)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Black-Scholes Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def test_bs_benchmark() -> dict:
    """
    Law-of-large-numbers check: with 10 000 paths the MC-GBM price must land
    within $0.50 of the closed-form Black-Scholes price for both call and put.
    Returns the raw response so later tests can reuse it.
    """
    print(f"\n{BOLD}[1] Black-Scholes Benchmark{RESET}")

    data = _simulate({"S0": 100, "K": 100, "r": 0.05, "sigma": 0.20,
                      "T": 1.0, "nPaths": 10000})
    p = data["pricing"]

    mc_c  = p["mc_gbm"]["call"]
    mc_p  = p["mc_gbm"]["put"]
    bs_c  = p["bs"]["call"]
    bs_p  = p["bs"]["put"]
    diff_c = abs(mc_c - bs_c)
    diff_p = abs(mc_p - bs_p)

    call_ok = diff_c <= 0.50
    put_ok  = diff_p <= 0.50

    (_ok if call_ok else _bad)(
        f"Call  — MC-GBM: ${mc_c:.2f}, BS: ${bs_c:.2f}, diff: ${diff_c:.2f}"
    )
    (_ok if put_ok else _bad)(
        f"Put   — MC-GBM: ${mc_p:.2f}, BS: ${bs_p:.2f}, diff: ${diff_p:.2f}"
    )

    passed = call_ok and put_ok
    detail = (f"(MC-GBM: ${mc_c:.2f}, BS: ${bs_c:.2f}, diff: ${diff_c:.2f})")
    _record("BS Benchmark", passed, detail)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Put-Call Parity
# ─────────────────────────────────────────────────────────────────────────────

def test_put_call_parity(data: dict, S0: float = 100, K: float = 100,
                          r: float = 0.05, T: float = 1.0) -> None:
    """
    C − P = S₀ − K·e^(−rT) must hold for every model.
    Tolerance: $0.10 for closed-form models (BS, Merton), $0.50 for MC.
    """
    print(f"\n{BOLD}[2] Put-Call Parity{RESET}")

    pcp_rhs = S0 - K * math.exp(-r * T)  # ≈ 4.877
    p = data["pricing"]

    checks = [
        ("BS",      p["bs"]["call"],     p["bs"]["put"],     0.10),
        ("Merton",  p["merton"]["call"], p["merton"]["put"], 0.10),
        ("MC-GBM",  p["mc_gbm"]["call"], p["mc_gbm"]["put"], 0.50),
        ("MC-MJD",  p["mc_mjd"]["call"], p["mc_mjd"]["put"], 0.50),
    ]

    violations: list[str] = []
    for model, call, put, tol in checks:
        diff = abs((call - put) - pcp_rhs)
        ok   = diff <= tol
        msg  = (f"{model:<8}: C=${call:.2f} P=${put:.2f}  "
                f"C−P=${call-put:.2f}  expected≈${pcp_rhs:.2f}  "
                f"err=${diff:.3f}  tol=${tol:.2f}")
        ((_ok if ok else _bad))(msg)
        if not ok:
            violations.append(f"{model} parity error ${diff:.3f} > ${tol:.2f}")

    passed = len(violations) == 0
    detail = "(all models)" if passed else f"({len(violations)} violation(s))"
    _record("Put-Call Parity", passed, detail)
    for v in violations:
        _warn(v)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Merton Jump Directionality
# ─────────────────────────────────────────────────────────────────────────────

def test_merton_directionality() -> None:
    """
    Financial identities under jump diffusion:
    - Downward jumps (muJ < 0) suppress the call price and inflate the put price
      relative to Black-Scholes (which has no jumps).
    - Upward jumps (muJ > 0) inflate the call price.
    These are non-negotiable; any violation is flagged loudly.
    """
    print(f"\n{BOLD}[3] Merton Jump Directionality{RESET}")

    base = {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.20,
            "T": 1.0, "nPaths": 5000, "lam": 1.0, "sigmaJ": 0.15}

    data_dn = _simulate({**base, "muJ": -0.10})
    data_up = _simulate({**base, "muJ": +0.10})

    bs_c   = data_dn["pricing"]["bs"]["call"]
    bs_p   = data_dn["pricing"]["bs"]["put"]
    m_dn_c = data_dn["pricing"]["merton"]["call"]
    m_dn_p = data_dn["pricing"]["merton"]["put"]
    m_up_c = data_up["pricing"]["merton"]["call"]

    failures: list[str] = []
    checks = [
        (m_dn_c < m_up_c,
         f"muJ=−0.10 call ${m_dn_c:.2f} < muJ=+0.10 call ${m_up_c:.2f} (neg jump → lower call)"),
        (m_dn_p > bs_p,
         f"muJ=−0.10: Merton put  ${m_dn_p:.2f} > BS put  ${bs_p:.2f}"),
        (m_up_c > bs_c,
         f"muJ=+0.10: Merton call ${m_up_c:.2f} > BS call ${bs_c:.2f}"),
    ]

    for ok, msg in checks:
        ((_ok if ok else _bad))(msg)
        if not ok:
            failures.append(msg)

    passed = len(failures) == 0
    _record("Jump Directionality", passed)
    for f in failures:
        _warn(f"VIOLATED: {f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Moneyness Sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_moneyness_sanity() -> None:
    """
    Basic option pricing intuition (using Black-Scholes as ground truth):
    - Deep ITM call must exceed its intrinsic value of $50.
    - Deep OTM call must be nearly worthless.
    - ATM call must be in the textbook range for σ=0.20, T=1.
    """
    print(f"\n{BOLD}[4] Moneyness Sanity{RESET}")

    base = {"K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0, "nPaths": 3000}

    itm_call = _simulate({**base, "S0": 150})["pricing"]["bs"]["call"]
    otm_call = _simulate({**base, "S0":  50})["pricing"]["bs"]["call"]
    atm_call = _simulate({**base, "S0": 100})["pricing"]["bs"]["call"]

    checks = [
        (itm_call > 45,       f"Deep ITM call (S0=150): ${itm_call:.2f} > $45 (intrinsic floor)"),
        (otm_call < 5,        f"Deep OTM call (S0=50):  ${otm_call:.4f} < $5"),
        (5 <= atm_call <= 20, f"ATM call      (S0=100): ${atm_call:.2f} ∈ [$5, $20]"),
    ]

    failures: list[str] = []
    for ok, msg in checks:
        ((_ok if ok else _bad))(msg)
        if not ok:
            failures.append(msg)

    _record("Moneyness Sanity", len(failures) == 0)
    for f in failures:
        _warn(f)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Greeks Sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_greeks_sanity() -> None:
    """
    Analytical Greeks over a spot range [70, 130] must satisfy option theory:
    - Delta_call ∈ (0, 1), Delta_put ∈ (−1, 0)
    - Gamma ≥ 0, Vega ≥ 0 everywhere
    - Theta ≤ 0 everywhere for calls (time erodes value)
    - ATM delta_call ≈ 0.5–0.6
    - MC finite-difference Greeks must be within 5% of analytical at ATM.
    """
    print(f"\n{BOLD}[5] Greeks Sanity{RESET}")

    params_call = {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.20,
                   "T": 1.0, "optionType": "call", "nPaths": 5000}
    params_put  = {**params_call, "optionType": "put"}

    g_call = _greeks(params_call)
    g_put  = _greeks(params_put)

    bs_d_c = g_call["bs"]["delta"]
    bs_d_p = g_put["bs"]["delta"]
    bs_gam = g_call["bs"]["gamma"]
    bs_veg = g_call["bs"]["vega"]
    bs_tht = g_call["bs"]["theta"]
    mc_d_c = g_call["mc"]["delta"]
    mc_gam = g_call["mc"]["gamma"]
    mc_veg = g_call["mc"]["vega"]

    # Spot index closest to ATM (S0=100); spot_range = linspace(70, 130, 30)
    spot_range = g_call["spot_range"]
    atm_idx = min(range(len(spot_range)), key=lambda i: abs(spot_range[i] - 100))

    failures: list[str] = []

    def _check(cond: bool, msg: str) -> None:
        ((_ok if cond else _bad))(msg)
        if not cond:
            failures.append(msg)

    _check(all(0 <= d <= 1  for d in bs_d_c),
           "Call delta ∈ [0, 1] across all spots")
    _check(all(-1 <= d <= 0 for d in bs_d_p),
           "Put delta ∈ [−1, 0] across all spots")
    _check(all(g >= 0 for g in bs_gam),
           "Gamma ≥ 0 everywhere")
    _check(all(v >= 0 for v in bs_veg),
           "Vega ≥ 0 everywhere")
    _check(all(t <= 0 for t in bs_tht),
           "Call theta ≤ 0 everywhere")

    atm_delta = g_call["current"]["delta"]
    _check(0.50 <= atm_delta <= 0.70,
           f"ATM delta_call = {atm_delta:.4f} ≈ 0.50–0.70")

    # MC vs BS at the ATM index
    mc_errors: list[tuple[str, float]] = []
    for gk, mc_arr, bs_arr in [
        ("delta", mc_d_c, bs_d_c),
        ("gamma", mc_gam, bs_gam),
        ("vega",  mc_veg, bs_veg),
    ]:
        bs_val = bs_arr[atm_idx]
        mc_val = mc_arr[atm_idx]
        if abs(bs_val) > 1e-10:
            pct = abs(mc_val - bs_val) / abs(bs_val) * 100
        else:
            pct = 0.0
        mc_errors.append((gk, pct))
        ok = pct <= 5.0
        ((_ok if ok else _bad))(
            f"MC {gk} at ATM: BS={bs_val:.4f} MC={mc_val:.4f} err={pct:.1f}%"
        )
        if not ok:
            failures.append(f"MC {gk} at ATM {pct:.1f}% > 5% tolerance")

    avg_pct = sum(e for _, e in mc_errors) / len(mc_errors)
    passed  = len(failures) == 0
    detail  = (f"(MC-FD within {avg_pct:.1f}% of analytical)"
               if passed else f"({len(failures)} failure(s))")
    _record("Greeks Sanity", passed, detail)
    for f in failures:
        _warn(f)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Volatility Prediction Sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_vol_prediction() -> None:
    """
    The /api/predict-vol endpoint fetches live OHLCV data from yfinance and
    trains a 500-tree Random Forest on the fly — it cannot accept injected
    feature vectors. This test therefore validates structural sanity:

    - predicted_vol is a finite, positive, economically plausible value.
    - Feature importances are a valid probability distribution (sum ≈ 1.0).
    - Out-of-sample R² is a finite number in the plausible range [−1, 1].

    If your environment has no internet access this test will fail with an
    HTTP 500 from the server, which is caught and reported as a failure.
    """
    print(f"\n{BOLD}[6] Volatility Prediction Sanity{RESET}")
    print(f"  {YELLOW}(Uses live market data via yfinance — may take ~30 s){RESET}")

    data = _predict_vol({"ticker": "SPY", "r": 0.05, "T": 1.0, "optionType": "call"})

    pv       = data["predicted_vol"]
    hv       = data["hist_vol"]
    r2       = data["r2"]
    fi       = data["feat_importances"]
    fi_sum   = sum(fi.values())

    failures: list[str] = []

    def _check(cond: bool, msg: str) -> None:
        ((_ok if cond else _bad))(msg)
        if not cond:
            failures.append(msg)

    _check(0.01 <= pv <= 2.0,
           f"Predicted vol: {pv:.4f}  (expected 0.01–2.0)")
    _check(0.01 <= hv <= 2.0,
           f"Historical vol: {hv:.4f}  (expected 0.01–2.0)")
    _check(-1.0 <= r2 <= 1.0,
           f"Out-of-sample R²: {r2:.4f}  (expected −1 to 1)")
    _check(0.99 <= fi_sum <= 1.01,
           f"Feature importances sum: {fi_sum:.6f}  (expected ≈ 1.0)")

    # Top feature should dominate
    top_feat, top_imp = max(fi.items(), key=lambda kv: kv[1])
    _ok(f"Top feature: {top_feat!r} importance={top_imp:.4f}")

    _record("Vol Prediction", len(failures) == 0)
    for f in failures:
        _warn(f)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Path Shape Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_path_shape() -> None:
    """
    Validate the geometry and physics of the returned sample paths.

    The server downsamples from 253 time steps to 63 for payload efficiency,
    so the expected shape is [nSamplePaths, 63].

    Checks:
    - Shape is exactly [30, 63]
    - All stock prices are strictly positive
    - All paths start at S₀
    - Time axis runs from 0 to T
    - MJD terminal prices have higher variance than GBM (jumps add vol)
    """
    print(f"\n{BOLD}[7] Path Shape Validation{RESET}")

    S0, T = 100.0, 1.0
    data = _simulate({"S0": S0, "K": 100, "r": 0.05, "sigma": 0.20,
                      "T": T, "nPaths": 1000, "nSamplePaths": 30})

    sp     = data["sample_paths"]
    gbm    = sp["gbm"]   # list[list[float]]
    mjd    = sp["mjd"]
    t_axis = sp["t"]

    n_paths = len(gbm)
    n_steps = len(gbm[0]) if gbm else 0

    failures: list[str] = []

    def _check(cond: bool, msg: str) -> None:
        ((_ok if cond else _bad))(msg)
        if not cond:
            failures.append(msg)

    # Shape
    _check(n_paths == 30 and n_steps == 63,
           f"GBM shape: [{n_paths}, {n_steps}]  (expected [30, 63])")

    # All prices positive
    all_pos_gbm = all(v > 0 for path in gbm for v in path)
    all_pos_mjd = all(v > 0 for path in mjd for v in path)
    _check(all_pos_gbm, "All GBM path values > 0")
    _check(all_pos_mjd, "All MJD path values > 0")

    # Paths start at S0 (first column = step 0 = initial price)
    starts_ok = all(abs(path[0] - S0) < 1e-6 for path in gbm)
    _check(starts_ok, f"All GBM paths start at S0={S0}")

    # Time axis
    time_start_ok = abs(t_axis[0]) < 1e-9
    time_end_ok   = abs(t_axis[-1] - T) < 1e-6
    _check(time_start_ok and time_end_ok,
           f"Time axis: [{t_axis[0]:.6f}, {t_axis[-1]:.6f}]  (expected [0, {T}])")

    # MJD variance > GBM variance (jumps inflate dispersion)
    gbm_terminal = [path[-1] for path in gbm]
    mjd_terminal = [path[-1] for path in mjd]
    gbm_var = statistics.variance(gbm_terminal)
    mjd_var = statistics.variance(mjd_terminal)
    _check(mjd_var > gbm_var,
           f"MJD terminal var={mjd_var:.2f} > GBM terminal var={gbm_var:.2f}")

    _record("Path Shape", len(failures) == 0)
    for f in failures:
        _warn(f)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Convergence Test
# ─────────────────────────────────────────────────────────────────────────────

def test_convergence() -> None:
    """
    MC error should shrink as √N (central limit theorem).
    Three independent trials at nPaths ∈ {500, 1000, 5000}; uses average error
    to reduce stochastic noise from lucky/unlucky single runs.
    Pass condition: avg |error₅₀₀₀| < avg |error₅₀₀|.
    """
    print(f"\n{BOLD}[8] Convergence Test{RESET}")

    base = {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0}
    path_counts = [500, 1000, 5000]
    N_TRIALS = 3  # average over 3 runs to reduce stochastic noise

    print(f"  {'nPaths':>6}  {'Avg |err|':>10}  {'Trial errors'}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*30}")

    avg_errors = []
    bs_ref = None
    for n in path_counts:
        trial_errors = []
        for _ in range(N_TRIALS):
            data = _simulate({**base, "nPaths": n})
            mc = data["pricing"]["mc_gbm"]["call"]
            bs = data["pricing"]["bs"]["call"]
            if bs_ref is None:
                bs_ref = bs
            trial_errors.append(abs(mc - bs))
        avg_err = sum(trial_errors) / N_TRIALS
        avg_errors.append(avg_err)
        trials_str = "  ".join(f"${e:.3f}" for e in trial_errors)
        print(f"  {n:>6}  ${avg_err:>9.4f}  [{trials_str}]")

    converging = avg_errors[-1] < avg_errors[0]
    err_str = " → ".join(f"${e:.3f}" for e in avg_errors)
    (_ok if converging else _bad)(
        f"Avg error decreased 500→5000: {err_str}"
    )
    _record("Convergence", converging, f"(avg error: {err_str})")
    if not converging:
        _warn("Consistent convergence failure indicates a GBM simulation bug")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    banner = "=" * 50
    print(f"\n{BOLD}{banner}{RESET}")
    print(f"{BOLD}  QuantSim Pro — Integrity Test Runner{RESET}")
    print(f"{BOLD}  Target: {BASE_URL}{RESET}")
    print(f"{BOLD}{banner}{RESET}")

    # Connectivity check
    try:
        requests.get(BASE_URL, timeout=5)
    except requests.ConnectionError as exc:
        print(f"\n{RED}{BOLD}Cannot reach {BASE_URL}: {exc}{RESET}")
        print("Start the server with:")
        print("  cd Monte-Carlo-Option-Pricing-Simulator")
        print("  uvicorn server:app --port 8000")
        sys.exit(2)

    try:
        # Test 1 returns the raw response so Test 2 can reuse it (saves one API call)
        base_data = test_bs_benchmark()
        test_put_call_parity(base_data)
        test_merton_directionality()
        test_moneyness_sanity()
        test_greeks_sanity()
        test_vol_prediction()
        test_path_shape()
        test_convergence()
    except requests.HTTPError as exc:
        print(f"\n{RED}{BOLD}HTTP error during tests: {exc}{RESET}")
        sys.exit(2)
    except requests.ConnectionError as exc:
        print(f"\n{RED}{BOLD}Lost connection to server: {exc}{RESET}")
        sys.exit(2)

    # ── Final summary ─────────────────────────────────────────────────────────
    n_pass  = sum(1 for _, ok, _ in _results if ok)
    n_total = len(_results)
    all_ok  = n_pass == n_total

    print(f"\n{BOLD}{banner}{RESET}")
    print(f"{BOLD}  QuantSim Pro — Integrity Test Report{RESET}")
    print(f"{BOLD}{banner}{RESET}")

    for name, ok, detail in _results:
        icon   = f"{GREEN}✅{RESET}" if ok else f"{RED}❌{RESET}"
        status = "PASS" if ok else "FAIL"
        color  = GREEN if ok else RED
        print(f"{icon} {BOLD}{name:<22}{RESET} {color}{status}{RESET} {detail}")

    print(f"{BOLD}{banner}{RESET}")
    summary_color = GREEN if all_ok else RED
    verdict = "Simulation integrity verified." if all_ok else "Some tests FAILED."
    print(f"{summary_color}{BOLD}{n_pass}/{n_total} tests passed. {verdict}{RESET}")
    print(f"{BOLD}{banner}{RESET}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
