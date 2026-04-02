import numpy as np
from scipy.stats import norm, poisson

# Geometric Brownian Motion simulation
def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths):
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    for i in range(n_paths):
        W = np.random.standard_normal(steps)
        W = np.cumsum(W) * np.sqrt(dt)
        X = (r - 0.5 * sigma ** 2) * t[1:] + sigma * W
        paths[i, 1:] = S0 * np.exp(X)
    return t, paths

# Monte Carlo European Option Pricing
def monte_carlo_option_price(paths, K, r, T, option_type='call'):
    S_T = paths[:, -1]
    if option_type == 'call':
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price

# Black-Scholes Analytical Price
def black_scholes_price(S0, K, r, sigma, T, option_type='call'):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price


# Merton Jump Diffusion path simulation
def simulate_mjd_paths(S0, r, sigma, T, steps, n_paths, lam, mu_j, sigma_j):
    """
    Simulate stock price paths under Merton's Jump Diffusion Model (1976).

    Risk-neutral dynamics:  dS/S = (r − λk̄) dt + σ dW + (J−1) dN
      k̄     = E[J−1] = exp(μⱼ + σⱼ²/2) − 1   (expected proportional jump size)
      dN    ~ Poisson(λ dt) per step
      log J ~ N(μⱼ, σⱼ²)

    The jump log-return per step is N(Nₜ·μⱼ, Nₜ·σⱼ²), computed exactly via
    vectorised Poisson draws — no Python loop over paths or steps.
    """
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    k_bar = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1

    # Diffusion component: (n_paths, steps)
    Z_diff = np.random.standard_normal((n_paths, steps))
    diffusion = (r - lam * k_bar - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_diff

    # Jump component: Poisson number of jumps per step, then aggregate log-jump
    N_jumps = np.random.poisson(lam * dt, size=(n_paths, steps))
    Z_jump = np.random.standard_normal((n_paths, steps))
    jump = N_jumps * mu_j + np.sqrt(N_jumps) * sigma_j * Z_jump

    log_returns = diffusion + jump
    paths = np.empty((n_paths, steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))

    return t, paths


# Merton analytical option price (infinite series, truncated)
def merton_price(S0, K, r, sigma, T, lam, mu_j, sigma_j, option_type='call', n_terms=50):
    """
    Merton (1976) closed-form European option price under jump diffusion.

    Conditions on exactly n jumps in [0,T] (Poisson-weighted BS prices):
      σₙ = √(σ² + n·σⱼ²/T)
      rₙ = r − λk̄ + n·(μⱼ + σⱼ²/2) / T
      weight = P(N(T) = n) = Poisson PMF at λT
    """
    k_bar = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1
    price = 0.0
    for n in range(n_terms):
        weight = poisson.pmf(n, lam * T)
        if weight < 1e-12:          # series has converged
            break
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / T)
        r_n = r - lam * k_bar + n * (mu_j + 0.5 * sigma_j ** 2) / T
        price += weight * black_scholes_price(S0, K, r_n, sigma_n, T, option_type)
    return price


# Black-Scholes Analytical Greeks
def black_scholes_greeks(S0, K, r, sigma, T, option_type='call'):
    """
    Analytical BS Greeks. S0 may be a scalar or numpy array.
    Theta is per calendar day; Vega is per 1 percentage-point change in vol.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    n_d1 = norm.pdf(d1)

    gamma = n_d1 / (S0 * sigma * np.sqrt(T))
    vega = S0 * n_d1 * np.sqrt(T) / 100.0  # per 1% change in vol

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-(S0 * n_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-(S0 * n_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


# Monte Carlo Greeks via finite differences, vectorized over a spot price range
def mc_greeks_range(S0_range, K, r, sigma, T, n_paths=10000, option_type='call', seed=42):
    """
    Compute MC Greeks via central finite differences for each spot in S0_range.

    Uses common random numbers (same Z reused across all bumps) to reduce
    variance in the finite-difference estimates.

    Returns a dict of 1-D arrays (length = len(S0_range)):
      delta  — per $1 move in spot
      gamma  — per $1^2 move in spot
      vega   — per 1 percentage-point change in vol
      theta  — per calendar day ($)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    # Reshape for broadcasting: (n_spots, 1) x (1, n_paths) → (n_spots, n_paths)
    S0 = np.asarray(S0_range, dtype=float).reshape(-1, 1)
    Z = Z.reshape(1, -1)

    dsig = 0.001        # 0.1 vol-point bump
    dT = 1.0 / 365.0   # 1 calendar day

    def sim_terminal(s, sig, tau):
        return s * np.exp((r - 0.5 * sig ** 2) * tau + sig * np.sqrt(tau) * Z)

    def price_arr(S_T, tau=T):
        disc = np.exp(-r * tau)
        if option_type == 'call':
            return disc * np.mean(np.maximum(S_T - K, 0.0), axis=1)
        else:
            return disc * np.mean(np.maximum(K - S_T, 0.0), axis=1)

    dS = S0 * 0.01  # 1% spot bump; shape (n_spots, 1)

    p_mid  = price_arr(sim_terminal(S0,       sigma,       T))
    p_up_S = price_arr(sim_terminal(S0 + dS,  sigma,       T))
    p_dn_S = price_arr(sim_terminal(S0 - dS,  sigma,       T))

    delta = (p_up_S - p_dn_S) / (2.0 * dS[:, 0])
    gamma = (p_up_S - 2.0 * p_mid + p_dn_S) / (dS[:, 0] ** 2)

    p_up_sig = price_arr(sim_terminal(S0, sigma + dsig, T))
    p_dn_sig = price_arr(sim_terminal(S0, sigma - dsig, T))
    vega = (p_up_sig - p_dn_sig) / (2.0 * dsig) / 100.0

    if T > dT:
        theta = price_arr(sim_terminal(S0, sigma, T - dT), tau=T - dT) - p_mid
    else:
        theta = np.zeros(len(S0_range))

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}
