import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import (simulate_gbm_paths, simulate_mjd_paths,
                   monte_carlo_option_price, black_scholes_price, merton_price)

# Set matplotlib dark theme
def set_dark_theme():
    plt.style.use('dark_background')
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = '#222222'
    plt.rcParams['axes.facecolor'] = '#222222'
    plt.rcParams['savefig.facecolor'] = '#222222'

# Parameters
S0 = 100      # Initial stock price
K = 100       # Strike price
r = 0.05      # Risk-free rate
sigma = 0.2   # Volatility
T = 1.0       # Time to maturity (years)
steps = 252   # Steps per path (daily)
n_paths = 1000

# Merton jump parameters
lam     = 1.0    # jump intensity (jumps per year)
mu_j    = -0.10  # mean log-jump size (negative = downward bias)
sigma_j = 0.15   # jump volatility

set_dark_theme()

# Simulate GBM paths
t, paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths)

# Simulate Merton Jump Diffusion paths
_, mjd_paths = simulate_mjd_paths(S0, r, sigma, T, steps, n_paths, lam, mu_j, sigma_j)

# Monte Carlo pricing
mc_call     = monte_carlo_option_price(paths,     K, r, T, option_type='call')
mc_put      = monte_carlo_option_price(paths,     K, r, T, option_type='put')
mc_mjd_call = monte_carlo_option_price(mjd_paths, K, r, T, option_type='call')
mc_mjd_put  = monte_carlo_option_price(mjd_paths, K, r, T, option_type='put')

# Black-Scholes pricing
bs_call = black_scholes_price(S0, K, r, sigma, T, option_type='call')
bs_put  = black_scholes_price(S0, K, r, sigma, T, option_type='put')

# Merton analytical pricing
merton_call = merton_price(S0, K, r, sigma, T, lam, mu_j, sigma_j, option_type='call')
merton_put  = merton_price(S0, K, r, sigma, T, lam, mu_j, sigma_j, option_type='put')

# Price differences vs Black-Scholes
diff_call     = mc_call     - bs_call
diff_put      = mc_put      - bs_put
diff_mjd_call = mc_mjd_call - bs_call
diff_mjd_put  = mc_mjd_put  - bs_put
diff_merton_call = merton_call - bs_call
diff_merton_put  = merton_put  - bs_put

# Prepare DataFrame for display
df_prices = pd.DataFrame({
    'Method':     ['MC (GBM)', 'MC (MJD)', 'Black-Scholes', 'Merton'],
    'Call Price': [mc_call, mc_mjd_call, bs_call, merton_call],
    'Put Price':  [mc_put,  mc_mjd_put,  bs_put,  merton_put],
})

# Animation setup
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Monte Carlo Stock Price Simulation  —  GBM vs Merton Jump Diffusion',
             fontsize=14, color='white')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')

# Label parameters (GBM + jump params)
params_text = (f"S₀: {S0}   K: {K}   r: {r}   σ: {sigma}   T: {T}\n"
               f"λ: {lam}   μⱼ: {mu_j}   σⱼ: {sigma_j}")
param_box = ax.text(0.02, 0.97, params_text, transform=ax.transAxes, fontsize=10,
                    color='white', va='top',
                    bbox=dict(facecolor='#333333', alpha=0.7))

# Option price display (four methods)
price_box = ax.text(0.98, 0.97, '', transform=ax.transAxes, fontsize=10,
                    color='white', va='top', ha='right',
                    bbox=dict(facecolor='#333333', alpha=0.7))

# GBM path lines (gold) and MJD path lines (coral)
gbm_lines = [ax.plot([], [], lw=1, alpha=0.55, color='#FFD700')[0] for _ in range(10)]
mjd_lines = [ax.plot([], [], lw=1, alpha=0.55, color='#FF6B35')[0] for _ in range(10)]

# Legend proxies
ax.plot([], [], color='#FFD700', lw=1.5, label='GBM')
ax.plot([], [], color='#FF6B35', lw=1.5, label='MJD')
ax.legend(loc='upper left', fontsize=9, framealpha=0.4,
          bbox_to_anchor=(0.02, 0.82))

ax.set_xlim(0, T)
y_min = min(np.min(paths[:10]), np.min(mjd_paths[:10]))
y_max = max(np.max(paths[:10]), np.max(mjd_paths[:10]))
ax.set_ylim(y_min * 0.97, y_max * 1.03)

# Animation function
def animate(i):
    for j, line in enumerate(gbm_lines):
        line.set_data(t[:i], paths[j, :i])
    for j, line in enumerate(mjd_lines):
        line.set_data(t[:i], mjd_paths[j, :i])
    price_box.set_text(
        f"── Call ──────────────\n"
        f"MC GBM:  {mc_call:.4f}\n"
        f"MC MJD:  {mc_mjd_call:.4f}\n"
        f"B-S:     {bs_call:.4f}\n"
        f"Merton:  {merton_call:.4f}\n"
        f"\n── Put ───────────────\n"
        f"MC GBM:  {mc_put:.4f}\n"
        f"MC MJD:  {mc_mjd_put:.4f}\n"
        f"B-S:     {bs_put:.4f}\n"
        f"Merton:  {merton_put:.4f}"
    )
    return gbm_lines + mjd_lines + [price_box]

# Print price comparison table before showing the plot
print("\nOption Price Comparison:")
print(df_prices.to_string(index=False))
print(f"\nCall diff vs B-S  |  MC-GBM: {diff_call:+.4f}  "
      f"MC-MJD: {diff_mjd_call:+.4f}  Merton: {diff_merton_call:+.4f}")
print(f"Put  diff vs B-S  |  MC-GBM: {diff_put:+.4f}  "
      f"MC-MJD: {diff_mjd_put:+.4f}  Merton: {diff_merton_put:+.4f}")

ani = animation.FuncAnimation(fig, animate, frames=steps+1, interval=20, blit=True)  # noqa: F841

plt.tight_layout()
plt.show()
