# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulator

```bash
# Install dependencies
pip install -r Monte-Carlo-Option-Pricing-Simulator/requirements.txt

# Run the simulator
cd Monte-Carlo-Option-Pricing-Simulator && python main.py
```

There are no automated tests or lint configuration in this project.

## Architecture

The project is a Monte Carlo Option Pricing Simulator for European Call and Put options. It lives in the `Monte-Carlo-Option-Pricing-Simulator/` subdirectory with two modules:

**`utils.py`** — pure financial math, no I/O or plotting:
- `simulate_gbm_paths()` — generates stock price paths via Geometric Brownian Motion: `S_t = S₀ · exp((r − 0.5σ²)t + σW_t)`
- `monte_carlo_option_price()` — discounted expected payoff from simulated terminal prices
- `black_scholes_price()` — analytical Black-Scholes pricing using `scipy.stats.norm.cdf`

**`main.py`** — orchestration and visualization:
- Defines simulation parameters (S0=100, K=100, r=5%, σ=20%, T=1yr, 252 steps, 1000 paths)
- Calls `utils.py` functions to generate paths and prices
- Produces an animated matplotlib figure (dark theme) showing 20 sample paths and live MC vs BS price comparison
- Prints a pandas comparison table to stdout

Data flows one-way: parameters → GBM paths → MC pricing + BS pricing → comparison/visualization.

**Important:** `main.py` has no `if __name__ == '__main__'` guard — all code executes at import time. Do not import from `main.py`; doing so will immediately trigger the simulation and open a plot window.
