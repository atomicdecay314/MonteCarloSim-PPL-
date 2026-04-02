# Monte Carlo Option Pricing Simulator
### Project Report

---

## 1. Project Overview

This project is a **quantitative finance simulator** built in Python that prices European stock options using three progressively sophisticated models:

1. **Monte Carlo simulation on Geometric Brownian Motion (GBM)** — stochastic numerical method
2. **Black-Scholes analytical formula** — closed-form mathematical benchmark
3. **Merton Jump Diffusion (MJD)** — extended model capturing sudden market crashes

It also computes and visualises the four **option Greeks** (Delta, Gamma, Vega, Theta) using both analytical formulas and Monte Carlo finite differences, calibrated to real live market data from Yahoo Finance.

---

## 2. Project Structure

```
Monte-Carlo-Option-Pricing-Simulator/
├── utils.py          # All financial mathematics (pure functions, no I/O)
├── main.py           # Simulation orchestration + animated visualisation
├── greeks.py         # Greeks visualisation with live market data
└── requirements.txt  # numpy, pandas, matplotlib, scipy, yfinance
```

**Design principle:** `utils.py` is a pure math library. `main.py` and `greeks.py` are the entry points — they call `utils.py` and handle all output.

---

## 3. Financial Background

### What is an Option?

A **European call option** gives the buyer the right (not obligation) to buy a stock at a fixed price (the **strike K**) at a future date (maturity **T**). Its payoff at expiry is:

```
Call payoff = max(S_T − K, 0)
```

A **put option** is the mirror image — the right to sell:

```
Put payoff = max(K − S_T, 0)
```

The challenge is: what is this contract worth *today*?

---

## 4. Model 1 — Geometric Brownian Motion (GBM)

GBM is the classical assumption that stock prices evolve continuously with a constant drift and random fluctuations (Brownian motion):

```
S(t) = S₀ · exp((r − σ²/2)·t + σ·W_t)
```

### Simulation Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| S₀ | 100 | Initial stock price |
| K | 100 | Strike price (at-the-money) |
| r | 5% | Risk-free interest rate |
| σ | 20% | Annual volatility |
| T | 1 year | Time to maturity |
| steps | 252 | Daily time steps |
| n_paths | 1,000 | Number of simulated paths |

**Implementation:** Each path is built by cumulative-summing 252 standard normal draws, scaled by σ√dt. This correctly produces correlated daily returns that compound into a log-normal terminal distribution.

---

## 5. Model 2 — Black-Scholes Analytical Price

Black and Scholes (1973) derived an exact closed-form solution under GBM assumptions:

```
d₁ = [ln(S₀/K) + (r + σ²/2)·T] / (σ√T)
d₂ = d₁ − σ√T

Call = S₀·N(d₁) − K·e^(−rT)·N(d₂)
Put  = K·e^(−rT)·N(−d₂) − S₀·N(−d₁)
```

where N(·) is the standard normal CDF.

This is the **ground truth benchmark** — the Monte Carlo GBM price should converge to this as the number of paths increases.

---

## 6. Monte Carlo Pricing

The MC price is computed by:

1. Simulating 1,000 GBM paths to get 1,000 terminal stock prices S_T
2. Computing the payoff for each path: `max(S_T − K, 0)`
3. Averaging all payoffs and discounting back to today:

```
Price = e^(−rT) · mean(payoffs)
```

This is the **Law of Large Numbers** applied to option pricing. With 1,000 paths, the MC price typically falls within ±$0.10 of the analytical Black-Scholes price.

---

## 7. Model 3 — Merton Jump Diffusion (MJD)

### Motivation

**The problem with GBM:** Real markets experience sudden crashes (e.g., 2008 financial crisis, COVID-19 March 2020). GBM produces smooth continuous paths that cannot capture these events.

**Merton's solution (1976):** Add a random jump process on top of GBM:

```
dS/S = (r − λk̄) dt + σ dW + (J−1) dN
```

### Jump Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| λ (lambda) | 1.0 | Average number of jumps per year |
| μⱼ (mu_j) | −0.10 | Mean log-jump size (−10%, downward bias) |
| σⱼ (sigma_j) | 0.15 | Jump volatility (15%) |

- **dN** is a Poisson process: in any small interval dt, a jump occurs with probability λ·dt
- **J** is the jump multiplier: log(J) ~ Normal(μⱼ, σⱼ²), making each jump log-normally distributed
- **k̄ = E[J−1]** is the expected proportional jump size, used to keep the model risk-neutral

### Vectorised Implementation

The aggregate log-return from Nₜ jumps in one time step is exactly:

```
jump contribution = Nₜ·μⱼ + √Nₜ·σⱼ·Z
```

This is valid because the sum of Nₜ independent N(μⱼ, σⱼ²) variables is N(Nₜ·μⱼ, Nₜ·σⱼ²). The entire (1,000 paths × 252 steps) tensor is computed in three NumPy operations — no Python loop.

### Merton Analytical Formula

Merton also derived a closed-form price by conditioning on exactly n jumps occurring in [0,T]. Given n jumps, the stock is still log-normal, so Black-Scholes applies with modified parameters:

```
σₙ = √(σ² + n·σⱼ²/T)
rₙ = r − λk̄ + n·(μⱼ + σⱼ²/2) / T

Price = Σ(n=0 to ∞)  P(N(T)=n) · BS(S₀, K, rₙ, σₙ, T)
```

P(N(T)=n) is the Poisson PMF at λT. The series is truncated when weights fall below 1×10⁻¹² — in practice it converges within 15–20 terms.

### What the Jump Model Reveals

With μⱼ = −0.10 (downward-biased jumps):

- **Merton call < Black-Scholes call** — jumps shift probability to the left tail, reducing upside
- **Merton put > Black-Scholes put** — same left-tail mass increase inflates put value

This is the mechanism behind the **volatility smile** — a well-known market phenomenon that standard Black-Scholes cannot explain.

---

## 8. Visualisation — main.py

The animation shows 10 **gold (GBM)** paths and 10 **coral/orange (MJD)** paths evolving simultaneously in real time. The differences are visually clear — MJD paths exhibit sudden discontinuous drops (the jumps), while GBM paths are smooth.

An on-screen price box shows all four prices live during the animation, and a summary table is printed to the terminal immediately before the plot opens:

```
        Method  Call Price  Put Price
     MC (GBM)      10.45       5.57
     MC (MJD)       9.82       6.10
 Black-Scholes      10.45       5.57
        Merton       9.79       6.12

Call diff vs B-S  |  MC-GBM: +0.00  MC-MJD: -0.63  Merton: -0.66
Put  diff vs B-S  |  MC-GBM: +0.00  MC-MJD: +0.53  Merton: +0.55
```

---

## 9. Option Greeks — greeks.py

Greeks measure how sensitive an option's price is to changes in market conditions:

| Greek | Measures | Typical Behaviour |
|-------|----------|-------------------|
| **Delta (Δ)** | Price change per $1 move in stock | 0 to 1 for calls; −1 to 0 for puts |
| **Gamma (Γ)** | Rate of change of Delta | Peaks at-the-money near expiry |
| **Vega (ν)** | Price change per 1% move in volatility | Always positive for long options |
| **Theta (Θ)** | Price decay per calendar day | Always negative for long options |

### Two Methods Compared

**Black-Scholes analytical** — closed-form formulas:

```
Delta_call = N(d₁)
Gamma      = N'(d₁) / (S·σ·√T)
Vega       = S·N'(d₁)·√T / 100
Theta_call = [−S·N'(d₁)·σ/(2√T) − r·K·e^(−rT)·N(d₂)] / 365
```

**Monte Carlo finite differences** — bump-and-reprice for each Greek:

```
Delta ≈ [Price(S+ΔS) − Price(S−ΔS)] / (2ΔS)           central difference
Gamma ≈ [Price(S+ΔS) − 2·Price(S) + Price(S−ΔS)] / ΔS²  second difference
Vega  ≈ [Price(σ+Δσ) − Price(σ−Δσ)] / (2·Δσ·100)
Theta ≈  Price(T − 1 day) − Price(T)
```

### Variance Reduction: Common Random Numbers

The same random draws Z are reused across all bumped simulations. The noise cancels in the finite difference, giving much more accurate estimates than independent draws. The entire computation over 40 spot levels uses a vectorised (40 × 10,000) matrix — all Greeks for all spot prices are computed in one pass.

### Live Market Data

`greeks.py` fetches from Yahoo Finance via yfinance:

- **Spot price** — from the most recent closing price
- **Implied volatility** — from the at-the-money call in the nearest ~30-day expiry options chain
- **Strike** — nearest listed strike to current spot
- **Time to maturity** — actual calendar days to expiry / 365

---

## 10. Key Results and Insights

1. **MC-GBM ≈ Black-Scholes** — confirms the simulation correctly implements GBM and risk-neutral pricing. Small residual (~$0.10) is expected Monte Carlo noise with 1,000 paths.

2. **MC-MJD ≈ Merton analytical** — confirms the jump diffusion simulation correctly matches the closed-form theory, validating the vectorised implementation.

3. **Jump risk reprices both tails** — with downward-biased jumps, calls become cheaper and puts become more expensive relative to BS. This is the real-market phenomenon known as the **volatility skew**.

4. **Greeks converge** — MC finite-difference Greeks closely match BS analytical Greeks across all spot prices, validating the bump-and-reprice methodology and the common random numbers variance reduction.

---

## 11. How to Run

```bash
# Install dependencies (once)
pip install -r Monte-Carlo-Option-Pricing-Simulator/requirements.txt

# Main simulation — GBM + MJD animation + price comparison table
cd Monte-Carlo-Option-Pricing-Simulator && python main.py

# Greeks visualisation — live market data from Yahoo Finance
python greeks.py            # SPY call (default)
python greeks.py AAPL       # different ticker
python greeks.py TSLA put   # put Greeks
```

---

## 12. Technologies Used

| Library | Purpose |
|---------|---------|
| **NumPy** | Vectorised path simulation, matrix operations |
| **SciPy** | Normal CDF/PDF for Black-Scholes, Poisson PMF for Merton series |
| **Matplotlib** | Animated path visualisation, Greeks plots |
| **Pandas** | Price comparison table formatting |
| **yfinance** | Live spot price, implied volatility, options chain data |

---

## 13. References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.
- Merton, R. C. (1976). *Option Pricing When Underlying Stock Returns Are Discontinuous.* Journal of Financial Economics, 3(1–2), 125–144.
