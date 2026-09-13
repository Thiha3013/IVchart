# ivlib — a vectorized implied volatility engine

Inverts Black-76 for implied volatility across a full equity option chain.
Validated against vendor marks on 548,163 real AAPL quotes; solves the whole set
in **19 ms**.

![validation](bench/validation.png)

## Results

| | |
|---|---|
| Throughput | **13.9M implied vols/sec** (Numba), 1.5M/s (pure NumPy) |
| Agreement with vendor marks | median ratio **1.0027**, 81.9% within 5% |
| Restricted to quotes the market pins to ±2 vol points (77.7%) | median **1.0043**, 92.5% within 5% |
| Solve coverage | 95.8% of quotes; the rest rejected as outside no-arbitrage bounds |
| Tests | 66 |

## What it does

```python
from ivlib import filter as qf, parity, surface

table = surface.build_iv_table(chain)      # quotes in, implied vol surface out
term  = surface.atm_term_structure(table)
iv30  = surface.constant_maturity(term, days=30)
```

The pipeline is four steps, one module each:

1. **`filter.py`** — drop crossed, zero-bid, sub-penny and wide-spread quotes,
   with a per-reason rejection waterfall so coverage is a reported number rather
   than an assumption.
2. **`parity.py`** — recover the forward price and discount factor from the
   quotes themselves. Since `C - P = df·(F - K)` is a straight line in `K`,
   regressing it per expiry gives `df = -slope` and `F = intercept / df`. No
   interest rate or dividend yield is downloaded; both are already in `F`.
   Median R² = 0.99998 across 7,845 fitted expiries.
3. **`solver.py`** / **`fast.py`** — invert Black-76 with a bracketed Newton that
   falls back to bisection whenever a step would leave the bracket, so it can
   neither diverge nor freeze. Quotes outside the no-arbitrage bounds admit no
   solution at all and are rejected explicitly.
4. **`surface.py`** — smiles, skew, term structure, and VIX-style constant-
   maturity series interpolated in total variance.

## The optimization ladder

Five implementations of the same inversion, same 262,796 real quotes, all
agreeing to within 1e-11:

| rung | M/s | vs L0 | source of the win |
|---|---|---|---|
| L0 scalar + `scipy.stats.norm` | 0.003 | 1× | the original shape |
| L1 scalar + `math.erf` | 0.155 | **52×** | cheaper normal CDF |
| L2 NumPy vectorized | 1.472 | 500× | memory layout |
| L3 NumPy + Corrado-Miller seed | 1.519 | 516× | fewer iterations |
| L4 Numba fused + parallel | **13.887** | **4,717×** | registers, 8 cores |

![ladder](bench/ladder.png)

Two results worth stating plainly, because both contradicted expectation:

- **The largest single jump is not vectorization.** `scipy.stats.norm.cdf` and
  `scipy.special.ndtr` compute the same function, but the first wraps it in
  distribution-object machinery that dominates a scalar loop. Removing it is
  worth 52× with no algorithmic change at all.
- **The algorithmic rung paid least.** A better initial guess was expected to be
  the interesting win and delivered 1.05×: mean iterations fall only 6.89 → 6.26,
  because the `|Δσ| < 1e-12` convergence test imposes a ~3-4 iteration floor and
  the mean is dragged by a tail (p50=6, p99=18) of elements needing bracketing
  steps regardless of where they start.

## Correctness

The dataset ships `C_IV`/`P_IV` columns computed by the data vendor — an
independent answer key. Running the pipeline under both time conventions:

```
T = DTE/365 (fixed)      median ratio 1.0027     81.9% within 5%
T = DTE/252 (original)   median ratio 0.8332      0.8% within 5%

predicted bias √(252/365) = 0.8309
observed                  = 0.8309
```

Residual disagreement is **not** solver error. Bucketing by how tightly each
quote's own bid-ask spread pins down a volatility (half-spread ÷ vega), agreement
tracks the market's own resolution monotonically — from 81% within 2% for the
tightest quotes down to 10% for the widest. A quote whose spread spans 10 vol
points cannot agree with anyone to better than 10 vol points.

## The surface

![surface](bench/surface.png)

262,752 implied volatilities across 570 trading days — 563× the 467 points the
original version produced, because a scalar loop could not afford the rest.

## Running it

```bash
pip install -e ".[dev,bench]"
python bench/ingest.py                 # CSV -> Parquet (195 MB -> 9.3 MB)
python bench/validate.py               # validate against vendor marks
python -m bench.ladder                 # the optimization ladder
python -m bench.plot_surface           # the surface figure
pytest                                 # 66 tests
```

## Notes on convention

- **Day count.** `T = calendar_days / 365`. The original divided a *calendar* day
  count by 252, a *trading* day denominator, overstating `T` by 45% and biasing
  implied vol low by exactly 0.8309. Both self-consistent alternatives —
  30/365 = 0.08219 and 21/252 = 0.08333 — agree to within 1.4%.
- **Prices.** Bid-ask mid, never last trade. On real data, using last trade moves
  median implied vol by ~7%.
- **Moneyness.** `k = ln(K/F)`, so strikes are comparable across days as the
  underlying moves.
- **OTM convention.** Puts below the forward, calls above — the liquid side.
