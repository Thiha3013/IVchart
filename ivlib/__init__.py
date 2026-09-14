"""ivlib -- vectorized implied volatility engine.

    pricing   Black-76 formulas, greeks, no-arbitrage bounds, initial guesses
    solver    the inversion: bracketed Newton, NumPy and Numba implementations
    market    quote filtering and the forward/discount implied by put-call parity
    surface   smiles, skew, term structure, constant-maturity series
"""
from ivlib import market, pricing, solver, surface  # noqa: F401

__version__ = "0.2.0"
