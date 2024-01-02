import numpy as np
import scipy.stats as si

"""Using Newton-Rhapson on Blackscholes to find IV"""

def d1(S, K, r, sigma, T):
    return (np.log(S/K) + (r + sigma**2 / 2) * (T/252)) / (sigma * np.sqrt(T/252))

def d2(S, K, r, sigma, T):
    return (np.log(S/K) + (r - sigma**2 / 2) * (T/252)) / (sigma * np.sqrt(T/252))


def call_price(S, K, r, sigma, T):
    return S * si.norm.cdf(d1(S, K, r, sigma, T), 0.0, 1.0) - K * np.exp(-r * T/252) * si.norm.cdf(d2(S, K, r, sigma, T), 0.0, 1.0)

def put_price(S, K, r, sigma, T):
    return K * np.exp(-r * T/252) * si.norm.cdf(-d2(S, K, r, sigma, T), 0.0, 1.0) - S * si.norm.cdf(-d1(S, K, r, sigma, T), 0.0, 1.0)


def call_vega(S, K, r, sigma, T):
    return S * np.sqrt(T/252) * si.norm.pdf(d1(S, K, r, sigma, T), 0.0, 1.0)

def put_vega(S, K, r, sigma, T):
    return S * np.sqrt(T/252) * si.norm.pdf(d1(S, K, r, sigma, T), 0.0, 1.0)


def call_imp_vol(S, K, r, T, C0, sigma_est, it=100):
    """
        S: initial stock price
        K:  strike price
        r:  risk-free rate
        T:  time to expiry
        C0: call option price
        sigma_est: realized volatility
    """
    for i in range(it):
        sigma_est -= ((call_price(S, K, r, sigma_est, T) - C0) / call_vega(S, K, r, sigma_est, T))
    return sigma_est

def put_imp_vol(S, K, r, T, P0, sigma_est, it=100):
    """
        S: initial stock price
        K:  strike price
        r:  risk-free rate
        T:  time to expiry
        P0: call option price
        sigma_est: realized volatility
    """
    for i in range(it):
        sigma_est -= ((put_price(S, K, r, sigma_est, T) - P0) / put_vega(S, K, r, sigma_est, T))
    return sigma_est