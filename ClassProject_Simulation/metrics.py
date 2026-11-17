"""
metrics.py - Post-processing of the simulated SIR samples.
"""

import numpy as np


def ecdf(x):
    """Empirical CDF: returns sorted values and their cumulative probabilities."""
    xs = np.sort(np.asarray(x))
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def prob_at_least(sir_db, threshold_db):
    """Fraction of users whose SIR is >= threshold_db."""
    return np.mean(np.asarray(sir_db) >= threshold_db)


def throughput(sir_db, total_bw_hz, reuse, gap_db=4.0):
    """Achievable bit rate (bit/s) of each user.

    R = (B / reuse) * log2(1 + SIR / Gamma)
    Each sector gets only B / reuse of the total bandwidth, and the
    constellations/codes are Gamma = gap_db away from capacity.
    """
    sir = 10 ** (np.asarray(sir_db) / 10)
    gap = 10 ** (gap_db / 10)
    return total_bw_hz / reuse * np.log2(1 + sir / gap)


def rate_attained_by(rates, fraction=0.97):
    """Bit rate attained by `fraction` of users (the (1-fraction) percentile)."""
    return np.percentile(rates, 100 * (1 - fraction))
