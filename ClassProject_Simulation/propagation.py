"""
propagation.py - Large-scale channel model (no multipath fading).

Channel power gain of a link of length d:
    G = d^(-nu) * 10^(S/10),   S ~ N(0, sigma^2) dB   (log-normal shadowing)
Thermal noise is neglected (interference-limited), so any constant in the
pathloss (antenna gains, reference distance, carrier frequency) cancels in
the SIR and is omitted.
"""

import numpy as np


def shadowing_db(rng, shape, sigma_db=8.0):
    """Independent log-normal shadow fading samples, in dB."""
    return sigma_db * rng.standard_normal(shape)


def channel_gain(d, shadow_db, nu):
    """Linear power gain d^(-nu) * 10^(S/10) of each link."""
    return d ** (-nu) * 10.0 ** (shadow_db / 10.0)
