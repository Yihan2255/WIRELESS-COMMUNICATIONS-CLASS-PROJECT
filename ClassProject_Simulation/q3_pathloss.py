"""
q3_pathloss.py - Question 3: repeats question 2 (reuse 3, optimal fractional
power control) for pathloss exponents nu = 3, 3.8 and 4.5 and plots the SIR
CDFs together.

Input : snap (same snapshots for every nu -> fair comparison)
Output: figures/q3_sir_pathloss.png, figures/q3_eta_sweep.png,
        dict {nu: (best_eta, P(SIR >= -5 dB))}
"""

import os

import matplotlib.pyplot as plt

from metrics import ecdf
from q2_power_control import ETAS, THRESHOLD_DB, optimal_eta
from simulator import compute_sir

NUS = (3.0, 3.8, 4.5)


def run(snap, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    results = {}
    fig_c, ax_c = plt.subplots(figsize=(7, 5))
    fig_s, ax_s = plt.subplots(figsize=(7, 4.5))
    for nu in NUS:
        best, probs = optimal_eta(snap, nu, 3)
        results[nu] = (best, probs.max())
        print(f"[Q3] nu = {nu}: best eta = {best:.2f}  "
              f"P(SIR >= -5 dB) = {100 * probs.max():.2f} %")
        x, F = ecdf(compute_sir(snap, nu, 3, best))
        ax_c.plot(x, F, label=f"ν = {nu}, η = {best:.2f}  ({100 * probs.max():.1f} % ≥ -5 dB)")
        ax_s.plot(ETAS, 100 * probs, label=f"ν = {nu}")
    ax_c.axvline(THRESHOLD_DB, color="k", ls="--", lw=0.8)
    ax_c.set(xlabel="SIR (dB)", ylabel="CDF", xlim=(-20, 60), ylim=(0, 1),
             title="Q3: uplink SIR, reuse 3, optimal fractional PC")
    ax_c.grid(alpha=0.3)
    ax_c.legend()
    fig_c.savefig(os.path.join(outdir, "q3_sir_pathloss.png"), dpi=150, bbox_inches="tight")
    ax_s.set(xlabel="Power-control exponent η", ylabel="Users with SIR ≥ -5 dB (%)",
             title="Q3: η sweep for each pathloss exponent (reuse 3)")
    ax_s.grid(alpha=0.3)
    ax_s.legend()
    fig_s.savefig(os.path.join(outdir, "q3_eta_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    plt.close(fig_s)
    return results
