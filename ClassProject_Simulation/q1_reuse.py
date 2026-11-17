"""
q1_reuse.py - Question 1: SIR CDF for reuse factors 1, 3 and 9, no power
control. Checks whether SIR >= -5 dB is achieved with 97 % probability.

Input : snap (from simulator.generate_snapshots)
Output: figures/q1_sir_reuse.png and a dict {reuse: P(SIR >= -5 dB)}
"""

import os

import matplotlib.pyplot as plt

from metrics import ecdf, prob_at_least
from simulator import compute_sir

THRESHOLD_DB = -5.0
TARGET = 0.97


def run(snap, nu=3.8, outdir="figures"):
    results = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for reuse in (1, 3, 9):
        sir = compute_sir(snap, nu=nu, reuse=reuse, eta=0.0)
        p = prob_at_least(sir, THRESHOLD_DB)
        results[reuse] = p
        x, F = ecdf(sir)
        ax.plot(x, F, label=f"Reuse {reuse}  (P[SIR ≥ -5 dB] = {100 * p:.1f} %)")
        print(f"[Q1] reuse {reuse}: P(SIR >= -5 dB) = {100 * p:.2f} %  "
              f"-> {'meets' if p >= TARGET else 'does NOT meet'} the 97 % target")
    ax.axvline(THRESHOLD_DB, color="k", ls="--", lw=0.8)
    ax.axhline(1 - TARGET, color="gray", ls=":", lw=0.8, label="3 % outage")
    ax.set(xlabel="SIR (dB)", ylabel="CDF", xlim=(-20, 60), ylim=(0, 1),
           title=f"Q1: uplink SIR, no power control (ν = {nu})")
    ax.grid(alpha=0.3)
    ax.legend()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "q1_sir_reuse.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return results
