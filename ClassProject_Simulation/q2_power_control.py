"""
q2_power_control.py - Question 2: fractional power control with reuse 3.
Sweeps the power-control exponent eta in [0, 1] and finds the one that
maximises the fraction of users with SIR >= -5 dB.

Functions
---------
optimal_eta(snap, nu, reuse, etas) -> (best_eta, probs array)
run(snap, nu, outdir)              -> dict with best eta and probability
Outputs: figures/q2_eta_sweep.png, figures/q2_sir_pc.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from metrics import ecdf, prob_at_least
from simulator import compute_sir

THRESHOLD_DB = -5.0
ETAS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def optimal_eta(snap, nu=3.8, reuse=3, etas=ETAS):
    """Exponent maximising P(SIR >= -5 dB), and the probability for every eta."""
    probs = np.array([prob_at_least(compute_sir(snap, nu, reuse, e), THRESHOLD_DB)
                      for e in etas])
    return etas[np.argmax(probs)], probs


def run(snap, nu=3.8, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    best, probs = optimal_eta(snap, nu, 3)
    print(f"[Q2] reuse 3, nu = {nu}: best eta = {best:.2f}  "
          f"P(SIR >= -5 dB) = {100 * probs.max():.2f} %  (eta=0: {100 * probs[0]:.2f} %)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ETAS, 100 * probs)
    ax.axvline(best, color="r", ls="--", label=f"optimum η = {best:.2f}")
    ax.set(xlabel="Power-control exponent η", ylabel="Users with SIR ≥ -5 dB (%)",
           title=f"Q2: reuse 3, ν = {nu}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(outdir, "q2_eta_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for eta, lab in ((0.0, "No power control (η = 0)"),
                     (best, f"Fractional PC, η = {best:.2f}")):
        x, F = ecdf(compute_sir(snap, nu, 3, eta))
        ax.plot(x, F, label=lab)
    ax.axvline(THRESHOLD_DB, color="k", ls="--", lw=0.8)
    ax.set(xlabel="SIR (dB)", ylabel="CDF", xlim=(-20, 50), ylim=(0, 1),
           title=f"Q2: uplink SIR, reuse 3, ν = {nu}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(outdir, "q2_sir_pc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"best_eta": best, "p_best": probs.max(), "p_no_pc": probs[0]}
