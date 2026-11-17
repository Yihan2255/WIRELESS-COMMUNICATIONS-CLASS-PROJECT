"""
q4_throughput.py - Question 4: throughput CDF for reuse 1, 3 and 9
(nu = 3.8, no power control, B = 100 MHz, SNR gap 4 dB).

Each sector only gets B / reuse of the bandwidth:
    R = (B / reuse) * log2(1 + SIR / Gamma)

Input : snap
Output: figures/q4_throughput.png, dict {reuse: (mean rate, 97 %-rate)} in Mb/s
"""

import os

import matplotlib.pyplot as plt

from metrics import ecdf, rate_attained_by, throughput
from simulator import compute_sir

BW_HZ = 100e6
GAP_DB = 4.0


def run(snap, nu=3.8, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    results = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for reuse in (1, 3, 9):
        rate = throughput(compute_sir(snap, nu, reuse, 0.0), BW_HZ, reuse, GAP_DB) / 1e6
        mean, r97 = rate.mean(), rate_attained_by(rate, 0.97)
        results[reuse] = (mean, r97)
        print(f"[Q4] reuse {reuse}: mean = {mean:7.2f} Mb/s   97 % of users >= {r97:6.2f} Mb/s")
        x, F = ecdf(rate)
        ax.plot(x, F, label=f"Reuse {reuse}: mean {mean:.1f} Mb/s, 97 % ≥ {r97:.2f} Mb/s")
    ax.set(xlabel="Throughput (Mb/s)", ylabel="CDF", xlim=(0, 1200), ylim=(0, 1),
           title="Q4: uplink throughput, B = 100 MHz, gap 4 dB, no PC")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(outdir, "q4_throughput.png"), dpi=150, bbox_inches="tight")
    # zoom on the lower tail, where the 97 % rate is read
    ax.set(xlim=(0, 30), ylim=(0, 0.2), title="Q4: lower tail (zoom)")
    ax.axhline(0.03, color="gray", ls=":", lw=0.8)
    fig.savefig(os.path.join(outdir, "q4_throughput_zoom.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return results
