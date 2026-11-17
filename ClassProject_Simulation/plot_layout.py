"""
plot_layout.py - Draws the 19-cell / 57-sector layout with one snapshot of
users, coloured by their reuse-9 frequency band. Useful to check the geometry.

Run:  python plot_layout.py   -> figures/layout.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import hexgrid
from simulator import frequency_band, generate_snapshots


def plot_layout(outdir="figures", R=1.0, seed=1):
    axial = hexgrid.cell_axial_coords(2)
    centers = hexgrid.cell_centers(axial, R)
    ang = np.deg2rad(np.arange(0, 361, 60))
    fig, ax = plt.subplots(figsize=(7, 7))
    for n, c in enumerate(centers):
        ax.plot(c[0] + R * np.cos(ang), c[1] + R * np.sin(ang), "k-", lw=1)
        for k in range(3):                       # sector borders
            a = 2 * np.pi / 3 * k
            ax.plot([c[0], c[0] + R * np.cos(a)], [c[1], c[1] + R * np.sin(a)],
                    "k:", lw=0.7)
        ax.text(c[0], c[1], str(n), ha="center", va="center", fontsize=8)

    snap = generate_snapshots(1, seed=seed)
    pos = hexgrid.sample_users(np.random.default_rng(seed), 1, centers, R)[0]
    band = frequency_band(snap, 9)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=band, cmap="tab10", s=25)
    ax.set_aspect("equal")
    ax.set_title("19 cells x 3 sectors, one user per sector (colour = reuse-9 band)")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="band")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "layout.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_layout()
