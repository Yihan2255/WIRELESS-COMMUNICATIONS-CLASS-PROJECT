"""
main.py - Runs the whole class project.

    python main.py [n_snapshots]      (default 20000)

Generates one common set of snapshots (positions + shadowing) and answers
questions 1-4; all figures are written to ./figures and all numbers to
./figures/results.txt.
"""

import os
import sys
from contextlib import redirect_stdout

import q1_reuse
import q2_power_control
import q3_pathloss
import q4_throughput
from plot_layout import plot_layout
from simulator import generate_snapshots


class Tee:
    """Writes to several streams at once (console + results file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main(n_snapshots=20000, outdir="figures", seed=0):
    os.makedirs(outdir, exist_ok=True)
    snap = generate_snapshots(n_snapshots, sigma_db=8.0, seed=seed)
    with open(os.path.join(outdir, "results.txt"), "w", encoding="utf-8") as f, \
            redirect_stdout(Tee(sys.stdout, f)):
        print(f"Snapshots: {n_snapshots} (x3 central sectors = {3 * n_snapshots} SIR samples)\n")
        plot_layout(outdir)
        q1_reuse.run(snap, nu=3.8, outdir=outdir)
        print()
        q2_power_control.run(snap, nu=3.8, outdir=outdir)
        print()
        q3_pathloss.run(snap, outdir=outdir)
        print()
        q4_throughput.run(snap, nu=3.8, outdir=outdir)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
