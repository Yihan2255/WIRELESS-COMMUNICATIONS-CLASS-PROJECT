"""
simulator.py - Monte-Carlo uplink SIR simulation.

Workflow
--------
1. generate_snapshots(): draws the random inputs of every snapshot
   (user positions and shadow fading). Drawing them once lets every
   configuration (reuse factor, power-control exponent, pathloss exponent)
   be evaluated on the same snapshots, so the curves compare fairly.
2. compute_sir(): for a given configuration, returns the uplink SIR measured
   at the three sector antennas of the central base station.

Model
-----
* One user per sector on the channel under study, uniform in its sector,
  served by its own sector antenna.
* Ideal 120-deg sector antennas: gain 1 inside the sector, 0 outside.
  A user interferes with central sector k only if (a) it uses the same
  frequency band and (b) it lies inside the angular span of antenna k.
* Shadowing is drawn independently for every user -> site link. Each user
  needs two values: towards its own site (used by power control) and towards
  the central site (used for the received power). For users of the central
  cell both are the same link.
* Fractional power control (uplink, no power cap; P0 cancels in the SIR):
      P_tx = G_own^(-eta)      (eta = 0: no PC, eta = 1: full inversion)
  where G_own is the channel gain (pathloss + shadowing) to the serving BS.
"""

import numpy as np

import hexgrid
from propagation import channel_gain, shadowing_db

N_SEC = hexgrid.N_SECTORS


def generate_snapshots(n_snapshots, sigma_db=8.0, R=1.0, rings=2, seed=0):
    """Draw the random inputs for `n_snapshots` snapshots.

    Output: dict with
      d_own, d_cen   : (n_snap, n_users) distance to own / central BS
      s_own, s_cen   : (n_snap, n_users) shadowing (dB) to own / central BS
      beam           : (n_snap, n_users) central sector antenna covering the user
      cell, sector   : (n_users,) cell index and sector index of each user
      axial          : (n_cells, 2) axial coordinates of the cells
    """
    rng = np.random.default_rng(seed)
    axial = hexgrid.cell_axial_coords(rings)
    centers = hexgrid.cell_centers(axial, R)
    n_users = len(centers) * N_SEC

    pos = hexgrid.sample_users(rng, n_snapshots, centers, R)
    cell = np.repeat(np.arange(len(centers)), N_SEC)
    sector = np.tile(np.arange(N_SEC), len(centers))

    rel_own = pos - centers[cell][None, :, :]
    d_own = np.linalg.norm(rel_own, axis=-1)
    d_cen = np.linalg.norm(pos, axis=-1)          # central BS at the origin
    beam = hexgrid.sector_of_angle(np.arctan2(pos[..., 1], pos[..., 0]))

    s_own = shadowing_db(rng, (n_snapshots, n_users), sigma_db)
    s_cen = shadowing_db(rng, (n_snapshots, n_users), sigma_db)
    central = cell == 0
    s_cen[:, central] = s_own[:, central]         # same physical link

    return dict(d_own=d_own, d_cen=d_cen, s_own=s_own, s_cen=s_cen,
                beam=beam, cell=cell, sector=sector, axial=axial)


def frequency_band(snap, reuse):
    """Frequency band used by each user for total reuse factor 1, 3 or 9.

    reuse 1 : same band everywhere
    reuse 3 : band = sector index (universal reuse across cells)
    reuse 9 : 3-cell reuse across cells x 3 bands across sectors
    """
    if reuse == 1:
        return np.zeros_like(snap["sector"])
    if reuse == 3:
        return snap["sector"].copy()
    if reuse == 9:
        group = hexgrid.cell_reuse_group(snap["axial"], 3)
        return group[snap["cell"]] * N_SEC + snap["sector"]
    raise ValueError("reuse must be 1, 3 or 9")


def compute_sir(snap, nu=3.8, reuse=1, eta=0.0):
    """Uplink SIR (dB) at the 3 sector antennas of the central cell.

    Inputs : snap (from generate_snapshots), pathloss exponent nu,
             reuse factor (1, 3, 9), power-control exponent eta.
    Output : 1-D array of n_snapshots * 3 SIR samples in dB
             (the 3 central sectors are statistically identical).
    """
    band = frequency_band(snap, reuse)
    p_tx = channel_gain(snap["d_own"], snap["s_own"], nu) ** (-eta)
    p_rx = p_tx * channel_gain(snap["d_cen"], snap["s_cen"], nu)

    sir = []
    for k in range(N_SEC):                      # central users are 0, 1, 2
        co_channel = band == band[k]
        co_channel[k] = False                   # the desired user itself
        mask = co_channel[None, :] & (snap["beam"] == k)
        interference = np.sum(p_rx * mask, axis=1)
        sir.append(p_rx[:, k] / interference)
    return 10 * np.log10(np.concatenate(sir))
