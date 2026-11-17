"""
hexgrid.py - Geometry of the simulation universe.

19 hexagonal cells (central cell + 2 rings), each split into 3 identical
120-degree sectors. Every base station is at its cell centre and has 3
sector antennas pointing at the sectors.

Conventions
-----------
* Cell radius R (centre-to-vertex). SIR does not depend on R, so R = 1.
* Hexagon vertices are at angles 0, 60, ..., 300 deg from the cell centre.
* Sector k (k = 0, 1, 2) spans the angles [120k, 120k + 120) deg, i.e. it is
  the rhombus with corners: centre, vertex at 120k, vertex at 120k+60,
  vertex at 120k+120.
* Cells are indexed with axial coordinates (i, j): centre = sqrt(3) R (i e1 + j e2),
  with e1 at 30 deg and e2 at 90 deg. Index 0 is always the central cell.
"""

import numpy as np

N_SECTORS = 3


def cell_axial_coords(rings=2):
    """Axial coordinates (i, j) of all cells within `rings` rings of the centre.

    Output: int array (n_cells, 2); row 0 is the central cell (0, 0).
    For rings = 2 there are 19 cells.
    """
    coords = []
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            # hex distance for this basis (neighbours are ±(1,0), ±(0,1), ±(1,-1))
            if (abs(i) + abs(j) + abs(i + j)) // 2 <= rings:
                coords.append((i, j))
    coords.sort(key=lambda c: (abs(c[0]) + abs(c[1]) + abs(c[0] + c[1]), c))
    return np.array(coords, dtype=int)


def cell_centers(axial, R=1.0):
    """Cartesian centres (n_cells, 2) of the cells given in axial coordinates."""
    e1 = np.sqrt(3) * R * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    e2 = np.sqrt(3) * R * np.array([0.0, 1.0])
    return axial[:, [0]] * e1 + axial[:, [1]] * e2


def cell_reuse_group(axial, cell_reuse):
    """Frequency group of each cell for a cell-level reuse factor (1 or 3).

    For reuse 3 the classic 3-cell cluster is used: group = (i - j) mod 3,
    which guarantees that adjacent cells never share a group. Co-channel
    cells are then at distance D = 3R.
    """
    if cell_reuse == 1:
        return np.zeros(len(axial), dtype=int)
    if cell_reuse == 3:
        return (axial[:, 0] - axial[:, 1]) % 3
    raise ValueError("cell_reuse must be 1 or 3")


def sample_users(rng, n_snapshots, centers, R=1.0):
    """Place one user uniformly at random in every sector of every cell.

    Inputs : rng (np.random.Generator), n_snapshots, centers (n_cells, 2), R.
    Output : positions, array (n_snapshots, n_cells * 3, 2).
             User u belongs to cell u // 3 and sector u % 3.

    A sector is a rhombus spanned by two edges of length R at 120 deg, so a
    uniform point is a*v0 + b*v2 with a, b ~ U(0,1); it is then rotated to
    the right sector and shifted to the cell centre.
    """
    n_cells = len(centers)
    a = rng.random((n_snapshots, n_cells, N_SECTORS))
    b = rng.random((n_snapshots, n_cells, N_SECTORS))
    # rhombus of sector 0: v0 at 0 deg, v2 at 120 deg
    x = R * (a - 0.5 * b)
    y = R * (np.sqrt(3) / 2) * b
    # rotate by 120k deg for sector k
    rot = 2 * np.pi / 3 * np.arange(N_SECTORS)
    c, s = np.cos(rot), np.sin(rot)
    xr = x * c - y * s
    yr = x * s + y * c
    pos = np.stack([xr, yr], axis=-1) + centers[None, :, None, :]
    return pos.reshape(n_snapshots, n_cells * N_SECTORS, 2)


def sector_of_angle(angle):
    """Index (0, 1, 2) of the 120-deg sector antenna that covers `angle` (rad)."""
    return (np.floor(np.mod(angle, 2 * np.pi) / (2 * np.pi / 3)).astype(int)) % 3
