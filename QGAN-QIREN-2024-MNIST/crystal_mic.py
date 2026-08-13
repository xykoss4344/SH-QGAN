"""Shared crystal decode + minimum-image-convention distance check.

Six eval_*.py files each carried their own copy of this and they disagreed:
eval_v4.py used no periodic wrap at all (reporting 47% where the periodic
check gives 5.9%), and every copy skipped the 15A -> 10A re-box that
datasets/view_atoms_mgmno.py:back_to_10_cell performs. On real structures that
omission cost 1.92 -> 1.28 A median min-distance.

One definition, used by probe_collapse.py and every eval.
"""
import numpy as np
from ase import Atoms

# Slot layout of the 28-atom representation: Mg 0:8, Mn 8:16, O 16:28.
SPECIES_MAP = ['Mg'] * 8 + ['Mn'] * 8 + ['O'] * 12

# datasets/make_representation.py:go_to_15_cell places atoms in a 15A box
# offset by 2.5A, so real occupied coords live in [1/6, 5/6]. Undo that to get
# true fractional coordinates before applying the real lattice.
BOX_OFFSET, BOX_SCALE = 1.0 / 6.0, 2.0 / 3.0

VALID_THRESHOLD = 1.0  # Angstrom

# Species-aware hard floors (Angstrom): distances below these are unphysical for
# Mg-Mn-O, roughly 0.8x the typical bond length in the relevant oxides. The flat
# 1.0A validity bar says nothing about chemistry -- an Mg-O pair at 1.1A passes it
# and is still nonsense. Training against these targets chemical plausibility, and
# clearing them clears 1.0A for free.
_MIN_SEP = {
    ('Mg', 'Mg'): 2.4, ('Mg', 'Mn'): 2.3, ('Mg', 'O'): 1.7,
    ('Mn', 'Mn'): 2.2, ('Mn', 'O'): 1.6, ('O', 'O'): 2.0,
}


def min_separation_matrix(species_map=SPECIES_MAP):
    """(28, 28) matrix of per-pair minimum allowed separation in Angstrom."""
    n = len(species_map)
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            a, b = species_map[i], species_map[j]
            m[i, j] = _MIN_SEP.get((a, b)) or _MIN_SEP[(b, a)]
    return m


def decode(coords_90, label_28, species_map=SPECIES_MAP):
    """90-dim vector -> (species, fractional coords, 6 cell params).

    Row 0 is lengths/30, row 1 is angles/180, rows 2: are the 28 atom slots.
    `label_28` selects which slots are occupied.
    """
    arr = np.asarray(coords_90, dtype=float).reshape(30, 3)
    lengths = np.clip(arr[0] * 30.0, 2.0, 30.0)
    angles = np.clip(arr[1] * 180.0, 30.0, 150.0)
    occ = np.asarray(label_28).reshape(-1).astype(bool)
    frac = (arr[2:][occ] - BOX_OFFSET) / BOX_SCALE
    species = [species_map[i] for i in range(len(occ)) if occ[i]]
    return species, frac, np.concatenate([lengths, angles])


def min_dist(coords_90, label_28, species_map=SPECIES_MAP):
    """Minimum interatomic distance in Angstrom under PBC.

    Returns 0.0 (i.e. invalid) for degenerate cells or fewer than two atoms.
    """
    species, frac, cell = decode(coords_90, label_28, species_map)
    if len(species) < 2:
        return 0.0
    try:
        atoms = Atoms(symbols=species, scaled_positions=frac, cell=cell, pbc=True)
        d = atoms.get_all_distances(mic=True)
    except Exception:
        return 0.0
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def validity(coords, labels, threshold=VALID_THRESHOLD, species_map=SPECIES_MAP):
    """(fraction valid, per-sample min distances) for a batch of structures."""
    dists = np.array([min_dist(c, l, species_map) for c, l in zip(coords, labels)])
    return float((dists >= threshold).mean()), dists


def _demo():
    """Real structures must be ~100% valid. They are the ground truth."""
    import pickle, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'datasets', 'mgmno_100.pickle')
    raw = pickle.load(open(path, 'rb'))
    coords = [np.array(c).flatten() for c, l in raw]
    labels = [np.array(l).flatten() for c, l in raw]
    frac_valid, dists = validity(coords, labels)
    print(f'real data: {frac_valid * 100:.1f}% valid, median min-dist {np.median(dists):.3f} A')
    assert frac_valid > 0.99, f'decode is broken: real data only {frac_valid:.3f} valid'
    assert np.median(dists) > 1.8, f'median {np.median(dists):.3f} A, expected ~1.92'
    print('OK')


if __name__ == '__main__':
    _demo()
