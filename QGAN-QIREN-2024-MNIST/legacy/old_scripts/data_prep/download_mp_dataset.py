import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
download_mp_dataset.py
======================
Downloads all Mg-Mn-O crystal structures from the Materials Project,
converts them to the (coords_90, labels_28) format used by train_crystal.py,
and saves as datasets/mgmno_mp.pickle.

Usage:
    py -3.12 download_mp_dataset.py

Label format (28-dim binary vector):
    [0:8]   = Mg slot occupancy  (1 if Mg atom in slot i)
    [8:16]  = Mn slot occupancy  (1 if Mn atom in slot i)
    [16:28] = O  slot occupancy  (1 if O  atom in slot i)

Coord format (90-dim float vector):
    30 atoms x 3 fractional coords, zero-padded.
    Atom order: Mg (up to 8), Mn (up to 8), O (up to 12).
"""

import os, sys, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
from mp_api.client import MPRester

MP_API_KEY = os.environ.get('MP_API_KEY', 'hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA')

MG_SLOTS = 8
MN_SLOTS = 8
O_SLOTS  = 12
MAX_ATOMS = MG_SLOTS + MN_SLOTS + O_SLOTS  # 28 -> but we use 30 rows in output

def structure_to_tensors(structure):
    """
    Convert a pymatgen Structure (Mg-Mn-O only, ≤28 atoms each type)
    to (coords_90, labels_28).

    Returns None if the structure contains elements other than Mg/Mn/O,
    or has too many atoms of any type.
    """
    allowed = {'Mg', 'Mn', 'O'}
    mg_pos, mn_pos, o_pos = [], [], []

    for site in structure:
        el = site.species_string
        if el not in allowed:
            return None  # skip mixed/other-element structures
        fp = list(site.frac_coords)
        if el == 'Mg':
            mg_pos.append(fp)
        elif el == 'Mn':
            mn_pos.append(fp)
        elif el == 'O':
            o_pos.append(fp)

    # Truncate to slot limits
    mg_pos = mg_pos[:MG_SLOTS]
    mn_pos = mn_pos[:MN_SLOTS]
    o_pos  = o_pos[:O_SLOTS]

    # Must have at least 1 of each element
    if not mg_pos or not mn_pos or not o_pos:
        return None

    # Build 28-dim binary label (slot occupancy)
    label = np.zeros(28, dtype=np.float32)
    for i in range(len(mg_pos)):
        label[i] = 1.0
    for i in range(len(mn_pos)):
        label[8 + i] = 1.0
    for i in range(len(o_pos)):
        label[16 + i] = 1.0

    # Build 90-dim coord vector (30 rows x 3 cols, zero-padded)
    # Row order: Mg rows, Mn rows, O rows, zero padding to 30 rows
    all_pos = mg_pos + mn_pos + o_pos  # up to 28 positions
    n = len(all_pos)
    coords = np.zeros((30, 3), dtype=np.float32)
    for i, pos in enumerate(all_pos):
        coords[i] = pos
    coords_flat = coords.flatten()  # (90,)

    return coords_flat, label


def download_dataset(output_path='datasets/mgmno_mp.pickle'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Connecting to Materials Project API...")
    with MPRester(MP_API_KEY) as mpr:
        print("Fetching Mg-Mn-O entries (this may take a minute)...")
        docs = mpr.materials.summary.search(
            chemsys='Mg-Mn-O',
            fields=['structure', 'material_id', 'formula_pretty', 'energy_above_hull']
        )
    print(f"Retrieved {len(docs)} entries from Materials Project.")

    dataset = []
    skipped_elements = 0
    skipped_empty    = 0
    converted        = 0

    for doc in docs:
        if doc.structure is None:
            continue
        result = structure_to_tensors(doc.structure)
        if result is None:
            skipped_elements += 1
            continue
        coords_flat, label = result
        dataset.append((coords_flat, label))
        converted += 1

    print(f"\nConversion complete:")
    print(f"  Converted   : {converted}")
    print(f"  Skipped (bad elements / empty) : {skipped_elements}")
    print(f"  Total saved : {len(dataset)}")

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\nSaved to {output_path}")

    # Quick sanity check
    c, l = dataset[0]
    print(f"Sample coords shape: {c.shape}  Label shape: {l.shape}")
    print(f"Label example: {l}")


if __name__ == '__main__':
    download_dataset()
