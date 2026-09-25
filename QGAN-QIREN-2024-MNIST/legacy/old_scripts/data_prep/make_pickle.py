import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
make_pickle.py  (QINR-QGAN version)
====================================
Converts augmented mgmno_N.npy → mgmno_N.pickle in the format
expected by train_crystal.py.

Each sample: (image [30,3], label [28,1])
  image[0:2]   = cell (lengths/30, angles/180)
  image[2:10]  = Mg positions (8 slots, zero-padded)
  image[10:18] = Mn positions (8 slots, zero-padded)
  image[18:30] = O  positions (12 slots, zero-padded)
  label        = binary occupancy per slot (row_sum > 0.4)

Usage:
    py make_pickle.py 1000
    → reads  datasets/mgmno_1000.npy
    → writes datasets/mgmno_1000.pickle
"""

import os
import sys
import numpy as np
import pickle
from tqdm import tqdm

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
THRESHOLD   = 0.4


def make_label(image):
    """
    Given image (30, 3), produce label (28, 1) with binary occupancy.
    Rows 2-9:  Mg slots → label[0:8]
    Rows 10-17: Mn slots → label[8:16]
    Rows 18-29: O  slots → label[16:28]
    """
    pos = image[2:, :]
    mg_occ = (np.sum(pos[:8, :],   axis=1) > THRESHOLD).astype(np.float32).reshape(8,  1)
    mn_occ = (np.sum(pos[8:16, :], axis=1) > THRESHOLD).astype(np.float32).reshape(8,  1)
    o_occ  = (np.sum(pos[16:, :],  axis=1) > THRESHOLD).astype(np.float32).reshape(12, 1)
    return np.vstack([mg_occ, mn_occ, o_occ])


def main():
    if len(sys.argv) < 2:
        print("Usage: py make_pickle.py <N>")
        print("Example: py make_pickle.py 1000")
        sys.exit(1)

    n = sys.argv[1]
    npy_path    = os.path.join(DATASET_DIR, f"mgmno_{n}.npy")
    pickle_path = os.path.join(DATASET_DIR, f"mgmno_{n}.pickle")

    print(f"Loading: {npy_path}")
    if not os.path.isfile(npy_path):
        print(f"ERROR: {npy_path} not found. Run data_augmentation.py first.")
        sys.exit(1)

    data = np.load(npy_path, allow_pickle=True)
    print(f"Loaded {len(data)} samples, first shape: {data[0].shape}")

    output  = []
    skipped = 0
    for i in tqdm(range(len(data)), desc="Building pickle"):
        img = data[i]
        if img.shape != (30, 3):
            skipped += 1
            continue
        label = make_label(img)
        output.append((img, label))

    print(f"\nValid samples: {len(output)}  |  Skipped: {skipped}")

    with open(pickle_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved: {pickle_path}")

    # Verification
    with open(pickle_path, "rb") as f:
        verify = pickle.load(f)
    s_img, s_lbl = verify[0]
    print(f"\nVerification:")
    print(f"  Samples     : {len(verify)}")
    print(f"  Image shape : {s_img.shape}")
    print(f"  Label shape : {s_lbl.shape}")
    print(f"  Atom count  : {int(s_lbl.sum())}")


if __name__ == "__main__":
    main()
