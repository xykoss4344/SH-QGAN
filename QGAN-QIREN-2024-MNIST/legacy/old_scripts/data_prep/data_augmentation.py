import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
data_augmentation.py  (QINR-QGAN version)
==========================================
1:1 port of the classical CrystalGAN augmentation pipeline.
Augments raw Mg-Mn-O structures to N samples per composition
using permutation + rotation + translation invariance.

Usage:
    py data_augmentation.py 1000
    → produces datasets/mgmno_1000.npy + datasets/mgmno_names_1000
    Then run make_pickle.py to convert to .pickle training format.

Requires:
    - datasets/unique_sc_mgmno_comp_dict   (composition → image dict)
    - datasets/make_representation.py       (image ↔ atoms conversion)
    - datasets/view_atoms_mgmno.py          (image → ASE Atoms)
"""

import numpy as np
import os
import sys
import pickle
from tqdm import tqdm

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")

# Add datasets/ to path so we can import its modules
sys.path.insert(0, DATASET_DIR)
import make_representation
import view_atoms_mgmno


# ── Augmentation functions (identical to classical GAN) ───────────────────────

def permutation(image):
    """Randomly shuffle atom slot ordering within each element group."""
    c  = image[:2, :]       # cell params (2 rows)
    mg = image[2:10, :]     # Mg positions (8 slots)
    mn = image[10:18, :]    # Mn positions (8 slots)
    o  = image[18:, :]      # O  positions (12 slots)

    mg_index = np.random.choice(8,  8,  replace=False)
    mn_index = np.random.choice(8,  8,  replace=False)
    o_index  = np.random.choice(12, 12, replace=False)

    new_mg = mg[mg_index, :]
    new_mn = mn[mn_index, :]
    new_o  = o[o_index, :]

    return np.vstack((c, new_mg, new_mn, new_o))


def do_translation(image, n=None):
    """Apply random fractional translation in all 3 axes + re-permute."""
    cellcell = image[:2, :]
    atoms, image = view_atoms_mgmno.view_atoms(image, view=False)
    atoms0 = atoms.copy()
    pos  = atoms0.get_positions()
    cell = atoms0.get_cell()

    delta = np.random.uniform(0, 1, size=3).reshape(1, 3)
    delta = np.multiply(np.linalg.norm(cell, axis=1), delta)
    new_pos = pos + delta

    atoms.set_positions(new_pos)
    new_atoms = atoms.copy()
    new_image = make_representation.do_feature(new_atoms)

    temp = new_image[2:, :]
    final_new_image = np.vstack((cellcell, temp))
    final_new_image = permutation(final_new_image)
    return final_new_image


def remain(image_list, b):
    """Fill remainder to reach exact target count via random translations."""
    m  = len(image_list)
    mm = np.arange(m)
    index_list = np.random.choice(mm, b)
    remain_list = []
    name_list   = []
    for index in index_list:
        image = image_list[index][0]
        name  = image_list[index][1]
        image_t = do_translation(image)
        remain_list.append(image_t)
        name_list.append(name)
    return remain_list, name_list


# ── Main augmentation loop ────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: py data_augmentation.py <target_per_composition>")
        print("Example: py data_augmentation.py 1000")
        sys.exit(1)

    ag_number = int(sys.argv[1])

    dict_path = os.path.join(DATASET_DIR, "unique_sc_mgmno_comp_dict")
    print(f"Loading composition dictionary from: {dict_path}")
    comp_image_dict = np.load(dict_path, allow_pickle=True)
    comp_list = comp_image_dict.keys()

    final = []
    names = []

    print(f"Target: {ag_number} structures per composition")
    print("=" * 60)

    for ii, comp in enumerate(comp_list):
        comp_number = int(len(comp_image_dict[comp]))
        print(f"\nComposition: {comp}  ({comp_number} original structures)")

        if ag_number <= comp_number:
            print("  → augmentation not required (enough samples)")
            continue

        # Determine rotation multiplier based on underrepresentation
        a = ag_number / comp_number
        b = ag_number % comp_number

        if comp_number <= ag_number / 4:
            n_r = 3
        elif comp_number <= ag_number / 2:
            n_r = 1
        else:
            n_r = 0

        comp_number_rot = comp_number * (n_r + 1)
        n_t = int(ag_number / comp_number_rot - 1)
        t_c = ag_number - (n_t + 1) * comp_number_rot

        final_images = []
        final_names  = []

        print(f"  n_rotations={n_r}, n_translations={n_t}, remainder={t_c}")
        print(f"  expected total = {comp_number * (n_r+1) * (n_t+1) + t_c}")

        for i in range(len(comp_image_dict[comp])):
            image = comp_image_dict[comp][i][0]
            name  = comp_image_dict[comp][i][1]

            image_after_rotation = [image]
            name_after_rotation  = [name]

            # Axis-swap rotations
            x_r = image[:, 0].reshape(30, 1)
            y_r = image[:, 1].reshape(30, 1)
            z_r = image[:, 2].reshape(30, 1)
            r1 = np.hstack((y_r, x_r, z_r))
            r2 = np.hstack((z_r, y_r, x_r))
            r3 = np.hstack((x_r, z_r, y_r))

            if n_r == 3:
                image_after_rotation += [r1, r2, r3]
                name_after_rotation  += [name, name, name]
            elif n_r == 1:
                image_after_rotation += [r2]
                name_after_rotation  += [name]

            m = len(image_after_rotation)
            image_after_rotation = np.array(image_after_rotation)

            # Translation augmentations
            image_after_translation = []
            name_after_translation  = []

            for iii in range(m):
                image_ = image_after_rotation[iii]
                image_after_translation.append(image_)
                name_after_translation.append(name)
                for iiii in range(n_t):
                    t_image = do_translation(image_)
                    image_after_translation.append(t_image)
                    name_after_translation.append(name)

            final_images += image_after_translation
            final_names  += name_after_translation

        # Fill remainder
        remain_list, remain_list_name = remain(comp_image_dict[comp], t_c)
        print(f"  before remainder: {len(final_images)}, +{len(remain_list)} = {len(final_images)+len(remain_list)}")
        final_images += remain_list
        final_names  += remain_list_name

        final += final_images
        names += final_names

    final = np.array(final)
    print(f"\n{'='*60}")
    print(f"Total augmented dataset shape: {final.shape}")

    out_npy   = os.path.join(DATASET_DIR, f"mgmno_{ag_number}.npy")
    out_names = os.path.join(DATASET_DIR, f"mgmno_names_{ag_number}")

    np.save(out_npy, final)
    with open(out_names, "wb") as f:
        pickle.dump(names, f)

    print(f"Saved: {out_npy}")
    print(f"Saved: {out_names}")
    print(f"\nNext step: py make_pickle.py {ag_number}")


if __name__ == "__main__":
    main()
