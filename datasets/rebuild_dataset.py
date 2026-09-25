"""Rebuild mgmno_100(.pickle/_aug) with correct supercells.

The supercell bug (see 3.supercell.py): ASE repeat() interleaves species and
do_feature assumed Mg..Mn..O order, so 1374 of the 2627 structures in
unique_sc_mgmno.npy had Mn and O on each other's sites -- 2-4 eV/atom above the
same crystal in its primitive cell. Everything trained before 2026-09-26 saw
them.

The first 1253 rows of the original unique_sc_mgmno.npy are the primitive cells
(step 4 stacks unique, then supercells) and are intact; they are kept as
unique_mgmno.npy. This regenerates the supercells from them, then re-runs steps
5-7, the distribution-loss quantile targets, and the novelty reference.

    cd datasets && python rebuild_dataset.py
"""
import importlib.util
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def load_script(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'), fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def min_pbc_dist(row):
    """Exact periodic minimum interatomic distance (pymatgen), in A."""
    from eval_wave import one_structure
    lab = (np.abs(row.reshape(30, 3)[2:]).sum(-1) > 0.4).astype(np.float32)
    st = one_structure(row.reshape(-1).astype(np.float32), lab)
    d = st.distance_matrix
    return d[d > 1e-8].min()


def main():
    import view_atoms_mgmno
    sc = load_script('3.supercell.py')

    prims = np.load('unique_mgmno.npy')
    names = pickle.load(open('unique_mgmno_name_list', 'rb'))
    assert len(prims) == len(names) == len(set(names)), 'primitives must be unique'

    new_sc, new_names, parent = [], [], []
    for i, (img, name) in enumerate(zip(prims, names)):
        atoms, _ = view_atoms_mgmno.view_atoms(img, view=False)
        s = atoms.get_chemical_symbols()
        if s.count('Mg') <= 4 and s.count('Mn') <= 4 and s.count('O') <= 6:
            new_sc.append(sc.do_supercell(img))
            new_names += [name] * 3
            parent += [i] * 3
    new_sc = np.vstack(new_sc)
    print(f'{len(prims)} primitives -> {len(new_sc)} supercells')

    # Self-check: a correct supercell has exactly its primitive's minimum
    # interatomic distance. The bug moved atoms onto the wrong sites, which
    # changes it. Check every supercell, not a sample.
    bad = 0
    for k, (row, p) in enumerate(zip(new_sc, parent)):
        if abs(min_pbc_dist(row) - min_pbc_dist(prims[p])) > 1e-3:
            bad += 1
    print(f'supercells whose min distance differs from their primitive: {bad}')
    assert bad == 0, 'supercell geometry does not match its primitive'

    np.save('unique_sc_mgmno.npy', np.vstack([prims, new_sc]))
    with open('unique_sc_mgmno_name_list', 'wb') as f:
        pickle.dump(list(names) + new_names, f)

    py = sys.executable
    run = lambda *a: subprocess.run([py, *a], check=True, stdout=subprocess.DEVNULL)
    run('5.make_comp_dict.py')
    # Two independent augmentation draws, as the original pair was.
    run('6.data_augmentation_mgmno.py', '100', '1')
    run('7.make_label.py')
    shutil.move('mgmno_100.pickle', 'mgmno_100_aug.pickle')
    run('6.data_augmentation_mgmno.py', '100', '0')
    run('7.make_label.py')
    for f in ('mgmno_100.npy', 'mgmno_names_100', 'unique_sc_mgmno_comp_dict'):
        if os.path.exists(f):
            os.remove(f)

    # Distribution-loss targets: 65 quantiles of the real per-atom NN distance
    # and volume per atom (the recipe reproduces the old files exactly on the
    # old data).
    import torch
    from crystal_physics import lattice_matrix, nn_distances
    raw = pickle.load(open('mgmno_100.pickle', 'rb'))
    c = torch.tensor(np.array([np.array(x).flatten() for x, _ in raw]), dtype=torch.float32)
    lab = torch.tensor(np.array([np.array(l).flatten() for _, l in raw]), dtype=torch.float32)
    nn = torch.cat([nn_distances(c[i:i + 512], lab[i:i + 512]) for i in range(0, len(c), 512)])
    v = nn[lab > 0.5]
    v = v[v < 50]
    vpa = torch.linalg.det(lattice_matrix(c)).abs() / lab.sum(1).clamp(min=1)
    q = torch.linspace(0, 1, 65)
    np.save('real_nn_quantiles.npy', torch.quantile(v, q).numpy())
    np.save('real_vpa_quantiles.npy', torch.quantile(vpa, q).numpy())
    print(f'nn median {v.median():.3f} A, vpa median {vpa.median():.2f} A^3/atom')
    from crystal_physics import save_nn_class_quantiles
    save_nn_class_quantiles(c, lab, HERE)

    run('make_novelty_reference.py')
    print('done: mgmno_100.pickle, mgmno_100_aug.pickle, quantiles, novelty_ref_idx.npy')


if __name__ == '__main__':
    main()
