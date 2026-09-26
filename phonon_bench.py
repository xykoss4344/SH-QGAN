"""Dynamical stability (PhononBench-style): no imaginary phonon modes.

PhononBench (arXiv 2512.21227) calls a generated crystal dynamically stable if,
after relaxation, its phonon spectrum has no imaginary modes -- i.e. it is a
true local minimum, which E_hull alone does not check. They use MatterSim and
report MatterGen 45%, average 32% over 7 models.

Here: CHGNet (not MatterSim -- stated wherever this is reported) relaxation to
fmax 0.02, then phonopy finite displacements (0.01 A) in a supercell of at
least ~8 A per side, frequencies on an 8x8x8 mesh. Stable if the lowest
frequency is above -0.1 THz (tolerance for numerical noise near Gamma).

    python phonon_bench.py <dir of CIFs> [more dirs]  -> prints and writes phonon_<dir>.csv
"""
import csv
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

TOL_THZ = -0.1


def supercell_matrix(lattice, target=8.0):
    return np.diag([max(1, int(np.ceil(target / L))) for L in lattice.abc])


def is_dynamically_stable(st, calc, relaxer):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    r = relaxer.relax(st, fmax=0.02, steps=500, verbose=False)
    st = r['final_structure']
    unit = PhonopyAtoms(symbols=[str(s.specie) for s in st], cell=st.lattice.matrix,
                        scaled_positions=st.frac_coords)
    ph = Phonopy(unit, supercell_matrix=supercell_matrix(st.lattice))
    ph.generate_displacements(distance=0.01)
    from ase import Atoms
    forces = []
    for sc in ph.supercells_with_displacements:
        atoms = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        atoms.calc = calc
        forces.append(atoms.get_forces())
    ph.forces = forces
    ph.produce_force_constants()
    ph.run_mesh([8, 8, 8])
    fmin = float(ph.get_mesh_dict()['frequencies'].min())
    return fmin > TOL_THZ, fmin, len(st)


def main(dirs):
    from chgnet.model import CHGNet, StructOptimizer
    from chgnet.model.dynamics import CHGNetCalculator
    from pymatgen.core import Structure
    model = CHGNet.load(verbose=False)
    calc = CHGNetCalculator(model=model)
    relaxer = StructOptimizer(model=model)
    for d in dirs:
        rows = []
        for f in sorted(glob.glob(os.path.join(d, '*.cif'))):
            try:
                ok, fmin, n = is_dynamically_stable(Structure.from_file(f), calc, relaxer)
            except Exception as e:
                ok, fmin, n = False, float('nan'), -1
            rows.append([os.path.basename(f), n, fmin, ok])
        out = os.path.join(os.path.dirname(os.path.abspath(d)), f'phonon_{os.path.basename(d.rstrip("/"))}.csv')
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'n_atoms', 'min_freq_THz', 'dynamically_stable'])
            w.writerows(rows)
        n_ok = sum(r[3] for r in rows)
        print(f'{d}: {n_ok}/{len(rows)} dynamically stable ({n_ok / max(len(rows), 1):.0%}) -> {out}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
