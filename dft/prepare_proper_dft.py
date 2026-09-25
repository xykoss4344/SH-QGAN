import os
import json
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar, Incar, Kpoints

# Define paths
Q_DIR = os.path.dirname(os.path.abspath(__file__))
MACE_RESULTS = os.path.join(Q_DIR, 'results_analysis', 'dft_mace', 'mace_results.json')
CIF_DIR = os.path.join(Q_DIR, 'results_analysis', 'dft_mace')
OUT_DIR = os.path.join(Q_DIR, 'vasp_proper_dft_runs')

# INCAR matching the classical GAN standard (from generate_vasp_inputs.py)
INCAR_DICT = {
    'ISTART':   0,
    'ICHARG':   2,
    'ENCUT':    520,
    'PREC':     'Accurate',
    'EDIFF':    1e-5,
    'NELM':     100,
    'ISMEAR':   0,
    'SIGMA':    0.05,
    'IBRION':   2,
    'ISIF':     3,
    'NSW':      300,
    'EDIFFG':   -0.02,
    'POTIM':    0.3,
    'LDAU':     True,
    'LDAUTYPE': 2,
    'LDAUL':    '0 2 -1',
    'LDAUU':    '0.0 3.9 0.0',
    'LDAUJ':    '0.0 0.0 0.0',
    'LDAUPRINT': 1,
    'NCORE':    4,
    'LWAVE':    False,
    'LCHARG':   False,
    'LORBIT':   11,
}

def main():
    if not os.path.exists(MACE_RESULTS):
        print(f"File not found: {MACE_RESULTS}")
        return

    with open(MACE_RESULTS, 'r') as f:
        results = json.load(f)

    # Pick 5 of the absolute best valid structures
    selected_indices = [1, 2, 3, 4, 5] 
    selected_results = [r for r in results if r['rank'] in selected_indices]
    
    # If any specific rank is missing, simply fall back to the top 5 available
    if len(selected_results) < 5:
        valid_results = [r for r in results if 'error' not in r]
        selected_results = valid_results[:5]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Setting up proper DFT VASP runs in: {OUT_DIR}")

    for idx, r in enumerate(selected_results):
        rank = r['rank']
        form = r['formula']
        c_idx = r['idx']
        
        cif_name = f"rank{rank:02d}_idx{c_idx}_{form}.cif"
        cif_path = os.path.join(CIF_DIR, cif_name)
        
        if not os.path.exists(cif_path):
            print(f"Warning: CIF not found {cif_path}")
            continue
            
        struct = Structure.from_file(cif_path)
        
        struct_dir = os.path.join(OUT_DIR, f'crystal_rank{rank:02d}')
        os.makedirs(struct_dir, exist_ok=True)
        
        Poscar(struct).write_file(os.path.join(struct_dir, 'POSCAR'))
        Incar(INCAR_DICT).write_file(os.path.join(struct_dir, 'INCAR'))
        Kpoints.automatic_density(struct, kppa=1000).write_file(os.path.join(struct_dir, 'KPOINTS'))
        
        print(f"  [{idx+1}/4] Generated VASP inputs for Rank {rank} ({form}) -> {struct_dir}")

    print("\nProper DFT VASP inputs are ready.")
    print("Please add the necessary POTCAR files and run VASP on your compute cluster.")

if __name__ == "__main__":
    main()
