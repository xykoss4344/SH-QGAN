import os
import glob
from ase.io import read, write

# Directories
CIF_DIR = os.path.join("results_analysis", "dft_mace")
OUT_DIR = os.path.join("results_analysis", "qe_inputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Pseudopotentials mapping for standard PBE
pseudos = {
    'Mg': 'Mg.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Mn': 'Mn.pbe-spn-kjpaw_psl.0.3.1.UPF',
    'O':  'O.pbe-n-kjpaw_psl.1.0.0.UPF'
}

# Find all CIFs
cif_files = glob.glob(os.path.join(CIF_DIR, "*.cif"))

if not cif_files:
    print(f"No .cif files found in {CIF_DIR}")
    exit(1)

print(f"Found {len(cif_files)} CIF files. Generating Quantum Espresso inputs...")

# Base Quantum Espresso parameters for vc-relax
input_data = {
    'control': {
        'calculation': 'vc-relax',
        'pseudo_dir': './pseudo/',
        'prefix': 'qgan_struct',
        'outdir': './out/',
        'restart_mode': 'from_scratch',
    },
    'system': {
        'ecutwfc': 60,
        'ecutrho': 480,
        'occupations': 'smearing',
        'smearing': 'cold',
        'degauss': 0.02,
    },
    'electrons': {
        'conv_thr': 1e-6,
        'mixing_beta': 0.7
    },
    'ions': {
        'ion_dynamics': 'bfgs'
    },
    'cell': {
        'cell_dynamics': 'bfgs'
    }
}

for cif_path in cif_files:
    basename = os.path.basename(cif_path).replace('.cif', '')
    atoms = read(cif_path)
    
    # Check if elements have mapped pseudopotentials
    for elem in set(atoms.get_chemical_symbols()):
        if elem not in pseudos:
            print(f"Warning: No pseudopotential defined for {elem} in {basename}")
            
    out_folder = os.path.join(OUT_DIR, basename)
    os.makedirs(out_folder, exist_ok=True)
    
    in_path = os.path.join(out_folder, 'pw_relax.in')
    
    # Write using ASE ESPRESSO exporter
    write(in_path, atoms, format='espresso-in',
          pseudopotentials=pseudos,
          input_data=input_data,
          kpts=(4, 4, 4), # K-points grid 4x4x4
          koffset=(0, 0, 0))
          
    # Generate a simple download script for pseudopotentials
    pseudo_sh = os.path.join(out_folder, 'download_pseudos.sh')
    with open(pseudo_sh, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("mkdir -p pseudo\n")
        f.write("cd pseudo\n")
        f.write("wget -nc https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-n-kjpaw_psl.1.0.0.UPF\n")
        f.write("wget -nc https://pseudopotentials.quantum-espresso.org/upf_files/Mn.pbe-spn-kjpaw_psl.0.3.1.UPF\n")
        f.write("wget -nc https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF\n")
        f.write("cd ..\n")
        
    print(f"Generated {in_path}")

print("\nDone! You can upload the folders in results_analysis/qe_inputs to your HPC cluster.")
