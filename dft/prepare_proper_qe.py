import os
import json
from ase.io import read

Q_DIR = os.path.dirname(os.path.abspath(__file__))
MACE_RESULTS = os.path.join(Q_DIR, 'results_analysis', 'dft_mace', 'mace_results.json')
CIF_DIR = os.path.join(Q_DIR, 'results_analysis', 'dft_mace')
OUT_DIR = os.path.join(Q_DIR, 'results_analysis', 'qe_proper_dft_runs')

pseudos = {
    'Mg': 'Mg.pbe-n-kjpaw_psl.0.3.0.UPF',
    'Mn': 'Mn.pbe-spn-kjpaw_psl.0.3.1.UPF',
    'O':  'O.pbe-n-kjpaw_psl.1.0.0.UPF'
}

def write_qe_input(atoms, filepath, step='relax'):
    elems = list(set(atoms.get_chemical_symbols()))
    masses = {'Mg': 24.305, 'Mn': 54.938, 'O': 15.999}
    
    with open(filepath, 'w') as f:
        f.write("&CONTROL\n")
        f.write(f"  calculation = '{'vc-relax' if step == 'relax' else 'scf'}',\n")
        f.write("  pseudo_dir = '../../pseudo/',\n")
        f.write("  outdir = './out/',\n")
        f.write("  prefix = 'qgan',\n")
        f.write("&END\n")
        
        f.write("&SYSTEM\n")
        f.write("  ibrav = 0,\n")
        f.write(f"  nat = {len(atoms)},\n")
        f.write(f"  ntyp = {len(elems)},\n")
        f.write("  ecutwfc = 60.0,\n")
        f.write("  ecutrho = 480.0,\n")
        f.write("  occupations = 'smearing',\n")
        f.write("  smearing = 'cold',\n")
        f.write("  degauss = 0.02,\n")
        
        if step == 'relax':
            # Modern QE > 7.3 handles Hubbard U in a separate card at the end
            pass
        else:
            f.write("  input_dft = 'hse',\n")
            # Downsampled q-grid for HSE speed
            f.write("  nqx1 = 2, nqx2 = 2, nqx3 = 2,\n") 
        f.write("&END\n")
        
        f.write("&ELECTRONS\n")
        f.write("  conv_thr = 1.0d-6,\n")
        f.write("  mixing_beta = 0.7,\n")
        f.write("&END\n")
        
        if step == 'relax':
            f.write("&IONS\n  ion_dynamics = 'bfgs',\n&END\n")
            f.write("&CELL\n  cell_dynamics = 'bfgs',\n&END\n")
        
        f.write("\nATOMIC_SPECIES\n")
        for el in elems:
            f.write(f" {el} {masses[el]} {pseudos[el]}\n")
            
        f.write("\nCELL_PARAMETERS angstrom\n")
        for row in atoms.cell:
            f.write(f" {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
            
        f.write("\nATOMIC_POSITIONS angstrom\n")
        for el, pos in zip(atoms.get_chemical_symbols(), atoms.positions):
            f.write(f" {el} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
        f.write("\nK_POINTS automatic\n")
        if step == 'relax':
            f.write(" 4 4 4 0 0 0\n")
        else:
            f.write(" 6 6 6 0 0 0\n")
            
        if step == 'relax':
            f.write("\nHUBBARD {ortho-atomic}\n")
            f.write("U Mn-3d 3.9\n")

def main():
    if not os.path.exists(MACE_RESULTS):
        print(f"File not found: {MACE_RESULTS}")
        return

    with open(MACE_RESULTS, 'r') as f:
        results = json.load(f)

    valid = [r for r in results if 'error' not in r]
    selected = valid[:5]
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Download pseudos natively
    pseudo_sh = os.path.join(OUT_DIR, 'download_pseudos.bat')
    with open(pseudo_sh, 'w') as f:
        f.write("@echo off\n")
        f.write("if not exist pseudo mkdir pseudo\n")
        f.write("cd pseudo\n")
        f.write("curl -s -O -J -L https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-n-kjpaw_psl.0.3.0.UPF\n")
        f.write("curl -s -O -J -L https://pseudopotentials.quantum-espresso.org/upf_files/Mn.pbe-spn-kjpaw_psl.0.3.1.UPF\n")
        f.write("curl -s -O -J -L https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF\n")
        f.write("cd ..\n")
        
    for r in selected:
        rank = r['rank']
        form = r['formula']
        c_idx = r['idx']
        
        cif_path = os.path.join(CIF_DIR, f"rank{rank:02d}_idx{c_idx}_{form}.cif")
        run_folder = os.path.join(OUT_DIR, f'crystal_rank{rank:02d}')
        os.makedirs(run_folder, exist_ok=True)
        
        try:
            atoms = read(cif_path)
        except Exception:
            continue
            
        # Step 1: PBE+U
        s1_folder = os.path.join(run_folder, 'step1_pbe_u_relax')
        os.makedirs(s1_folder, exist_ok=True)
        write_qe_input(atoms, os.path.join(s1_folder, 'pw_relax.in'), step='relax')
        
        # Step 2: HSE
        s2_folder = os.path.join(run_folder, 'step2_hse_static')
        os.makedirs(s2_folder, exist_ok=True)
        write_qe_input(atoms, os.path.join(s2_folder, 'pw_hse_static.in'), step='hse')
        
    print(f"Generated rigorous Quantum Espresso multi-step inputs in {OUT_DIR}")

if __name__ == "__main__":
    main()
