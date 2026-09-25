import os
import subprocess
import glob
import sys
from ase import Atoms
from prepare_proper_qe import write_qe_input

def extract_relaxed_geometry(out_path):
    with open(out_path, 'r') as f:
        lines = f.readlines()
    
    cell = []
    positions = []
    symbols = []
    in_cell = False
    in_pos = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('CELL_PARAMETERS'):
            cell = []
            in_cell = True
            in_pos = False
            continue
        elif line.startswith('ATOMIC_POSITIONS'):
            positions = []
            symbols = []
            in_pos = True
            in_cell = False
            continue
        elif line.startswith('End final coordinates') or line.startswith('Writing output data'):
            in_pos = False
            in_cell = False
            
        if in_cell:
            parts = line.split()
            if len(parts) >= 3:
                try: cell.append([float(p) for p in parts[:3]])
                except: in_cell = False
        elif in_pos:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    symbols.append(parts[0])
                    positions.append([float(p) for p in parts[1:4]])
                except: in_pos = False

    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

BASE_DIR = os.path.abspath(r"C:\Users\Adminb\OneDrive\Documents\Projects\qgan\QINR-QGAN\QGAN-QIREN-2024-MNIST\results_analysis\qe_proper_dft_runs")
PW_EXE = r"C:\Users\Adminb\OneDrive\Documents\Projects\qgan\QINR-QGAN\QGAN-QIREN-2024-MNIST\qe_binaries\bin\pw.exe"

def run_qe(input_file, cwd):
    out_file = input_file.replace('.in', '.out')
    cmd = [PW_EXE, '-in', input_file]
    print(f"Executing: {' '.join(cmd)} in {cwd}")
    with open(os.path.join(cwd, out_file), 'w') as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)

def main():
    print(f"============================================================")
    print(f"Starting Quantum ESPRESSO Automated Pipeline (Local Windows)")
    print(f"Executable: {PW_EXE}")
    print(f"============================================================\n")
    
    crystal_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "crystal_rank*")))
    for cdir in crystal_dirs:
        rank_name = os.path.basename(cdir)
        print(f"\n############################################################")
        print(f"                   Processing {rank_name}")
        print(f"############################################################")
        
        s1_dir = os.path.join(cdir, 'step1_pbe_u_relax')
        s1_in = 'pw_relax.in'
        s1_out = 'pw_relax.out'
        s1_out_path = os.path.join(s1_dir, s1_out)
        
        skip_step1 = False
        if os.path.exists(s1_out_path):
            with open(s1_out_path, 'r', encoding='utf-8', errors='ignore') as f:
                if "JOB DONE" in f.read():
                    skip_step1 = True
                    
        if skip_step1:
            print("-> Step 1 output already finished! Skipping PBE+U...")
        else:
            print("-> Step 1: Initiating PBE+U Geometric Relaxation. This will take a while...")
            run_qe(s1_in, s1_dir)
        
        print("\n-> Extracting structurally relaxed geometry parameters from PBE+U out file...")
        try:
            # Bypass ASE completely and read the raw QE output text directly
            relaxed_atoms = extract_relaxed_geometry(s1_out_path)
            if len(relaxed_atoms) == 0:
                raise ValueError("Parsed zero atoms. Is the pw_relax.out file incomplete?")
        except Exception as e:
            print(f"[FATAL ERROR] Failed to extract custom geometry from {s1_out_path}. The relaxation likely crashed or failed to converge: {e}")
            continue
            
        s2_dir = os.path.join(cdir, 'step2_hse_static')
        s2_in_path = os.path.join(s2_dir, 'pw_hse_static.in')
        s2_out_path = os.path.join(s2_dir, 'pw_hse_static.out')
        
        print("-> Step 2 Pre-flight: Overwriting static template with the newly relaxed atom coordinates...")
        # We reuse the exact formatting writer we established previously, guaranteeing identical phys-params
        write_qe_input(relaxed_atoms, s2_in_path, step='hse')
        print("   [Success] Geometry correctly patched into step2_hse_static.in.")
        
        if not os.path.exists(s2_out_path):
            print("-> Step 2: Initiating HSE06 Hybrid-Functional Static Electronic Structure calculation. This is highly compute-intensive...")
            run_qe('pw_hse_static.in', s2_dir)
        else:
            print("-> Step 2 output already exists! Skipping HSE computation...")
            
        print(f"\n[COMPLETE] Pipeline finished fully for {rank_name}.")
        
if __name__ == "__main__":
    main()
