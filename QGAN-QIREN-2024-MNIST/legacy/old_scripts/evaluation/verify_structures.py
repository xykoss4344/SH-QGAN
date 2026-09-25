import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import glob
import numpy as np
from ase.io import read

def verify_crystal_distances(cif_path, min_distance_threshold=1.0):
    try:
        atoms = read(cif_path)
        distances = atoms.get_all_distances(mic=True) # mic=True accounts for periodic boundary conditions
        
        # We ignore the diagonal (distance from an atom to itself, which is 0)
        np.fill_diagonal(distances, np.inf)
        
        min_distance = np.min(distances)
        if min_distance < min_distance_threshold:
            print(f"❌ [INVALID] {os.path.basename(cif_path)}: Atoms overlap! Minimum distance is {min_distance:0.3f} Å")
            return False
        else:
            print(f"✅ [VALID]   {os.path.basename(cif_path)}: Geometry looks okay! Minimum distance is {min_distance:0.3f} Å")
            return True
    except Exception as e:
        print(f"Error reading {cif_path}: {e}")
        return False

if __name__ == "__main__":
    cif_folder = "./generated_crystals"
    cif_files = glob.glob(os.path.join(cif_folder, "*.cif"))
    
    print("--- Starting Crystal Validity Check ---")
    valid_count = 0
    for file in cif_files:
         if verify_crystal_distances(file):
             valid_count += 1
             
    print(f"\nTotal Valid: {valid_count}/{len(cif_files)}")
