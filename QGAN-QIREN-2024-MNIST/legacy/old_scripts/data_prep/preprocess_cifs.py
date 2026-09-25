import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import glob
import pickle
import numpy as np
from ase.io import read

def cif_to_tensor(cif_path, max_atoms=30):
    """Converts a CIF to a flat 96-dimensional array."""
    # 1. Read CIF file
    atoms = read(cif_path)
    
    # 2. Extract Lattice Parameters [a, b, c, alpha, beta, gamma]
    # cellpar() returns 6 geometric properties of the crystal box
    lattice_params = atoms.cell.cellpar() 
    
    # 3. Extract Atomic Coordinates
    # We use "scaled" (fractional) positions so they stay between 0.0 and 1.0
    coords = atoms.get_scaled_positions()
    
    # 4. Standardize atom count (Pad or truncate to exactly max_atoms)
    num_atoms = len(coords)
    if num_atoms < max_atoms:
        padding = np.zeros((max_atoms - num_atoms, 3)) # Pad with (0,0,0)
        coords = np.vstack((coords, padding))
    else:
        coords = coords[:max_atoms]
        
    # Flatten the coordinates from shape (30, 3) -> (90,)
    flat_coords = coords.flatten()
    
    # 5. Combine! Result is shape (96,)
    combined_data = np.concatenate([lattice_params, flat_coords])
    
    # Optional: We mock a 28-dim condition label to match your existing code
    dummy_label = np.zeros(28) 
    
    return np.array(combined_data, dtype=np.float32), np.array(dummy_label, dtype=np.float32)

def process_folder(cif_folder, output_filepath):
    dataset = []
    cif_files = glob.glob(os.path.join(cif_folder, "*.cif"))
    
    for file in cif_files:
        try:
            crystal_vector, label_vector = cif_to_tensor(file)
            dataset.append((crystal_vector, label_vector))
        except Exception as e:
            print(f"Skipping {file}: {e}")
            
    # Save as a pickle file format expected by train_crystal.py
    with open(output_filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Successfully processed {len(dataset)} crystals into {output_filepath}")

if __name__ == "__main__":
    # Example usage
    # process_folder('./my_raw_cifs_folder', './crystal_dataset_96dim.pickle')
    pass
