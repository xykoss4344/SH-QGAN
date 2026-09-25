import torch
import numpy as np
from torch.utils.data import Dataset
import os

# Try to import RDKit for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None

class MolecularDataset(Dataset):
    """
    Dataset that converts SMILES strings into 28x28 image representations.
    Default mode 'matrix' creates an Adjacency Matrix (Graph) representation.
    """
    def __init__(self, smiles_file=None, image_size=28, mode='matrix'):
        self.image_size = image_size
        self.mode = mode
        self.smiles_list = []
        
        # 1. Load Data
        if smiles_file and os.path.exists(smiles_file):
            try:
                with open(smiles_file, 'r') as f:
                    # Assumes one SMILES per line
                    self.smiles_list = [line.strip().split()[0] for line in f if line.strip()]
                print(f"Loaded {len(self.smiles_list)} molecules from {smiles_file}")
            except Exception as e:
                print(f"Error loading file {smiles_file}: {e}")
        
        # 2. Fallback / Dummy Data if empty (for testing)
        if not self.smiles_list:
            if smiles_file:
                print(f"Warning: Could not load molecules from {smiles_file}. Using dummy data.")
            else:
                print("No molecule file provided. Using dummy data (Benzene, Ethanol, etc).")
            
            # Simple list of common small molecules
            self.smiles_list = [
                "c1ccccc1", # Benzene
                "CCO",      # Ethanol
                "CC(=O)O",  # Acetic Acid
                "CCN",      # Ethylamine
                "C1CCCCC1", # Cyclohexane
                "C=C-C=C",  # Butadiene
                "C#N",      # HCN
                "C(=O)N",   # Formamide
            ] * 50 # Duplicate to create a 'dataset'

    def smiles_to_matrix(self, smiles):
        """
        Converts a SMILES string to a padded Adjacency Matrix (28x28).
        This captures the topological structure of the molecule.
        """
        if not RDKIT_AVAILABLE:
            return np.eye(self.image_size, dtype=np.float32)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                return np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
            # Get Adjacency Matrix (0 or 1)
            adj = Chem.GetAdjacencyMatrix(mol)
            
            # Add Identity (Self-loop) so diagonal is 1
            adj = adj + np.eye(adj.shape[0])
            
            # Create container
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
            # Crop dimensions if molecule is larger than image_size
            h, w = adj.shape
            h_tgt = min(h, self.image_size)
            w_tgt = min(w, self.image_size)
            
            # Fill matrix
            img[:h_tgt, :w_tgt] = adj[:h_tgt, :w_tgt]
            
            # Normalize to [-1, 1] range for GAN (Tanh activation)
            # 0 -> -1 (Background)
            # 1 ->  1 (Atom/Bond)
            img = (img * 2.0) - 1.0
            
            return img
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        # Generate 'Image'
        if self.mode == 'matrix':
            img_array = self.smiles_to_matrix(smiles)
        else:
            # Placeholder for other modes
            img_array = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Convert to Tensor [Channels, Height, Width] -> [1, 28, 28]
        img_tensor = torch.tensor(img_array).float().unsqueeze(0)
        
        # Return image and dummy label (0) to match MNIST format
        return img_tensor, 0

def load_molecules(file_path="molecules.txt", image_size=28):
    """Helper function to load the dataset"""
    return MolecularDataset(smiles_file=file_path, image_size=image_size)

if __name__ == "__main__":
    # Test the dataset
    ds = MolecularDataset()
    img, lbl = ds[0]
    print(f"Output Img Shape: {img.shape}")
    print(f"Value Range: {img.min()} to {img.max()}")
