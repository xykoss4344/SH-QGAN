import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import pickle
import torch
import numpy as np

# Import QINR first so it doesn't clash with models.py in the other folder
from models.QINR_Crystal import PQWGAN_CC_Crystal

# Add the path to the original repo to import view_atoms
original_repo_path = r"c:\Users\Adminb\OneDrive\Documents\Projects\qgan\QINR-QGAN\QGAN-QIREN-2024-MNIST\datasets"
sys.path.append(original_repo_path)
from view_atoms_mgmno import view_atoms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load some labels from the dataset to condition the generation
    dataset_path = "datasets/mgmno_100.pickle"
    with open(dataset_path, 'rb') as f:
        raw_data = pickle.load(f)
        
    labels = []
    for c, l in raw_data[:5]: # take first 5 samples
        labels.append(l.flatten())
        
    labels = torch.tensor(np.array(labels, dtype=np.float32)).to(device)
    
    # 2. Init model and load checkpoint
    z_dim = 16
    label_dim = 28
    data_dim = 90
    gen_input_dim = z_dim + label_dim
    critic_input_dim = data_dim + label_dim
    
    gan = PQWGAN_CC_Crystal(
        input_dim_g=gen_input_dim,
        output_dim=data_dim,
        input_dim_d=critic_input_dim,
        hidden_features=6, # from args
        hidden_layers=2, # from args
        spectrum_layer=2, # from args
        use_noise=0.0 # from args
    )
    generator = gan.generator.to(device)
    
    # Load checkpoint
    checkpoint_path = "./results_crystal_qgan/checkpoint_90.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # 3. Generate fake images
    z = torch.randn(labels.shape[0], z_dim).to(device)
    gen_input = torch.cat([z, labels], dim=1)
    
    with torch.no_grad():
        fake_imgs = generator(gen_input).cpu().numpy()
        
    # 4. Save to CIF using ASE
    os.makedirs("./generated_crystals", exist_ok=True)
    
    for i in range(len(fake_imgs)):
        img = fake_imgs[i]
        
        # view_atoms from the other repo takes the generated 90-dim array
        atoms, _ = view_atoms(img, view=False)
        
        # Save as CIF
        out_path = f"./generated_crystals/sample_{i}.cif"
        atoms.write(out_path)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
