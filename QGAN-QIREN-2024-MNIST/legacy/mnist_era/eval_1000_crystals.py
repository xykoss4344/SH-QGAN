import os
import sys
import pickle
import torch
import numpy as np
import random
import glob

from models.QINR_Crystal import PQWGAN_CC_Crystal
original_repo_path = r"c:\Users\Adminb\OneDrive\Documents\Projects\qgan\QINR-QGAN\QGAN-QIREN-2024-MNIST\datasets"
sys.path.append(original_repo_path)
from view_atoms_mgmno import view_atoms

def verify_crystal_distances(atoms, min_distance_threshold=1.0):
    try:
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        min_distance = np.min(distances)
        return min_distance >= min_distance_threshold, min_distance
    except Exception as e:
        return False, 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_path = "datasets/mgmno_100.pickle"
    with open(dataset_path, 'rb') as f:
        raw_data = pickle.load(f)
        
    labels = []
    # Randomly sample 1000 labels
    for _ in range(1000):
        c, l = random.choice(raw_data)
        labels.append(l.flatten())
        
    labels = torch.tensor(np.array(labels, dtype=np.float32)).to(device)
    
    z_dim = 16
    label_dim = 28
    data_dim = 90
    gen_input_dim = z_dim + label_dim
    critic_input_dim = data_dim + label_dim
    
    gan = PQWGAN_CC_Crystal(
        input_dim_g=gen_input_dim,
        output_dim=data_dim,
        input_dim_d=critic_input_dim,
        hidden_features=6,
        hidden_layers=2,
        spectrum_layer=2,
        use_noise=0.0
    )
    generator = gan.generator.to(device)
    
    # Check what checkpoints we have
    checkpoint_path = "./results_crystal_qgan/checkpoint_90.pt"
    if not os.path.exists(checkpoint_path):
        ckpts = glob.glob("./results_crystal_qgan/checkpoint_*.pt")
        if ckpts:
            # Sort by modification time and get the latest
            checkpoint_path = sorted(ckpts, key=os.path.getmtime)[-1]
        else:
            print("No checkpoints found! Please train the model first.")
            return

    print(f"Loading checkpoint from {checkpoint_path}")
    # Weights_only=False for pickle loading safety if necessary, depending on PyTorch version. We'll use default load.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    print("Generating 1000 crystal structures...")
    valid_count = 0
    total_count = 1000
    batch_size = 100
    
    min_dists = []
    
    with torch.no_grad():
        for i in range(0, total_count, batch_size):
            batch_labels = labels[i:i+batch_size]
            z = torch.randn(batch_labels.shape[0], z_dim).to(device)
            gen_input = torch.cat([z, batch_labels], dim=1)
            fake_imgs = generator(gen_input).cpu().numpy()
            
            for j in range(len(fake_imgs)):
                img = fake_imgs[j]
                atoms, _ = view_atoms(img, view=False)
                is_valid, min_dist = verify_crystal_distances(atoms)
                if is_valid:
                    valid_count += 1
                min_dists.append(min_dist)
            
            print(f"Processed {i+batch_size}/{total_count}...")
            
    print("\n===========================================")
    print("        PHYSICAL VALIDATION RESULTS       ")
    print("===========================================")
    print(f"Total Crystals Evaluated : {total_count}")
    print(f"Total Valid (min distance >= 1.0 Å) : {valid_count}")
    print(f"Validity Percentage : {(valid_count/total_count)*100:.2f}%")
    print(f"Average Minimum Interatomic Distance : {np.mean(min_dists):.3f} Å")
    print("===========================================")

if __name__ == '__main__':
    main()
