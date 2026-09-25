import os
import sys
import pickle
import torch
import numpy as np
import random
import glob

# Try importing the advanced structural and thermodynamic libraries
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not found. SSIM calculations will be skipped. Run 'pip install scikit-image'")
    SKIMAGE_AVAILABLE = False

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from chgnet.model.dynamics import StructOptimizer
    PYMATGEN_AVAILABLE = True
except ImportError:
    print("Warning: pymatgen, mp-api, or chgnet not found. E_hull, Relaxation, and StructureMatcher will be skipped.")
    print("Run 'pip install pymatgen mp-api chgnet' to enable these metrics.")
    PYMATGEN_AVAILABLE = False

from models.QINR_Crystal import PQWGAN_CC_Crystal

original_repo_path = r"c:\Users\Adminb\OneDrive\Documents\Projects\qgan\QINR-QGAN\QGAN-QIREN-2024-MNIST\datasets"
sys.path.append(original_repo_path)
try:
    from view_atoms_mgmno import view_atoms
except ImportError:
    print("Could not import view_atoms from datasets/view_atoms_mgmno.py")
    sys.exit(1)

def compute_distance_matrix(atoms):
    """Computes a padded 30x30 interatomic distance matrix for an ASE atoms object."""
    distances = atoms.get_all_distances(mic=True)
    padded = np.zeros((30, 30))
    n = min(30, distances.shape[0])
    padded[:n, :n] = distances[:n, :n]
    return padded

def compute_ssim(dist_mat_1, dist_mat_2):
    """Computes SSIM between two distance matrices."""
    # Data range is set to approx max distance across a 15A or 10A cell, e.g., ~25.0
    return ssim(dist_mat_1, dist_mat_2, data_range=25.0)

def evaluate_ehull(structures, api_key=None):
    """
    Relaxes structures using CHGNet and queries Materials Project for E_hull.
    structures: List of pymatgen.core.Structure objects
    Returns: List of (relaxed_structure, formation_energy, e_hull) tuples
    """
    if not api_key:
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            print("Error: MP_API_KEY environment variable not set. Cannot fetch phase diagram.")
            return []
            
    print("Initializing CHGNet Relaxer...")
    try:
        relaxer = StructOptimizer()
    except Exception as e:
        print(f"Failed to initialize CHGNet relaxer: {e}")
        return []

    print("Fetching Mg-Mn-O Phase Diagram from Materials Project...")
    try:
        with MPRester(api_key) as mpr:
            entries = mpr.get_entries_in_chemsys(["Mg", "Mn", "O"])
            pd = PhaseDiagram(entries)
    except Exception as e:
        print(f"Failed to fetch phase diagram from Materials Project: {e}")
        return []

    results = []
    
    for i, struct in enumerate(structures):
        print(f"Relaxing structure {i+1}/{len(structures)}...")
        try:
            relax_result = relaxer.relax(struct, verbose=False)
            relaxed_struct = relax_result["final_structure"]
            # Energy predicted by CHGNet
            energy = relax_result["trajectory"].energies[-1] / len(relaxed_struct)
            
            # Form ComputedEntry to query the PhaseDiagram
            from pymatgen.entries.computed_entries import ComputedEntry
            comp = relaxed_struct.composition
            
            # Predict Ehull using the calculated energy.
            # (Note: Requires MLIP and MP energies to be compatible or locally referenced)
            entry = ComputedEntry(composition=comp, energy=energy * comp.num_atoms)
            ehull = pd.get_e_above_hull(entry)
            
            results.append((relaxed_struct, energy, ehull))
            
        except Exception as e:
            print(f" Error relaxing structure {i+1}: {e}")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Advanced Metrics for Crystal GAN")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of crystals to generate and evaluate")
    parser.add_argument("--dataset", type=str, default="datasets/mgmno_100.pickle")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(args.dataset):
        print(f"Dataset not found at: {args.dataset}")
        return

    with open(args.dataset, 'rb') as f:
        raw_data = pickle.load(f)
        
    labels = []
    real_atoms_list = []
    
    # We sample a few random reference structures from the training set to serve as our 'Reference Database'
    ref_count = min(100, len(raw_data))
    print(f"Extracting {ref_count} reference structures from dataset for comparison...")
    
    sample_indices = random.sample(range(len(raw_data)), ref_count)
    for idx in sample_indices:
        c, l = raw_data[idx]
        if len(labels) < args.num_samples:
             labels.append(l.flatten())
        if SKIMAGE_AVAILABLE or PYMATGEN_AVAILABLE:
            atoms, _ = view_atoms(c.flatten(), view=False)
            real_atoms_list.append(atoms)
            
    # Pad labels if dataset didn't have enough
    while len(labels) < args.num_samples:
        labels.append(labels[0])
        
    labels = torch.tensor(np.array(labels, dtype=np.float32)).to(device)
    
    # Check what checkpoints we have
    checkpoint_path = "./results_crystal_qgan/checkpoint_390.pt"
    if not os.path.exists(checkpoint_path):
        ckpts = glob.glob("./results_crystal_qgan/checkpoint_*.pt")
        if ckpts:
            checkpoint_path = sorted(ckpts, key=os.path.getmtime)[-1]
        else:
            print("No checkpoints found! Please train the model first.")
            return

    print(f"Loading checkpoint from {checkpoint_path}")
    z_dim = 16
    label_dim = 28
    data_dim = 90
    
    gan = PQWGAN_CC_Crystal(
        input_dim_g=z_dim + label_dim, output_dim=data_dim, input_dim_d=data_dim + label_dim,
        hidden_features=6, hidden_layers=2, spectrum_layer=2, use_noise=0.0
    )
    generator = gan.generator.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    print(f"Generating {args.num_samples} crystal structures for evaluation...")
    with torch.no_grad():
        z = torch.randn(labels.shape[0], z_dim).to(device)
        gen_input = torch.cat([z, labels], dim=1)
        fake_imgs = generator(gen_input).cpu().numpy()
        
    generated_atoms = []
    for img in fake_imgs:
        try:
            atoms, _ = view_atoms(img, view=False)
            generated_atoms.append(atoms)
        except Exception as e:
            print(f"Skipping an invalid generated structure: {e}")
        
    print("\n" + "="*60)
    print("             ADVANCED METRICS RESULTS")
    print("="*60)

    # 1. Structural Similarity Index Measure (SSIM)
    if SKIMAGE_AVAILABLE:
        print("\n--- 1. Structural Similarity Index Measure (SSIM) ---")
        ssim_scores = []
        for g_atom in generated_atoms:
            g_dist = compute_distance_matrix(g_atom)
            r_atom = random.choice(real_atoms_list)
            r_dist = compute_distance_matrix(r_atom)
            
            score = compute_ssim(g_dist, r_dist)
            ssim_scores.append(score)
            
        avg_ssim = np.sum(ssim_scores) / len(ssim_scores)
        print(f"Average SSIM score : {avg_ssim:.4f}")
        print("(Higher SSIM indicates the generated structure's global patterns better match real data representation.)")
    
    # 2. Structural Dissimilarity using StructureMatcher
    if PYMATGEN_AVAILABLE:
        print("\n--- 2. Structural Dissimilarity (Pymatgen StructureMatcher) ---")
        ref_structs = [AseAtomsAdaptor.get_structure(a) for a in real_atoms_list]
        gen_structs = [AseAtomsAdaptor.get_structure(a) for a in generated_atoms]
        
        matcher = StructureMatcher()
        match_count = 0
        dissimilarities = []
        
        for i, g_struct in enumerate(gen_structs):
            # Find Best Match Displacement
            best_rms = float('inf')
            is_match = False
            for ref in ref_structs:
                if matcher.fit(g_struct, ref):
                    is_match = True
                    rms_dist = matcher.get_rms_dist(g_struct, ref)
                    if rms_dist and rms_dist[0] < best_rms:
                        best_rms = rms_dist[0]
                        
            if is_match:
                match_count += 1
                dissimilarities.append(best_rms)
                
        print(f"Found {match_count} distinct structural matches with reference dataset (out of {len(gen_structs)}).")
        if match_count > 0:
            print(f"Average RMS Dissimilarity of Matches: {np.mean(dissimilarities):.5f} Å")
        else:
            print("The model generated mostly completely new structures (no exact matches with default StructureMatcher thresholds).")

        # 3. Energy above Convex Hull
        print("\n--- 3. DFT Surrogation: Energy Above Convex Hull (E_hull) ---")
        api_key = os.environ.get("MP_API_KEY")
        if api_key:
            print("Running CHGNet relaxations and Materials Project Ehull queries...")
            results = evaluate_ehull(gen_structs, api_key=api_key)
            
            if len(results) > 0:
                metastable_cnt = sum(1 for _, _, e in results if e <= 0.200) # <= 200 meV/atom
                synthesizable_cnt = sum(1 for _, _, e in results if e <= 0.080) # <= 80 meV/atom
                
                print("\n  >> Thermodynamical Stability Breakdown <<")
                print(f"Total Successfully Relaxed   : {len(results)}")
                print(f"Theoretically Metastable   : {metastable_cnt} (E_hull <= 200 meV/atom)")
                print(f"Potentially Synthesizable  : {synthesizable_cnt}  (E_hull <= 80 meV/atom)")
                
                avg_ehull = np.mean([e for _, _, e in results])
                print(f"Average E_hull of batch    : {avg_ehull:.3f} eV/atom")
        else:
            print("Skipped! To calculate E_hull and perform structural relaxations:")
            print("  1. Get an API key from materialsproject.org")
            print("  2. Set environment variable: $env:MP_API_KEY='your_key'")
            print("  3. Run the script again.")
            
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()
