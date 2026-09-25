import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
quick_ehull_qgan.py
===================
Fast E_hull pre-screen (min interatomic distance) for a single QINR-QGAN checkpoint.
Mirrors crystalGAN/quick_ehull.py for 1:1 comparison.

Usage:
    py -3.12 quick_ehull_qgan.py --checkpoint results_crystal_qgan/checkpoint_390.pt
    py -3.12 quick_ehull_qgan.py --checkpoint results_crystal_qgan/checkpoint_390.pt --n_gen 500
"""

import argparse, os, sys, pickle
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, DATASET_DIR)

from models.QINR_Crystal import PQWGAN_CC_Crystal
from view_atoms_mgmno import view_atoms

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',     type=str,   required=True,
                    help='Path to QINR-QGAN checkpoint_N.pt file')
parser.add_argument('--n_gen',          type=int,   default=200,
                    help='Number of structures to generate')
parser.add_argument('--min_dist',       type=float, default=1.0,
                    help='Minimum interatomic distance threshold (Angstrom)')
parser.add_argument('--z_dim',          type=int,   default=16)
parser.add_argument('--hidden_features',type=int,   default=6)
parser.add_argument('--hidden_layers',  type=int,   default=2)
parser.add_argument('--spectrum_layer', type=int,   default=2)
parser.add_argument('--use_noise',      type=float, default=0.0)
parser.add_argument('--dataset',        type=str,   default=None,
                    help='Pickle path for sampling real labels. '
                         'Default: datasets/mgmno_1000.pickle')
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Checkpoint : {opt.checkpoint}")
print(f"Device     : {device}")

# ── Build QINR-QGAN generator ─────────────────────────────────────────────────
data_dim  = 90
label_dim = 28
z_dim     = opt.z_dim

gen_input_dim    = z_dim + label_dim       # 16 + 28 = 44
critic_input_dim = data_dim + label_dim    # 90 + 28 = 118

gan = PQWGAN_CC_Crystal(
    input_dim_g     = gen_input_dim,
    output_dim      = data_dim,
    input_dim_d     = critic_input_dim,
    hidden_features = opt.hidden_features,
    hidden_layers   = opt.hidden_layers,
    spectrum_layer  = opt.spectrum_layer,
    use_noise       = opt.use_noise,
)
generator = gan.generator.to(device)

ckpt = torch.load(opt.checkpoint, map_location=device, weights_only=False)
generator.load_state_dict(ckpt['generator'])
generator.to(device).eval()

epoch_loaded = ckpt.get('epoch', '?')
print(f"Loaded generator (epoch {epoch_loaded})")

# ── Label sampling: draw from training distribution ───────────────────────────
_pickle_path = opt.dataset or os.path.join(DATASET_DIR, "mgmno_1000.pickle")
_real_labels = None
if os.path.isfile(_pickle_path):
    with open(_pickle_path, "rb") as f:
        _raw = pickle.load(f)
    _real_labels = np.array([x[1] for x in _raw], dtype=np.float32).reshape(-1, 28)

def sample_labels(n):
    if _real_labels is not None:
        idx = np.random.choice(len(_real_labels), n, replace=True)
        return torch.tensor(_real_labels[idx], dtype=torch.float32, device=device)
    # Fallback: random binary labels
    mg_i = np.random.randint(0, 8,  n)
    mn_i = np.random.randint(0, 8,  n)
    o_i  = np.random.randint(0, 12, n)
    lbl  = np.zeros((n, 28), dtype=np.float32)
    for j in range(n):
        lbl[j, :mg_i[j]+1]           = 1.0
        lbl[j, 8:8+mn_i[j]+1]        = 1.0
        lbl[j, 16:16+o_i[j]+1]       = 1.0
    return torch.tensor(lbl, dtype=torch.float32, device=device)

# ── Generate structures ───────────────────────────────────────────────────────
gen_images_list = []
with torch.no_grad():
    for start in range(0, opt.n_gen, 32):
        bs = min(32, opt.n_gen - start)
        z  = torch.randn(bs, z_dim, device=device)
        labels = sample_labels(bs)
        gen_input = torch.cat([z, labels], dim=1)   # (bs, 44)
        fake = generator(gen_input)                  # (bs, 90)
        gen_images_list.append(fake.cpu().numpy())

gen_images = np.concatenate(gen_images_list, axis=0).reshape(opt.n_gen, 30, 3)

# ── E_hull pre-screen: min interatomic distance ───────────────────────────────
min_dists   = []
valid_count = 0
for img in gen_images:
    try:
        atoms, _ = view_atoms(img, view=False)
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        md = float(d.min())
    except Exception:
        md = 0.0
    min_dists.append(md)
    if md >= opt.min_dist:
        valid_count += 1

validity_pct = valid_count / opt.n_gen * 100
mean_dist    = np.mean(min_dists)

print(f"\nResults for {os.path.basename(opt.checkpoint)}:")
print(f"  Valid (min dist >= {opt.min_dist} A) : {valid_count}/{opt.n_gen}  ({validity_pct:.1f}%)")
print(f"  Mean min interatomic distance       : {mean_dist:.3f} A")
