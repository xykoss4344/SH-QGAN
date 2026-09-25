"""
Quick evaluation of QGAN v3 checkpoint.
- Generates 1000 crystals
- Computes SSIM vs 100 real references
- Min interatomic distance screen (>= 1.0 A)
- Prints summary table
"""
# Grouped into a subfolder during the 2026-08-30 reorg; core modules stay at the
# package root because eight files import them. This keeps them importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, pickle, torch
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from models.QINR_Crystal import PQWGAN_CC_Crystal

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = './results_crystal_qgan_v3/checkpoint_370.pt'
DATASET     = 'datasets/mgmno_100_aug.pickle'
N_GEN       = 1000
N_REF       = 100
Z_DIM       = 64
LABEL_DIM   = 28
DATA_DIM    = 90
DIST_THRESH = 1.0   # Angstrom

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load real data ─────────────────────────────────────────────────────────────
with open(DATASET, 'rb') as f:
    raw = pickle.load(f)

coords_all, labels_all = [], []
for c, l in raw:
    coords_all.append(np.array(c).flatten())
    labels_all.append(np.array(l).flatten())
coords_all  = np.array(coords_all,  dtype=np.float32)
labels_all  = np.array(labels_all,  dtype=np.float32)

idx_ref = np.random.choice(len(coords_all), N_REF, replace=False)
real_ref     = coords_all[idx_ref]          # (N_REF, 90)
labels_ref   = labels_all[idx_ref]          # (N_REF, 28)

# ── Load generator ─────────────────────────────────────────────────────────────
gan = PQWGAN_CC_Crystal(
    input_dim_g   = Z_DIM + LABEL_DIM,
    output_dim    = DATA_DIM,
    input_dim_d   = DATA_DIM + LABEL_DIM,
    hidden_features = 8,
    hidden_layers   = 3,
    spectrum_layer  = 1,
    use_noise       = 0.0,
)
ckpt = torch.load(CHECKPOINT, map_location=device)
gan.generator.load_state_dict(ckpt['generator'])
gen = gan.generator.to(device).eval()
print(f'Loaded checkpoint: {CHECKPOINT}')

# ── Generate crystals ──────────────────────────────────────────────────────────
print(f'Generating {N_GEN} crystals...')
gen_coords = []
bs = 128
label_tensor = torch.from_numpy(labels_all[:N_GEN]).to(device)

with torch.no_grad():
    for start in range(0, N_GEN, bs):
        end  = min(start + bs, N_GEN)
        lbls = label_tensor[start:end]
        z    = torch.randn(end - start, Z_DIM, device=device)
        inp  = torch.cat([z, lbls], dim=1)
        out  = gen(inp).cpu().numpy()
        gen_coords.append(out)

gen_coords = np.concatenate(gen_coords, axis=0)   # (N_GEN, 90)
print(f'Generated shape: {gen_coords.shape}')
print(f'Output range: [{gen_coords.min():.4f}, {gen_coords.max():.4f}]')

# ── SSIM ───────────────────────────────────────────────────────────────────────
from skimage.metrics import structural_similarity as ssim

def compute_ssim_1d(a, b):
    """SSIM on two 1-D vectors treated as single-channel 1×N images."""
    dr = max(a.max() - a.min(), b.max() - b.min(), 1e-6)
    return ssim(a, b, data_range=dr)

print('Computing SSIM (max over real refs per generated crystal)...')
ssim_scores = []
for g in gen_coords:
    best = max(compute_ssim_1d(g, r) for r in real_ref)
    ssim_scores.append(best)
ssim_scores = np.array(ssim_scores)

# ── Min-distance validity screen ───────────────────────────────────────────────
def min_interatomic_dist(coords_90):
    """
    Rows 2-29 (28 rows) are fractional atom positions (x,y,z in [0,1]).
    Use a simple Euclidean distance in fractional space as a proxy.
    Real validity needs ASE; this is a fast pre-screen.
    """
    positions = coords_90.reshape(30, 3)[2:]   # (28, 3)
    min_d = np.inf
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_d:
                min_d = d
    return min_d

print('Computing min interatomic distances...')
min_dists = np.array([min_interatomic_dist(g) for g in gen_coords])
valid_mask = min_dists >= DIST_THRESH
validity   = valid_mask.mean() * 100

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print('=' * 52)
print(f'  QGAN v3  —  Checkpoint 370  (Quick Eval, N={N_GEN})')
print('=' * 52)
print(f'  SSIM  mean : {ssim_scores.mean():.4f}')
print(f'  SSIM  std  : {ssim_scores.std():.4f}')
print(f'  SSIM  max  : {ssim_scores.max():.4f}')
print(f'  SSIM  min  : {ssim_scores.min():.4f}')
print(f'  SSIM > 0.9 : {(ssim_scores > 0.9).mean()*100:.1f}%')
print(f'  SSIM > 0.7 : {(ssim_scores > 0.7).mean()*100:.1f}%')
print()
print(f'  Min-dist validity (>={DIST_THRESH}A) : {validity:.1f}%  ({valid_mask.sum()}/{N_GEN})')
print(f'  Min-dist mean : {min_dists.mean():.4f}')
print(f'  Min-dist std  : {min_dists.std():.4f}')
print('=' * 52)
