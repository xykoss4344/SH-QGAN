"""
Full evaluation of QGAN v4 (split-head generator).
  - Generates N_GEN crystals from checkpoint_490
  - Min interatomic distance in Angstrom (proper unit-cell conversion)
  - CHGNet energy prediction → E_above_hull via pymatgen PhaseDiagram
  - SSIM vs real references
  - Summary table printed to console + saved as eval_v4_report.txt
"""
# Grouped into a subfolder during the 2026-08-30 reorg; core modules stay at the
# package root because eight files import them. This keeps them importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys, os, pickle, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from models.QINR_Crystal import PQWGAN_CC_Crystal
from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, CompoundPhaseDiagram
from pymatgen.core.composition import Composition
from chgnet.model import CHGNet
from skimage.metrics import structural_similarity as ssim

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = './results_crystal_qgan_v4/checkpoint_490.pt'
DATASET     = 'datasets/mgmno_100_aug.pickle'
N_GEN       = 500       # total generated
N_REF_SSIM  = 100       # real refs for SSIM
N_EHULL     = 200       # subset for CHGNet (slow)
Z_DIM       = 64
LABEL_DIM   = 28
DATA_DIM    = 90
DIST_THRESH = 1.0       # Angstrom validity cutoff
OUT_REPORT  = './results_crystal_qgan_v4/eval_v4_report.txt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET, 'rb') as f:
    raw = pickle.load(f)
coords_all = np.array([np.array(c).flatten() for c,l in raw], dtype=np.float32)
labels_all = np.array([np.array(l).flatten() for c,l in raw], dtype=np.float32)
print(f'Dataset: {len(coords_all)} samples')

# ── Load generator ────────────────────────────────────────────────────────────
gan = PQWGAN_CC_Crystal(
    input_dim_g=Z_DIM+LABEL_DIM, output_dim=DATA_DIM, input_dim_d=DATA_DIM+LABEL_DIM,
    hidden_features=8, hidden_layers=3, spectrum_layer=1, use_noise=0.0,
)
ckpt = torch.load(CHECKPOINT, map_location=device)
gan.generator.load_state_dict(ckpt['generator'])
gen = gan.generator.to(device).eval()
print(f'Loaded: {CHECKPOINT}')

# ── Generate crystals ─────────────────────────────────────────────────────────
print(f'Generating {N_GEN} crystals...')
gen_coords, gen_labels = [], []
bs = 128
with torch.no_grad():
    for start in range(0, N_GEN, bs):
        end  = min(start+bs, N_GEN)
        lbls = torch.from_numpy(labels_all[start:end]).to(device)
        z    = torch.randn(end-start, Z_DIM, device=device)
        out  = gen(torch.cat([z, lbls], dim=1)).cpu().numpy()
        gen_coords.append(out)
        gen_labels.append(labels_all[start:end])
gen_coords = np.concatenate(gen_coords)
gen_labels = np.concatenate(gen_labels)
print(f'Generated range: [{gen_coords.min():.4f}, {gen_coords.max():.4f}]')

# ── Lattice builder ───────────────────────────────────────────────────────────
def build_lattice_matrix(coords_90):
    arr     = coords_90.reshape(30, 3)
    lengths = arr[0] * 30          # Angstrom
    angles  = arr[1] * 180         # degrees
    a, b, c = np.clip(lengths, 1.0, 30.0)
    al, be, ga = np.radians(np.clip(angles, 10.0, 170.0))
    ax = a
    bx = b * np.cos(ga);  by = b * np.sin(ga)
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be)*np.cos(ga)) / (np.sin(ga) + 1e-9)
    cz = np.sqrt(max(c**2 - cx**2 - cy**2, 1e-6))
    return np.array([[ax, bx, cx], [0, by, cy], [0, 0, cz]])

# ── Min interatomic distance ──────────────────────────────────────────────────
def min_dist_angstrom(coords_90, label_28):
    M    = build_lattice_matrix(coords_90)
    occ  = label_28.astype(bool)
    frac = coords_90.reshape(30, 3)[2:][occ]   # occupied fractional positions
    if len(frac) < 2:
        return 0.0
    cart = (M @ frac.T).T
    dists = [np.linalg.norm(cart[i]-cart[j])
             for i in range(len(cart)) for j in range(i+1, len(cart))]
    return float(min(dists))

print('Computing min-distance validity...')
min_dists  = np.array([min_dist_angstrom(gen_coords[i], gen_labels[i]) for i in range(N_GEN)])
valid_mask = min_dists >= DIST_THRESH
validity   = valid_mask.mean() * 100
print(f'Validity ({DIST_THRESH}A): {validity:.1f}%  mean-dist={min_dists.mean():.3f}A')

# ── Build pymatgen Structure ──────────────────────────────────────────────────
# Slot mapping: Mg=slots 0:8, Mn=slots 8:16, O=slots 16:28
SPECIES_MAP = ['Mg']*8 + ['Mn']*8 + ['O']*12

def to_pymatgen(coords_90, label_28):
    arr     = coords_90.reshape(30, 3)
    lengths = np.clip(arr[0]*30, 1.0, 30.0)
    angles  = np.clip(arr[1]*180, 10.0, 170.0)
    lattice = Lattice.from_parameters(*lengths, *angles)
    occ     = label_28.astype(bool)
    species = [SPECIES_MAP[i] for i in range(28) if occ[i]]
    frac    = arr[2:][occ]
    if len(species) == 0:
        return None
    return Structure(lattice, species, frac, coords_are_cartesian=False)

# ── CHGNet E_above_hull ───────────────────────────────────────────────────────
# We use CHGNet to predict formation energy, then compute E_hull using
# pymatgen PhaseDiagram with elemental reference energies from CHGNet.
print(f'Running CHGNet on {N_EHULL} valid structures...')
chgnet = CHGNet.load()

# Elemental reference energies from CHGNet (eV/atom)
REF_ELEMENTS = ['Mg', 'Mn', 'O']
ref_energies = {}
for el in REF_ELEMENTS:
    # Single-atom bulk references
    lat  = Lattice.cubic(3.0)
    strc = Structure(lat, [el], [[0,0,0]])
    pred = chgnet.predict_structure(strc)
    ref_energies[el] = float(pred['e'])
    print(f'  Ref energy {el}: {ref_energies[el]:.4f} eV/atom')

ehull_values = []
n_chgnet_ok  = 0
valid_indices = np.where(valid_mask)[0][:N_EHULL]

for idx in valid_indices:
    try:
        struct = to_pymatgen(gen_coords[idx], gen_labels[idx])
        if struct is None:
            continue
        pred   = chgnet.predict_structure(struct)
        e_tot  = float(pred['e']) * len(struct)   # eV total

        # Formation energy per atom
        comp = struct.composition
        e_ref = sum(comp[el] * ref_energies[str(el)] for el in comp)
        e_form = (e_tot - e_ref) / len(struct)

        # Build phase diagram entries
        entries = [PDEntry(Composition(el), ref_energies[el]) for el in REF_ELEMENTS]
        entries.append(PDEntry(comp, e_tot / len(struct)))
        try:
            pd = PhaseDiagram(entries)
            e_hull = pd.get_e_above_hull(entries[-1])
        except Exception:
            e_hull = max(e_form, 0.0)   # fallback: formation energy

        ehull_values.append(e_hull)
        n_chgnet_ok += 1
    except Exception as e:
        pass

ehull_values = np.array(ehull_values) if ehull_values else np.array([999.0])
print(f'CHGNet succeeded on {n_chgnet_ok}/{len(valid_indices)} structures')

# ── SSIM ─────────────────────────────────────────────────────────────────────
print('Computing SSIM...')
real_ref = coords_all[np.random.choice(len(coords_all), N_REF_SSIM, replace=False)]

def ssim_1d(a, b):
    dr = max(a.max()-a.min(), b.max()-b.min(), 1e-6)
    return ssim(a, b, data_range=dr)

ssim_scores = np.array([max(ssim_1d(g, r) for r in real_ref) for g in gen_coords])

# ── Report ────────────────────────────────────────────────────────────────────
lines = []
lines.append('=' * 58)
lines.append('  QGAN v4  —  Checkpoint 490  —  Split-Head Generator')
lines.append('=' * 58)
lines.append(f'  Generated crystals        : {N_GEN}')
lines.append(f'  Output range              : [{gen_coords.min():.4f}, {gen_coords.max():.4f}]')
lines.append('')
lines.append('  MIN INTERATOMIC DISTANCE (Angstrom)')
lines.append(f'    Valid (>={DIST_THRESH}A)          : {validity:.1f}%  ({valid_mask.sum()}/{N_GEN})')
lines.append(f'    Mean dist               : {min_dists.mean():.3f} A')
lines.append(f'    Std  dist               : {min_dists.std():.3f} A')
lines.append(f'    Min  dist               : {min_dists.min():.3f} A')
lines.append(f'    Max  dist               : {min_dists.max():.3f} A')
lines.append('')
lines.append('  SSIM (max vs 100 real refs)')
lines.append(f'    Mean                    : {ssim_scores.mean():.4f}')
lines.append(f'    Std                     : {ssim_scores.std():.4f}')
lines.append(f'    > 0.9                   : {(ssim_scores>0.9).mean()*100:.1f}%')
lines.append(f'    > 0.7                   : {(ssim_scores>0.7).mean()*100:.1f}%')
lines.append(f'    > 0.5                   : {(ssim_scores>0.5).mean()*100:.1f}%')
lines.append('')
lines.append(f'  E_ABOVE_HULL via CHGNet  (n={n_chgnet_ok} valid structures)')
lines.append(f'    Mean                    : {ehull_values.mean():.4f} eV/atom')
lines.append(f'    Std                     : {ehull_values.std():.4f} eV/atom')
lines.append(f'    Stable (<0.1 eV/atom)   : {(ehull_values<0.1).mean()*100:.1f}%')
lines.append(f'    Near-stable (<0.5 eV)   : {(ehull_values<0.5).mean()*100:.1f}%')
lines.append(f'    Min                     : {ehull_values.min():.4f} eV/atom')
lines.append('=' * 58)

report = '\n'.join(lines)
print('\n' + report)
with open(OUT_REPORT, 'w') as f:
    f.write(report + '\n')
print(f'\nSaved: {OUT_REPORT}')
