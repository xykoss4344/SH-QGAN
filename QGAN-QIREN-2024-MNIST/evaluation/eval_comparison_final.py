"""
Final comparison: Classical GAN vs QGAN v4
- 4800 crystals each
- Min interatomic distance validity (Angstrom)
- SSIM vs 200 real references
- E_above_hull via CHGNet on valid structures
- Side-by-side table
"""
# Grouped into a subfolder during the 2026-08-30 reorg; core modules stay at the
# package root because eight files import them. This keeps them importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys, os, pickle, warnings, importlib.util
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
Q_DIR     = os.path.dirname(__file__)
CLS_DIR   = ('C:/Users/Adminb/OneDrive/Documents/Projects/crystalGan/'
             'Composition-Conditioned-Crystal-GAN/Composition_Conditioned_Crystal_GAN')
DATASET_Q = os.path.join(Q_DIR, 'datasets/mgmno_100_aug.pickle')
DATASET_C = os.path.join(CLS_DIR, 'mgmno_100_aug.pickle') if os.path.exists(
            os.path.join(CLS_DIR, 'mgmno_100_aug.pickle')) else os.path.join(CLS_DIR, 'mgmno_100.pickle')

Q_CKPT    = os.path.join(Q_DIR, 'results_crystal_qgan_v4/checkpoint_490.pt')
CLS_GEN   = os.path.join(CLS_DIR, 'model_cwgan_mgmno_v2/generator_490')

N_GEN        = 4800
N_REF_SSIM   = 200
N_EHULL_MAX  = 500    # CHGNet subset (valid structures only, up to this many)
DIST_THRESH  = 1.0
OUT_REPORT   = os.path.join(Q_DIR, 'results_crystal_qgan_v4/comparison_report.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f'Loading dataset: {DATASET_Q}')
with open(DATASET_Q, 'rb') as f:
    raw = pickle.load(f)
coords_all = np.array([np.array(c).flatten() for c,l in raw], dtype=np.float32)
labels_all = np.array([np.array(l).flatten() for c,l in raw], dtype=np.float32)
print(f'  {len(coords_all)} samples, coords {coords_all.shape}, labels {labels_all.shape}')

# Tile labels to cover N_GEN
idx = np.tile(np.arange(len(labels_all)), (N_GEN // len(labels_all) + 1))[:N_GEN]
labels_gen = labels_all[idx]   # (N_GEN, 28)

# ── Load QGAN ────────────────────────────────────────────────────────────────
print('Loading QGAN v4...')
sys.path.insert(0, Q_DIR)
from models.QINR_Crystal import PQWGAN_CC_Crystal
qgan = PQWGAN_CC_Crystal(
    input_dim_g=64+28, output_dim=90, input_dim_d=90+28,
    hidden_features=8, hidden_layers=3, spectrum_layer=1, use_noise=0.0,
)
ckpt = torch.load(Q_CKPT, map_location=device)
qgan.generator.load_state_dict(ckpt['generator'])
q_gen = qgan.generator.to(device).eval()
print(f'  Loaded: {Q_CKPT}')

# ── Load Classical GAN ────────────────────────────────────────────────────────
print('Loading Classical GAN...')
spec = importlib.util.spec_from_file_location('cls_models', os.path.join(CLS_DIR, 'models.py'))
cls_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cls_mod)

class _Opt:
    latent_dim = 512
    input_dim  = 512 + 28 + 1

c_gen = cls_mod.Generator(_Opt()).to(device)
raw_ckpt = torch.load(CLS_GEN, map_location=device)
# Checkpoint may be wrapped as {'model': state_dict, ...} or raw state_dict
state_dict = raw_ckpt['model'] if isinstance(raw_ckpt, dict) and 'model' in raw_ckpt else raw_ckpt
c_gen.load_state_dict(state_dict)
c_gen.eval()
print(f'  Loaded: {CLS_GEN}')

# ── Generate from QGAN ───────────────────────────────────────────────────────
print(f'\nGenerating {N_GEN} quantum crystals...')
q_coords = []
with torch.no_grad():
    for s in range(0, N_GEN, 128):
        e = min(s+128, N_GEN)
        lbls = torch.from_numpy(labels_gen[s:e]).to(device)
        z    = torch.randn(e-s, 64, device=device)
        out  = q_gen(torch.cat([z, lbls], dim=1)).cpu().numpy()
        q_coords.append(out)
q_coords = np.concatenate(q_coords)   # (N_GEN, 90)
print(f'  range [{q_coords.min():.4f}, {q_coords.max():.4f}]')

# ── Generate from Classical GAN ───────────────────────────────────────────────
print(f'Generating {N_GEN} classical crystals...')
c_coords = []
with torch.no_grad():
    for s in range(0, N_GEN, 128):
        e    = min(s+128, N_GEN)
        lbls = torch.from_numpy(labels_gen[s:e]).to(device)
        z    = torch.randn(e-s, 512, device=device)
        c1   = lbls[:, 0:8]
        c2   = lbls[:, 8:16]
        c3   = lbls[:, 16:28]
        c4   = lbls.sum(dim=1, keepdim=True).float() / 28.0
        out  = c_gen(z, c1, c2, c3, c4)    # (batch,1,30,3)
        c_coords.append(out.view(e-s, 90).cpu().numpy())
c_coords = np.concatenate(c_coords)   # (N_GEN, 90)
print(f'  range [{c_coords.min():.4f}, {c_coords.max():.4f}]')

# ── Helpers ───────────────────────────────────────────────────────────────────
SPECIES_MAP = ['Mg']*8 + ['Mn']*8 + ['O']*12

def build_lattice_matrix(coords_90):
    arr     = coords_90.reshape(30, 3)
    lengths = np.clip(arr[0]*30, 1.0, 30.0)
    angles  = np.clip(arr[1]*180, 10.0, 170.0)
    a,b,c   = lengths
    al,be,ga = np.radians(angles)
    bx = b*np.cos(ga); by = b*np.sin(ga)
    cx = c*np.cos(be)
    cy = c*(np.cos(al)-np.cos(be)*np.cos(ga))/(np.sin(ga)+1e-9)
    cz = np.sqrt(max(c**2-cx**2-cy**2, 1e-6))
    return np.array([[a, bx, cx],[0, by, cy],[0, 0, cz]])

def min_dist_angstrom(coords_90, label_28):
    M    = build_lattice_matrix(coords_90)
    occ  = label_28.astype(bool)
    frac = coords_90.reshape(30,3)[2:][occ]
    if len(frac) < 2: return 0.0
    cart = (M @ frac.T).T
    dists = [np.linalg.norm(cart[i]-cart[j])
             for i in range(len(cart)) for j in range(i+1,len(cart))]
    return float(min(dists))

def ssim_1d(a, b):
    from skimage.metrics import structural_similarity as ssim
    dr = max(a.max()-a.min(), b.max()-b.min(), 1e-6)
    return ssim(a, b, data_range=dr)

# ── Min-dist validity ─────────────────────────────────────────────────────────
print('\nComputing min-distance validity...')
q_dists = np.array([min_dist_angstrom(q_coords[i], labels_gen[i]) for i in range(N_GEN)])
c_dists = np.array([min_dist_angstrom(c_coords[i], labels_gen[i]) for i in range(N_GEN)])
q_valid = q_dists >= DIST_THRESH
c_valid = c_dists >= DIST_THRESH
print(f'  Quantum  valid: {q_valid.mean()*100:.1f}%  Classical valid: {c_valid.mean()*100:.1f}%')

# ── SSIM ──────────────────────────────────────────────────────────────────────
print(f'Computing SSIM vs {N_REF_SSIM} real references...')
ref_idx  = np.random.choice(len(coords_all), N_REF_SSIM, replace=False)
real_ref = coords_all[ref_idx]

# Sample 500 for SSIM (too slow for 4800)
ssim_sample = 500
q_ssim_scores = np.array([max(ssim_1d(q_coords[i], r) for r in real_ref)
                           for i in range(ssim_sample)])
c_ssim_scores = np.array([max(ssim_1d(c_coords[i], r) for r in real_ref)
                           for i in range(ssim_sample)])
print(f'  Q SSIM mean={q_ssim_scores.mean():.4f}  C SSIM mean={c_ssim_scores.mean():.4f}')

# ── CHGNet E_above_hull ───────────────────────────────────────────────────────
from chgnet.model import CHGNet
from pymatgen.core import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

print(f'\nRunning CHGNet (up to {N_EHULL_MAX} valid structures per model)...')
chgnet = CHGNet.load()

REF_ELS = ['Mg','Mn','O']
ref_e   = {}
for el in REF_ELS:
    s = Structure(Lattice.cubic(3.0), [el], [[0,0,0]])
    ref_e[el] = float(chgnet.predict_structure(s)['e'])
print(f'  Ref energies: Mg={ref_e["Mg"]:.4f}  Mn={ref_e["Mn"]:.4f}  O={ref_e["O"]:.4f} eV/atom')

def to_pymatgen(coords_90, label_28):
    arr  = coords_90.reshape(30,3)
    lat  = Lattice.from_parameters(*np.clip(arr[0]*30,1,30), *np.clip(arr[1]*180,10,170))
    occ  = label_28.astype(bool)
    sp   = [SPECIES_MAP[i] for i in range(28) if occ[i]]
    frac = arr[2:][occ]
    return Structure(lat, sp, frac) if sp else None

def ehull_batch(coords, labels, valid_mask, max_n):
    vals, n_ok = [], 0
    for i in np.where(valid_mask)[0]:
        if n_ok >= max_n: break
        try:
            st = to_pymatgen(coords[i], labels[i])
            if st is None: continue
            pred  = chgnet.predict_structure(st)
            e_tot = float(pred['e']) * len(st)
            comp  = st.composition
            e_ref = sum(comp[el]*ref_e[str(el)] for el in comp)
            e_f   = (e_tot - e_ref) / len(st)
            ents  = [PDEntry(Composition(el), ref_e[el]) for el in REF_ELS]
            ents.append(PDEntry(comp, e_tot/len(st)))
            try:
                eh = PhaseDiagram(ents).get_e_above_hull(ents[-1])
            except Exception:
                eh = max(e_f, 0.0)
            vals.append(eh); n_ok += 1
        except Exception:
            pass
    return np.array(vals) if vals else np.array([999.0]), n_ok

print('  CHGNet on quantum valid structures...')
q_ehull, q_nok = ehull_batch(q_coords, labels_gen, q_valid, N_EHULL_MAX)
print(f'    Quantum: {q_nok} structures  mean={q_ehull.mean():.4f} eV/atom')

print('  CHGNet on classical valid structures...')
c_ehull, c_nok = ehull_batch(c_coords, labels_gen, c_valid, N_EHULL_MAX)
print(f'    Classical: {c_nok} structures  mean={c_ehull.mean():.4f} eV/atom')

# ── Report ────────────────────────────────────────────────────────────────────
lines = []
sep = '=' * 64
lines += [sep,
          '  Classical GAN  vs  QGAN v4  —  Full Comparison  (N=4800)',
          sep,
          f'  {"Metric":<38}  {"Classical":>10}  {"Quantum":>10}',
          '-' * 64]

def row(name, cv, qv): return f'  {name:<38}  {cv:>10}  {qv:>10}'

lines += [
    row('Generated crystals',              f'{N_GEN}',            f'{N_GEN}'),
    row('Checkpoint epoch',                '490',                  '490'),
    '',
    '  MIN INTERATOMIC DISTANCE (Angstrom)',
    row(f'  Valid (>={DIST_THRESH} A)  [%]',
        f'{c_valid.mean()*100:.1f}%', f'{q_valid.mean()*100:.1f}%'),
    row('  Mean dist  [A]',               f'{c_dists.mean():.3f}', f'{q_dists.mean():.3f}'),
    row('  Std  dist  [A]',               f'{c_dists.std():.3f}',  f'{q_dists.std():.3f}'),
    '',
    f'  SSIM  (max vs {N_REF_SSIM} real refs, n={ssim_sample} sampled)',
    row('  Mean',                          f'{c_ssim_scores.mean():.4f}', f'{q_ssim_scores.mean():.4f}'),
    row('  Std',                           f'{c_ssim_scores.std():.4f}',  f'{q_ssim_scores.std():.4f}'),
    row('  > 0.7  [%]',                   f'{(c_ssim_scores>0.7).mean()*100:.1f}%',
                                           f'{(q_ssim_scores>0.7).mean()*100:.1f}%'),
    row('  > 0.5  [%]',                   f'{(c_ssim_scores>0.5).mean()*100:.1f}%',
                                           f'{(q_ssim_scores>0.5).mean()*100:.1f}%'),
    '',
    f'  E_ABOVE_HULL via CHGNet  (n={c_nok} / {q_nok} valid structures)',
    row('  Mean  [eV/atom]',              f'{c_ehull.mean():.4f}', f'{q_ehull.mean():.4f}'),
    row('  Std   [eV/atom]',              f'{c_ehull.std():.4f}',  f'{q_ehull.std():.4f}'),
    row('  Stable  (<0.1 eV/atom)  [%]', f'{(c_ehull<0.1).mean()*100:.1f}%',
                                          f'{(q_ehull<0.1).mean()*100:.1f}%'),
    row('  Near-stable (<0.5 eV)   [%]', f'{(c_ehull<0.5).mean()*100:.1f}%',
                                          f'{(q_ehull<0.5).mean()*100:.1f}%'),
    row('  Min   [eV/atom]',             f'{c_ehull.min():.4f}', f'{q_ehull.min():.4f}'),
    sep,
]

report = '\n'.join(lines)
print('\n' + report)
with open(OUT_REPORT, 'w') as f:
    f.write(report + '\n')
print(f'\nSaved: {OUT_REPORT}')
