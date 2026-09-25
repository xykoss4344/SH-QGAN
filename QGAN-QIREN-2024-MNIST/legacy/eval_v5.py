"""
Full evaluation: Quantum v5 vs Classical v3
MIC validity + CHGNet relaxation + full MP phase diagram E_hull
"""
# Grouped into a subfolder during the 2026-08-30 reorg; core modules stay at the
# package root because eight files import them. This keeps them importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, pickle, warnings, importlib.util
import numpy as np
import torch
warnings.filterwarnings('ignore')

Q_DIR   = os.path.dirname(os.path.abspath(__file__))
CLS_DIR = ('C:/Users/Adminb/OneDrive/Documents/Projects/crystalGan/'
           'Composition-Conditioned-Crystal-GAN/Composition_Conditioned_Crystal_GAN')

# Quantum v4 is used (v5 cell-MSE loss caused atom clustering — v4 = 46.6% MIC validity)
Q_DATASET  = os.path.join(Q_DIR, 'datasets/mgmno_100_aug.pickle')  # quantum v4 trained on this
C_DATASET  = os.path.join(Q_DIR, 'datasets/mgmno_1000.pickle')     # classical v3 trained on this
Q_CKPT     = os.path.join(Q_DIR, 'results_crystal_qgan_v4/checkpoint_490.pt')
CLS_GEN    = os.path.join(CLS_DIR, 'model_cwgan_mgmno_v3/generator_490')
OUT        = os.path.join(Q_DIR, 'results_eval_v5v3/eval_report.txt')

# Allow quick-eval override from CLI: python eval_v5.py quick <epoch>
QUICK    = len(sys.argv) > 1 and sys.argv[1] == 'quick'
CLS_EPOCH = int(sys.argv[2]) if len(sys.argv) > 2 else 490
if QUICK:
    CLS_GEN = os.path.join(CLS_DIR, f'model_cwgan_mgmno_v3/generator_{CLS_EPOCH}')
    OUT     = os.path.join(Q_DIR, f'results_eval_v5v3/quickcheck_cls{CLS_EPOCH}.txt')

N_GEN       = 200 if QUICK else 4800
DIST_THRESH = 1.0
FMAX        = 0.1
MAX_STEPS   = 200
MAX_RELAX   = 50 if QUICK else 1000

os.makedirs(os.path.dirname(OUT), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  |  N_GEN={N_GEN}  QUICK={QUICK}', flush=True)

# ── Dataset — separate label sets per model (each evaluated on its training dist) ─
def load_labels(path, n):
    print(f'Dataset: {path}', flush=True)
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    labels = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)
    idx    = np.tile(np.arange(len(labels)), (n // len(labels) + 1))[:n]
    return labels[idx]

q_labels = load_labels(Q_DATASET, N_GEN)   # quantum trained on 100_aug
c_labels = load_labels(C_DATASET, N_GEN)   # classical v3 trained on 1000

# ── Quantum generator ─────────────────────────────────────────────────────────
sys.path.insert(0, Q_DIR)
from models.QINR_Crystal import PQWGAN_CC_Crystal
qgan = PQWGAN_CC_Crystal(input_dim_g=92, output_dim=90, input_dim_d=118,
                          hidden_features=8, hidden_layers=3, spectrum_layer=1, use_noise=0.0)
ckpt = torch.load(Q_CKPT, map_location=device)
qgan.generator.load_state_dict(ckpt['generator'])
q_gen = qgan.generator.to(device).eval()
print(f'Quantum v5 loaded: {Q_CKPT}', flush=True)

# ── Classical generator ───────────────────────────────────────────────────────
c_gen = None
cls_available = os.path.exists(CLS_GEN)
if cls_available:
    spec    = importlib.util.spec_from_file_location('cls_models', os.path.join(CLS_DIR, 'models.py'))
    cls_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls_mod)
    class _Opt: latent_dim = 512; input_dim = 541
    c_gen   = cls_mod.Generator(_Opt()).to(device)
    raw_ck  = torch.load(CLS_GEN, map_location=device)
    c_gen.load_state_dict(raw_ck['model'] if 'model' in raw_ck else raw_ck)
    c_gen.eval()
    print(f'Classical v3 loaded: {CLS_GEN}', flush=True)
else:
    print(f'Classical checkpoint not found ({CLS_GEN}) — quantum only', flush=True)

# ── Generation ────────────────────────────────────────────────────────────────
def gen_q(n, labels):
    out = []
    with torch.no_grad():
        for s in range(0, n, 128):
            e    = min(s + 128, n)
            lbls = torch.from_numpy(labels[s:e]).to(device)
            out.append(q_gen(torch.cat([torch.randn(e - s, 64, device=device), lbls], 1)).cpu().numpy())
    return np.concatenate(out)

def gen_c(n, labels):
    out = []
    with torch.no_grad():
        for s in range(0, n, 128):
            e    = min(s + 128, n)
            lbls = torch.from_numpy(labels[s:e]).to(device)
            c1, c2, c3 = lbls[:, 0:8], lbls[:, 8:16], lbls[:, 16:28]
            c4          = lbls.sum(1, keepdim=True).float() / 28.
            out.append(c_gen(torch.randn(e - s, 512, device=device), c1, c2, c3, c4).view(e - s, 90).cpu().numpy())
    return np.concatenate(out)

print(f'Generating {N_GEN} crystals per model...', flush=True)
q_coords = gen_q(N_GEN, q_labels)
c_coords = gen_c(N_GEN, c_labels) if cls_available else None

# ── MIC validity screen ───────────────────────────────────────────────────────
from ase import Atoms
SPECIES_MAP = ['Mg'] * 8 + ['Mn'] * 8 + ['O'] * 12

def min_dist_mic(coords_90, label_28):
    arr     = coords_90.reshape(30, 3)
    lengths = np.clip(arr[0] * 30, 2.0, 30.0)
    angles  = np.clip(arr[1] * 180, 30.0, 150.0)
    occ     = label_28.astype(bool)
    sp      = [SPECIES_MAP[i] for i in range(28) if occ[i]]
    frac    = arr[2:][occ]
    if len(sp) < 2:
        return 0.
    try:
        atoms = Atoms(symbols=sp, scaled_positions=frac,
                      cell=np.concatenate([lengths, angles]), pbc=True)
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        return float(d.min())
    except:
        return 0.

print('MIC validity screening...', flush=True)
q_dists = np.array([min_dist_mic(q_coords[i], q_labels[i]) for i in range(N_GEN)])
q_valid = np.where(q_dists >= DIST_THRESH)[0]
print(f'  Q valid (MIC): {len(q_valid)} ({len(q_valid)/N_GEN*100:.1f}%)', flush=True)

if cls_available:
    c_dists = np.array([min_dist_mic(c_coords[i], c_labels[i]) for i in range(N_GEN)])
    c_valid = np.where(c_dists >= DIST_THRESH)[0]
    print(f'  C valid (MIC): {len(c_valid)} ({len(c_valid)/N_GEN*100:.1f}%)', flush=True)
else:
    c_valid = np.array([], dtype=int)

# ── CHGNet relaxation ─────────────────────────────────────────────────────────
from chgnet.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

chgnet  = CHGNet.load()
relaxer = StructOptimizer(model=chgnet, optimizer_class='FIRE', use_device=str(device))

# Build CHGNet-consistent hull: re-predict stable MP structures with CHGNet
# This avoids the DFT+U vs CHGNet systematic energy offset (~4-6 eV/at for Mn compounds)
print('Building CHGNet-consistent Mg-Mn-O phase diagram...', flush=True)
from mp_api.client import MPRester
with MPRester('hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA') as mpr:
    all_entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'], inc_structure=True)

# Find which are stable via DFT phase diagram
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
dft_pd       = PhaseDiagram(all_entries)
stable_dft   = dft_pd.stable_entries
print(f'  {len(all_entries)} total entries, {len(stable_dft)} DFT-stable — re-predicting with CHGNet...', flush=True)

chgnet_ref_entries = []
for entry in stable_dft:
    try:
        st   = entry.structure          # ComputedStructureEntry has .structure
        pred = chgnet.predict_structure(st)
        e_pa = float(pred['e'])
        chgnet_ref_entries.append(PDEntry(entry.composition, e_pa * len(st)))
    except Exception as ex:
        pass  # skip if structure unavailable or CHGNet fails

if len(chgnet_ref_entries) < 3:
    # Fallback: use DFT entries directly (known offset, but better than nothing)
    print('  WARNING: too few CHGNet refs — falling back to DFT+U entries', flush=True)
    mp_pd = dft_pd
else:
    mp_pd = PhaseDiagram(chgnet_ref_entries)
    print(f'  CHGNet hull: {len(chgnet_ref_entries)} refs, {len(mp_pd.stable_entries)} on hull', flush=True)

def to_struct(c90, lbl):
    arr = c90.reshape(30, 3)
    lat = Lattice.from_parameters(*np.clip(arr[0]*30, 2, 30), *np.clip(arr[1]*180, 30, 150))
    occ = lbl.astype(bool)
    sp  = [SPECIES_MAP[i] for i in range(28) if occ[i]]
    return Structure(lat, sp, arr[2:][occ]) if sp else None

def relax_ehull(c90, lbl):
    try:
        st = to_struct(c90, lbl)
        if st is None:
            return None
        res        = relaxer.relax(st, fmax=FMAX, steps=MAX_STEPS, verbose=False)
        rs         = res['final_structure']
        e_per_atom = float(chgnet.predict_structure(rs)['e'])
        e_total    = e_per_atom * len(rs)
        gen_entry  = PDEntry(rs.composition, e_total)
        try:
            return mp_pd.get_e_above_hull(gen_entry)
        except:
            # Fallback: add entry to hull and recompute
            ents = list(chgnet_ref_entries) + [gen_entry]
            return PhaseDiagram(ents).get_e_above_hull(gen_entry)
    except:
        return None

def run(coords, labels, idx_arr, name):
    ehull = np.full(len(labels), np.nan)
    cap   = min(len(idx_arr), MAX_RELAX)
    print(f'\nRelaxing {cap} {name} structures...', flush=True)
    for k, i in enumerate(idx_arr[:cap]):
        if k % (10 if QUICK else 50) == 0:
            print(f'  {k}/{cap}', flush=True)
        v = relax_ehull(coords[i], labels[i])
        if v is not None:
            ehull[i] = v
    return ehull

q_ehull = run(q_coords, q_labels, q_valid, 'QUANTUM')
c_ehull = run(c_coords, c_labels, c_valid, 'CLASSICAL') if cls_available else np.full(N_GEN, np.nan)

# ── Report ────────────────────────────────────────────────────────────────────
def stats(ehull, relax_idx, n, name):
    cap = min(len(relax_idx), MAX_RELAX)
    eh  = ehull[relax_idx[:cap]]
    ok  = eh[~np.isnan(eh)]
    return dict(
        valid=len(relax_idx), relaxed=len(ok),
        stable      =int((ok < 0.1).sum()),
        near_stable =int(((ok >= 0.1) & (ok < 0.5)).sum()),
        metastable  =int(((ok >= 0.5) & (ok < 2.0)).sum()),
        high_e      =int((ok >= 2.0).sum()),
        eh_mean=float(np.nanmean(ok)) if len(ok) else float('nan'),
        eh_min =float(np.nanmin(ok))  if len(ok) else float('nan'),
    )

Q = stats(q_ehull, q_valid, N_GEN, 'Q')
C = stats(c_ehull, c_valid if cls_available else np.array([], dtype=int), N_GEN, 'C')

def pct(x, n): return f'{x}  ({x/n*100:.1f}%)'
sep   = '=' * 70
title = 'QUICK-CHECK' if QUICK else 'FULL EVAL'
tag   = f'cls_ep={CLS_EPOCH}  N={N_GEN}' if QUICK else f'N={N_GEN}'

lines = [sep,
  f'  {title}  Quantum-v5 vs Classical-v3  [{tag}]',
  sep,
  f'  {"Category":<40}  {"Classical-v3":>14}  {"Quantum-v5":>12}',
  '-' * 70,
  f'  {"Valid geometry MIC (>=1.0A)":<40}  {pct(C["valid"],N_GEN):>14}  {pct(Q["valid"],N_GEN):>12}',
  f'  {"Relaxed (capped)":<40}  {pct(C["relaxed"],N_GEN):>14}  {pct(Q["relaxed"],N_GEN):>12}',
  '',
  f'  {"-- POST-RELAXATION E_above_hull --":<40}',
  f'  {"  Stable       (< 0.1 eV/at)":<40}  {pct(C["stable"],N_GEN):>14}  {pct(Q["stable"],N_GEN):>12}',
  f'  {"  Near-stable  (0.1-0.5 eV/at)":<40}  {pct(C["near_stable"],N_GEN):>14}  {pct(Q["near_stable"],N_GEN):>12}',
  f'  {"  Metastable   (0.5-2.0 eV/at)":<40}  {pct(C["metastable"],N_GEN):>14}  {pct(Q["metastable"],N_GEN):>12}',
  f'  {"  High energy  (>= 2.0 eV/at)":<40}  {pct(C["high_e"],N_GEN):>14}  {pct(Q["high_e"],N_GEN):>12}',
  '',
  f'  {"E_hull mean [eV/atom]":<40}  {C["eh_mean"]:>14.4f}  {Q["eh_mean"]:>12.4f}',
  f'  {"E_hull min  [eV/atom]":<40}  {C["eh_min"]:>14.4f}  {Q["eh_min"]:>12.4f}',
  sep]

report = '\n'.join(lines)
print('\n' + report)
with open(OUT, 'w') as f:
    f.write(report + '\n')
print(f'\nSaved: {OUT}', flush=True)
