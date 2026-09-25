"""
Three-way ablation eval:
  1. Classical CNN  (Composition-Conditioned, aug100)
  2. Classical ablation (split-head, no quantum, aug100)   ← new
  3. QGAN v4  (split-head, quantum, aug100)

All trained on mgmno_100_aug. Same decode/relax logic as eval_fair.py.
"""
# Grouped into a subfolder during the 2026-08-30 reorg; core modules stay at the
# package root because eight files import them. This keeps them importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys, os, pickle, warnings, importlib.util
import numpy as np
import torch
warnings.filterwarnings('ignore')

QDIR    = os.path.dirname(os.path.abspath(__file__))
CLS_DIR = ('C:/Users/Adminb/OneDrive/Documents/Projects/crystalGan/'
           'Composition-Conditioned-Crystal-GAN/Composition_Conditioned_Crystal_GAN')

DATASET  = os.path.join(QDIR, 'datasets/mgmno_100_aug.pickle')
Q_CKPT   = os.path.join(QDIR, 'results_crystal_qgan_v4/checkpoint_490.pt')
ABL_CKPT = os.path.join(QDIR, 'results_crystal_classical_ablation/checkpoint_490.pt')
CLS_GEN  = os.path.join(CLS_DIR, 'model_cwgan_mgmno_aug100/generator_490')
OUT      = os.path.join(QDIR, 'results_eval_ablation/ablation_report.txt')

N_GEN     = 4800
DIST_THRESH = 1.0
FMAX      = 0.1
MAX_STEPS = 200
MAX_RELAX = 1000

os.makedirs(os.path.dirname(OUT), exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  |  N_GEN={N_GEN}', flush=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
with open(DATASET, 'rb') as f:
    raw = pickle.load(f)
labels_all = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)
idx        = np.tile(np.arange(len(labels_all)), (N_GEN // len(labels_all) + 1))[:N_GEN]
labels_gen = labels_all[idx]

# ── Load models ───────────────────────────────────────────────────────────────
sys.path.insert(0, QDIR)

from models.QINR_Crystal import PQWGAN_CC_Crystal
qgan = PQWGAN_CC_Crystal(input_dim_g=92, output_dim=90, input_dim_d=118,
                          hidden_features=8, hidden_layers=3, spectrum_layer=1, use_noise=0.0)
qgan.generator.load_state_dict(torch.load(Q_CKPT, map_location=device)['generator'])
q_gen = qgan.generator.to(device).eval()
print('QGAN v4 loaded', flush=True)

from models.Classical_Crystal import Classical_CC_Crystal
abl = Classical_CC_Crystal(input_dim_g=92, output_dim=90, input_dim_d=118,
                            hidden_features=8, hidden_layers=3)
abl.generator.load_state_dict(torch.load(ABL_CKPT, map_location=device)['generator'])
a_gen = abl.generator.to(device).eval()
print('Classical ablation loaded', flush=True)

spec    = importlib.util.spec_from_file_location('cls_models', os.path.join(CLS_DIR, 'models.py'))
cls_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls_mod)
class _Opt: latent_dim = 512; input_dim = 541
c_gen = cls_mod.Generator(_Opt()).to(device)
raw_ck = torch.load(CLS_GEN, map_location=device)
c_gen.load_state_dict(raw_ck['model'] if 'model' in raw_ck else raw_ck)
c_gen.eval()
print('Classical CNN loaded', flush=True)

# ── Generation ────────────────────────────────────────────────────────────────
def gen_splithead(gen, n):
    out = []
    with torch.no_grad():
        for s in range(0, n, 128):
            e    = min(s + 128, n)
            lbls = torch.from_numpy(labels_gen[s:e]).to(device)
            z    = torch.randn(e - s, 64, device=device)
            out.append(gen(torch.cat([z, lbls], 1)).cpu().numpy())
    return np.concatenate(out)

def gen_cls(n):
    out = []
    with torch.no_grad():
        for s in range(0, n, 128):
            e    = min(s + 128, n)
            lbls = torch.from_numpy(labels_gen[s:e]).to(device)
            c1, c2, c3 = lbls[:, 0:8], lbls[:, 8:16], lbls[:, 16:28]
            c4   = lbls.sum(1, keepdim=True).float() / 28.
            fake = c_gen(torch.randn(e - s, 512, device=device), c1, c2, c3, c4)
            out.append(fake.view(e - s, 90).cpu().numpy())
    return np.concatenate(out)

print(f'Generating {N_GEN} structures per model...', flush=True)
q_coords   = gen_splithead(q_gen, N_GEN)
abl_coords = gen_splithead(a_gen, N_GEN)
cls_coords = gen_cls(N_GEN)

# ── MIC validity (same logic as eval_fair.py) ─────────────────────────────────
from ase import Atoms
SPECIES_MAP = ['Mg']*8 + ['Mn']*8 + ['O']*12

def min_dist_mic(coords_90, label_28):
    arr     = coords_90.reshape(30, 3)
    lengths = np.clip(arr[0] * 30, 2.0, 30.0)
    angles  = np.clip(arr[1] * 180, 30.0, 150.0)
    occ     = label_28.astype(bool)
    sp      = [SPECIES_MAP[i] for i in range(28) if occ[i]]
    frac    = arr[2:][occ]
    if len(sp) < 2: return 0.
    try:
        atoms = Atoms(symbols=sp, scaled_positions=frac,
                      cell=np.concatenate([lengths, angles]), pbc=True)
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        return float(d.min())
    except: return 0.

print('MIC validity screening...', flush=True)
q_dists   = np.array([min_dist_mic(q_coords[i],   labels_gen[i]) for i in range(N_GEN)])
abl_dists = np.array([min_dist_mic(abl_coords[i], labels_gen[i]) for i in range(N_GEN)])
cls_dists = np.array([min_dist_mic(cls_coords[i], labels_gen[i]) for i in range(N_GEN)])

q_valid   = np.where(q_dists   >= DIST_THRESH)[0]
abl_valid = np.where(abl_dists >= DIST_THRESH)[0]
cls_valid = np.where(cls_dists >= DIST_THRESH)[0]

print(f'  QGAN v4 valid:            {len(q_valid)} ({len(q_valid)/N_GEN*100:.1f}%)', flush=True)
print(f'  Classical ablation valid: {len(abl_valid)} ({len(abl_valid)/N_GEN*100:.1f}%)', flush=True)
print(f'  Classical CNN valid:      {len(cls_valid)} ({len(cls_valid)/N_GEN*100:.1f}%)', flush=True)

# ── CHGNet relaxation + hull ──────────────────────────────────────────────────
from chgnet.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from mp_api.client import MPRester

chgnet  = CHGNet.load()
relaxer = StructOptimizer(model=chgnet, optimizer_class='FIRE', use_device=str(device))

print('Building CHGNet-consistent Mg-Mn-O phase diagram...', flush=True)
with MPRester('hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA') as mpr:
    all_entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'], inc_structure=True)
dft_pd     = PhaseDiagram(all_entries)
stable_dft = dft_pd.stable_entries
chgnet_ref_entries = []
for entry in stable_dft:
    try:
        pred = chgnet.predict_structure(entry.structure)
        chgnet_ref_entries.append(PDEntry(entry.composition, float(pred['e']) * len(entry.structure)))
    except: pass
mp_pd = PhaseDiagram(chgnet_ref_entries) if len(chgnet_ref_entries) >= 3 else dft_pd
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
        if st is None: return None, None
        res     = relaxer.relax(st, fmax=FMAX, steps=MAX_STEPS, verbose=False)
        rs      = res['final_structure']
        e_total = float(chgnet.predict_structure(rs)['e']) * len(rs)
        entry   = PDEntry(rs.composition, e_total)
        try:   eh = mp_pd.get_e_above_hull(entry)
        except:
            eh = PhaseDiagram(list(chgnet_ref_entries) + [entry]).get_e_above_hull(entry)
        return eh, rs
    except: return None, None

def run(coords, labels, idx_arr, name):
    ehull = np.full(N_GEN, np.nan)
    structs = [None] * N_GEN
    cap   = min(len(idx_arr), MAX_RELAX)
    print(f'\nRelaxing {cap} {name} structures...', flush=True)
    for k, i in enumerate(idx_arr[:cap]):
        if k % 50 == 0: print(f'  {k}/{cap}', flush=True)
        v, s = relax_ehull(coords[i], labels[i])
        if v is not None: 
            ehull[i] = v
            structs[i] = s
    return ehull, structs

q_ehull, q_structs     = run(q_coords,   labels_gen, q_valid,   'QGAN-v4')
abl_ehull, abl_structs = run(abl_coords, labels_gen, abl_valid, 'Classical-ablation')
cls_ehull, cls_structs = run(cls_coords, labels_gen, cls_valid, 'Classical-CNN')

# ── Report ────────────────────────────────────────────────────────────────────
def stats(ehull, idx_arr):
    cap = min(len(idx_arr), MAX_RELAX)
    eh  = ehull[idx_arr[:cap]]
    ok  = eh[~np.isnan(eh)]
    return dict(valid=len(idx_arr), relaxed=len(ok),
                stable=int((ok<0.1).sum()), near=int(((ok>=0.1)&(ok<0.5)).sum()),
                meta=int(((ok>=0.5)&(ok<2.0)).sum()), high=int((ok>=2.0).sum()),
                mean=float(np.nanmean(ok)) if len(ok) else float('nan'),
                mn  =float(np.nanmin(ok))  if len(ok) else float('nan'))

Q = stats(q_ehull,   q_valid)
A = stats(abl_ehull, abl_valid)
C = stats(cls_ehull, cls_valid)

def pct(x): return f'{x}  ({x/N_GEN*100:.1f}%)'
sep = '=' * 80

lines = [
    sep,
    f'  THREE-WAY ABLATION EVAL  [mgmno_100_aug  N={N_GEN}]',
    sep,
    f'  {"Category":<42} {"Cls-CNN":>14} {"Cls-Ablation":>14} {"QGAN-v4":>10}',
    '-' * 80,
    f'  {"Valid geometry MIC (>=1.0A)":<42} {pct(C["valid"]):>14} {pct(A["valid"]):>14} {pct(Q["valid"]):>10}',
    f'  {"Relaxed":<42} {pct(C["relaxed"]):>14} {pct(A["relaxed"]):>14} {pct(Q["relaxed"]):>10}',
    '',
    f'  {"-- POST-RELAXATION E_above_hull --":<42}',
    f'  {"  Stable       (< 0.1 eV/at)":<42} {pct(C["stable"]):>14} {pct(A["stable"]):>14} {pct(Q["stable"]):>10}',
    f'  {"  Near-stable  (0.1-0.5 eV/at)":<42} {pct(C["near"]):>14} {pct(A["near"]):>14} {pct(Q["near"]):>10}',
    f'  {"  Metastable   (0.5-2.0 eV/at)":<42} {pct(C["meta"]):>14} {pct(A["meta"]):>14} {pct(Q["meta"]):>10}',
    f'  {"  High energy  (>= 2.0 eV/at)":<42} {pct(C["high"]):>14} {pct(A["high"]):>14} {pct(Q["high"]):>10}',
    '',
    f'  {"E_hull mean [eV/atom]":<42} {C["mean"]:>14.4f} {A["mean"]:>14.4f} {Q["mean"]:>10.4f}',
    f'  {"E_hull min  [eV/atom]":<42} {C["mn"]:>14.4f} {A["mn"]:>14.4f} {Q["mn"]:>10.4f}',
    sep,
]

report = '\n'.join(lines)
print('\n' + report, flush=True)
with open(OUT, 'w') as f:
    f.write(report + '\n')
print(f'\nSaved text report: {OUT}', flush=True)

cache_path = OUT.replace('.txt', '.pkl')
with open(cache_path, 'wb') as f:
    pickle.dump({'q_ehull': q_ehull, 'q_structs': q_structs, 
                 'abl_ehull': abl_ehull, 'abl_structs': abl_structs,
                 'cls_ehull': cls_ehull, 'cls_structs': cls_structs}, f)
print(f'Saved structural cache: {cache_path}', flush=True)
