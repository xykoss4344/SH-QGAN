"""
Recompute E_hull using MIC-correct validity screen + CHGNet relaxation.
"""
# Grouped into a subfolder during the 2026-08-30 reorg; core modules stay at the
# package root because eight files import them. This keeps them importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, pickle, warnings, importlib.util
import numpy as np
import torch
warnings.filterwarnings('ignore')

Q_DIR   = os.path.dirname(__file__)
CLS_DIR = ('C:/Users/Adminb/OneDrive/Documents/Projects/crystalGan/'
           'Composition-Conditioned-Crystal-GAN/Composition_Conditioned_Crystal_GAN')

DATASET = os.path.join(Q_DIR, 'datasets/mgmno_100_aug.pickle')
Q_CKPT  = os.path.join(Q_DIR, 'results_crystal_qgan_v4/checkpoint_490.pt')
CLS_GEN = os.path.join(CLS_DIR, 'model_cwgan_mgmno_v2/generator_490')
OUT     = os.path.join(Q_DIR, 'results_crystal_qgan_v4/relaxed_mic_report.txt')

N_GEN       = 4800
DIST_THRESH = 1.0
FMAX        = 0.1
MAX_STEPS   = 200
MAX_RELAX   = 1000   # cap per model to keep runtime reasonable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Dataset ───────────────────────────────────────────────────────────────────
with open(DATASET, 'rb') as f: raw = pickle.load(f)
coords_all = np.array([np.array(c).flatten() for c,l in raw], dtype=np.float32)
labels_all = np.array([np.array(l).flatten() for c,l in raw], dtype=np.float32)
idx = np.tile(np.arange(len(labels_all)), (N_GEN // len(labels_all) + 1))[:N_GEN]
labels_gen = labels_all[idx]

# ── Models ────────────────────────────────────────────────────────────────────
sys.path.insert(0, Q_DIR)
from models.QINR_Crystal import PQWGAN_CC_Crystal
qgan = PQWGAN_CC_Crystal(input_dim_g=92, output_dim=90, input_dim_d=118,
                          hidden_features=8, hidden_layers=3, spectrum_layer=1, use_noise=0.0)
qgan.generator.load_state_dict(torch.load(Q_CKPT, map_location=device)['generator'])
q_gen = qgan.generator.to(device).eval()

spec = importlib.util.spec_from_file_location('cls_models', os.path.join(CLS_DIR,'models.py'))
cls_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls_mod)
class _Opt: latent_dim=512; input_dim=541
c_gen = cls_mod.Generator(_Opt()).to(device)
raw_ck = torch.load(CLS_GEN, map_location=device)
c_gen.load_state_dict(raw_ck['model'] if 'model' in raw_ck else raw_ck); c_gen.eval()

# ── Generate ──────────────────────────────────────────────────────────────────
def gen_q(n):
    out=[]
    with torch.no_grad():
        for s in range(0,n,128):
            e=min(s+128,n); lbls=torch.from_numpy(labels_gen[s:e]).to(device)
            out.append(q_gen(torch.cat([torch.randn(e-s,64,device=device),lbls],1)).cpu().numpy())
    return np.concatenate(out)

def gen_c(n):
    out=[]
    with torch.no_grad():
        for s in range(0,n,128):
            e=min(s+128,n); lbls=torch.from_numpy(labels_gen[s:e]).to(device)
            c1,c2,c3=lbls[:,0:8],lbls[:,8:16],lbls[:,16:28]
            c4=lbls.sum(1,keepdim=True).float()/28.
            out.append(c_gen(torch.randn(e-s,512,device=device),c1,c2,c3,c4).view(e-s,90).cpu().numpy())
    return np.concatenate(out)

print(f'Generating {N_GEN} crystals per model...')
q_coords = gen_q(N_GEN)
c_coords = gen_c(N_GEN)

# ── MIC validity screen ───────────────────────────────────────────────────────
from ase import Atoms
SPECIES_MAP = ['Mg']*8 + ['Mn']*8 + ['O']*12

def min_dist_mic(coords_90, label_28):
    arr=coords_90.reshape(30,3)
    lengths=np.clip(arr[0]*30, 2.0, 30.0)
    angles =np.clip(arr[1]*180, 30.0, 150.0)
    occ=label_28.astype(bool)
    sp=[SPECIES_MAP[i] for i in range(28) if occ[i]]
    frac=arr[2:][occ]
    if len(sp)<2: return 0.
    try:
        atoms=Atoms(symbols=sp, scaled_positions=frac,
                    cell=np.concatenate([lengths,angles]), pbc=True)
        d=atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        return float(d.min())
    except: return 0.

print('MIC validity screening...')
q_dists = np.array([min_dist_mic(q_coords[i], labels_gen[i]) for i in range(N_GEN)])
c_dists = np.array([min_dist_mic(c_coords[i], labels_gen[i]) for i in range(N_GEN)])
q_valid = np.where(q_dists >= DIST_THRESH)[0]
c_valid = np.where(c_dists >= DIST_THRESH)[0]
print(f'  Q valid (MIC): {len(q_valid)} ({len(q_valid)/N_GEN*100:.1f}%)')
print(f'  C valid (MIC): {len(c_valid)} ({len(c_valid)/N_GEN*100:.1f}%)')

# Cap at MAX_RELAX
q_relax_idx = q_valid[:MAX_RELAX]
c_relax_idx = c_valid[:MAX_RELAX]

# ── CHGNet relaxation ─────────────────────────────────────────────────────────
from chgnet.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

chgnet  = CHGNet.load()
relaxer = StructOptimizer(model=chgnet, optimizer_class='FIRE', use_device=str(device))

# ── Full MP phase diagram for proper E_above_hull ─────────────────────────────
print('Fetching full Mg-Mn-O phase diagram from Materials Project...')
from mp_api.client import MPRester
with MPRester('hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA') as mpr:
    mp_entries = mpr.get_entries_in_chemsys(['Mg','Mn','O'])
mp_pd = PhaseDiagram(mp_entries)
print(f'  {len(mp_entries)} MP entries, {len(mp_pd.stable_entries)} stable phases on hull')

def to_struct(c90, lbl):
    arr=c90.reshape(30,3)
    lat=Lattice.from_parameters(*np.clip(arr[0]*30,2,30), *np.clip(arr[1]*180,30,150))
    occ=lbl.astype(bool); sp=[SPECIES_MAP[i] for i in range(28) if occ[i]]
    return Structure(lat, sp, arr[2:][occ]) if sp else None

def relax_ehull(c90, lbl):
    try:
        st=to_struct(c90, lbl)
        if st is None: return None
        res=relaxer.relax(st, fmax=FMAX, steps=MAX_STEPS, verbose=False)
        rs=res['final_structure']
        e_per_atom=float(chgnet.predict_structure(rs)['e'])
        # Add generated entry to MP phase diagram and compute E_above_hull
        gen_entry = PDEntry(rs.composition, e_per_atom)
        try:
            return mp_pd.get_e_above_hull(gen_entry)
        except Exception:
            # Fallback: local 3-component diagram
            ents = list(mp_entries) + [gen_entry]
            return PhaseDiagram(ents).get_e_above_hull(gen_entry)
    except: return None

def run(coords, labels, idx, name):
    ehull=np.full(N_GEN, np.nan)
    print(f'\nRelaxing {len(idx)} {name} structures...')
    for k,i in enumerate(idx):
        if k%50==0: print(f'  {k}/{len(idx)}', flush=True)
        v=relax_ehull(coords[i], labels[i])
        if v is not None: ehull[i]=v
    return ehull

q_ehull = run(q_coords, labels_gen, q_relax_idx, 'QUANTUM')
c_ehull = run(c_coords, labels_gen, c_relax_idx, 'CLASSICAL')

# ── Report ────────────────────────────────────────────────────────────────────
def cat(ehull, relax_idx, valid_idx, n, name):
    eh=ehull[relax_idx]; ok=eh[~np.isnan(eh)]
    return dict(
        valid=len(valid_idx), relaxed=len(ok),
        stable      =int((ok<0.1).sum()),
        near_stable =int(((ok>=0.1)&(ok<0.5)).sum()),
        metastable  =int(((ok>=0.5)&(ok<2.0)).sum()),
        high_e      =int((ok>=2.0).sum()),
        eh_mean=float(np.nanmean(ok)) if len(ok) else float('nan'),
        eh_min =float(np.nanmin(ok))  if len(ok) else float('nan'),
    )

Q=cat(q_ehull, q_relax_idx, q_valid, N_GEN, 'Q')
C=cat(c_ehull, c_relax_idx, c_valid, N_GEN, 'C')

def pct(x,n): return f'{x}  ({x/n*100:.1f}%)'
sep='='*70
lines=[sep,
  f'  RELAXED + MIC  Stability Report  (N={N_GEN}, relaxed up to {MAX_RELAX})',
  sep,
  f'  {"Category":<40}  {"Classical":>12}  {"Quantum":>12}',
  '-'*70,
  f'  {"Valid geometry MIC (>=1.0A)":<40}  {pct(C["valid"],N_GEN):>12}  {pct(Q["valid"],N_GEN):>12}',
  f'  {"Successfully relaxed":<40}  {pct(C["relaxed"],N_GEN):>12}  {pct(Q["relaxed"],N_GEN):>12}',
  '',
  f'  {"-- POST-RELAXATION E_hull --":<40}',
  f'  {"  Stable       (< 0.1 eV/at)":<40}  {pct(C["stable"],N_GEN):>12}  {pct(Q["stable"],N_GEN):>12}',
  f'  {"  Near-stable  (0.1-0.5 eV/at)":<40}  {pct(C["near_stable"],N_GEN):>12}  {pct(Q["near_stable"],N_GEN):>12}',
  f'  {"  Metastable   (0.5-2.0 eV/at)":<40}  {pct(C["metastable"],N_GEN):>12}  {pct(Q["metastable"],N_GEN):>12}',
  f'  {"  High energy  (>= 2.0 eV/at)":<40}  {pct(C["high_e"],N_GEN):>12}  {pct(Q["high_e"],N_GEN):>12}',
  '',
  f'  {"E_hull mean (relaxed) [eV/atom]":<40}  {C["eh_mean"]:>12.4f}  {Q["eh_mean"]:>12.4f}',
  f'  {"E_hull min  (relaxed) [eV/atom]":<40}  {C["eh_min"]:>12.4f}  {Q["eh_min"]:>12.4f}',
  sep]

report='\n'.join(lines)
print('\n'+report)
with open(OUT,'w') as f: f.write(report+'\n')
print(f'\nSaved: {OUT}')
