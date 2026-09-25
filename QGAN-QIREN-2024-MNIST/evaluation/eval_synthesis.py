"""
Categorize all 4800 generated crystals (both models) into:
  - Invalid geometry  : min dist < 1.0 A
  - High energy       : valid geometry but E_hull >= 2.0 eV/atom
  - Metastable        : 0.5 <= E_hull < 2.0 eV/atom
  - Near-stable       : 0.1 <= E_hull < 0.5 eV/atom
  - Stable            : E_hull < 0.1 eV/atom

Runs CHGNet on ALL geometrically valid structures.
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

DATASET  = os.path.join(Q_DIR, 'datasets/mgmno_100_aug.pickle')
Q_CKPT   = os.path.join(Q_DIR, 'results_crystal_qgan_v4/checkpoint_490.pt')
CLS_GEN  = os.path.join(CLS_DIR, 'model_cwgan_mgmno_v2/generator_490')
OUT      = os.path.join(Q_DIR, 'results_crystal_qgan_v4/synthesis_report.txt')

N_GEN       = 4800
DIST_THRESH = 1.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Dataset ───────────────────────────────────────────────────────────────────
with open(DATASET, 'rb') as f: raw = pickle.load(f)
coords_all = np.array([np.array(c).flatten() for c,l in raw], dtype=np.float32)
labels_all = np.array([np.array(l).flatten() for c,l in raw], dtype=np.float32)
idx = np.tile(np.arange(len(labels_all)), (N_GEN // len(labels_all) + 1))[:N_GEN]
labels_gen = labels_all[idx]

# ── Load models ───────────────────────────────────────────────────────────────
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
c_gen.load_state_dict(raw_ck['model'] if 'model' in raw_ck else raw_ck)
c_gen.eval()
print('Models loaded.')

# ── Generate ──────────────────────────────────────────────────────────────────
def gen_quantum(n):
    out = []
    with torch.no_grad():
        for s in range(0, n, 128):
            e = min(s+128, n)
            lbls = torch.from_numpy(labels_gen[s:e]).to(device)
            z = torch.randn(e-s, 64, device=device)
            out.append(q_gen(torch.cat([z,lbls],1)).cpu().numpy())
    return np.concatenate(out)

def gen_classical(n):
    out = []
    with torch.no_grad():
        for s in range(0, n, 128):
            e = min(s+128, n)
            lbls = torch.from_numpy(labels_gen[s:e]).to(device)
            z = torch.randn(e-s, 512, device=device)
            c1,c2,c3 = lbls[:,0:8], lbls[:,8:16], lbls[:,16:28]
            c4 = lbls.sum(1,keepdim=True).float()/28.0
            out.append(c_gen(z,c1,c2,c3,c4).view(e-s,90).cpu().numpy())
    return np.concatenate(out)

print(f'Generating {N_GEN} crystals per model...')
q_coords = gen_quantum(N_GEN);   print('  Quantum done.')
c_coords = gen_classical(N_GEN); print('  Classical done.')

# ── Min-dist ──────────────────────────────────────────────────────────────────
SPECIES_MAP = ['Mg']*8 + ['Mn']*8 + ['O']*12

def build_M(coords_90):
    arr = coords_90.reshape(30,3)
    a,b,c = np.clip(arr[0]*30,1,30)
    al,be,ga = np.radians(np.clip(arr[1]*180,10,170))
    bx=b*np.cos(ga); by=b*np.sin(ga)
    cx=c*np.cos(be)
    cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/(np.sin(ga)+1e-9)
    cz=np.sqrt(max(c**2-cx**2-cy**2,1e-6))
    return np.array([[a,bx,cx],[0,by,cy],[0,0,cz]])

def min_dist(coords_90, lbl):
    M=build_M(coords_90); occ=lbl.astype(bool)
    frac=coords_90.reshape(30,3)[2:][occ]
    if len(frac)<2: return 0.0
    cart=(M@frac.T).T
    return float(min(np.linalg.norm(cart[i]-cart[j])
                     for i in range(len(cart)) for j in range(i+1,len(cart))))

print('Computing distances...')
q_dists = np.array([min_dist(q_coords[i],labels_gen[i]) for i in range(N_GEN)])
c_dists = np.array([min_dist(c_coords[i],labels_gen[i]) for i in range(N_GEN)])
q_valid = q_dists >= DIST_THRESH
c_valid = c_dists >= DIST_THRESH
print(f'  Q valid: {q_valid.sum()}  C valid: {c_valid.sum()}')

# ── CHGNet on ALL valid structures ────────────────────────────────────────────
from chgnet.model import CHGNet
from pymatgen.core import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

chgnet = CHGNet.load()
REF_ELS = ['Mg','Mn','O']
ref_e = {}
for el in REF_ELS:
    ref_e[el] = float(chgnet.predict_structure(Structure(Lattice.cubic(3.),[el],[[0,0,0]]))['e'])

def to_struct(coords_90, lbl):
    arr=coords_90.reshape(30,3)
    lat=Lattice.from_parameters(*np.clip(arr[0]*30,1,30),*np.clip(arr[1]*180,10,170))
    occ=lbl.astype(bool)
    sp=[SPECIES_MAP[i] for i in range(28) if occ[i]]
    frac=arr[2:][occ]
    return Structure(lat,sp,frac) if sp else None

def get_ehull(coords_90, lbl):
    try:
        st=to_struct(coords_90,lbl)
        if st is None: return None
        pred=chgnet.predict_structure(st)
        e_tot=float(pred['e'])*len(st)
        comp=st.composition
        e_ref=sum(comp[el]*ref_e[str(el)] for el in comp)
        e_f=(e_tot-e_ref)/len(st)
        ents=[PDEntry(Composition(el),ref_e[el]) for el in REF_ELS]
        ents.append(PDEntry(comp,e_tot/len(st)))
        try:    return PhaseDiagram(ents).get_e_above_hull(ents[-1])
        except: return max(e_f,0.0)
    except: return None

def run_chgnet(coords, labels, valid_mask, name):
    indices = np.where(valid_mask)[0]
    print(f'  Running CHGNet on {len(indices)} {name} valid structures...')
    ehull = np.full(len(coords), np.nan)
    for k, i in enumerate(indices):
        if k % 50 == 0: print(f'    {k}/{len(indices)}', flush=True)
        v = get_ehull(coords[i], labels[i])
        if v is not None: ehull[i] = v
    return ehull

q_ehull = run_chgnet(q_coords, labels_gen, q_valid, 'quantum')
c_ehull = run_chgnet(c_coords, labels_gen, c_valid, 'classical')

# ── Categorize ────────────────────────────────────────────────────────────────
def categorize(dists, ehull, n):
    invalid   = (dists < DIST_THRESH).sum()
    valid_idx = np.where(dists >= DIST_THRESH)[0]
    eh_valid  = ehull[valid_idx]
    eh_ok     = eh_valid[~np.isnan(eh_valid)]

    stable      = (eh_ok < 0.1).sum()
    near_stable = ((eh_ok >= 0.1) & (eh_ok < 0.5)).sum()
    metastable  = ((eh_ok >= 0.5) & (eh_ok < 2.0)).sum()
    high_e      = (eh_ok >= 2.0).sum()
    chgnet_fail = np.isnan(eh_valid).sum()

    return dict(
        n=n, invalid=int(invalid),
        valid_geom=int(len(valid_idx)),
        stable=int(stable), near_stable=int(near_stable),
        metastable=int(metastable), high_e=int(high_e),
        chgnet_fail=int(chgnet_fail),
        ehull_mean=float(np.nanmean(ehull[valid_idx])) if len(valid_idx) else float('nan'),
        ehull_min=float(np.nanmin(ehull[valid_idx]))   if len(valid_idx) else float('nan'),
    )

Q = categorize(q_dists, q_ehull, N_GEN)
C = categorize(c_dists, c_ehull, N_GEN)

# ── Report ────────────────────────────────────────────────────────────────────
def pct(x, n): return f'{x}  ({x/n*100:.1f}%)'

lines = []
sep = '=' * 66
lines += [sep,
          f'  Crystal Synthesis Report  —  N={N_GEN} per model  (Epoch 490)',
          sep,
          f'  {"Category":<36}  {"Classical":>12}  {"Quantum":>12}',
          '-'*66,
          f'  {"Total generated":<36}  {pct(C["n"],N_GEN):>12}  {pct(Q["n"],N_GEN):>12}',
          '',
          f'  {"-- GEOMETRY SCREEN (min dist >= 1.0 A) --":<36}',
          f'  {"Invalid geometry  (dist < 1.0 A)":<36}  {pct(C["invalid"],N_GEN):>12}  {pct(Q["invalid"],N_GEN):>12}',
          f'  {"Valid geometry":<36}  {pct(C["valid_geom"],N_GEN):>12}  {pct(Q["valid_geom"],N_GEN):>12}',
          '',
          f'  {"-- ENERGY SCREEN (CHGNet E_hull) --":<36}',
          f'  {"  Stable       (E_hull < 0.1 eV/at)":<36}  {pct(C["stable"],N_GEN):>12}  {pct(Q["stable"],N_GEN):>12}',
          f'  {"  Near-stable  (0.1–0.5 eV/at)":<36}  {pct(C["near_stable"],N_GEN):>12}  {pct(Q["near_stable"],N_GEN):>12}',
          f'  {"  Metastable   (0.5–2.0 eV/at)":<36}  {pct(C["metastable"],N_GEN):>12}  {pct(Q["metastable"],N_GEN):>12}',
          f'  {"  High energy  (>= 2.0 eV/at)":<36}  {pct(C["high_e"],N_GEN):>12}  {pct(Q["high_e"],N_GEN):>12}',
          '',
          f'  {"E_hull mean  [eV/atom]":<36}  {C["ehull_mean"]:>12.4f}  {Q["ehull_mean"]:>12.4f}',
          f'  {"E_hull min   [eV/atom]":<36}  {C["ehull_min"]:>12.4f}  {Q["ehull_min"]:>12.4f}',
          sep]

report = '\n'.join(lines)
print('\n' + report)
with open(OUT,'w') as f: f.write(report+'\n')
print(f'\nSaved: {OUT}')
