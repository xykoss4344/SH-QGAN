"""
Relax all geometrically valid generated crystals with CHGNet StructOptimizer,
then compute E_above_hull on the relaxed structures.
Runs on both quantum (v4) and classical models.
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
OUT     = os.path.join(Q_DIR, 'results_crystal_qgan_v4/relaxed_stability_report.txt')

N_GEN        = 4800
DIST_THRESH  = 1.0
FMAX         = 0.1    # eV/A convergence
MAX_STEPS    = 200
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
qgan = PQWGAN_CC_Crystal(input_dim_g=92,output_dim=90,input_dim_d=118,
                          hidden_features=8,hidden_layers=3,spectrum_layer=1,use_noise=0.0)
qgan.generator.load_state_dict(torch.load(Q_CKPT,map_location=device)['generator'])
q_gen = qgan.generator.to(device).eval()

spec = importlib.util.spec_from_file_location('cls_models', os.path.join(CLS_DIR,'models.py'))
cls_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls_mod)
class _Opt: latent_dim=512; input_dim=541
c_gen = cls_mod.Generator(_Opt()).to(device)
raw_ck = torch.load(CLS_GEN, map_location=device)
c_gen.load_state_dict(raw_ck['model'] if 'model' in raw_ck else raw_ck)
c_gen.eval()

# ── Generate ──────────────────────────────────────────────────────────────────
def gen_q(n):
    out=[]
    with torch.no_grad():
        for s in range(0,n,128):
            e=min(s+128,n); lbls=torch.from_numpy(labels_gen[s:e]).to(device)
            z=torch.randn(e-s,64,device=device)
            out.append(q_gen(torch.cat([z,lbls],1)).cpu().numpy())
    return np.concatenate(out)

def gen_c(n):
    out=[]
    with torch.no_grad():
        for s in range(0,n,128):
            e=min(s+128,n); lbls=torch.from_numpy(labels_gen[s:e]).to(device)
            z=torch.randn(e-s,512,device=device)
            c1,c2,c3=lbls[:,0:8],lbls[:,8:16],lbls[:,16:28]
            c4=lbls.sum(1,keepdim=True).float()/28.
            out.append(c_gen(z,c1,c2,c3,c4).view(e-s,90).cpu().numpy())
    return np.concatenate(out)

print(f'Generating {N_GEN} crystals per model...')
q_coords = gen_q(N_GEN)
c_coords = gen_c(N_GEN)

# ── Geometry screen ───────────────────────────────────────────────────────────
SPECIES_MAP = ['Mg']*8 + ['Mn']*8 + ['O']*12

def build_M(c90):
    arr=c90.reshape(30,3); a,b,c=np.clip(arr[0]*30,1,30)
    al,be,ga=np.radians(np.clip(arr[1]*180,10,170))
    bx=b*np.cos(ga); by=b*np.sin(ga); cx2=c*np.cos(be)
    cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/(np.sin(ga)+1e-9)
    cz=np.sqrt(max(c**2-cx2**2-cy**2,1e-6))
    return np.array([[a,bx,cx2],[0,by,cy],[0,0,cz]])

def min_dist(c90, lbl):
    M=build_M(c90); occ=lbl.astype(bool); frac=c90.reshape(30,3)[2:][occ]
    if len(frac)<2: return 0.
    cart=(M@frac.T).T
    return float(min(np.linalg.norm(cart[i]-cart[j])
                     for i in range(len(cart)) for j in range(i+1,len(cart))))

print('Geometry screening...')
q_dists=np.array([min_dist(q_coords[i],labels_gen[i]) for i in range(N_GEN)])
c_dists=np.array([min_dist(c_coords[i],labels_gen[i]) for i in range(N_GEN)])
q_valid=np.where(q_dists>=DIST_THRESH)[0]
c_valid=np.where(c_dists>=DIST_THRESH)[0]
print(f'  Q valid: {len(q_valid)}   C valid: {len(c_valid)}')

# ── CHGNet relaxation + E_hull ────────────────────────────────────────────────
from chgnet.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

chgnet  = CHGNet.load()
relaxer = StructOptimizer(model=chgnet, optimizer_class='FIRE', use_device=str(device))

REF_ELS = ['Mg','Mn','O']
ref_e   = {el: float(chgnet.predict_structure(
               Structure(Lattice.cubic(3.),[el],[[0,0,0]]))['e']) for el in REF_ELS}
print(f'Ref energies: {ref_e}')

def to_struct(c90, lbl):
    arr=c90.reshape(30,3)
    lat=Lattice.from_parameters(*np.clip(arr[0]*30,1,30),*np.clip(arr[1]*180,10,170))
    occ=lbl.astype(bool); sp=[SPECIES_MAP[i] for i in range(28) if occ[i]]
    return Structure(lat,sp,arr[2:][occ]) if sp else None

def relax_and_ehull(c90, lbl):
    st = to_struct(c90, lbl)
    if st is None: return None, None
    try:
        res = relaxer.relax(st, fmax=FMAX, steps=MAX_STEPS, verbose=False)
        relax_st = res['final_structure']
        e_tot    = float(chgnet.predict_structure(relax_st)['e']) * len(relax_st)
        comp     = relax_st.composition
        e_ref    = sum(comp[el]*ref_e[str(el)] for el in comp)
        e_f      = (e_tot - e_ref) / len(relax_st)
        ents     = [PDEntry(Composition(el),ref_e[el]) for el in REF_ELS]
        ents.append(PDEntry(comp, e_tot/len(relax_st)))
        try:    eh = PhaseDiagram(ents).get_e_above_hull(ents[-1])
        except: eh = max(e_f, 0.)
        return eh, e_f
    except Exception as ex:
        return None, None

def run_relaxation(coords, labels, valid_idx, name):
    ehull_arr = np.full(N_GEN, np.nan)
    ef_arr    = np.full(N_GEN, np.nan)
    print(f'\nRelaxing {len(valid_idx)} {name} structures...')
    for k, i in enumerate(valid_idx):
        if k % 20 == 0: print(f'  {k}/{len(valid_idx)}', flush=True)
        eh, ef = relax_and_ehull(coords[i], labels[i])
        if eh is not None:
            ehull_arr[i] = eh
            ef_arr[i]    = ef
    return ehull_arr, ef_arr

q_ehull, q_ef = run_relaxation(q_coords, labels_gen, q_valid, 'QUANTUM')
c_ehull, c_ef = run_relaxation(c_coords, labels_gen, c_valid, 'CLASSICAL')

# ── Categorize ────────────────────────────────────────────────────────────────
def cat(ehull_arr, valid_idx, n, name):
    eh   = ehull_arr[valid_idx]
    ok   = eh[~np.isnan(eh)]
    fail = np.isnan(eh).sum()
    return dict(
        name=name, n=n,
        valid=len(valid_idx),
        relaxed=len(ok),
        stable       = int((ok<0.1).sum()),
        near_stable  = int(((ok>=0.1)&(ok<0.5)).sum()),
        metastable   = int(((ok>=0.5)&(ok<2.0)).sum()),
        high_e       = int((ok>=2.0).sum()),
        relax_fail   = int(fail),
        eh_mean = float(np.nanmean(ehull_arr[valid_idx])),
        eh_min  = float(np.nanmin(ehull_arr[valid_idx])),
    )

Q = cat(q_ehull, q_valid, N_GEN, 'Quantum')
C = cat(c_ehull, c_valid, N_GEN, 'Classical')

def pct(x,n): return f'{x}  ({x/n*100:.1f}%)'
sep = '=' * 68

lines = [sep,
  f'  RELAXED Stability Report  (CHGNet FIRE, fmax={FMAX})  N={N_GEN}',
  sep,
  f'  {"Category":<38}  {"Classical":>12}  {"Quantum":>12}',
  '-'*68,
  f'  {"Total generated":<38}  {pct(N_GEN,N_GEN):>12}  {pct(N_GEN,N_GEN):>12}',
  f'  {"Valid geometry (>=1.0A)":<38}  {pct(C["valid"],N_GEN):>12}  {pct(Q["valid"],N_GEN):>12}',
  f'  {"Successfully relaxed":<38}  {pct(C["relaxed"],N_GEN):>12}  {pct(Q["relaxed"],N_GEN):>12}',
  '',
  f'  {"-- POST-RELAXATION E_hull --":<38}',
  f'  {"  Stable       (E_hull < 0.1 eV/at)":<38}  {pct(C["stable"],N_GEN):>12}  {pct(Q["stable"],N_GEN):>12}',
  f'  {"  Near-stable  (0.1-0.5 eV/at)":<38}  {pct(C["near_stable"],N_GEN):>12}  {pct(Q["near_stable"],N_GEN):>12}',
  f'  {"  Metastable   (0.5-2.0 eV/at)":<38}  {pct(C["metastable"],N_GEN):>12}  {pct(Q["metastable"],N_GEN):>12}',
  f'  {"  High energy  (>= 2.0 eV/at)":<38}  {pct(C["high_e"],N_GEN):>12}  {pct(Q["high_e"],N_GEN):>12}',
  '',
  f'  {"E_hull mean (relaxed) [eV/atom]":<38}  {C["eh_mean"]:>12.4f}  {Q["eh_mean"]:>12.4f}',
  f'  {"E_hull min  (relaxed) [eV/atom]":<38}  {C["eh_min"]:>12.4f}  {Q["eh_min"]:>12.4f}',
  sep]

report = '\n'.join(lines)
print('\n' + report)
with open(OUT,'w') as f: f.write(report+'\n')
print(f'\nSaved: {OUT}')
