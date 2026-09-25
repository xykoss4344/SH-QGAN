"""
Full evaluation of Classical-fair model (trained on mgmno_100_aug, same as quantum v4).
Generates 4800 crystals, MIC screen, CHGNet relax, E_above_hull report.
"""
import sys, os, pickle, warnings, importlib.util
import numpy as np
import torch
warnings.filterwarnings('ignore')

Q_DIR   = os.path.dirname(os.path.abspath(__file__))
CLS_DIR = ('C:/Users/Adminb/OneDrive/Documents/Projects/crystalGan/'
           'Composition-Conditioned-Crystal-GAN/Composition_Conditioned_Crystal_GAN')

DATASET  = os.path.join(Q_DIR, 'datasets/mgmno_100_aug.pickle')
CLS_GEN  = os.path.join(CLS_DIR, 'model_cwgan_mgmno_fair/generator_490')
OUT      = os.path.join(Q_DIR, 'results_eval_fair/cls_fair_report.txt')

N_GEN       = 4800
DIST_THRESH = 1.0
FMAX        = 0.1
MAX_STEPS   = 200
MAX_RELAX   = 1000

os.makedirs(os.path.dirname(OUT), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  |  N_GEN={N_GEN}', flush=True)
print(f'Dataset: {DATASET}', flush=True)
print(f'Classical model: {CLS_GEN}', flush=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
with open(DATASET, 'rb') as f:
    raw = pickle.load(f)
labels_all = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)
idx        = np.tile(np.arange(len(labels_all)), (N_GEN // len(labels_all) + 1))[:N_GEN]
labels_gen = labels_all[idx]

# ── Classical generator ───────────────────────────────────────────────────────
spec    = importlib.util.spec_from_file_location('cls_models', os.path.join(CLS_DIR, 'models.py'))
cls_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls_mod)
class _Opt: latent_dim = 512; input_dim = 541
c_gen = cls_mod.Generator(_Opt()).to(device)
raw_ck = torch.load(CLS_GEN, map_location=device)
c_gen.load_state_dict(raw_ck['model'] if 'model' in raw_ck else raw_ck)
c_gen.eval()
print(f'Classical loaded.', flush=True)

# ── Generation ────────────────────────────────────────────────────────────────
SPECIES_MAP = ['Mg']*8 + ['Mn']*8 + ['O']*12

def gen_c(n):
    out = []
    with torch.no_grad():
        for s in range(0, n, 128):
            e    = min(s + 128, n)
            lbls = torch.from_numpy(labels_gen[s:e]).to(device)
            c1, c2, c3 = lbls[:, 0:8], lbls[:, 8:16], lbls[:, 16:28]
            c4          = lbls.sum(1, keepdim=True).float() / 28.
            out.append(c_gen(torch.randn(e-s, 512, device=device), c1, c2, c3, c4).view(e-s, 90).cpu().numpy())
    return np.concatenate(out)

print(f'Generating {N_GEN} crystals...', flush=True)
c_coords = gen_c(N_GEN)

# ── MIC validity screen ───────────────────────────────────────────────────────
from ase import Atoms

def min_dist_mic(coords_90, label_28):
    arr     = coords_90.reshape(30, 3)
    lengths = np.clip(arr[0]*30, 2.0, 30.0)
    angles  = np.clip(arr[1]*180, 30.0, 150.0)
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
c_dists = np.array([min_dist_mic(c_coords[i], labels_gen[i]) for i in range(N_GEN)])
c_valid = np.where(c_dists >= DIST_THRESH)[0]
print(f'  C valid (MIC): {len(c_valid)} ({len(c_valid)/N_GEN*100:.1f}%)', flush=True)

# ── CHGNet + CHGNet-consistent hull ──────────────────────────────────────────
from chgnet.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

chgnet  = CHGNet.load()
relaxer = StructOptimizer(model=chgnet, optimizer_class='FIRE', use_device=str(device))

print('Building CHGNet-consistent Mg-Mn-O phase diagram...', flush=True)
from mp_api.client import MPRester
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
        if st is None: return None
        res       = relaxer.relax(st, fmax=FMAX, steps=MAX_STEPS, verbose=False)
        rs        = res['final_structure']
        e_total   = float(chgnet.predict_structure(rs)['e']) * len(rs)
        gen_entry = PDEntry(rs.composition, e_total)
        try:   return mp_pd.get_e_above_hull(gen_entry)
        except:
            ents = list(chgnet_ref_entries) + [gen_entry]
            return PhaseDiagram(ents).get_e_above_hull(gen_entry)
    except: return None

ehull = np.full(N_GEN, np.nan)
cap   = min(len(c_valid), MAX_RELAX)
print(f'\nRelaxing {cap} CLASSICAL-fair structures...', flush=True)
for k, i in enumerate(c_valid[:cap]):
    if k % 50 == 0: print(f'  {k}/{cap}', flush=True)
    v = relax_ehull(c_coords[i], labels_gen[i])
    if v is not None: ehull[i] = v

# ── Report ────────────────────────────────────────────────────────────────────
eh  = ehull[c_valid[:cap]]
ok  = eh[~np.isnan(eh)]
C = dict(valid=len(c_valid), relaxed=len(ok),
         stable=int((ok<0.1).sum()), near=int(((ok>=0.1)&(ok<0.5)).sum()),
         meta=int(((ok>=0.5)&(ok<2.0)).sum()), high=int((ok>=2.0).sum()),
         mean=float(np.nanmean(ok)) if len(ok) else float('nan'),
         mn  =float(np.nanmin(ok))  if len(ok) else float('nan'))

def pct(x, n): return f'{x}  ({x/n*100:.1f}%)'
sep = '=' * 60

lines = [sep,
  f'  CLASSICAL-fair  [mgmno_100_aug  N={N_GEN}]',
  sep,
  f'  {"Valid geometry MIC (>=1.0A)":<38}  {pct(C["valid"],N_GEN):>10}',
  f'  {"Relaxed":<38}  {pct(C["relaxed"],N_GEN):>10}',
  '',
  f'  {"-- POST-RELAXATION E_above_hull --":<38}',
  f'  {"  Stable       (< 0.1 eV/at)":<38}  {pct(C["stable"],N_GEN):>10}',
  f'  {"  Near-stable  (0.1-0.5 eV/at)":<38}  {pct(C["near"],N_GEN):>10}',
  f'  {"  Metastable   (0.5-2.0 eV/at)":<38}  {pct(C["meta"],N_GEN):>10}',
  f'  {"  High energy  (>= 2.0 eV/at)":<38}  {pct(C["high"],N_GEN):>10}',
  '',
  f'  {"E_hull mean [eV/atom]":<38}  {C["mean"]:>10.4f}',
  f'  {"E_hull min  [eV/atom]":<38}  {C["mn"]:>10.4f}',
  sep]

report = '\n'.join(lines)
print('\n' + report, flush=True)
with open(OUT, 'w') as f: f.write(report + '\n')
print(f'\nSaved: {OUT}', flush=True)
