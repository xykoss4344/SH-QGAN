"""
Cross-validate top-10 quantum near-stable structures with MACE-MP-0.
Compares CHGNet E_hull vs MACE-MP-0 E_hull on the same structures.
Saves results to results_analysis/dft_mace/
"""
import os, sys, pickle, warnings, json
import numpy as np
import torch
warnings.filterwarnings('ignore')

Q_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(Q_DIR, 'results_analysis', 'dft_mace')
CACHE  = os.path.join(Q_DIR, 'results_analysis', 'relaxed_structures.pkl')
os.makedirs(OUT, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}', flush=True)

# ── Load relaxation cache ─────────────────────────────────────────────────────
print('Loading CHGNet relaxation cache...', flush=True)
with open(CACHE, 'rb') as f:
    cache = pickle.load(f)

q_ehull   = cache['q_ehull']   # list len=4800, float or nan
q_structs = cache['q_structs'] # list len=4800, Structure or None

# Collect valid near-stable entries (0.0–0.5 eV/at)
near = []
for i, (eh, st) in enumerate(zip(q_ehull, q_structs)):
    if st is not None and hasattr(st, 'sites') and eh is not None:
        try:
            eh_f = float(eh)
            if 0.0 <= eh_f < 0.5 and eh_f == eh_f:  # not nan
                near.append((i, eh_f, st))
        except: pass
near.sort(key=lambda x: x[1])
top10 = near[:10]
print(f'Selected {len(top10)} structures (lowest E_hull in near-stable range)', flush=True)
for i, (idx, eh, st) in enumerate(top10):
    print(f'  {i+1:2d}. idx={idx:4d}  CHGNet E_hull={eh*1000:.1f} meV/at  '
          f'formula={st.composition.reduced_formula}  natoms={len(st)}', flush=True)

# ── Build CHGNet-consistent hull (reuse from original eval) ──────────────────
print('\nBuilding CHGNet-consistent Mg-Mn-O hull...', flush=True)
from chgnet.model import CHGNet
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from mp_api.client import MPRester

chgnet = CHGNet.load()
with MPRester('hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA') as mpr:
    all_entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'], inc_structure=True)
dft_pd     = PhaseDiagram(all_entries)
stable_dft = dft_pd.stable_entries
chgnet_refs = []
for entry in stable_dft:
    try:
        pred = chgnet.predict_structure(entry.structure)
        chgnet_refs.append(PDEntry(entry.composition, float(pred['e']) * len(entry.structure)))
    except: pass
chgnet_pd = PhaseDiagram(chgnet_refs) if len(chgnet_refs) >= 3 else dft_pd
print(f'  CHGNet hull: {len(chgnet_refs)} refs', flush=True)

# ── Build MACE-consistent hull ────────────────────────────────────────────────
print('Building MACE-MP-0 Mg-Mn-O hull...', flush=True)
from mace.calculators import mace_mp as load_mace
from ase.optimize import FIRE as ASE_FIRE
from ase.filters import ExpCellFilter
from pymatgen.io.ase import AseAtomsAdaptor

mace_calc = load_mace(model='medium', dispersion=False, default_dtype='float64', device=device)
adaptor = AseAtomsAdaptor()

def mace_energy_per_atom(pmg_structure):
    """Get MACE total energy per atom for a pymatgen Structure."""
    atoms = adaptor.get_atoms(pmg_structure)
    atoms.calc = mace_calc
    return atoms.get_potential_energy() / len(atoms)

mace_refs = []
for entry in stable_dft:
    try:
        e_pa = mace_energy_per_atom(entry.structure)
        mace_refs.append(PDEntry(entry.composition, e_pa * len(entry.structure)))
    except: pass
mace_pd = PhaseDiagram(mace_refs) if len(mace_refs) >= 3 else None
print(f'  MACE hull: {len(mace_refs)} refs', flush=True)

# ── Relax top-10 with MACE and compute E_hull ────────────────────────────────
print('\nRelaxing top-10 quantum structures with MACE-MP-0...', flush=True)
results = []

for rank, (idx, chgnet_eh, pmg_struct) in enumerate(top10):
    print(f'\n  [{rank+1}/10] idx={idx}  formula={pmg_struct.composition.reduced_formula}  '
          f'CHGNet E_hull={chgnet_eh*1000:.1f} meV/at', flush=True)
    try:
        # Convert to ASE, attach MACE calculator, relax
        atoms = adaptor.get_atoms(pmg_struct)
        atoms.calc = mace_calc

        ecf = ExpCellFilter(atoms)
        dyn = ASE_FIRE(ecf, logfile=None)
        dyn.run(fmax=0.05, steps=300)

        e_total = atoms.get_potential_energy()
        e_pa    = e_total / len(atoms)

        # Get relaxed pymatgen structure
        relaxed_pmg = adaptor.get_structure(atoms)
        mace_entry  = PDEntry(relaxed_pmg.composition, e_total)

        # E_hull from MACE-consistent PD
        if mace_pd is not None:
            try:
                mace_eh = mace_pd.get_e_above_hull(mace_entry)
            except:
                tmp_pd  = PhaseDiagram(mace_refs + [mace_entry])
                mace_eh = tmp_pd.get_e_above_hull(mace_entry)
        else:
            mace_eh = float('nan')

        print(f'    MACE E/atom={e_pa:.4f} eV  E_hull={mace_eh*1000:.1f} meV/at', flush=True)

        results.append({
            'rank':              rank + 1,
            'idx':               int(idx),
            'formula':           pmg_struct.composition.reduced_formula,
            'natoms':            len(pmg_struct),
            'chgnet_ehull_meV':  round(chgnet_eh * 1000, 2),
            'mace_ehull_meV':    round(mace_eh * 1000, 2) if not np.isnan(mace_eh) else None,
            'mace_e_per_atom':   round(e_pa, 6),
            'agreement':         'stable'      if (mace_eh is not None and mace_eh < 0.1) else
                                 'near-stable' if (mace_eh is not None and mace_eh < 0.5) else
                                 'metastable'  if (mace_eh is not None and mace_eh < 2.0) else 'high',
        })

        # Save relaxed structure as CIF
        cif_path = os.path.join(OUT, f'rank{rank+1:02d}_idx{idx}_{relaxed_pmg.composition.reduced_formula}.cif')
        relaxed_pmg.to(fmt='cif', filename=cif_path)
        print(f'    Saved CIF: {os.path.basename(cif_path)}', flush=True)

    except Exception as e:
        print(f'    ERROR: {e}', flush=True)
        results.append({'rank': rank+1, 'idx': int(idx), 'error': str(e)})

# ── Save JSON results ─────────────────────────────────────────────────────────
json_path = os.path.join(OUT, 'mace_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved JSON: {json_path}', flush=True)

# ── Print report ──────────────────────────────────────────────────────────────
sep = '=' * 70
print(f'\n{sep}')
print(f'  MACE-MP-0 Cross-Validation  —  Top-10 Quantum v4 Near-Stable')
print(f'  Second ML potential trained on Materials Project DFT (150k structures)')
print(sep)
print(f'  {"Rank":<5} {"Formula":<18} {"N":>4}  {"CHGNet":>12}  {"MACE-MP-0":>12}  {"Agreement"}')
print(f'  {"-"*5} {"-"*18} {"-"*4}  {"-"*12}  {"-"*12}  {"-"*10}')
for r in results:
    if 'error' in r:
        print(f'  {r["rank"]:<5} {"ERROR":<18}  —             —             —')
    else:
        chg = f'{r["chgnet_ehull_meV"]:.1f} meV/at'
        mac = f'{r["mace_ehull_meV"]:.1f} meV/at' if r["mace_ehull_meV"] is not None else 'N/A'
        print(f'  {r["rank"]:<5} {r["formula"]:<18} {r["natoms"]:>4}  {chg:>12}  {mac:>12}  {r.get("agreement","?")}')
print(sep)

# ── Plot CHGNet vs MACE scatter ───────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,
    'lines.linewidth': 1.5,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

valid = [r for r in results if 'error' not in r and r['mace_ehull_meV'] is not None]
if valid:
    chg_vals  = [r['chgnet_ehull_meV'] for r in valid]
    mace_vals = [r['mace_ehull_meV']   for r in valid]
    labels    = [r['formula']           for r in valid]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sc = ax.scatter(chg_vals, mace_vals, c='#f78166', s=120, zorder=5, edgecolors='black', linewidths=0.8)
    for x, y, lbl in zip(chg_vals, mace_vals, labels):
        ax.annotate(lbl, (x, y), textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color='black', alpha=0.7)

    # Parity line
    all_v = chg_vals + mace_vals
    lo, hi = min(all_v) - 10, max(all_v) + 10
    ax.plot([lo, hi], [lo, hi], '--', color='blue', linewidth=1.5, label='Parity', zorder=3)
    ax.axhline(100, color='gray', linewidth=0.8, linestyle=':')
    ax.axvline(100, color='gray', linewidth=0.8, linestyle=':')
    ax.fill_between([lo, 100], lo, 100, alpha=0.1, color='green')
    ax.text(lo + 5, 95, 'stable\n< 100 meV/at', color='green', fontsize=8, va='top')

    ax.set_xlabel('CHGNet  E_above_hull  (meV/at)')
    ax.set_ylabel('MACE-MP-0  E_above_hull  (meV/at)')
    ax.set_title('CHGNet vs MACE-MP-0 Cross-Validation\nTop-10 Quantum v4 Near-Stable Structures')
    ax.legend(loc='lower right')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    plot_path = os.path.join(OUT, 'chgnet_vs_mace_scatter.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved scatter plot: {plot_path}', flush=True)

print('\nDone.', flush=True)
