"""
Focused E_hull evaluation for checkpoint_400.pt (closest to epoch 390).
Generates 50 structures, relaxes up to 20 with CHGNet, plots bar chart.
"""
import os, sys, pickle, random, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
sys.path.insert(0, 'datasets')

import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.QINR_Crystal import PQWGAN_CC_Crystal
from view_atoms_mgmno import view_atoms

MP_API_KEY = os.environ.get('MP_API_KEY', 'hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA')
device = torch.device('cpu')
Z_DIM, LABEL_DIM, DATA_DIM, NUM = 16, 28, 90, 50

# --- Load reference dataset ---
with open('datasets/mgmno_mp.pickle', 'rb') as f:
    raw_data = pickle.load(f)
print(f'Loaded {len(raw_data)} MP reference structures')

labels = []
for idx in random.sample(range(len(raw_data)), len(raw_data)):
    c, l = raw_data[idx]
    labels.append(np.array(l).flatten())

while len(labels) < NUM:
    labels.append(labels[0])
labels_t = torch.tensor(np.array(labels[:NUM], dtype=np.float32)).to(device)

# --- Load checkpoint_400 ---
CKPT = './results_crystal_qgan/checkpoint_400.pt'
print(f'Loading {CKPT}')
cd = torch.load(CKPT, map_location=device)
gan = PQWGAN_CC_Crystal(Z_DIM + LABEL_DIM, DATA_DIM, DATA_DIM + LABEL_DIM,
                         hidden_features=6, hidden_layers=2,
                         spectrum_layer=2, use_noise=0.0)
gan.generator.load_state_dict(cd['generator'])
gan.generator.eval()

# --- Generate structures ---
with torch.no_grad():
    z = torch.randn(NUM, Z_DIM).to(device)
    fake_flat = gan.generator(torch.cat([z, labels_t], dim=1)).cpu().numpy()

gen_atoms, failed = [], 0
for img in fake_flat:
    try:
        a, _ = view_atoms(img, view=False)
        gen_atoms.append(a)
    except:
        failed += 1

print(f'Valid structures: {len(gen_atoms)}/{NUM} | Failed to parse: {failed}')
if not gen_atoms:
    print('All structures invalid — cannot compute E_hull. Exiting.')
    sys.exit(1)

# --- E_hull via CHGNet ---
try:
    from chgnet.model.dynamics import StructOptimizer
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.io.ase import AseAtomsAdaptor

    N_RELAX = min(20, len(gen_atoms))
    print(f'\nFetching Mg-Mn-O phase diagram from Materials Project...')
    with MPRester(MP_API_KEY) as mpr:
        entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'])
    pd_obj = PhaseDiagram(entries)
    relaxer = StructOptimizer()

    gen_structs = [AseAtomsAdaptor.get_structure(a) for a in gen_atoms[:N_RELAX]]
    print(f'Relaxing {N_RELAX} structures with CHGNet (this may take a few minutes)...\n')

    ehull_vals, labels_plot = [], []
    for i, s in enumerate(gen_structs):
        try:
            res = relaxer.relax(s, verbose=False)
            e_pa = res['trajectory'].energies[-1] / len(res['final_structure'])
            entry = ComputedEntry(
                res['final_structure'].composition,
                e_pa * res['final_structure'].composition.num_atoms
            )
            ehull = pd_obj.get_e_above_hull(entry) * 1000  # eV → meV/atom
            ehull_vals.append(ehull)
            labels_plot.append(f"#{i+1}\n{res['final_structure'].composition.reduced_formula}")
            status = ('✅ synth' if ehull <= 80 else '🟡 meta' if ehull <= 200 else '❌ unstable')
            print(f"  [{i+1:2d}/{N_RELAX}] {res['final_structure'].composition.reduced_formula:20s}  "
                  f"E_hull = {ehull:7.1f} meV/atom  {status}")
        except Exception as e:
            print(f"  [{i+1:2d}/{N_RELAX}] ERROR: {e}")

    if ehull_vals:
        synth = sum(1 for e in ehull_vals if e <= 80)
        meta  = sum(1 for e in ehull_vals if e <= 200)
        unstable = len(ehull_vals) - meta
        mean_e = np.mean(ehull_vals)

        print(f'\n{"="*60}')
        print(f'RESULTS (Epoch ~390 / checkpoint_400):')
        print(f'  Structures relaxed  : {len(ehull_vals)}/{N_RELAX}')
        print(f'  Mean E_hull         : {mean_e:.1f} meV/atom')
        print(f'  Synthesizable ≤80   : {synth}/{len(ehull_vals)} ({100*synth/len(ehull_vals):.0f}%)')
        print(f'  Metastable ≤200     : {meta}/{len(ehull_vals)} ({100*meta/len(ehull_vals):.0f}%)')
        print(f'  Unstable >200       : {unstable}/{len(ehull_vals)} ({100*unstable/len(ehull_vals):.0f}%)')
        print(f'{"="*60}')

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Energy Above Convex Hull — Epoch ~390 (checkpoint_400, 112k Dataset)',
                     fontsize=13, fontweight='bold')

        colors = ['mediumseagreen' if e <= 80 else 'gold' if e <= 200 else 'tomato'
                  for e in ehull_vals]
        bars = axes[0].bar(range(len(ehull_vals)), ehull_vals, color=colors, edgecolor='white', linewidth=0.8)
        axes[0].axhline(80,  color='steelblue', linestyle='--', lw=2, label='Synthesizable (80 meV/atom)')
        axes[0].axhline(200, color='red',       linestyle='--', lw=2, label='Metastable (200 meV/atom)')
        axes[0].axhline(mean_e, color='white',  linestyle=':',  lw=1.5, label=f'Mean: {mean_e:.1f} meV/atom')
        axes[0].set_xlabel('Structure Index'); axes[0].set_ylabel('E_hull (meV/atom)')
        axes[0].set_title('Per-Structure E_hull'); axes[0].legend(fontsize=9)
        axes[0].grid(axis='y', alpha=0.25)

        # Summary donut
        sizes = [synth, meta - synth, unstable]
        clrs  = ['mediumseagreen', 'gold', 'tomato']
        lbls  = [f'Synth ≤80\n{synth}', f'Meta ≤200\n{meta-synth}', f'Unstable\n{unstable}']
        wedges, texts = axes[1].pie(
            [max(s, 0.001) for s in sizes], colors=clrs, labels=lbls,
            startangle=90, wedgeprops=dict(width=0.55), textprops={'fontsize': 11}
        )
        axes[1].set_title(f'Distribution ({len(ehull_vals)} structures)')

        plt.tight_layout()
        plt.savefig('eval_ehull_400.png', dpi=150)
        print('\nSaved eval_ehull_400.png')
    else:
        print('No E_hull values computed.')

except ImportError as e:
    print(f'Missing package: {e}')
except Exception as e:
    print(f'E_hull error: {e}')

print('\nDone.')
