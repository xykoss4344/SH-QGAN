import os, sys, pickle, random, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0,'.')
sys.path.insert(0,'datasets')
import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.QINR_Crystal import PQWGAN_CC_Crystal
from view_atoms_mgmno import view_atoms

MP_API_KEY = os.environ.get('MP_API_KEY', 'hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA')
# Quantum circuit generator runs on CPU (PennyLane backend limitation)
device = torch.device('cpu')
Z_DIM, LABEL_DIM, DATA_DIM, NUM = 16, 28, 90, 50

with open('datasets/mgmno_mp.pickle','rb') as f:
    raw_data = pickle.load(f)
print(f'Dataset: {len(raw_data)} MP structures')

labels, real_atoms = [], []
for idx in random.sample(range(len(raw_data)), len(raw_data)):
    c, l = raw_data[idx]
    labels.append(np.array(l).flatten())
    try:
        a, _ = view_atoms(np.array(c).flatten(), view=False)
        real_atoms.append(a)
    except: pass

while len(labels) < NUM:
    labels.append(labels[0])
labels_t = torch.tensor(np.array(labels[:NUM], dtype=np.float32)).to(device)

CKPT = './results_crystal_qgan_v4/checkpoint_450.pt'
print(f'Loading {CKPT}')
cd = torch.load(CKPT, map_location=device)
gan = PQWGAN_CC_Crystal(Z_DIM+LABEL_DIM, DATA_DIM, DATA_DIM+LABEL_DIM,
                        hidden_features=8, hidden_layers=3, spectrum_layer=1, use_noise=0.0)
gan.generator.load_state_dict(cd['generator'])
gan.generator.eval()

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

print(f'Valid: {len(gen_atoms)}/{NUM} | Failed: {failed}')
if not gen_atoms:
    print('All structures invalid — exiting')
    sys.exit(1)

# SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    def dm(atoms):
        d = atoms.get_all_distances(mic=True)
        m = np.zeros((30,30)); n=min(30,d.shape[0]); m[:n,:n]=d[:n,:n]; return m
    scores = [ssim(dm(g), dm(random.choice(real_atoms)), data_range=25.0) for g in gen_atoms]
    print(f'SSIM  -- Mean:{np.mean(scores):.4f}  Min:{min(scores):.4f}  Max:{max(scores):.4f}')
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.suptitle('SSIM Analysis (Epoch 450, MP Dataset)', fontsize=18, fontweight='bold')
    axes[0].hist(scores, bins=15, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(np.mean(scores), color='red', linestyle='--', lw=2.5, label=f'Mean: {np.mean(scores):.3f}')
    axes[0].axvline(0.65, color='orange', linestyle=':', lw=2.5, label='Classical GAN (~0.65)')
    axes[0].set_title('SSIM Score Distribution', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Structural Similarity Index (SSIM)', fontsize=15, fontweight='bold'); axes[0].legend(fontsize=14)
    axes[1].imshow(np.hstack([dm(gen_atoms[0]), np.ones((30,2))*25, dm(real_atoms[0])]), cmap='viridis', vmin=0, vmax=25)
    axes[1].set_title('Distance Matrix: Generated | Real', fontsize=16, fontweight='bold'); axes[1].axis('off')
    plt.tight_layout(); plt.savefig('logs/eval_ssim.png', dpi=150)
    print('Saved logs/eval_ssim.png')
    sys.exit(0)
except Exception as e:
    print(f'SSIM error: {e}')

# StructureMatcher
try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    refs = [AseAtomsAdaptor.get_structure(a) for a in real_atoms]
    gens = [AseAtomsAdaptor.get_structure(a) for a in gen_atoms]
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    matches = 0
    for g in gens:
        for r in refs:
            try:
                if matcher.fit(g, r): matches += 1; break
            except: pass
    novelty = 100*(len(gens)-matches)/len(gens)
    print(f'Novelty: {novelty:.1f}%  ({len(gens)-matches}/{len(gens)} novel, {matches} matched)')
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle('Novelty Analysis (Epoch 450, MP Dataset)', fontsize=13, fontweight='bold')
    bars = axes[0].bar(['Novel','Matched'], [len(gens)-matches, matches], color=['mediumseagreen','tomato'], width=0.5)
    for bar, v in zip(bars, [len(gens)-matches, matches]):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.2, str(v), ha='center', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Structures'); axes[0].set_title('Structural Novelty')
    axes[1].text(0.5, 0.5, f'{novelty:.1f}% Novel\n{len(gens)-matches} of {len(gens)} structures\nare genuinely new',
                 ha='center', va='center', transform=axes[1].transAxes, fontsize=16, color='mediumseagreen', fontweight='bold')
    axes[1].axis('off')
    plt.tight_layout(); plt.savefig('eval_novelty.png', dpi=120)
    print('Saved eval_novelty.png')
except Exception as e:
    print(f'StructureMatcher error: {e}')

# E_hull
try:
    from chgnet.model.dynamics import StructOptimizer
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.io.ase import AseAtomsAdaptor
    print('Fetching Mg-Mn-O phase diagram...')
    with MPRester(MP_API_KEY) as mpr:
        entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'])
    pd_obj = PhaseDiagram(entries)
    relaxer = StructOptimizer()
    ehull_vals, N = [], min(10, len(gen_atoms))
    gen_structs = [AseAtomsAdaptor.get_structure(a) for a in gen_atoms[:N]]
    print(f'Relaxing {N} structures with CHGNet...')
    for i, s in enumerate(gen_structs):
        try:
            res = relaxer.relax(s, verbose=False)
            e_pa = res['trajectory'].energies[-1] / len(res['final_structure'])
            entry = ComputedEntry(res['final_structure'].composition, e_pa * res['final_structure'].composition.num_atoms)
            ehull = pd_obj.get_e_above_hull(entry) * 1000
            ehull_vals.append(ehull)
            print(f'  Structure {i+1}: {ehull:.1f} meV/atom')
        except Exception as e:
            print(f'  Structure {i+1}: Error - {e}')
    if ehull_vals:
        synth = sum(1 for e in ehull_vals if e<=80)
        meta  = sum(1 for e in ehull_vals if e<=200)
        print(f'E_hull -- Mean:{np.mean(ehull_vals):.1f} meV/atom | Synth(<=80):{synth}/{N} | Meta(<=200):{meta}/{N}')
        fig, ax = plt.subplots(figsize=(11,5))
        bc = ['mediumseagreen' if e<=80 else 'gold' if e<=200 else 'tomato' for e in ehull_vals]
        ax.bar(range(len(ehull_vals)), ehull_vals, color=bc, edgecolor='white')
        ax.axhline(80, color='blue', linestyle='--', lw=1.5, label='Synthesizable (80 meV/atom)')
        ax.axhline(200, color='red', linestyle='--', lw=1.5, label='Metastable (200 meV/atom)')
        ax.set_title('Energy Above Convex Hull (Epoch 450, MP Dataset)', fontweight='bold')
        ax.set_xlabel('Crystal Index'); ax.set_ylabel('E_hull (meV/atom)'); ax.legend()
        plt.tight_layout(); plt.savefig('eval_ehull.png', dpi=120)
        print('Saved eval_ehull.png')
except Exception as e:
    print(f'E_hull error: {e}')

# Loss curve
import csv
rows = []
with open('results_crystal_qgan/training_loss_history.csv') as f:
    for row in csv.DictReader(f):
        rows.append({k: float(v) for k, v in row.items()})

eps = [r['epoch'] for r in rows]
fig, axes = plt.subplots(2, 3, figsize=(18,10))
fig.suptitle('QINR-QGAN Training Loss (WGAN-GP + InfoGAN, MP Dataset, 500 Epochs)', fontsize=14, fontweight='bold')
axes = axes.flatten()
plots = [
    ('d_loss',       'Critic Loss (L_D)',          'tomato',       'L_D = Wasserstein + GP - Q_real\nNegative = critic in learning regime'),
    ('wasserstein',  'Wasserstein Distance',        'steelblue',    'W = E[D(real)] - E[D(fake)]\nPositive + rising = critic improving'),
    ('q_real_loss',  'Q-Head Loss on Real (Q_real)','darkorange',   'Q-Head learning composition from real\nShould stabilise at a low value'),
    ('q_fake_loss',  'Q-Head Loss on Fake (Q_fake)','hotpink',      'Generator composition consistency\nShould decrease as G learns'),
    ('total_g_loss', 'Generator Loss (L_G)',        'mediumpurple', 'L_G = -E[D(G(z))] + Q_fake\nShould trend negative'),
]
for ax, (col, title, color, interp) in zip(axes, plots):
    vals = [r[col] for r in rows]
    ax.plot(eps, vals, color=color, linewidth=2)
    ax.set_title(title, fontweight='bold'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.97, interp, transform=ax.transAxes, fontsize=8, va='top', color='gray', style='italic')
axes[-1].axis('off')
plt.tight_layout(); plt.savefig('eval_loss_curve.png', dpi=120)
print('Saved eval_loss_curve.png')
print('\nDONE — all evaluation plots saved.')
