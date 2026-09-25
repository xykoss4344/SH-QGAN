"""
Full QINR-QGAN Evaluation Script
Compares QINR-QGAN (checkpoint_400) vs Classical GAN baseline (Epoch 390)

Metrics:
  1. SSIM (max per generated structure vs training set) + random baseline
  2. Structural Novelty (StructureMatcher, ltol=0.3, stol=0.5, angle_tol=10)
  3. Geometric Validity (min interatomic distance >= 1.0 A)
  4. Loss Curves (Wasserstein, G_loss, D_loss vs epoch)

Generates 200 structures with composition labels sampled as Classical GAN:
  Mg: uniform int [0,8], Mn: uniform int [0,8], O: uniform int [0,12]
"""

import os, sys, pickle, random, csv, warnings, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
sys.path.insert(0, 'datasets')

import torch
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from models.QINR_Crystal import PQWGAN_CC_Crystal
from view_atoms_mgmno import view_atoms

# ============================================================
# Config
# ============================================================
CKPT         = './results_crystal_qgan/checkpoint_400.pt'
TRAIN_PICKLE = 'datasets/mgmno_1000.pickle'
NUM_GEN      = 200          # structures to generate
Z_DIM        = 16
LABEL_DIM    = 28
DATA_DIM     = 90
device       = torch.device('cpu')

# Classical GAN baseline (Epoch 390) — provided by user
CLASSICAL = {
    'ssim_mean':       0.4010,
    'ssim_std':        0.1108,
    'ssim_random':    -0.0066,
    'novelty_pct':   100.0,
    'validity_pct':   29.5,
    'mean_min_dist':   0.831,
}

# ============================================================
# 1. Build composition-conditioned labels same way as Classical GAN
#    Mg: 0-8, Mn: 0-8, O: 0-12  (uniform int, one-hot occupancy)
# ============================================================
def make_label(n_mg, n_mn, n_o):
    """28-dim binary occupancy vector: 8 Mg + 8 Mn + 12 O slots"""
    lbl = np.zeros(28, dtype=np.float32)
    lbl[:n_mg]          = 1.0   # Mg slots
    lbl[8:8+n_mn]       = 1.0   # Mn slots
    lbl[16:16+n_o]      = 1.0   # O  slots
    return lbl

labels_np = []
for _ in range(NUM_GEN):
    n_mg = random.randint(0, 8)
    n_mn = random.randint(0, 8)
    n_o  = random.randint(0, 12)
    # ensure at least one atom per species (avoid empty structures)
    if n_mg == 0 and n_mn == 0: n_mg = random.randint(1, 4)
    if n_o  == 0:                n_o  = random.randint(1, 4)
    labels_np.append(make_label(n_mg, n_mn, n_o))

labels_t = torch.tensor(np.array(labels_np, dtype=np.float32)).to(device)

# ============================================================
# 2. Load model & generate
# ============================================================
print(f'[1/5] Loading {CKPT} ...')
cd  = torch.load(CKPT, map_location=device)
gan = PQWGAN_CC_Crystal(
    Z_DIM + LABEL_DIM, DATA_DIM, DATA_DIM + LABEL_DIM,
    hidden_features=6, hidden_layers=2, spectrum_layer=2, use_noise=0.0
)
gan.generator.load_state_dict(cd['generator'])
gan.generator.eval()

print(f'[2/5] Generating {NUM_GEN} structures ...')
t0 = time.time()
with torch.no_grad():
    z        = torch.randn(NUM_GEN, Z_DIM).to(device)
    fake_flat = gan.generator(torch.cat([z, labels_t], dim=1)).cpu().numpy()
print(f'      Generation took {time.time()-t0:.1f}s')

# Parse generated atoms
gen_atoms, gen_imgs, failed_parse = [], [], 0
for img in fake_flat:
    try:
        # img is shape (90,) = 30*3 flattened
        img_reshaped = img.reshape(30, 3)
        a, _ = view_atoms(img_reshaped, view=False)
        gen_atoms.append(a)
        gen_imgs.append(img_reshaped)
    except:
        failed_parse += 1

print(f'      Parseable structures: {len(gen_atoms)}/{NUM_GEN} (failed: {failed_parse})')

# ============================================================
# 3. Load training set for SSIM & novelty comparisons
# ============================================================
print(f'[3/5] Loading training set {TRAIN_PICKLE} ...')
with open(TRAIN_PICKLE, 'rb') as f:
    train_data = pickle.load(f)

# Sample up to 500 training structures for comparison (speed)
sample_size = min(500, len(train_data))
train_sample = random.sample(train_data, sample_size)
train_atoms, train_imgs = [], []
for img_arr, _ in train_sample:
    try:
        img_arr = np.array(img_arr)
        if img_arr.shape == (30, 3):
            a, _ = view_atoms(img_arr, view=False)
            train_atoms.append(a)
            train_imgs.append(img_arr)
    except:
        pass
print(f'      Training sample: {len(train_atoms)} atoms objects')

# ============================================================
# Metric A: SSIM
# ============================================================
print('[4/5] Computing metrics ...')
ssim_scores, random_scores = [], []

try:
    from skimage.metrics import structural_similarity as ssim

    def dist_matrix(atoms, n=30):
        d = atoms.get_all_distances(mic=True)
        m = np.zeros((n, n))
        k = min(n, d.shape[0])
        m[:k, :k] = d[:k, :k]
        return m

    print('      SSIM: computing distance matrices for training set ...')
    train_dms = [dist_matrix(a) for a in train_atoms]

    print(f'      SSIM: scoring {len(gen_atoms)} generated structures (max over training sample) ...')
    for ga in gen_atoms:
        try:
            gdm = dist_matrix(ga)
            scores_vs_train = [ssim(gdm, tdm, data_range=25.0) for tdm in train_dms]
            ssim_scores.append(max(scores_vs_train))   # max similarity to ANY training structure
        except Exception as e:
            pass

    # Random baseline: random pairs of training structures
    print('      SSIM: computing random baseline ...')
    for _ in range(min(200, len(train_dms))):
        a, b = random.sample(train_dms, 2)
        try:
            random_scores.append(ssim(a, b, data_range=25.0))
        except:
            pass

    ssim_mean = np.mean(ssim_scores)  if ssim_scores  else float('nan')
    ssim_std  = np.std(ssim_scores)   if ssim_scores  else float('nan')
    ssim_rand = np.mean(random_scores) if random_scores else float('nan')
    print(f'      SSIM (QINR-QGAN): {ssim_mean:.4f} ± {ssim_std:.4f}  |  Random baseline: {ssim_rand:.4f}')

except Exception as e:
    print(f'      SSIM error: {e}')
    ssim_mean = ssim_std = ssim_rand = float('nan')

# ============================================================
# Metric B: Structural Novelty
# ============================================================
novelty_pct = float('nan')
n_novel = n_matched = 0
try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    refs = []
    for a in train_atoms:
        try:
            refs.append(AseAtomsAdaptor.get_structure(a))
        except:
            pass

    print(f'      Novelty: matching {len(gen_atoms)} generated vs {len(refs)} training (may take a while) ...')
    gens_pm = []
    for a in gen_atoms:
        try:
            gens_pm.append(AseAtomsAdaptor.get_structure(a))
        except:
            gens_pm.append(None)

    for g in gens_pm:
        if g is None:
            n_novel += 1
            continue
        matched = False
        for r in refs:
            try:
                if matcher.fit(g, r):
                    matched = True
                    break
            except:
                pass
        if matched:
            n_matched += 1
        else:
            n_novel += 1

    novelty_pct = 100.0 * n_novel / len(gens_pm) if gens_pm else float('nan')
    print(f'      Novelty: {novelty_pct:.1f}%  ({n_novel} novel, {n_matched} matched)')

except Exception as e:
    print(f'      Novelty error: {e}')

# ============================================================
# Metric C: Geometric Validity (min interatomic distance >= 1.0 A)
# ============================================================
valid_geom, min_dists = 0, []
for a in gen_atoms:
    try:
        d = a.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        min_d = d.min()
        min_dists.append(min_d)
        if min_d >= 1.0:
            valid_geom += 1
    except:
        pass

validity_pct   = 100.0 * valid_geom / len(gen_atoms) if gen_atoms else float('nan')
mean_min_dist  = np.mean(min_dists) if min_dists else float('nan')
print(f'      Geometric Validity: {validity_pct:.1f}%  |  Mean min dist: {mean_min_dist:.3f} A')

# ============================================================
# 5. Print comparison table
# ============================================================
print('\n' + '='*70)
print(f'  QINR-QGAN vs Classical GAN — Evaluation Results')
print(f'  QINR-QGAN: checkpoint_400.pt (~Epoch 390) | 200 generated structures')
print('='*70)
print(f"{'Metric':<30} {'QINR-QGAN':>15} {'Classical GAN':>15}")
print('-'*70)
print(f"{'SSIM Mean':<30} {ssim_mean:>15.4f} {CLASSICAL['ssim_mean']:>15.4f}")
print(f"{'SSIM Std':<30} {ssim_std:>15.4f} {CLASSICAL['ssim_std']:>15.4f}")
print(f"{'SSIM Random Baseline':<30} {ssim_rand:>15.4f} {CLASSICAL['ssim_random']:>15.4f}")
print(f"{'Structural Novelty %':<30} {novelty_pct:>14.1f}% {CLASSICAL['novelty_pct']:>14.1f}%")
print(f"{'Geometric Validity %':<30} {validity_pct:>14.1f}% {CLASSICAL['validity_pct']:>14.1f}%")
print(f"{'Mean Min Interatomic Dist (A)':<30} {mean_min_dist:>15.3f} {CLASSICAL['mean_min_dist']:>15.3f}")
print('='*70)

# ============================================================
# 6. Plots
# ============================================================

# --- Plot 1: SSIM Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('SSIM Analysis — QINR-QGAN vs Classical GAN (Epoch ~390)', fontsize=13, fontweight='bold')

if ssim_scores:
    axes[0].hist(ssim_scores, bins=20, color='mediumpurple', edgecolor='white', alpha=0.85, label='QINR-QGAN')
    axes[0].axvline(ssim_mean, color='white', linestyle='--', lw=2, label=f'QINR Mean: {ssim_mean:.4f}')
    axes[0].axvline(CLASSICAL['ssim_mean'], color='tomato', linestyle='--', lw=2,
                    label=f"Classical GAN: {CLASSICAL['ssim_mean']:.4f}")
    axes[0].axvline(ssim_rand, color='gray', linestyle=':', lw=1.5,
                    label=f'Random baseline: {ssim_rand:.4f}')
axes[0].set_title('SSIM Score Distribution'); axes[0].set_xlabel('Max SSIM vs Training'); axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.2)

# Bar comparison
cats = ['QINR-QGAN\n(~Ep 390)', 'Classical GAN\n(Ep 390)', 'Random\nBaseline']
vals = [ssim_mean, CLASSICAL['ssim_mean'], ssim_rand]
errs = [ssim_std,  CLASSICAL['ssim_std'],  0.0]
cols = ['mediumpurple', 'tomato', 'gray']
bars = axes[1].bar(cats, vals, color=cols, edgecolor='white', width=0.5, yerr=errs, capsize=5)
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f'{v:.4f}', ha='center', fontsize=11, fontweight='bold', color='white')
axes[1].set_ylabel('Mean Max SSIM'); axes[1].set_title('SSIM Comparison'); axes[1].grid(axis='y', alpha=0.2)
axes[1].set_ylim(min(0, min(vals)) - 0.05, max(vals) + 0.1)
plt.tight_layout(); plt.savefig('ssim_distribution.png', dpi=150); plt.close()
print('Saved ssim_distribution.png')

# --- Plot 2: Structural Dissimilarity (Novelty) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Structural Novelty — QINR-QGAN vs Classical GAN', fontsize=13, fontweight='bold')

q_novel   = n_novel;  q_matched = n_matched
c_novel   = round(CLASSICAL['novelty_pct'] * len(gen_atoms) / 100)
c_matched = len(gen_atoms) - c_novel

for ax, (title, nov, mat, col) in zip(axes, [
    ('QINR-QGAN (~Ep 390)', q_novel, q_matched, 'mediumpurple'),
    ('Classical GAN (Ep 390)', c_novel, c_matched, 'tomato'),
]):
    sizes = [nov, mat]
    labels_pie = [f'Novel\n{nov}', f'Matched\n{mat}']
    wedges, texts = ax.pie([max(s,0.001) for s in sizes],
                            colors=[col, 'gray'], labels=labels_pie, startangle=90,
                            wedgeprops=dict(width=0.55), textprops={'fontsize': 11})
    pct = 100*nov/(nov+mat) if (nov+mat)>0 else 0
    ax.set_title(f'{title}\n{pct:.1f}% Novel', fontweight='bold')

plt.tight_layout(); plt.savefig('structural_dissimilarity.png', dpi=150); plt.close()
print('Saved structural_dissimilarity.png')

# --- Plot 3: Geometric Validity (E_hull pre-screen) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Geometric Validity Pre-Screen (Min Interatomic Distance ≥ 1.0 Å)', fontsize=13, fontweight='bold')

# Histogram of min distances
if min_dists:
    axes[0].hist(min_dists, bins=25, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(1.0, color='red', linestyle='--', lw=2, label='Validity threshold (1.0 Å)')
    axes[0].axvline(mean_min_dist, color='white', linestyle=':', lw=1.5,
                    label=f'Mean: {mean_min_dist:.3f} Å')
    axes[0].axvline(CLASSICAL['mean_min_dist'], color='tomato', linestyle='--', lw=2,
                    label=f"Classical GAN mean: {CLASSICAL['mean_min_dist']:.3f} Å")
axes[0].set_xlabel('Min Interatomic Distance (Å)'); axes[0].set_title('QINR-QGAN Min Dist Distribution')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2)

# Bar: validity %
cats_v = ['QINR-QGAN\n(~Ep 390)', 'Classical GAN\n(Ep 390)']
vals_v = [validity_pct, CLASSICAL['validity_pct']]
cols_v = ['steelblue', 'tomato']
bars_v = axes[1].bar(cats_v, vals_v, color=cols_v, edgecolor='white', width=0.5)
for bar, v in zip(bars_v, vals_v):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+1, f'{v:.1f}%',
                 ha='center', fontsize=13, fontweight='bold', color='white')
axes[1].set_ylabel('Geometric Validity (%)'); axes[1].set_title('Validity Rate Comparison')
axes[1].set_ylim(0, 110); axes[1].grid(axis='y', alpha=0.2)

# Bar: mean min dist
cats_d = ['QINR-QGAN\n(~Ep 390)', 'Classical GAN\n(Ep 390)']
vals_d = [mean_min_dist, CLASSICAL['mean_min_dist']]
bars_d = axes[2].bar(cats_d, vals_d, color=['steelblue', 'tomato'], edgecolor='white', width=0.5)
axes[2].axhline(1.0, color='white', linestyle='--', lw=2, label='Valid threshold (1.0 Å)')
for bar, v in zip(bars_d, vals_d):
    axes[2].text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f} Å',
                 ha='center', fontsize=13, fontweight='bold', color='white')
axes[2].set_ylabel('Mean Min Distance (Å)'); axes[2].set_title('Mean Min Interatomic Distance')
axes[2].set_ylim(0, max(vals_d)+0.3); axes[2].legend(fontsize=9); axes[2].grid(axis='y', alpha=0.2)

plt.tight_layout(); plt.savefig('ehull_prescreen.png', dpi=150); plt.close()
print('Saved ehull_prescreen.png')

# --- Plot 4: Loss Curves ---
rows_csv = []
with open('results_crystal_qgan/training_loss_history.csv') as f:
    for row in csv.DictReader(f):
        rows_csv.append({k: float(v) for k, v in row.items()})

eps  = [r['epoch']      for r in rows_csv]
wass = [r['wasserstein']  for r in rows_csv]
dloss= [r['d_loss']       for r in rows_csv]
gloss= [r['total_g_loss'] for r in rows_csv]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('QINR-QGAN Training Loss Curves (112k Augmented Dataset, 500 Epochs)',
             fontsize=13, fontweight='bold')

axes[0].plot(eps, wass,  color='steelblue',   lw=2)
axes[0].axvline(400, color='white', linestyle='--', lw=1.5, label='checkpoint_400 (eval)')
axes[0].set_title('Wasserstein Distance', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('W Distance'); axes[0].grid(alpha=0.2); axes[0].legend()

axes[1].plot(eps, dloss, color='tomato',       lw=2)
axes[1].axvline(400, color='white', linestyle='--', lw=1.5, label='checkpoint_400')
axes[1].set_title('Critic (Discriminator) Loss', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].grid(alpha=0.2); axes[1].legend()

axes[2].plot(eps, gloss, color='mediumpurple', lw=2)
axes[2].axvline(400, color='white', linestyle='--', lw=1.5, label='checkpoint_400')
axes[2].set_title('Generator Loss', fontweight='bold')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss'); axes[2].grid(alpha=0.2); axes[2].legend()

plt.tight_layout(); plt.savefig('loss_curves.png', dpi=150); plt.close()
print('Saved loss_curves.png')

print('\n[5/5] All metrics computed and plots saved.')
print('Outputs: ssim_distribution.png, structural_dissimilarity.png, ehull_prescreen.png, loss_curves.png')
