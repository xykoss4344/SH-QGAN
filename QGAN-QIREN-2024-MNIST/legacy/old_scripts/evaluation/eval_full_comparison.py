import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
eval_full_comparison.py
=======================
Full side-by-side evaluation of Classical CrystalGAN vs QINR-QGAN.

Matched epoch : 250  (both models have a checkpoint at this epoch)
N generated   : 4 800 crystals per model

Outputs
-------
  comparison_loss_curves.png        – loss history overlaid (3 panels)
  comparison_ssim_dist.png          – SSIM histogram, both models
  comparison_ehull_screen.png       – min-dist histogram, both models
  comparison_novelty.png            – novelty / reconstruction / invalid pie, both models
  full_comparison_table.csv         – master results table
  full_comparison_report.txt        – plain-text report
"""

import os, sys, json, pickle, glob, re, warnings, argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from skimage.metrics import structural_similarity as ssim_fn
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
QGAN_DIR      = os.path.dirname(os.path.abspath(__file__))
CLASSICAL_DIR = (r"C:\Users\Adminb\OneDrive\Documents\Projects\crystalGan"
                 r"\Composition-Conditioned-Crystal-GAN"
                 r"\Composition_Conditioned_Crystal_GAN")

EPOCH = 250
N_GEN = 4800        # crystals to generate per model
BATCH = 32
N_SSIM_REF  = 200   # real structures used as SSIM reference
N_MATCH_TEST = 300  # structures for StructureMatcher (subset of N_GEN)
MIN_DIST_THR = 1.0  # Angstroms

# Quantum paths
QGAN_CKPT    = os.path.join(QGAN_DIR, "results_crystal_qgan_v2",
                             f"checkpoint_{EPOCH}.pt")
QGAN_LOSS_CSV = os.path.join(QGAN_DIR, "results_crystal_qgan",
                              "training_loss_history.csv")
QGAN_DATASET  = os.path.join(QGAN_DIR, "datasets", "mgmno_1000.pickle")
QGAN_DATASET_DIR = os.path.join(QGAN_DIR, "datasets")

# Classical paths
CLS_CKPT     = os.path.join(CLASSICAL_DIR, "model_cwgan_mgmno_v2",
                             f"generator_{EPOCH}")
CLS_LOSS_JSON = os.path.join(CLASSICAL_DIR, "training_loss_log.json")
CLS_DATASET   = os.path.join(CLASSICAL_DIR, "mgmno_1000.pickle")
CLS_VIEW_DIR  = CLASSICAL_DIR  # view_atoms_mgmno.py lives here

OUT_DIR   = QGAN_DIR   # all outputs land next to this script
CACHE_Q   = os.path.join(OUT_DIR, f'_cache_q_ep{EPOCH}.npz')
CACHE_C   = os.path.join(OUT_DIR, f'_cache_c_ep{EPOCH}.npz')

# ── Device ────────────────────────────────────────────────────────────────────
cuda   = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(f"Device: {device}")

# ── Imports (project-local) ───────────────────────────────────────────────────
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, QGAN_DATASET_DIR)
sys.path.insert(0, QGAN_DIR)

from models.QINR_Crystal import PQWGAN_CC_Crystal  # quantum model package

# Load classical models.py directly by file path (avoids name collision)
_cls_models = load_module('cls_models',
                           os.path.join(CLASSICAL_DIR, 'models.py'))
ClassicalGenerator = _cls_models.Generator

# Use QGAN's view_atoms_mgmno (has better error handling than classical version)
from datasets.view_atoms_mgmno import view_atoms

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def image_to_ase(img):
    try:
        atoms, _ = view_atoms(img, view=False)
        return atoms
    except Exception:
        return None

def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

def compute_ssim_scores(gen_images, real_sample, label=''):
    """Max-SSIM of each generated structure against N_SSIM_REF real structures."""
    scores = []
    n = len(gen_images)
    for i, gen in enumerate(gen_images):
        gn = norm01(gen)
        best = max(ssim_fn(gn, norm01(r), data_range=1.0, win_size=3)
                   for r in real_sample)
        scores.append(best)
        if (i + 1) % 500 == 0:
            print(f"  [{label}] SSIM {i+1}/{n} ...")
    return np.array(scores)

def min_dist_screen(gen_images, label=''):
    """Minimum interatomic distance for each generated structure."""
    dists, valid_count = [], 0
    for i, img in enumerate(gen_images):
        a = image_to_ase(img)
        if a is None:
            dists.append(0.0)
            continue
        try:
            d = a.get_all_distances(mic=True)
            np.fill_diagonal(d, np.inf)
            md = float(d.min())
        except Exception:
            md = 0.0
        dists.append(md)
        if md >= MIN_DIST_THR:
            valid_count += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{label}] Dist screen {i+1}/{len(gen_images)} ...")
    return np.array(dists), valid_count

def structural_novelty(gen_images, ref_structs, label=''):
    """Pymatgen StructureMatcher novelty/reconstruction/invalid on N_MATCH_TEST samples."""
    from pymatgen.core import Structure, Lattice
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10,
                               primitive_cell=True, scale=True)
    novel = match = invalid = 0
    rms_list = []
    for i, img in enumerate(gen_images[:N_MATCH_TEST]):
        a = image_to_ase(img)
        if a is None:
            invalid += 1; continue
        try:
            cell  = a.get_cell().tolist()
            pos   = a.get_scaled_positions().tolist()
            symbs = a.get_chemical_symbols()
            gs = Structure(Lattice(cell), symbs, pos)
        except Exception:
            invalid += 1; continue
        matched = False
        for ref in ref_structs:
            try:
                if matcher.fit(gs, ref):
                    matched = True
                    rms = matcher.get_rms_dist(gs, ref)
                    if rms is not None:
                        r = rms[0] if isinstance(rms, (tuple, list)) else rms
                        rms_list.append(float(r))
                    break
            except Exception:
                continue
        if matched: match += 1
        else:       novel += 1
        if (i + 1) % 100 == 0:
            print(f"  [{label}] Matcher {i+1}/{N_MATCH_TEST} ...")
    evaluated = N_MATCH_TEST - invalid
    novelty_pct = novel / evaluated * 100 if evaluated else 0
    reconst_pct = match / evaluated * 100 if evaluated else 0
    mean_rms    = float(np.mean(rms_list)) if rms_list else float('nan')
    return novel, match, invalid, novelty_pct, reconst_pct, mean_rms


# =============================================================================
# SECTION 1 — LOAD DATASETS & BUILD REFERENCE STRUCTURES
# =============================================================================
print("\n" + "="*60)
print("  Loading datasets")
print("="*60)

with open(QGAN_DATASET, "rb") as f:
    training_data = pickle.load(f)
real_images = np.array([x[0] for x in training_data])   # (N, 30, 3)
real_labels = np.array([x[1] for x in training_data]).reshape(len(training_data), 28)
print(f"  Training samples: {len(real_images)}")

real_sample = real_images[:N_SSIM_REF]   # reference for SSIM

# Build pymatgen reference set for StructureMatcher
from pymatgen.core import Structure, Lattice
ref_structs = []
for img in real_images[:100]:
    a = image_to_ase(img)
    if a is not None:
        try:
            ref_structs.append(Structure(
                Lattice(a.get_cell().tolist()),
                a.get_chemical_symbols(),
                a.get_scaled_positions().tolist()))
        except Exception:
            pass
print(f"  Reference structures for matcher: {len(ref_structs)}")


# =============================================================================
# SECTION 2 — GENERATE: QUANTUM (QINR-QGAN) at epoch 250
# =============================================================================
print("\n" + "="*60)
print(f"  Quantum generator  (epoch {EPOCH})")
print("="*60)

gan = PQWGAN_CC_Crystal(
    input_dim_g=16 + 28, output_dim=90,
    input_dim_d=90 + 28,
    hidden_features=6, hidden_layers=2,
    spectrum_layer=2,  use_noise=0.0)
q_gen = gan.generator.to(device)
ckpt  = torch.load(QGAN_CKPT, map_location=device, weights_only=False)
q_gen.load_state_dict(ckpt['generator'])
q_gen.eval()
print(f"  Loaded: {os.path.basename(QGAN_CKPT)}  (epoch {ckpt.get('epoch','?')})")

def sample_q_labels(n):
    idx = np.random.choice(len(real_labels), n, replace=True)
    return torch.tensor(real_labels[idx], dtype=torch.float32, device=device)

if os.path.exists(CACHE_Q):
    q_images = np.load(CACHE_Q)['images']
    print(f"  Loaded from cache ({CACHE_Q})")
else:
    q_images_list = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N_GEN, BATCH):
            bs   = min(BATCH, N_GEN - start)
            z    = torch.randn(bs, 16, device=device)
            lbl  = sample_q_labels(bs)
            fake = q_gen(torch.cat([z, lbl], dim=1))
            q_images_list.append(fake.cpu().numpy().reshape(bs, 30, 3))
    q_images = np.concatenate(q_images_list, axis=0)
    np.savez_compressed(CACHE_Q, images=q_images)
    print(f"  Generated {N_GEN} structures in {time.time()-t0:.1f}s")


# =============================================================================
# SECTION 3 — GENERATE: CLASSICAL GAN at epoch 250
# =============================================================================
print("\n" + "="*60)
print(f"  Classical generator  (epoch {EPOCH})")
print("="*60)

class Opt:
    latent_dim = 512
    input_dim  = 512 + 28 + 1

c_gen = ClassicalGenerator(Opt()).to(device)
ckpt_data = torch.load(CLS_CKPT, map_location=device, weights_only=False)
state_dict = ckpt_data['model'] if isinstance(ckpt_data, dict) and 'model' in ckpt_data else ckpt_data
c_gen.load_state_dict(state_dict)
c_gen.eval()
print(f"  Loaded: {os.path.basename(CLS_CKPT)}")

def sample_c_labels(n):
    mg_i = np.random.randint(0, 8,  n)
    mn_i = np.random.randint(0, 8,  n)
    o_i  = np.random.randint(0, 12, n)
    def oh(idx, k):
        m = np.zeros((n, k), dtype=np.float32)
        m[np.arange(n), idx] = 1.0
        return torch.tensor(m, device=device)
    nat = torch.tensor((mg_i + mn_i + o_i + 3) / 28.0,
                       dtype=torch.float32, device=device).unsqueeze(-1)
    return oh(mg_i, 8), oh(mn_i, 8), oh(o_i, 12), nat

if os.path.exists(CACHE_C):
    c_images = np.load(CACHE_C)['images']
    print(f"  Loaded from cache ({CACHE_C})")
else:
    c_images_list = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N_GEN, BATCH):
            bs = min(BATCH, N_GEN - start)
            z  = torch.randn(bs, 512, device=device)
            c_mg, c_mn, c_o, nat = sample_c_labels(bs)
            fake = c_gen(z, c_mg, c_mn, c_o, nat)
            c_images_list.append(fake.cpu().numpy().reshape(bs, 30, 3))
    c_images = np.concatenate(c_images_list, axis=0)
    np.savez_compressed(CACHE_C, images=c_images)
    print(f"  Generated {N_GEN} structures in {time.time()-t0:.1f}s")


# =============================================================================
# SECTION 4 — SSIM
# =============================================================================
print("\n" + "="*60)
print("  Computing SSIM  (4 800 × 200 comparisons each model)")
print("="*60)

print("  [Quantum SSIM]")
t0 = time.time()
q_ssim = compute_ssim_scores(q_images, real_sample, label='Quantum')
print(f"  Done in {time.time()-t0:.0f}s  | mean={q_ssim.mean():.4f}  std={q_ssim.std():.4f}")

print("  [Classical SSIM]")
t0 = time.time()
c_ssim = compute_ssim_scores(c_images, real_sample, label='Classical')
print(f"  Done in {time.time()-t0:.0f}s  | mean={c_ssim.mean():.4f}  std={c_ssim.std():.4f}")

rng = np.random.default_rng(42)
baseline_ssim = float(np.mean([
    ssim_fn(rng.random((30,3)), rng.random((30,3)), data_range=1.0, win_size=3)
    for _ in range(200)]))
print(f"  Random baseline SSIM: {baseline_ssim:.4f}")


# =============================================================================
# SECTION 5 — MIN-DIST E_HULL PRE-SCREEN
# =============================================================================
print("\n" + "="*60)
print("  Min interatomic distance (E_hull pre-screen)")
print("="*60)

print("  [Quantum]")
q_dists, q_valid = min_dist_screen(q_images, label='Quantum')
print("  [Classical]")
c_dists, c_valid = min_dist_screen(c_images, label='Classical')

q_validity_pct = q_valid / N_GEN * 100
c_validity_pct = c_valid / N_GEN * 100
print(f"  Quantum  valid: {q_valid}/{N_GEN} ({q_validity_pct:.1f}%)  "
      f"mean dist={q_dists.mean():.3f} A")
print(f"  Classical valid: {c_valid}/{N_GEN} ({c_validity_pct:.1f}%)  "
      f"mean dist={c_dists.mean():.3f} A")


# =============================================================================
# SECTION 6 — STRUCTURAL NOVELTY (StructureMatcher, subset 300)
# =============================================================================
print("\n" + "="*60)
print(f"  StructureMatcher novelty (first {N_MATCH_TEST} generated structures each)")
print("="*60)

print("  [Quantum]")
q_novel, q_match, q_invalid, q_nov_pct, q_rec_pct, q_rms = \
    structural_novelty(q_images, ref_structs, label='Quantum')

print("  [Classical]")
c_novel, c_match, c_invalid, c_nov_pct, c_rec_pct, c_rms = \
    structural_novelty(c_images, ref_structs, label='Classical')


# =============================================================================
# SECTION 7 — LOSS CURVES FIGURE  (side-by-side, 3 panels each model)
# =============================================================================
print("\n" + "="*60)
print("  Plotting loss curves")
print("="*60)

# Load classical loss JSON
with open(CLS_LOSS_JSON, encoding='utf-8') as f:
    cls_records = json.load(f)
df_cls = pd.DataFrame(cls_records)   # columns: epoch, w_loss, g_loss, d_loss

# Load quantum loss CSV  (v1 full run, 500 epochs)
df_q = pd.read_csv(QGAN_LOSS_CSV)    # columns: epoch, d_loss, wasserstein, q_real_loss, q_fake_loss, total_g_loss

# Build comparison DataFrame aligned on epoch
df_q_plot = df_q.rename(columns={
    'wasserstein': 'w_loss',
    'total_g_loss': 'g_loss',
    'd_loss': 'd_loss'
})[['epoch', 'w_loss', 'g_loss', 'd_loss']]

PANEL_CFGS = [
    ('w_loss', 'Wasserstein Distance'),
    ('g_loss', 'Generator Loss'),
    ('d_loss', 'Discriminator Loss'),
]
WIN = 25   # moving-average window

fig = plt.figure(figsize=(18, 10), facecolor='#0d1117')
fig.suptitle(
    f'Training Loss Comparison — Classical GAN vs QINR-QGAN  (500 epochs)',
    fontsize=13, color='white', fontweight='bold', y=0.97)

gs_main = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                             top=0.91, bottom=0.08, left=0.06, right=0.97)

row_titles = ['Classical GAN', 'QINR-QGAN']
row_colors = ['#ff6d00', '#00e5ff']
data_rows  = [df_cls, df_q_plot]

for row, (df_r, rtitle, rcol) in enumerate(zip(data_rows, row_titles, row_colors)):
    for col, (key, panel_title) in enumerate(PANEL_CFGS):
        ax = fig.add_subplot(gs_main[row, col])
        ax.set_facecolor('#161b22')
        vals = df_r[key].values
        ax.plot(df_r['epoch'], vals, color=rcol, lw=0.8, alpha=0.50)
        smoothed = pd.Series(vals).rolling(WIN, min_periods=1).mean()
        ax.plot(df_r['epoch'], smoothed, color='white', lw=1.8, ls='--',
                label=f'MA-{WIN}')
        ax.axvline(EPOCH, color='#ffd700', lw=1.2, ls=':', alpha=0.8,
                   label=f'Epoch {EPOCH}')
        ax.set_title(f'[{rtitle}]  {panel_title}',
                     color=rcol, fontsize=8.5, fontweight='bold')
        ax.set_xlabel('Epoch', color='#aaa', fontsize=7.5)
        ax.set_ylabel('Loss',  color='#aaa', fontsize=7.5)
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.legend(fontsize=6.5, labelcolor='white', facecolor='#1a1a2e',
                  loc='upper right')

plt.savefig(os.path.join(OUT_DIR, 'comparison_loss_curves.png'),
            dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved -> comparison_loss_curves.png")


# =============================================================================
# SECTION 8 — SSIM HISTOGRAM (overlaid)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
ax.set_facecolor('#161b22')
ax.hist(q_ssim, bins=50, color='#00e5ff', alpha=0.65, edgecolor='none',
        label=f'QINR-QGAN  mean={q_ssim.mean():.4f}  std={q_ssim.std():.4f}')
ax.hist(c_ssim, bins=50, color='#ff6d00', alpha=0.65, edgecolor='none',
        label=f'Classical GAN  mean={c_ssim.mean():.4f}  std={c_ssim.std():.4f}')
ax.axvline(baseline_ssim, color='#ffd700', lw=1.8, ls='--',
           label=f'Random baseline  {baseline_ssim:.4f}')
ax.set_xlabel('Max SSIM vs Training Set', color='#aaa', fontsize=9)
ax.set_ylabel('Count', color='#aaa', fontsize=9)
ax.set_title(f'Structural Similarity Index (SSIM)  —  {N_GEN} generated crystals each',
             color='white', fontweight='bold', fontsize=10)
ax.tick_params(colors='#aaa')
for sp in ax.spines.values(): sp.set_edgecolor('#333')
ax.legend(labelcolor='white', facecolor='#1a1a2e', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'comparison_ssim_dist.png'),
            dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved -> comparison_ssim_dist.png")


# =============================================================================
# SECTION 9 — MIN-DIST HISTOGRAM (overlaid)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
ax.set_facecolor('#161b22')
ax.hist(q_dists, bins=50, color='#00e5ff', alpha=0.65, edgecolor='none',
        label=f'QINR-QGAN  valid={q_validity_pct:.1f}%')
ax.hist(c_dists, bins=50, color='#ff6d00', alpha=0.65, edgecolor='none',
        label=f'Classical GAN  valid={c_validity_pct:.1f}%')
ax.axvline(MIN_DIST_THR, color='#ffd700', lw=1.8, ls='--',
           label=f'Threshold = {MIN_DIST_THR} Å')
ax.set_xlabel('Min Interatomic Distance (Å)', color='#aaa', fontsize=9)
ax.set_ylabel('Count', color='#aaa', fontsize=9)
ax.set_title(f'Pre-DFT Structural Validity Screen  —  {N_GEN} generated crystals each',
             color='white', fontweight='bold', fontsize=10)
ax.tick_params(colors='#aaa')
for sp in ax.spines.values(): sp.set_edgecolor('#333')
ax.legend(labelcolor='white', facecolor='#1a1a2e', fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'comparison_ehull_screen.png'),
            dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved -> comparison_ehull_screen.png")


# =============================================================================
# SECTION 10 — NOVELTY PIE CHARTS (side-by-side)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='#0d1117')
for ax, (n, m, inv, title) in zip(
        [ax1, ax2],
        [(q_novel, q_match, q_invalid, 'QINR-QGAN'),
         (c_novel, c_match, c_invalid, 'Classical GAN')]):
    ax.set_facecolor('#0d1117')
    ax.pie([n, m, inv],
           labels=['Novel', 'Reconstruction', 'Invalid'],
           colors=['#00e5ff', '#ff6d00', '#555'],
           autopct='%1.1f%%', startangle=90,
           wedgeprops=dict(edgecolor='#0d1117'))
    ax.set_title(f'{title}\n(n={N_MATCH_TEST} structures)',
                 color='white', fontweight='bold', fontsize=10, pad=10)
    for t in ax.texts: t.set_color('white')
plt.suptitle('Structural Novelty vs Training Set',
             color='white', fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'comparison_novelty.png'),
            dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved -> comparison_novelty.png")


# =============================================================================
# SECTION 11 — MASTER COMPARISON TABLE
# =============================================================================
print("\n" + "="*60)
print("  Results table")
print("="*60)

# Final-epoch losses
q_w_final = float(df_q['wasserstein'].iloc[-1])
q_g_final = float(df_q['total_g_loss'].iloc[-1])
q_d_final = float(df_q['d_loss'].iloc[-1])

c_w_final = float(df_cls['w_loss'].iloc[-1])
c_g_final = float(df_cls['g_loss'].iloc[-1])
c_d_final = float(df_cls['d_loss'].iloc[-1])

# Loss at matched epoch 250
q_row = df_q[df_q['epoch'] == EPOCH]
c_row = df_cls[df_cls['epoch'] == EPOCH]
q_w_e = float(q_row['wasserstein'].iloc[0])  if len(q_row) else float('nan')
c_w_e = float(c_row['w_loss'].iloc[0])       if len(c_row) else float('nan')

rows = [
    # ── Experimental setup
    ('Matched epoch',                 str(EPOCH),            str(EPOCH)),
    ('Structures generated',          str(N_GEN),            str(N_GEN)),
    ('Latent dimension',              '16',                  '512'),
    ('Generator type',                'Hybrid QNN + MLP',    'Conv + MLP'),
    # ── SSIM
    ('SSIM mean',                     f'{q_ssim.mean():.4f}', f'{c_ssim.mean():.4f}'),
    ('SSIM std',                      f'{q_ssim.std():.4f}',  f'{c_ssim.std():.4f}'),
    ('SSIM random baseline',          f'{baseline_ssim:.4f}', f'{baseline_ssim:.4f}'),
    ('SSIM lift over baseline',       f'{q_ssim.mean()-baseline_ssim:+.4f}',
                                      f'{c_ssim.mean()-baseline_ssim:+.4f}'),
    # ── Validity (E_hull pre-screen)
    ('Valid structures (d_min >= 1A)',f'{q_valid} / {N_GEN}', f'{c_valid} / {N_GEN}'),
    ('Structural validity %',         f'{q_validity_pct:.1f}%', f'{c_validity_pct:.1f}%'),
    ('Mean min interatomic dist (A)', f'{q_dists.mean():.3f}',f'{c_dists.mean():.3f}'),
    # ── Novelty
    (f'Novel structures (of {N_MATCH_TEST})',
                                      f'{q_novel} ({q_nov_pct:.1f}%)',
                                      f'{c_novel} ({c_nov_pct:.1f}%)'),
    (f'Reconstructed (of {N_MATCH_TEST})',
                                      f'{q_match} ({q_rec_pct:.1f}%)',
                                      f'{c_match} ({c_rec_pct:.1f}%)'),
    (f'Invalid (of {N_MATCH_TEST})',  f'{q_invalid}',        f'{c_invalid}'),
    ('Mean RMS dist (matched)',       f'{q_rms:.4f}',         f'{c_rms:.4f}'),
    # ── Loss at epoch 250
    (f'Wasserstein dist @ epoch {EPOCH}', f'{q_w_e:.4f}',   f'{c_w_e:.4f}'),
    # ── Final-epoch losses
    ('Wasserstein dist (final epoch)',f'{q_w_final:.4f}',    f'{c_w_final:.4f}'),
    ('Generator loss   (final epoch)',f'{q_g_final:.4f}',    f'{c_g_final:.4f}'),
    ('Discriminator loss(final epoch)',f'{q_d_final:.4f}',   f'{c_d_final:.4f}'),
]

df_tbl = pd.DataFrame(rows, columns=['Metric', 'QINR-QGAN', 'Classical GAN'])
df_tbl = df_tbl.set_index('Metric')

print(df_tbl.to_string())

tbl_path = os.path.join(OUT_DIR, 'full_comparison_table.csv')
df_tbl.to_csv(tbl_path)
print(f"\nSaved -> full_comparison_table.csv")

# ── Plain-text report
report = [
    "QINR-QGAN vs Classical GAN — Full Evaluation Report",
    "=" * 65,
    f"Matched epoch   : {EPOCH}",
    f"Structures      : {N_GEN} generated  |  {N_SSIM_REF} real SSIM refs",
    f"Novelty test    : first {N_MATCH_TEST} generated each",
    f"Validity thresh : min interatomic dist >= {MIN_DIST_THR} A",
    "",
    df_tbl.to_string(),
    "",
    "Files produced:",
    "  comparison_loss_curves.png",
    "  comparison_ssim_dist.png",
    "  comparison_ehull_screen.png",
    "  comparison_novelty.png",
    "  full_comparison_table.csv",
]
rpt_path = os.path.join(OUT_DIR, 'full_comparison_report.txt')
with open(rpt_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print("Saved -> full_comparison_report.txt")

print("\n" + "="*60)
print("  Evaluation complete.")
print("="*60)
