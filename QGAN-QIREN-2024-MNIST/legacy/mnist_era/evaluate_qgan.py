"""
evaluate_qgan.py
================
Standalone evaluation script for the QINR-QGAN crystal generator.
Runs SSIM, StructureMatcher, E_hull, and Training Loss Curve analysis.
Graphs open in interactive windows with printed interpretation summaries.

Usage:
    py -3.12 evaluate_qgan.py
    py -3.12 evaluate_qgan.py --num_samples 30 --mp_key YOUR_KEY
"""

import os, sys
# Fix Windows console Unicode encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pickle, glob, random, argparse, warnings
warnings.filterwarnings('ignore')

# ?? Working directory ??????????????????????????????????????????????
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'datasets'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.QINR_Crystal import PQWGAN_CC_Crystal
from view_atoms_mgmno import view_atoms

# ?? Args ???????????????????????????????????????????????????????????
parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=50)
parser.add_argument('--mp_key', type=str,
                    default=os.environ.get('MP_API_KEY', 'hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA'))
parser.add_argument('--skip_ehull', action='store_true',
                    help='Skip E_hull (slow ? requires CHGNet + MP API)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"  QINR-QGAN Evaluation Suite")
print(f"{'='*60}")
print(f"  Device    : {device}")
print(f"  Samples   : {args.num_samples}")
print(f"{'='*60}\n")

# ?? Optional deps ??????????????????????????????????????????????????
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_OK = True
    print("[OK] scikit-image ? SSIM enabled")
except ImportError:
    SKIMAGE_OK = False
    print("[!!] scikit-image not found (pip install scikit-image)")

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    PYMATGEN_OK = True
    print("[OK] pymatgen ? StructureMatcher enabled")
except ImportError:
    PYMATGEN_OK = False
    print("[!!] pymatgen not found (pip install pymatgen)")

try:
    from chgnet.model.dynamics import StructOptimizer
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    EHULL_OK = True
    print("[OK] chgnet + mp-api ? E_hull enabled")
except ImportError:
    EHULL_OK = False
    print("[~]  chgnet/mp-api not found ? E_hull skipped (pip install chgnet mp-api)")

# ??????????????????????????????????????????????????????????????????
# STEP 1: Load Model & Generate Crystals
# ??????????????????????????????????????????????????????????????????
print("\n[1/4] Generating crystals...")

Z_DIM, LABEL_DIM, DATA_DIM = 16, 28, 90

with open('datasets/mgmno_100.pickle', 'rb') as f:
    raw_data = pickle.load(f)
print(f"      Dataset loaded: {len(raw_data)} crystals")

labels, real_atoms_list = [], []
for idx in random.sample(range(len(raw_data)), min(150, len(raw_data))):
    c, l = raw_data[idx]
    if len(labels) < args.num_samples:
        labels.append(l.flatten())
    try:
        atoms, _ = view_atoms(c.flatten(), view=False)
        real_atoms_list.append(atoms)
    except Exception:
        pass

while len(labels) < args.num_samples:
    labels.append(labels[0])
labels_t = torch.tensor(np.array(labels, dtype=np.float32)).to(device)

# Auto-scan checkpoints newest-to-oldest to find one with valid structures
ckpts = sorted(glob.glob('./results_crystal_qgan/checkpoint_*.pt'), key=os.path.getmtime)
assert ckpts, "No checkpoints found! Train the model first."
print("Scanning checkpoints for valid structures...")
best_ckpt = None
best_epoch = None
for ckpt in reversed(ckpts):
    epoch_n = os.path.basename(ckpt).replace('checkpoint_','').replace('.pt','')
    try:
        cd = torch.load(ckpt, map_location=device)
        gan_test = PQWGAN_CC_Crystal(Z_DIM+LABEL_DIM, DATA_DIM, DATA_DIM+LABEL_DIM,
                            hidden_features=6, hidden_layers=2, spectrum_layer=2, use_noise=0.0)
        gan_test.generator.load_state_dict(cd['generator'])
        gan_test.generator.eval()
        with torch.no_grad():
            z_t = torch.randn(5, Z_DIM).to(device)
            l_t = torch.zeros(5, LABEL_DIM).to(device)
            ft = gan_test.generator(torch.cat([z_t,l_t],dim=1)).cpu().numpy()
        v = 0
        for img in ft:
            try: view_atoms(img, view=False); v+=1
            except: pass
        print(f"  Epoch {epoch_n}: {v}/5 valid")
        if v > 0:
            best_ckpt = ckpt
            best_epoch = epoch_n
            break
    except Exception as e:
        print(f"  Epoch {epoch_n}: load error ? {e}")

assert best_ckpt, "No checkpoint produced valid structures. Train more epochs."
CKPT = best_ckpt
epoch_num = best_epoch
print(f"Using checkpoint: epoch {epoch_num}")

gan = PQWGAN_CC_Crystal(Z_DIM+LABEL_DIM, DATA_DIM, DATA_DIM+LABEL_DIM,
                        hidden_features=6, hidden_layers=2,
                        spectrum_layer=2, use_noise=0.0)
generator = gan.generator.to(device)
ckpt_data = torch.load(CKPT, map_location=device)
generator.load_state_dict(ckpt_data['generator'])
generator.eval()

with torch.no_grad():
    z = torch.randn(args.num_samples, Z_DIM).to(device)
    fake_flat = generator(torch.cat([z, labels_t], dim=1)).cpu().numpy()

generated_atoms, failed = [], 0
for img in fake_flat:
    try:
        atoms, _ = view_atoms(img, view=False)
        generated_atoms.append(atoms)
    except Exception:
        failed += 1

print(f"      Valid structures: {len(generated_atoms)} | Skipped (invalid cell): {failed}")
if not generated_atoms:
    print("ERROR: All structures invalid. Training hasn't converged yet.")
    print("       Try re-running after more epochs complete.")
    sys.exit(1)

# Physical validity rate
validity_pct = 100 * len(generated_atoms) / args.num_samples
print(f"\n  [*] INTERPRETATION ? Physical Validity: {validity_pct:.1f}%")
print(f"     {len(generated_atoms)} of {args.num_samples} generated structures have valid unit cell geometry.")
if validity_pct < 50:
    print("     [~]  Low validity ? generator still learning cell parameters (normal at early epochs).")
elif validity_pct >= 80:
    print("     [OK] High validity ? generator has learned valid crystal geometry.")

# ??????????????????????????????????????????????????????????????????
# STEP 2: SSIM
# ??????????????????????????????????????????????????????????????????
print("\n[2/4] Computing SSIM scores...")

def dist_matrix(atoms):
    d = atoms.get_all_distances(mic=True)
    mat = np.zeros((30, 30))
    n = min(30, d.shape[0])
    mat[:n, :n] = d[:n, :n]
    return mat

if SKIMAGE_OK and real_atoms_list:
    ssim_scores = []
    for g_atom in generated_atoms:
        g_mat = dist_matrix(g_atom)
        r_mat = dist_matrix(random.choice(real_atoms_list))
        score = ssim(g_mat, r_mat, data_range=25.0)
        ssim_scores.append(score)

    avg_ssim = np.mean(ssim_scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'SSIM Analysis (Epoch {epoch_num})', fontsize=14, fontweight='bold')

    axes[0].hist(ssim_scores, bins=15, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(avg_ssim, color='red', linestyle='--', lw=2, label=f'Mean SSIM: {avg_ssim:.3f}')
    axes[0].axvline(0.65, color='orange', linestyle=':', lw=1.5, label='Classical GAN baseline (~0.65)')
    axes[0].set_title('Structural Similarity Score Distribution')
    axes[0].set_xlabel('SSIM Score (0=no match, 1=perfect match)')
    axes[0].set_ylabel('Number of Generated Structures')
    axes[0].legend()
    axes[0].text(0.02, 0.97,
        'Higher = generated crystals are more\nstructurally similar to real training data.',
        transform=axes[0].transAxes, fontsize=9, va='top', color='gray')

    g_ex = dist_matrix(generated_atoms[0])
    r_ex = dist_matrix(real_atoms_list[0])
    axes[1].imshow(np.hstack([r_ex, np.ones((30,2))*25, g_ex]), cmap='viridis', vmin=0, vmax=25)
    axes[1].set_title('Interatomic Distance Matrix\nReal (left)  |  Generated (right)')
    axes[1].axis('off')
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis',
                        norm=plt.Normalize(0, 25)), ax=axes[1], shrink=0.8)
    cb.set_label('Distance (?)')

    plt.tight_layout()
    plt.savefig('eval_ssim.png', dpi=120)
    plt.show()

    print(f"  [*] INTERPRETATION ? SSIM:")
    print(f"     Mean SSIM : {avg_ssim:.4f}")
    print(f"     Min / Max : {min(ssim_scores):.4f} / {max(ssim_scores):.4f}")
    if avg_ssim >= 0.65:
        print("     [OK] Above classical GAN baseline ? QGAN matches training data structure well.")
    elif avg_ssim >= 0.45:
        print("     [~]  Below classical GAN baseline ? generator still converging.")
    else:
        print("     ? Low SSIM ? generator needs more training epochs.")
    print("     (Saved: eval_ssim.png)")
else:
    print("     Skipped ? scikit-image not installed.")

# ??????????????????????????????????????????????????????????????????
# STEP 3: StructureMatcher (Novelty)
# ??????????????????????????????????????????????????????????????????
print("\n[3/4] Running StructureMatcher novelty check...")

if PYMATGEN_OK and real_atoms_list:
    ref_structs = [AseAtomsAdaptor.get_structure(a) for a in real_atoms_list]
    gen_structs = [AseAtomsAdaptor.get_structure(a) for a in generated_atoms]

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    match_count, rms_displacements = 0, []

    for g_struct in gen_structs:
        best_rms, is_match = float('inf'), False
        for ref in ref_structs:
            try:
                if matcher.fit(g_struct, ref):
                    is_match = True
                    rms = matcher.get_rms_dist(g_struct, ref)
                    if rms and rms[0] < best_rms:
                        best_rms = rms[0]
            except Exception:
                continue
        if is_match:
            match_count += 1
            rms_displacements.append(best_rms)

    novelty_pct = 100 * (len(gen_structs) - match_count) / len(gen_structs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Structural Novelty Analysis (Epoch {epoch_num})', fontsize=14, fontweight='bold')

    labels_bar = ['Novel\n(New Structures)', 'Matched\n(Similar to Training)']
    counts_bar = [len(gen_structs) - match_count, match_count]
    bar_colors = ['mediumseagreen', 'tomato']
    bars = axes[0].bar(labels_bar, counts_bar, color=bar_colors, edgecolor='white', width=0.5)
    axes[0].set_title('Novelty Breakdown')
    axes[0].set_ylabel('Number of Structures')
    for bar, count in zip(bars, counts_bar):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(count), ha='center', fontsize=14, fontweight='bold')
    axes[0].text(0.02, 0.97,
        f'Novelty: {novelty_pct:.1f}% of generated structures\nare genuinely new (not in training set)',
        transform=axes[0].transAxes, fontsize=9, va='top', color='gray')

    if rms_displacements:
        axes[1].hist(rms_displacements, bins=10, color='sandybrown', edgecolor='white', alpha=0.85)
        axes[1].set_title('RMS Displacement of Matched Structures')
        axes[1].set_xlabel('RMS Displacement (?) ? lower = closer to a training sample')
        axes[1].set_ylabel('Count')
    else:
        axes[1].text(0.5, 0.5, '[DONE] No structural matches found!\n(Maximum Novelty)',
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize=14, color='mediumseagreen', fontweight='bold')
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('eval_novelty.png', dpi=120)
    plt.show()

    print(f"  [*] INTERPRETATION ? Structural Novelty:")
    print(f"     Novel structures : {len(gen_structs)-match_count} / {len(gen_structs)} ({novelty_pct:.1f}%)")
    print(f"     Matched to training : {match_count}")
    if rms_displacements:
        print(f"     Avg RMS displacement : {np.mean(rms_displacements):.5f} ?")
    if novelty_pct >= 90:
        print("     [OK] Excellent novelty ? QGAN generates structurally unique crystals.")
    elif novelty_pct >= 70:
        print("     [OK] Good novelty ? most structures are genuinely new.")
    else:
        print("     [~]  Some mode collapse ? generator repeating similar structures.")
    print("     (Saved: eval_novelty.png)")
else:
    print("     Skipped ? pymatgen not installed.")

# ??????????????????????????????????????????????????????????????????
# STEP 4: E_hull
# ??????????????????????????????????????????????????????????????????
if args.skip_ehull:
    print("\n[4/4] E_hull skipped (--skip_ehull flag set).")
elif not EHULL_OK or not PYMATGEN_OK:
    print("\n[4/4] E_hull skipped ? missing dependencies.")
else:
    print("\n[4/4] Computing E_hull (thermodynamic stability)...")
    print("      This relaxes structures with CHGNet ? may take several minutes...")
    from pymatgen.entries.computed_entries import ComputedEntry

    relaxer = StructOptimizer()
    gen_structs_ehull = [AseAtomsAdaptor.get_structure(a) for a in generated_atoms]

    print("      Fetching Mg-Mn-O phase diagram from Materials Project...")
    with MPRester(args.mp_key) as mpr:
        entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'])
    pd_obj = PhaseDiagram(entries)

    ehull_results, N = [], min(10, len(gen_structs_ehull))
    print(f"      Relaxing {N} structures...")
    for i, struct in enumerate(gen_structs_ehull[:N]):
        try:
            result = relaxer.relax(struct, verbose=False)
            relaxed = result['final_structure']
            energy_per_atom = result['trajectory'].energies[-1] / len(relaxed)
            entry = ComputedEntry(relaxed.composition, energy_per_atom * relaxed.composition.num_atoms)
            ehull = pd_obj.get_e_above_hull(entry)
            ehull_results.append(ehull)
            print(f"      Structure {i+1}: {ehull*1000:.1f} meV/atom")
        except Exception as e:
            print(f"      Structure {i+1}: Error ? {e}")

    if ehull_results:
        ehull_meV = [e*1000 for e in ehull_results]

        fig, ax = plt.subplots(figsize=(11, 5))
        bar_colors = ['mediumseagreen' if e<=80 else 'gold' if e<=200 else 'tomato' for e in ehull_meV]
        ax.bar(range(len(ehull_meV)), ehull_meV, color=bar_colors, edgecolor='white')
        ax.axhline(80, color='blue', linestyle='--', lw=1.5, label='Synthesizable (80 meV/atom)')
        ax.axhline(200, color='red', linestyle='--', lw=1.5, label='Metastable (200 meV/atom)')
        ax.set_title(f'Energy Above Convex Hull ? Epoch {epoch_num}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Crystal Index')
        ax.set_ylabel('E_hull (meV/atom)')
        ax.legend()
        # Legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='mediumseagreen', label='Synthesizable (?80)'),
                           Patch(facecolor='gold', label='Metastable (?200)'),
                           Patch(facecolor='tomato', label='Unstable (>200)')]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig('eval_ehull.png', dpi=120)
        plt.show()

        synth = sum(1 for e in ehull_meV if e <= 80)
        meta  = sum(1 for e in ehull_meV if e <= 200)
        print(f"\n  [*] INTERPRETATION ? E_hull:")
        print(f"     Average E_hull          : {np.mean(ehull_meV):.1f} meV/atom")
        print(f"     Potentially synthesizable (?80 meV/atom)  : {synth}/{N}")
        print(f"     Theoretically metastable  (?200 meV/atom) : {meta}/{N}")
        if np.mean(ehull_meV) > 1000:
            print("     ? High E_hull ? generator has not learned thermodynamic stability.")
            print("        This is expected at early training stages. More epochs + energy-aware")
            print("        loss terms are needed to improve this metric.")
        elif np.mean(ehull_meV) > 200:
            print("     [~]  Moderate E_hull ? partially stable structures being generated.")
        else:
            print("     [OK] Low E_hull ? generator is producing thermodynamically competitive structures!")
        print("     (Saved: eval_ehull.png)")

# ??????????????????????????????????????????????????????????????????
# STEP 5: Training Loss Curve
# ??????????????????????????????????????????????????????????????????
print("\n[5/5] Plotting training loss curve...")

CSV_PATH = 'results_crystal_qgan/training_loss_history.csv'

if os.path.exists(CSV_PATH):
    import csv
    rows = []
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    epochs = [r['epoch'] for r in rows]
    has_q  = 'q_real_loss' in rows[0]

    ncols = 3 if has_q else 2
    fig, axes = plt.subplots(2, ncols, figsize=(6*ncols, 10))
    fig.suptitle('QINR-QGAN Training Loss (WGAN-GP + InfoGAN)', fontsize=15, fontweight='bold')
    axes = axes.flatten()

    plots = [
        ('d_loss',       'Critic Loss',              'tomato',
         'Should stay negative and stable.\nVery negative = critic dominating.'),
        ('g_wgan_loss',  'Generator WGAN Loss',      'steelblue',
         'Should gradually decrease.\nApproaching 0 = generator fooling critic.'),
        ('physics_loss', 'Physics Penalty',          'mediumseagreen',
         'Should drop to ~0 early.\n0 = no atomic collisions in generated crystals.'),
        ('total_g_loss', 'Total Generator Loss',     'mediumpurple',
         'Combined G + Physics + InfoGAN.\nOverall convergence indicator.'),
    ]
    if has_q:
        plots += [
            ('q_real_loss', 'Q-Head Loss (Real)',    'darkorange',
             'Should drop to ~0 quickly.\nProves Q-Head reads real compositions.'),
            ('q_fake_loss', 'Q-Head Loss (Fake)',    'hotpink',
             'Should decrease over training.\nProves generator embeds correct composition.'),
        ]

    for ax, (col, title, color, interp) in zip(axes, plots):
        vals = [r[col] for r in rows]
        ax.plot(epochs, vals, color=color, linewidth=2)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.97, interp, transform=ax.transAxes,
                fontsize=8, va='top', color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('eval_loss_curve.png', dpi=120)
    plt.show()

    print(f"  [*] INTERPRETATION ? Training Loss:")
    print(f"     Total epochs trained : {int(max(epochs))}")
    print(f"     Final Critic loss    : {rows[-1]['d_loss']:.4f}")
    print(f"     Final Generator loss : {rows[-1]['total_g_loss']:.4f}")
    print(f"     Final Physics penalty: {rows[-1]['physics_loss']:.4f}")
    if has_q:
        print(f"     Final Q_real loss    : {rows[-1]['q_real_loss']:.4f}")
        print(f"     Final Q_fake loss    : {rows[-1]['q_fake_loss']:.4f}")
    print("     (Saved: eval_loss_curve.png)")
else:
    print("     training_loss_history.csv not found (training still in progress).")
    ckpt_epochs = []
    for c in sorted(glob.glob('./results_crystal_qgan/checkpoint_*.pt')):
        try:
            ckpt_epochs.append(int(os.path.basename(c).replace('checkpoint_','').replace('.pt','')))
        except: pass

    if ckpt_epochs:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(ckpt_epochs, [1]*len(ckpt_epochs), s=150, color='steelblue',
                   zorder=5, label='Saved Checkpoint')
        ax.plot(ckpt_epochs, [1]*len(ckpt_epochs), '--', color='gray', alpha=0.5)
        ax.set_yticks([])
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_title(f'Training in progress ? Latest checkpoint: epoch {max(ckpt_epochs)}',
                     fontsize=12, fontweight='bold')
        ax.legend(); plt.tight_layout()
        plt.savefig('eval_loss_curve.png', dpi=120)
        plt.show()
        print(f"     Latest checkpoint: epoch {max(ckpt_epochs)}")

print(f"\n{'='*60}")
print("  Evaluation complete! Saved plots:")
print("    eval_ssim.png       ? SSIM scores")
print("    eval_novelty.png    ? StructureMatcher novelty")
print("    eval_ehull.png      ? Energy above convex hull")
print("    eval_loss_curve.png ? Training loss over epochs")
print(f"{'='*60}\n")
