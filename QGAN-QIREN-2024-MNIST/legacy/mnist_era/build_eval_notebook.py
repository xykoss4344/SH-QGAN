"""
Creates QGAN_Evaluation.ipynb - clean rewrite with all fixes baked in.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

PROJECT_DIR = r"c:\\Users\\Adminb\\OneDrive\\Documents\\Projects\\qgan\\QINR-QGAN\\QGAN-QIREN-2024-MNIST"

# ── Title ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# QINR-QGAN — Advanced Evaluation Notebook
Evaluates generated Mg-Mn-O crystal structures across 4 metrics:
**SSIM · StructureMatcher · E_hull · Training Loss Curve**

> ⚠️ **Always run cells top-to-bottom (or use Run All).** Each cell imports what it needs.
"""))

# ── Cell 0: Force kernel fresh-import ──────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Step 0 — Environment Setup\nRun this first. It sets the working directory and reloads any cached modules."))
cells.append(nbf.v4.new_code_cell(f"""\
import os, sys, importlib

# Force correct working directory
PROJECT_DIR = r"{PROJECT_DIR}"
os.chdir(PROJECT_DIR)
print(f"✅ Working directory: {{os.getcwd()}}")

# Add paths
for p in [PROJECT_DIR, os.path.join(PROJECT_DIR, 'datasets')]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Force reload view_atoms module if already cached (fixes stale kernel issue)
if 'view_atoms_mgmno' in sys.modules:
    importlib.reload(sys.modules['view_atoms_mgmno'])
    print("🔄 Reloaded view_atoms_mgmno module")

import pickle, glob, random, warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Check optional deps
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_OK = True; print("✅ scikit-image — SSIM enabled")
except ImportError:
    SKIMAGE_OK = False; print("❌ scikit-image missing")

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    PYMATGEN_OK = True; print("✅ pymatgen — StructureMatcher enabled")
except ImportError:
    PYMATGEN_OK = False; print("❌ pymatgen missing")

try:
    from chgnet.model.dynamics import StructOptimizer
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    EHULL_OK = True; print("✅ chgnet + mp-api — E_hull enabled")
except ImportError:
    EHULL_OK = False; print("⚠️  chgnet/mp-api missing — E_hull will be skipped")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥  Device: {{device}}")
"""))

# ── Cell 1: Generate crystals ───────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Step 1 — Load Model & Generate Crystals"))
cells.append(nbf.v4.new_code_cell(f"""\
import os, sys, pickle, glob, random, importlib
import torch, numpy as np

PROJECT_DIR = r"{PROJECT_DIR}"
os.chdir(PROJECT_DIR)
for p in [PROJECT_DIR, os.path.join(PROJECT_DIR, 'datasets')]:
    if p not in sys.path: sys.path.insert(0, p)

# Always reload to pick up any source fixes
import view_atoms_mgmno as _va_mod
importlib.reload(_va_mod)
from view_atoms_mgmno import view_atoms

from models.QINR_Crystal import PQWGAN_CC_Crystal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SAMPLES = 50
Z_DIM, LABEL_DIM, DATA_DIM = 16, 28, 90

# Load dataset
with open("datasets/mgmno_100.pickle", 'rb') as f:
    raw_data = pickle.load(f)

labels, real_atoms_list = [], []
for idx in random.sample(range(len(raw_data)), min(100, len(raw_data))):
    c, l = raw_data[idx]
    if len(labels) < NUM_SAMPLES:
        labels.append(l.flatten())
    try:
        atoms, _ = view_atoms(c.flatten(), view=False)
        real_atoms_list.append(atoms)
    except Exception:
        pass

while len(labels) < NUM_SAMPLES:
    labels.append(labels[0])
labels_t = torch.tensor(np.array(labels, dtype=np.float32)).to(device)

# Load latest checkpoint
ckpts = sorted(glob.glob("./results_crystal_qgan/checkpoint_*.pt"), key=os.path.getmtime)
assert ckpts, "No checkpoints found!"
CKPT = ckpts[-1]
print(f"📂 Checkpoint: {{CKPT}}")

gan = PQWGAN_CC_Crystal(Z_DIM+LABEL_DIM, DATA_DIM, DATA_DIM+LABEL_DIM,
                        hidden_features=6, hidden_layers=2, spectrum_layer=2, use_noise=0.0)
generator = gan.generator.to(device)
ckpt_data = torch.load(CKPT, map_location=device)
generator.load_state_dict(ckpt_data['generator'])
generator.eval()

with torch.no_grad():
    z = torch.randn(NUM_SAMPLES, Z_DIM).to(device)
    fake_flat = generator(torch.cat([z, labels_t], dim=1)).cpu().numpy()

generated_atoms, failed = [], 0
for img in fake_flat:
    try:
        atoms, _ = view_atoms(img, view=False)
        generated_atoms.append(atoms)
    except Exception:
        failed += 1

print(f"✅ Valid: {{len(generated_atoms)}} | Invalid cell params (skipped): {{failed}}")
assert len(generated_atoms) > 0, "All structures invalid — generator not converged yet, try a later checkpoint."
print(f"📦 Real reference structures loaded: {{len(real_atoms_list)}}")
"""))

# ── Cell 2: SSIM ───────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 2 — Structural Similarity (SSIM)
Each crystal is represented as a 30×30 interatomic distance matrix. SSIM compares these between generated and real crystals.
- `1.0` = perfect match · `0.0` = no similarity
- Classical GAN baseline on Mg-Mn-O: ~0.55–0.70"""))
cells.append(nbf.v4.new_code_cell("""\
import numpy as np, random, matplotlib.pyplot as plt

def dist_matrix(atoms):
    d = atoms.get_all_distances(mic=True)
    mat = np.zeros((30, 30))
    n = min(30, d.shape[0])
    mat[:n, :n] = d[:n, :n]
    return mat

if SKIMAGE_OK and real_atoms_list and generated_atoms:
    ssim_scores = []
    for g_atom in generated_atoms:
        g_mat = dist_matrix(g_atom)
        r_mat = dist_matrix(random.choice(real_atoms_list))
        score = ssim(g_mat, r_mat, data_range=25.0)
        ssim_scores.append(score)

    avg_ssim = np.mean(ssim_scores)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(ssim_scores, bins=15, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(avg_ssim, color='red', linestyle='--', lw=2, label=f'Mean: {avg_ssim:.3f}')
    axes[0].set_title('SSIM Score Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('SSIM'); axes[0].set_ylabel('Count'); axes[0].legend()

    g_ex = dist_matrix(generated_atoms[0])
    r_ex = dist_matrix(real_atoms_list[0])
    axes[1].imshow(np.hstack([r_ex, np.ones((30,2))*25, g_ex]), cmap='viridis', vmin=0, vmax=25)
    axes[1].set_title('Distance Matrix: Real (left) | Generated (right)', fontsize=12)
    axes[1].axis('off')
    plt.tight_layout(); plt.show()

    print(f"\\n📊 SSIM — Mean: {avg_ssim:.4f}  Min: {min(ssim_scores):.4f}  Max: {max(ssim_scores):.4f}")
else:
    print("SSIM skipped — scikit-image not installed or no structures loaded.")
"""))

# ── Cell 3: StructureMatcher ────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 3 — Structural Dissimilarity (StructureMatcher)
Compares full 3D structures. `0 matches` = maximum novelty (new crystals not in training set)."""))
cells.append(nbf.v4.new_code_cell("""\
import numpy as np, matplotlib.pyplot as plt

if PYMATGEN_OK and real_atoms_list and generated_atoms:
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
    labels_bar = ['Novel\\n(New)', 'Matched\\n(Similar to Training)']
    counts_bar = [len(gen_structs) - match_count, match_count]
    axes[0].bar(labels_bar, counts_bar, color=['mediumseagreen','tomato'], edgecolor='white', width=0.5)
    axes[0].set_title('Novelty Breakdown', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Structures')
    for i,v in enumerate(counts_bar): axes[0].text(i, v+0.2, str(v), ha='center', fontsize=13, fontweight='bold')

    if rms_displacements:
        axes[1].hist(rms_displacements, bins=10, color='sandybrown', edgecolor='white')
        axes[1].set_title('RMS Displacement of Matched Structures', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('RMS (Å)')
    else:
        axes[1].text(0.5, 0.5, 'No matches found\\n(Maximum Novelty!)', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=14)
        axes[1].axis('off')
    plt.tight_layout(); plt.show()

    print(f"\\n📊 StructureMatcher — Novel: {len(gen_structs)-match_count} ({novelty_pct:.1f}%)  |  Matched: {match_count}")
    if rms_displacements: print(f"   Avg RMS displacement: {np.mean(rms_displacements):.5f} Å")
else:
    print("StructureMatcher skipped — pymatgen not installed or no structures.")
"""))

# ── Cell 4: E_hull ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 4 — Energy Above Convex Hull (E_hull)
Uses CHGNet to relax structures then queries Materials Project for the Mg-Mn-O phase diagram.
| E_hull | Category |
|--------|----------|
| ≤ 80 meV/atom | Potentially synthesizable |
| ≤ 200 meV/atom | Theoretically metastable |
| > 200 meV/atom | Thermodynamically unstable |"""))
cells.append(nbf.v4.new_code_cell("""\
import os, numpy as np, matplotlib.pyplot as plt

MP_API_KEY = os.environ.get('MP_API_KEY', 'hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA')

if EHULL_OK and PYMATGEN_OK and generated_atoms:
    from pymatgen.entries.computed_entries import ComputedEntry
    print("Initializing CHGNet relaxer...")
    relaxer = StructOptimizer()
    gen_structs_ehull = [AseAtomsAdaptor.get_structure(a) for a in generated_atoms]
    print("Fetching Mg-Mn-O phase diagram from Materials Project...")
    with MPRester(MP_API_KEY) as mpr:
        entries = mpr.get_entries_in_chemsys(["Mg","Mn","O"])
    pd_obj = PhaseDiagram(entries)

    ehull_results, N = [], min(10, len(gen_structs_ehull))
    print(f"Relaxing {N} structures with CHGNet...")
    for i, struct in enumerate(gen_structs_ehull[:N]):
        try:
            result = relaxer.relax(struct, verbose=False)
            relaxed = result["final_structure"]
            energy_per_atom = result["trajectory"].energies[-1] / len(relaxed)
            entry = ComputedEntry(relaxed.composition, energy_per_atom * relaxed.composition.num_atoms)
            ehull = pd_obj.get_e_above_hull(entry)
            ehull_results.append(ehull)
            print(f"  Structure {i+1}: E_hull = {ehull*1000:.1f} meV/atom")
        except Exception as e:
            print(f"  Structure {i+1}: Error — {e}")

    if ehull_results:
        ehull_meV = [e*1000 for e in ehull_results]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(range(len(ehull_meV)), ehull_meV,
               color=['mediumseagreen' if e<=80 else 'gold' if e<=200 else 'tomato' for e in ehull_meV])
        ax.axhline(80, color='blue', linestyle='--', label='Synthesizable (80 meV/atom)')
        ax.axhline(200, color='red', linestyle='--', label='Metastable (200 meV/atom)')
        ax.set_title('Energy Above Convex Hull per Crystal', fontsize=14, fontweight='bold')
        ax.set_xlabel('Crystal Index'); ax.set_ylabel('E_hull (meV/atom)'); ax.legend()
        plt.tight_layout(); plt.show()
        print(f"\\n📊 E_hull — Synthesizable (≤80): {sum(1 for e in ehull_meV if e<=80)}")
        print(f"   Metastable (≤200): {sum(1 for e in ehull_meV if e<=200)}")
        print(f"   Average: {np.mean(ehull_meV):.1f} meV/atom")
else:
    print("E_hull skipped — install chgnet + mp-api, or no valid structures were generated.")
"""))

# ── Cell 5: Loss Curve ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 5 — Training Loss Curve
Shows Generator / Critic / Physics / InfoGAN Q-Head losses over epochs.
> `training_loss_history.csv` is saved automatically when training completes."""))
cells.append(nbf.v4.new_code_cell("""\
import os, glob, matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results_crystal_qgan/training_loss_history.csv"

if os.path.exists(CSV_PATH):
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    has_q = 'q_real_loss' in df.columns

    n_plots = 6 if has_q else 4
    fig, axes = plt.subplots(2, 3 if has_q else 2, figsize=(18 if has_q else 15, 10))
    axes = axes.flatten()
    fig.suptitle("QINR-QGAN Training Loss (WGAN-GP + InfoGAN)", fontsize=16, fontweight='bold')

    plots = [
        ('d_loss',      'Critic Loss',                  'tomato'),
        ('g_wgan_loss', 'Generator WGAN Loss',           'steelblue'),
        ('physics_loss','Physics Penalty',               'mediumseagreen'),
        ('total_g_loss','Total Generator Loss',          'mediumpurple'),
    ]
    if has_q:
        plots += [
            ('q_real_loss', 'Q-Head Loss (Real)',        'darkorange'),
            ('q_fake_loss', 'Q-Head Loss (Fake)',        'hotpink'),
        ]

    for ax, (col, title, color) in zip(axes, plots):
        if col in df.columns:
            ax.plot(df['epoch'], df[col], color=color, linewidth=2)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.show()
    print(f"📊 Training epochs: {df['epoch'].max()}")
    print(f"   Final Critic:     {df['d_loss'].iloc[-1]:.4f}")
    print(f"   Final G Total:    {df['total_g_loss'].iloc[-1]:.4f}")
    if has_q:
        print(f"   Final Q_real:     {df['q_real_loss'].iloc[-1]:.4f}")
        print(f"   Final Q_fake:     {df['q_fake_loss'].iloc[-1]:.4f}")
else:
    ckpts = sorted(glob.glob("./results_crystal_qgan/checkpoint_*.pt"))
    ckpt_epochs = []
    for c in ckpts:
        try: ckpt_epochs.append(int(os.path.basename(c).replace("checkpoint_","").replace(".pt","")))
        except: pass

    fig, ax = plt.subplots(figsize=(10,3))
    ax.scatter(ckpt_epochs, [1]*len(ckpt_epochs), s=120, color='steelblue', zorder=5, label='Checkpoint')
    ax.plot(ckpt_epochs, [1]*len(ckpt_epochs), '--', color='gray', alpha=0.5)
    ax.set_yticks([]); ax.set_xlabel('Epoch')
    ax.set_title('Training still in progress — showing saved checkpoints', fontweight='bold')
    ax.legend(); plt.tight_layout(); plt.show()
    print(f"Latest checkpoint: epoch {max(ckpt_epochs) if ckpt_epochs else 'none'}")
    print("Full loss curve available after training completes.")
"""))

nb.cells.extend(cells)
nbf.write(nb, 'QGAN_Evaluation.ipynb')
print("✅ QGAN_Evaluation.ipynb rebuilt successfully.")
