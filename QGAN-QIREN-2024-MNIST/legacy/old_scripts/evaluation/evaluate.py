import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
evaluate.py  (QINR-QGAN)
=========================
Direct port of the classical CrystalGAN evaluate.py.
Identical pipeline, metrics, and plot style — only the model loading and
checkpoint format differ to accommodate PQWGAN_CC_Crystal.

Run from inside QGAN-QIREN-2024-MNIST/:

    py -3.12 evaluate.py
    py -3.12 evaluate.py --checkpoint results_crystal_qgan/checkpoint_390.pt

Produces (same filenames as classical, prefixed with qgan_ for side-by-side):
  - qgan_loss_curves.png
  - qgan_ssim_distribution.png
  - qgan_structural_dissimilarity.png
  - qgan_ehull_prescreen.png
  - qgan_evaluation_summary.csv
  - qgan_evaluation_results.txt
"""

import os, sys, csv, pickle, glob, re, warnings, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as compute_ssim
from ase import Atoms
warnings.filterwarnings("ignore")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to a specific checkpoint .pt file. '
                         'Default: latest in results_crystal_qgan/')
parser.add_argument('--n_gen',      type=int, default=200,
                    help='Number of structures to generate')
parser.add_argument('--z_dim',           type=int,   default=16)
parser.add_argument('--hidden_features', type=int,   default=6)
parser.add_argument('--hidden_layers',   type=int,   default=2)
parser.add_argument('--spectrum_layer',  type=int,   default=2)
parser.add_argument('--use_noise',       type=float, default=0.0)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(SCRIPT_DIR, "datasets")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results_crystal_qgan")
LOSS_CSV     = os.path.join(RESULTS_DIR, "training_loss_history.csv")
PICKLE_PATH  = os.path.join(DATASET_DIR, "mgmno_1000.pickle")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, DATASET_DIR)

from models.QINR_Crystal import PQWGAN_CC_Crystal
from view_atoms_mgmno import view_atoms

cuda   = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

print("=" * 60)
print("  QINR-QGAN Crystal Evaluation")
print("=" * 60)
print(f"  Device          : {device}")
print(f"  Loss CSV        : {os.path.isfile(LOSS_CSV)}")
print(f"  Training data   : {os.path.isfile(PICKLE_PATH)}")
print(f"  Results dir     : {os.path.isdir(RESULTS_DIR)}")
print()


# ══════════════════════════════════════════════════════════════════════════════
#  1. LOSS vs EPOCH PLOT
#  Same 3-subplot layout as classical GAN.
#  Column mapping: wasserstein → w_loss, total_g_loss → g_loss, d_loss → d_loss
# ══════════════════════════════════════════════════════════════════════════════
print("── [1/4] Loss Curves ─────────────────────────────────────")

df_loss = None
if os.path.isfile(LOSS_CSV):
    df_loss = pd.read_csv(LOSS_CSV)
    print(f"  Epochs logged: {len(df_loss)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1117")
    fig.suptitle("QINR-QGAN Training Losses",
                 fontsize=14, color="white", fontweight="bold")

    # Column names in QGAN CSV → display titles matching classical GAN panels
    plot_configs = [
        ("wasserstein",  "Wasserstein Distance  (D_real - D_fake)", "#00e5ff"),
        ("total_g_loss", "Generator Loss  (WGAN + Q_fake)",         "#76ff03"),
        ("d_loss",       "Discriminator Loss  (critic + GP + Q)",   "#ff6d00"),
    ]
    window = max(1, len(df_loss) // 20)

    for ax, (col, title, color) in zip(axes, plot_configs):
        if col not in df_loss.columns:
            ax.text(0.5, 0.5, f"'{col}' not in CSV", ha='center', va='center',
                    color='white', transform=ax.transAxes)
            continue
        ax.set_facecolor("#161b22")
        ax.plot(df_loss["epoch"], df_loss[col], color=color, linewidth=1.0, alpha=0.7)
        smoothed = df_loss[col].rolling(window, min_periods=1).mean()
        ax.plot(df_loss["epoch"], smoothed, color="white", linewidth=1.8,
                linestyle="--", alpha=0.8, label=f"MA-{window}")
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xlabel("Epoch", color="#aaaaaa")
        ax.set_ylabel("Loss",  color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e")

    plt.tight_layout()
    plt.savefig("qgan_loss_curves.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved → qgan_loss_curves.png")
else:
    print("  WARNING: training_loss_history.csv not found — skipping loss plot.")


# ══════════════════════════════════════════════════════════════════════════════
#  2. LOAD MODEL & GENERATE STRUCTURES
#  Same generation count (N_GEN=200) and label sampling strategy as classical.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── [2/4] Loading model & generating structures ───────────")

# ── Resolve checkpoint ────────────────────────────────────────────────────────
checkpoints = sorted(
    glob.glob(os.path.join(RESULTS_DIR, "checkpoint_*.pt")),
    key=lambda p: int(re.search(r"checkpoint_(\d+)", os.path.basename(p)).group(1))
)
if not checkpoints:
    raise FileNotFoundError(f"No checkpoint_*.pt found in {RESULTS_DIR}")

latest_ckpt = args.checkpoint if args.checkpoint else checkpoints[-1]
print(f"  Checkpoint: {os.path.basename(latest_ckpt)}")

# ── Model dimensions ──────────────────────────────────────────────────────────
data_dim  = 90    # 30 × 3  crystal image, flattened
label_dim = 28    # binary occupancy: 8 Mg + 8 Mn + 12 O slots
z_dim     = args.z_dim

gen_input_dim    = z_dim + label_dim       # noise + label  → 44
critic_input_dim = data_dim + label_dim    # data  + label  → 118

# ── Build PQWGAN_CC_Crystal ───────────────────────────────────────────────────
gan = PQWGAN_CC_Crystal(
    input_dim_g     = gen_input_dim,
    output_dim      = data_dim,
    input_dim_d     = critic_input_dim,
    hidden_features = args.hidden_features,
    hidden_layers   = args.hidden_layers,
    spectrum_layer  = args.spectrum_layer,
    use_noise       = args.use_noise,
)
generator = gan.generator.to(device)

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt_data  = torch.load(latest_ckpt, map_location=device, weights_only=False)
generator.load_state_dict(ckpt_data['generator'])
generator.to(device).eval()
epoch_loaded = ckpt_data.get('epoch', '?')
print(f"  Loaded generator (epoch {epoch_loaded})")

# ── Load training data ────────────────────────────────────────────────────────
with open(PICKLE_PATH, "rb") as f:
    training_data = pickle.load(f)
real_images = np.array([x[0] for x in training_data])   # (N, 30, 3)
real_labels = np.array([x[1] for x in training_data])   # (N, 28, ...)
real_labels = real_labels.reshape(len(real_labels), 28)
print(f"  Training samples: {len(real_images)}")

# ── Label sampling — identical probability structure to classical GAN ──────────
def sample_labels(n):
    """
    Sample composition labels with the same slot structure as the classical GAN.
    Classical GAN used separate one-hot vectors for c_mg, c_mn, c_o.
    Here we concatenate them into a 28-dim binary vector, matching train_crystal.py.
    """
    mg_i = np.random.randint(0, 8,  n)
    mn_i = np.random.randint(0, 8,  n)
    o_i  = np.random.randint(0, 12, n)
    lbl  = np.zeros((n, 28), dtype=np.float32)
    for j in range(n):
        lbl[j, mg_i[j]]              = 1.0   # one-hot Mg slot
        lbl[j, 8  + mn_i[j]]         = 1.0   # one-hot Mn slot
        lbl[j, 16 + o_i[j]]          = 1.0   # one-hot O  slot
    return torch.tensor(lbl, dtype=torch.float32, device=device)

N_GEN = args.n_gen
gen_images_list = []
with torch.no_grad():
    for start in range(0, N_GEN, 32):
        bs        = min(32, N_GEN - start)
        z         = torch.randn(bs, z_dim, device=device)
        labels    = sample_labels(bs)
        gen_input = torch.cat([z, labels], dim=1)    # (bs, 44)
        fake      = generator(gen_input)              # (bs, 90)
        gen_images_list.append(fake.cpu().numpy())

gen_images = np.concatenate(gen_images_list, axis=0).reshape(N_GEN, 30, 3)
print(f"  Generated {N_GEN} structures")


def image_to_ase(img):
    try:
        atoms, _ = view_atoms(img, view=False)
        return atoms
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  3. METRIC: SSIM  (identical to classical GAN — win_size=3, data_range=1.0)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── [3/4] SSIM ────────────────────────────────────────────")

def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

N_REAL = min(200, len(real_images))
real_sample = real_images[:N_REAL]

ssim_scores = []
for i, gen in enumerate(gen_images):
    gen_n = norm01(gen)
    best  = max(compute_ssim(gen_n, norm01(r), data_range=1.0, win_size=3) for r in real_sample)
    ssim_scores.append(best)
    if (i + 1) % 50 == 0:
        print(f"  SSIM: {i+1}/{N_GEN} done...")

ssim_mean = np.mean(ssim_scores)
ssim_std  = np.std(ssim_scores)

rng = np.random.default_rng(42)
baseline_ssim = np.mean([
    compute_ssim(rng.random((30, 3)), rng.random((30, 3)), data_range=1.0, win_size=3)
    for _ in range(100)
])

print(f"  Mean SSIM : {ssim_mean:.4f} +/- {ssim_std:.4f}")
print(f"  Baseline  : {baseline_ssim:.4f}")

fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
ax.set_facecolor("#161b22")
ax.hist(ssim_scores, bins=30, color="#00e5ff", edgecolor="none", alpha=0.8,
        label=f"QINR-QGAN  mean={ssim_mean:.3f}")
ax.axvline(baseline_ssim, color="#ff6d00", linewidth=1.8, linestyle="--",
           label="Random baseline")
ax.set_xlabel("Max SSIM vs Training Set", color="#aaaaaa")
ax.set_ylabel("Count", color="#aaaaaa")
ax.set_title("Structural Similarity Index Measure (SSIM)", color="white", fontweight="bold")
ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.legend(labelcolor="white", facecolor="#1a1a2e")
plt.tight_layout()
plt.savefig("qgan_ssim_distribution.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved → qgan_ssim_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  4. METRIC: STRUCTURAL DISSIMILARITY (StructureMatcher)
#  Identical parameters to classical GAN: ltol=0.3, stol=0.5, angle_tol=10
# ══════════════════════════════════════════════════════════════════════════════
print("\n── [4a/4] Structural Dissimilarity ───────────────────────")
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher

def ase_to_pmg(atoms):
    cell  = atoms.get_cell().tolist()
    pos   = atoms.get_scaled_positions().tolist()
    symbs = atoms.get_chemical_symbols()
    return Structure(Lattice(cell), symbs, pos)

N_REF = min(100, len(real_images))
print(f"  Building reference set ({N_REF} real structures)...")
ref_structs = []
for img in real_images[:N_REF]:
    a = image_to_ase(img)
    if a is not None:
        try: ref_structs.append(ase_to_pmg(a))
        except Exception: pass
print(f"  Reference structures: {len(ref_structs)}")

matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10,
                           primitive_cell=True, scale=True)

N_TEST = min(100, N_GEN)
novel = match = invalid = 0
rms_list = []

for i, img in enumerate(gen_images[:N_TEST]):
    a = image_to_ase(img)
    if a is None: invalid += 1; continue
    try:   gs = ase_to_pmg(a)
    except Exception: invalid += 1; continue

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
        except Exception: continue
    if matched: match += 1
    else:       novel += 1

    if (i + 1) % 25 == 0:
        print(f"  StructureMatcher: {i+1}/{N_TEST} done...")

evaluated   = N_TEST - invalid
novelty_pct = novel / evaluated * 100 if evaluated else 0
reconst_pct = match / evaluated * 100 if evaluated else 0
mean_rms    = np.mean(rms_list) if rms_list else float("nan")

print(f"  Novel : {novel}/{evaluated}  ({novelty_pct:.1f}%)")
print(f"  Match : {match}/{evaluated}  ({reconst_pct:.1f}%)")
print(f"  Mean RMS dissimilarity (matched): {mean_rms:.4f}")

fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0d1117")
ax.set_facecolor("#0d1117")
ax.pie([novel, match, invalid],
       labels=["Novel", "Reconstruction", "Invalid"],
       colors=["#00e5ff", "#ff6d00", "#555555"],
       autopct="%1.1f%%", startangle=90,
       wedgeprops=dict(edgecolor="#0d1117"))
ax.set_title("Structural Dissimilarity: Generated vs Training Set",
             color="white", fontweight="bold", pad=12)
for t in ax.texts: t.set_color("white")
plt.tight_layout()
plt.savefig("qgan_structural_dissimilarity.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved → qgan_structural_dissimilarity.png")


# ══════════════════════════════════════════════════════════════════════════════
#  5. METRIC: E_HULL PRE-SCREENING (min interatomic distance)
#  Identical threshold to classical GAN: MIN_DIST = 1.0 Å
# ══════════════════════════════════════════════════════════════════════════════
print("\n── [4b/4] E_hull Pre-screening ───────────────────────────")
MIN_DIST = 1.0   # Angstroms — same as classical GAN

min_dists = []
valid_count = 0
for img in gen_images:
    a = image_to_ase(img)
    if a is None:
        min_dists.append(0.0)
        continue
    try:
        d = a.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        md = float(d.min())
    except Exception:
        md = 0.0
    min_dists.append(md)
    if md >= MIN_DIST:
        valid_count += 1

validity_pct = valid_count / N_GEN * 100
print(f"  Valid (min dist >= {MIN_DIST} A): {valid_count}/{N_GEN}  ({validity_pct:.1f}%)")
print(f"  Mean min distance            : {np.mean(min_dists):.3f} A")

fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
ax.set_facecolor("#161b22")
ax.hist(min_dists, bins=30, color="#76ff03", edgecolor="none", alpha=0.8)
ax.axvline(MIN_DIST, color="#ff6d00", linewidth=1.8, linestyle="--",
           label=f"Threshold = {MIN_DIST} A")
ax.set_xlabel("Min Interatomic Distance (A)", color="#aaaaaa")
ax.set_ylabel("Count", color="#aaaaaa")
ax.set_title("Pre-DFT Structural Validity Screen (E_hull pre-qualification)",
             color="white", fontweight="bold")
ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.legend(labelcolor="white", facecolor="#1a1a2e")
plt.tight_layout()
plt.savefig("qgan_ehull_prescreen.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved → qgan_ehull_prescreen.png")


# ══════════════════════════════════════════════════════════════════════════════
#  6. SUMMARY TABLE  (same schema as classical GAN's evaluation_summary.csv)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Summary ───────────────────────────────────────────────")

w_final = "--"
g_final = "--"
if df_loss is not None:
    if 'wasserstein' in df_loss.columns:
        w_final = f"{df_loss['wasserstein'].iloc[-1]:.4f}"
    if 'total_g_loss' in df_loss.columns:
        g_final = f"{df_loss['total_g_loss'].iloc[-1]:.4f}"

summary = {
    "Metric": [
        "SSIM (mean +/- std)",
        "SSIM Baseline (random)",
        "Structural Novelty (%)",
        "Structural Reconstruction (%)",
        "Structural Validity -- min dist >= 1.0 A (%)",
        "Mean Min Interatomic Distance (A)",
        "Wasserstein Distance (final epoch)",
        "Generator Loss (final epoch)",
    ],
    "Classical GAN": [
        "-- (fill from classical evaluate.py)",
        f"{baseline_ssim:.4f}",
        "--", "--", "--", "--", "--", "--",
    ],
    "QINR-QGAN": [
        f"{ssim_mean:.4f} +/- {ssim_std:.4f}",
        f"{baseline_ssim:.4f}",
        f"{novelty_pct:.1f}%",
        f"{reconst_pct:.1f}%",
        f"{validity_pct:.1f}%",
        f"{np.mean(min_dists):.3f} A",
        w_final,
        g_final,
    ],
}

df_summary = pd.DataFrame(summary).set_index("Metric")
print(df_summary.to_string())
df_summary.to_csv("qgan_evaluation_summary.csv")

report_lines = [
    "QINR-QGAN Crystal Evaluation Report",
    "=" * 60,
    f"Checkpoint : {os.path.basename(latest_ckpt)} (epoch {epoch_loaded})",
    f"Generated  : {N_GEN} structures",
    "",
    df_summary.to_string(),
    "",
    "E_hull DFT Template (fill after VASP relaxation):",
    f"  Pre-DFT valid structures : {valid_count} / {N_GEN}  ({validity_pct:.1f}%)",
    "  DFT-relaxed              : ____ / " + str(valid_count),
    "  E_hull <= 200 meV/atom (metastable)    : ____  (____%) ",
    "  E_hull <=  80 meV/atom (synthesizable) : ____  (____%) ",
]
with open("qgan_evaluation_results.txt", "w") as f:
    f.write("\n".join(report_lines))

print("\n" + "=" * 60)
print("  Evaluation complete. Files saved:")
for fn in ["qgan_loss_curves.png", "qgan_ssim_distribution.png",
           "qgan_structural_dissimilarity.png", "qgan_ehull_prescreen.png",
           "qgan_evaluation_summary.csv", "qgan_evaluation_results.txt"]:
    print(f"  • {fn}")
print("=" * 60)
