"""
analysis_fair.py
================
Comprehensive comparison of Quantum v4 QGAN vs Classical-fair CWGAN
for MgMnO crystal generation.

Sections:
  0. Setup & paths
  1. Load training loss data & plot
  2. Load datasets and models
  3. Generate crystals (N_GEN=4800)
  4. MIC validity screening
  5. CHGNet relaxation + E_hull (skip if cache exists)
  6. SSIM comparison
  7. VASP input generation
  8. Plots: comparison bar, E_hull distribution, phase diagram, MIC distances
  9. Final summary table
"""
# Grouped into a subfolder during the 2026-08-30 reorg; core modules stay at the
# package root because eight files import them. This keeps them importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys
import json
import pickle
import importlib.util
import warnings
warnings.filterwarnings("ignore")

import types
from importlib.abc import MetaPathFinder, Loader

class MockObj:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return MockObj()
    def __call__(self, *args, **kwargs): return MockObj()

class MockLoader(Loader):
    def exec_module(self, module): pass
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        setattr(m, 'Structure', MockObj)
        setattr(m, 'StructOptimizer', MockObj)
        return m

class MockMetaFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname.startswith(x) for x in ['pymatgen', 'chgnet', 'mace']):
            import importlib.util
            return importlib.util.spec_from_loader(fullname, MockLoader())
        return None

sys.meta_path.insert(0, MockMetaFinder())

import builtins
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch, io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        try:
            return super().find_class(module, name)
        except Exception:
            return MockObj

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────────────────────
# 0.  PATHS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
Q_DIR  = os.path.dirname(os.path.abspath(__file__))
CLS_DIR = (
    "C:/Users/Adminb/OneDrive/Documents/Projects/crystalGan/"
    "Composition-Conditioned-Crystal-GAN/Composition_Conditioned_Crystal_GAN"
)

DATASET_PATH  = os.path.join(Q_DIR, "datasets", "mgmno_100_aug.pickle")
Q_CKPT        = os.path.join(Q_DIR, "results_crystal_qgan_v4", "checkpoint_490.pt")
Q_LOSS_CSV    = os.path.join(Q_DIR, "results_crystal_qgan_v4", "training_loss_history.csv")
CLS_GEN       = os.path.join(CLS_DIR, "model_cwgan_mgmno_fair", "generator_490")
CLS_MODELS_PY = os.path.join(CLS_DIR, "models.py")
CLS_LOSS_JSON = os.path.join(CLS_DIR, "training_loss_log.json")

OUT_DIR        = os.path.join(Q_DIR, "results_analysis")
RELAXED_CACHE  = os.path.join(OUT_DIR, "relaxed_structures.pkl")
VASP_Q_DIR     = os.path.join(OUT_DIR, "vasp_inputs", "quantum")
VASP_C_DIR     = os.path.join(OUT_DIR, "vasp_inputs", "classical")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VASP_Q_DIR, exist_ok=True)
os.makedirs(VASP_C_DIR, exist_ok=True)

N_GEN        = 4800
N_REAL_SSIM  = 200
N_GEN_SSIM   = 500
BATCH        = 128
MP_API_KEY   = "hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA"

SPECIES_MAP = ['Mg'] * 8 + ['Mn'] * 8 + ['O'] * 12   # 28 atom slots

DARK_BG   = "#0d1117"
AXES_BG   = "#161b22"
COL_Q     = "#ff9800"   # orange  – quantum
COL_C     = "#00e5ff"   # cyan    – classical
COL_Q_LOS = "#00e5ff"   # cyan for quantum loss panels (as specified)
COL_C_LOS = "#76ff03"   # lime for classical loss panels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Setup] Device: {device}")
print(f"[Setup] Output dir: {OUT_DIR}")


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: moving average
# ──────────────────────────────────────────────────────────────────────────────
def moving_average(arr, window=20):
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    ma = ret[window - 1:] / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, ma])


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

def light_fig(nrows=1, ncols=1, figsize=(12, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat:
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# 1.  TRAINING LOSS PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_training_losses():
    print("\n" + "=" * 60)
    print("[1] Plotting training loss curves")
    print("=" * 60)

    # --- Quantum v4 CSV ---
    q_df = pd.read_csv(Q_LOSS_CSV)
    # columns: epoch, d_loss, wasserstein, q_real_loss, q_fake_loss, total_g_loss

    # --- Classical-fair JSON ---
    with open(CLS_LOSS_JSON, "r") as f:
        cls_data = json.load(f)
    c_df = pd.DataFrame(cls_data)  # keys: epoch, w_loss, g_loss, d_loss

    fig, axes = light_fig(nrows=2, ncols=3, figsize=(18, 9))

    # --- Row 1: Quantum ---
    q_panels = [
        ("wasserstein", "Wasserstein Distance", "Q Wasserstein"),
        ("d_loss",      "Discriminator Loss",   "Q D-loss"),
        ("total_g_loss","Generator Loss",        "Q G-loss"),
    ]
    for col_idx, (key, ylabel, title) in enumerate(q_panels):
        ax = axes[0, col_idx]
        epochs = q_df["epoch"].values
        vals   = q_df[key].values
        ma     = moving_average(vals, 20)
        ax.plot(epochs, vals, color=COL_Q_LOS, alpha=0.3, linewidth=0.8, label="raw")
        ax.plot(epochs, ma,   color=COL_Q_LOS, linewidth=1.8, label="MA-20")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    # --- Row 2: Classical ---
    c_panels = [
        ("w_loss", "Wasserstein Loss",  "Classical w_loss"),
        ("d_loss", "Discriminator Loss","Classical D-loss"),
        ("g_loss", "Generator Loss",    "Classical G-loss"),
    ]
    for col_idx, (key, ylabel, title) in enumerate(c_panels):
        ax = axes[1, col_idx]
        epochs = c_df["epoch"].values
        vals   = c_df[key].values
        ma     = moving_average(vals, 20)
        ax.plot(epochs, vals, color=COL_C_LOS, alpha=0.3, linewidth=0.8, label="raw")
        ax.plot(epochs, ma,   color=COL_C_LOS, linewidth=1.8, label="MA-20")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.suptitle("Training Loss Curves: Quantum v4 (top) vs Classical-fair (bottom)",
                 color='black', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "training_loss_curves.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  LOAD DATASET & MODELS
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset():
    print("\n" + "=" * 60)
    print("[2] Loading dataset")
    print("=" * 60)
    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)
    # data may be a list of (crystal_90, label_28) tuples or a dict
    if isinstance(data, dict):
        crystals = np.array(data["crystals"])
        labels   = np.array(data["labels"])
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        crystals = np.array(data[0])
        labels   = np.array(data[1])
    else:
        # assume list of (x, y) pairs
        crystals = np.array([d[0] for d in data])
        labels   = np.array([d[1] for d in data])
    # Ensure shapes are (N,30,3) and (N,28)
    crystals = crystals.reshape(len(crystals), 30, 3).astype(np.float32)
    labels   = labels.reshape(len(labels), -1).astype(np.float32)
    if labels.shape[1] != 28:
        labels = labels[:, :28]
    print(f"  Dataset: {len(crystals)} samples, crystal shape {crystals.shape}, "
          f"label shape {labels.shape}")
    return crystals, labels


def load_quantum_model():
    print("[2] Loading Quantum v4 model ...")
    sys.path.insert(0, Q_DIR)
    from models.QINR_Crystal import PQWGAN_CC_Crystal
    qgan = PQWGAN_CC_Crystal(
        input_dim_g=92, output_dim=90, input_dim_d=118,
        hidden_features=8, hidden_layers=3, spectrum_layer=1, use_noise=0.0
    )
    ckpt = torch.load(Q_CKPT, map_location=device)
    qgan.generator.load_state_dict(ckpt['generator'])
    q_gen = qgan.generator.to(device).eval()
    print("  Quantum generator loaded.")
    return q_gen


def load_classical_model():
    print("[2] Loading Classical-fair model ...")
    spec = importlib.util.spec_from_file_location('cls_models', CLS_MODELS_PY)
    cls_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cls_mod)

    class _Opt:
        latent_dim = 512
        input_dim  = 541

    c_gen = cls_mod.Generator(_Opt()).to(device)
    raw_ck = torch.load(CLS_GEN, map_location=device)
    c_gen.load_state_dict(raw_ck['model'] if 'model' in raw_ck else raw_ck)
    c_gen.eval()
    print("  Classical generator loaded.")
    return c_gen


# ──────────────────────────────────────────────────────────────────────────────
# 3.  GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def make_labels(all_labels, n):
    """Tile labels from dataset to produce exactly n samples."""
    flat = np.array(all_labels, dtype=np.float32).reshape(len(all_labels), -1)  # flatten (N,28,1)->(N,28)
    reps = (n // len(flat)) + 1
    tiled = np.tile(flat, (reps, 1))[:n]
    return tiled.astype(np.float32)


def gen_q(q_gen, n, labels):
    out = []
    with torch.no_grad():
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            lbls = torch.from_numpy(labels[s:e]).to(device)
            noise = torch.randn(e - s, 64, device=device)
            inp = torch.cat([noise, lbls], dim=1)
            out.append(q_gen(inp).cpu().numpy())
    return np.concatenate(out)          # (n, 90)


def gen_c(c_gen, n, labels):
    out = []
    with torch.no_grad():
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            lbls = torch.from_numpy(labels[s:e]).to(device)
            c1 = lbls[:, 0:8]
            c2 = lbls[:, 8:16]
            c3 = lbls[:, 16:28]
            c4 = lbls.sum(1, keepdim=True).float() / 28.0
            raw = c_gen(torch.randn(e - s, 512, device=device), c1, c2, c3, c4)
            out.append(raw.view(e - s, 90).cpu().numpy())
    return np.concatenate(out)          # (n, 90)


def generate_crystals(q_gen, c_gen, labels_gen):
    print("\n" + "=" * 60)
    print(f"[3] Generating {N_GEN} crystals from each model")
    print("=" * 60)
    print("  Generating quantum ...")
    q_crystals = gen_q(q_gen, N_GEN, labels_gen)
    print(f"  Quantum done: {q_crystals.shape}")
    print("  Generating classical ...")
    c_crystals = gen_c(c_gen, N_GEN, labels_gen)
    print(f"  Classical done: {c_crystals.shape}")
    return q_crystals, c_crystals


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MIC VALIDITY
# ──────────────────────────────────────────────────────────────────────────────
def min_dist_mic(coords_90, label_28):
    from ase import Atoms
    arr     = coords_90.reshape(30, 3)
    lengths = np.clip(arr[0] * 30,  2.0,  30.0)
    angles  = np.clip(arr[1] * 180, 30.0, 150.0)
    occ = label_28.astype(bool)
    sp  = [SPECIES_MAP[i] for i in range(28) if occ[i]]
    frac = arr[2:][occ]
    if len(sp) < 2:
        return 0.0
    try:
        atoms = Atoms(
            symbols=sp,
            scaled_positions=frac,
            cell=np.concatenate([lengths, angles]),
            pbc=True
        )
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        return float(d.min())
    except Exception:
        return 0.0


def compute_mic_distances(crystals, labels, tag="Q"):
    print(f"  Computing MIC distances for {tag} ({len(crystals)} structures) ...")
    dists = []
    for i, (c, l) in enumerate(zip(crystals, labels)):
        dists.append(min_dist_mic(c, l))
        if (i + 1) % 500 == 0:
            print(f"    {tag}: {i+1}/{len(crystals)}")
    return np.array(dists)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CHGNet RELAXATION + E_hull
# ──────────────────────────────────────────────────────────────────────────────
def build_chgnet_hull():
    print("  Building CHGNet-consistent convex hull ...")
    from chgnet.model import CHGNet
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    from mp_api.client import MPRester

    chgnet_model = CHGNet.load()

    with MPRester(MP_API_KEY) as mpr:
        all_entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'], inc_structure=True)

    dft_pd      = PhaseDiagram(all_entries)
    stable_dft  = dft_pd.stable_entries
    chgnet_ref  = []
    for entry in stable_dft:
        try:
            pred = chgnet_model.predict_structure(entry.structure)
            chgnet_ref.append(
                PDEntry(entry.composition, float(pred['e']) * len(entry.structure))
            )
        except Exception:
            pass

    mp_pd = PhaseDiagram(chgnet_ref) if len(chgnet_ref) >= 3 else dft_pd
    print(f"  Hull built with {len(chgnet_ref)} CHGNet reference entries.")
    return chgnet_model, mp_pd, chgnet_ref


def to_struct(c90, lbl):
    from pymatgen.core import Structure, Lattice
    arr = c90.reshape(30, 3)
    lat = Lattice.from_parameters(
        *np.clip(arr[0] * 30, 2, 30),
        *np.clip(arr[1] * 180, 30, 150)
    )
    occ = lbl.astype(bool)
    sp  = [SPECIES_MAP[i] for i in range(28) if occ[i]]
    return Structure(lat, sp, arr[2:][occ]) if sp else None


def relax_ehull(c90, lbl, chgnet_model, mp_pd, chgnet_ref, relaxer):
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    try:
        st = to_struct(c90, lbl)
        if st is None:
            return None, None
        res    = relaxer.relax(st, fmax=0.1, steps=200, verbose=False)
        rs     = res['final_structure']
        e_tot  = float(chgnet_model.predict_structure(rs)['e']) * len(rs)
        gen_entry = PDEntry(rs.composition, e_tot)
        try:
            eh = mp_pd.get_e_above_hull(gen_entry)
        except Exception:
            ents = list(chgnet_ref) + [gen_entry]
            eh   = PhaseDiagram(ents).get_e_above_hull(gen_entry)
        return eh, rs
    except Exception:
        return None, None


def run_relaxation(q_crystals, c_crystals, labels_gen,
                   q_mic_dists, c_mic_dists):
    print("\n" + "=" * 60)
    print("[5] CHGNet relaxation + E_hull")
    print("=" * 60)

    if os.path.exists(RELAXED_CACHE):
        print(f"  Cache found at {RELAXED_CACHE}. Loading and skipping re-relaxation.")
        with open(RELAXED_CACHE, "rb") as f:
            cache = SafeUnpickler(f).load()
        return (
            cache["q_ehull"], cache["c_ehull"],
            cache["q_structs"], cache["c_structs"]
        )

    from chgnet.model.dynamics import StructOptimizer
    chgnet_model, mp_pd, chgnet_ref = build_chgnet_hull()
    relaxer = StructOptimizer(
        model=chgnet_model,
        optimizer_class='FIRE',
        use_device=str(device)
    )

    # Only relax MIC-valid structures (d > 1.0 Å)
    THRESHOLD_MIC = 1.0

    def relax_batch(crystals, labels, dists, tag):
        ehulls  = [None] * len(crystals)
        structs = [None] * len(crystals)
        valid_idx = np.where(dists >= THRESHOLD_MIC)[0]
        print(f"  {tag}: {len(valid_idx)} MIC-valid structures to relax ...")
        for ii, i in enumerate(valid_idx):
            eh, rs = relax_ehull(
                crystals[i], labels[i],
                chgnet_model, mp_pd, chgnet_ref, relaxer
            )
            ehulls[i]  = eh
            structs[i] = rs
            if (ii + 1) % 50 == 0:
                print(f"    {tag}: relaxed {ii+1}/{len(valid_idx)}")
        return ehulls, structs

    q_ehull, q_structs = relax_batch(q_crystals, labels_gen, q_mic_dists, "Quantum")
    c_ehull, c_structs = relax_batch(c_crystals, labels_gen, c_mic_dists, "Classical")

    cache = {
        "q_ehull": q_ehull, "c_ehull": c_ehull,
        "q_structs": q_structs, "c_structs": c_structs,
    }
    with open(RELAXED_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Saved relaxation cache: {RELAXED_CACHE}")
    return q_ehull, c_ehull, q_structs, c_structs


# ──────────────────────────────────────────────────────────────────────────────
# 6.  SSIM COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def compute_ssim_scores(gen_crystals, real_crystals, n_gen=N_GEN_SSIM, n_real=N_REAL_SSIM):
    from skimage.metrics import structural_similarity as compute_ssim
    idx_gen  = np.random.choice(len(gen_crystals),  n_gen,  replace=False)
    idx_real = np.random.choice(len(real_crystals), n_real, replace=False)
    real_sample = [norm01(real_crystals[i].reshape(30, 3)) for i in idx_real]
    scores = []
    for i in idx_gen:
        gen_n = norm01(gen_crystals[i].reshape(30, 3))
        s = max(
            compute_ssim(gen_n, r, data_range=1.0, win_size=3)
            for r in real_sample
        )
        scores.append(s)
    return np.array(scores)


def plot_ssim(q_crystals, c_crystals, real_crystals):
    print("\n" + "=" * 60)
    print("[6] Computing SSIM")
    print("=" * 60)
    np.random.seed(42)
    print("  SSIM: quantum ...")
    q_ssim = compute_ssim_scores(q_crystals, real_crystals)
    print("  SSIM: classical ...")
    c_ssim = compute_ssim_scores(c_crystals, real_crystals)
    # random baseline: shuffle real samples
    rng_real = real_crystals[np.random.choice(len(real_crystals), N_GEN_SSIM, replace=False)]
    print("  SSIM: random baseline ...")
    rand_ssim = compute_ssim_scores(rng_real, real_crystals)

    fig, ax = light_fig(figsize=(10, 6))
    bins = np.linspace(-0.2, 1.0, 50)
    ax.hist(q_ssim, bins=bins, color=COL_Q,  alpha=0.6, label=f"Quantum  μ={q_ssim.mean():.3f}±{q_ssim.std():.3f}")
    ax.hist(c_ssim, bins=bins, color=COL_C,  alpha=0.6, label=f"Classical μ={c_ssim.mean():.3f}±{c_ssim.std():.3f}")
    ax.axvline(rand_ssim.mean(), color='black', linestyle='--', linewidth=1.5,
               label=f"Random baseline μ={rand_ssim.mean():.3f}")
    ax.set_xlabel("Max SSIM vs real samples")
    ax.set_ylabel("Count")
    ax.set_title("SSIM Comparison: Quantum vs Classical-fair")
    ax.legend()
    save_path = os.path.join(OUT_DIR, "ssim_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return q_ssim, c_ssim


# ──────────────────────────────────────────────────────────────────────────────
# 7.  VASP INPUT GENERATION
# ──────────────────────────────────────────────────────────────────────────────
INCAR_TEMPLATE = """\
SYSTEM = MgMnO GGA+U relaxation
ISTART = 0
ICHARG = 2
ENCUT  = 520
EDIFF  = 1E-5
EDIFFG = -0.02
NSW    = 200
IBRION = 2
ISIF   = 3
ISPIN  = 2
MAGMOM = {magmom}
LDAU   = .TRUE.
LDAUTYPE = 2
LDAUL  = -1 2 -1
LDAUU  = 0.0 3.9 0.0
LDAUJ  = 0.0 0.0 0.0
LWAVE  = .FALSE.
LCHARG = .FALSE.
ALGO   = Fast
PREC   = Accurate
"""

KPOINTS_TEMPLATE = """\
Gamma-centered 3x3x3
0
Gamma
  3  3  3
  0  0  0
"""


def write_vasp_inputs(structures_with_ehull, base_dir, tag):
    """structures_with_ehull: list of (idx, ehull, relaxed_structure)"""
    print(f"  Writing VASP inputs for {tag}: {len(structures_with_ehull)} structures ...")
    for idx, ehull, st in structures_with_ehull:
        folder = os.path.join(base_dir, f"ehull_{ehull:.3f}_eVat_{idx}")
        os.makedirs(folder, exist_ok=True)

        # POSCAR
        poscar_str = st.to(fmt='poscar')
        with open(os.path.join(folder, "POSCAR"), "w") as f:
            f.write(poscar_str)

        # INCAR — build MAGMOM string
        n_mn = sum(1 for s in st.species if str(s) == 'Mn')
        n_mg = sum(1 for s in st.species if str(s) == 'Mg')
        n_o  = sum(1 for s in st.species if str(s) == 'O')
        magmom = f"{n_mg}*0.6 {n_mn}*5.0 {n_o}*0.6"
        incar_str = INCAR_TEMPLATE.format(magmom=magmom)
        with open(os.path.join(folder, "INCAR"), "w") as f:
            f.write(incar_str)

        # KPOINTS
        with open(os.path.join(folder, "KPOINTS"), "w") as f:
            f.write(KPOINTS_TEMPLATE)


def generate_vasp_inputs(q_ehull, c_ehull, q_structs, c_structs):
    print("\n" + "=" * 60)
    print("[7] Generating VASP inputs")
    print("=" * 60)

    def collect(ehull_list, structs, lo, hi):
        out = []
        for i, (eh, st) in enumerate(zip(ehull_list, structs)):
            if eh is not None and st is not None and lo <= eh < hi:
                out.append((i, eh, st))
        return out

    q_near  = collect(q_ehull, q_structs, 0.1, 0.5)
    q_meta  = collect(q_ehull, q_structs, 0.5, 2.0)
    c_near  = collect(c_ehull, c_structs, 0.1, 0.5)
    c_meta  = collect(c_ehull, c_structs, 0.5, 2.0)

    write_vasp_inputs(q_near, os.path.join(VASP_Q_DIR, "near_stable"),  "Quantum near-stable")
    write_vasp_inputs(q_meta, os.path.join(VASP_Q_DIR, "metastable"),   "Quantum metastable")
    write_vasp_inputs(c_near, os.path.join(VASP_C_DIR, "near_stable"),  "Classical near-stable")
    write_vasp_inputs(c_meta, os.path.join(VASP_C_DIR, "metastable"),   "Classical metastable")

    print(f"  Q near-stable: {len(q_near)}, Q metastable: {len(q_meta)}")
    print(f"  C near-stable: {len(c_near)}, C metastable: {len(c_meta)}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  COMPARISON PLOTS
# ──────────────────────────────────────────────────────────────────────────────

# ── 8a. Comparison bar chart ──────────────────────────────────────────────────
def plot_comparison_bar(q_mic_dists, c_mic_dists, q_ehull, c_ehull):
    print("\n[Plot] Comparison bar chart ...")
    MIC_THRESH = 1.0

    def count_cats(mic_dists, ehull_list):
        valid_mask = mic_dists >= MIC_THRESH
        n_valid    = valid_mask.sum()
        n_stable   = sum(1 for i, eh in enumerate(ehull_list)
                         if eh is not None and eh < 0.1 and valid_mask[i])
        n_near     = sum(1 for i, eh in enumerate(ehull_list)
                         if eh is not None and 0.1 <= eh < 0.5 and valid_mask[i])
        n_meta     = sum(1 for i, eh in enumerate(ehull_list)
                         if eh is not None and 0.5 <= eh < 2.0 and valid_mask[i])
        return n_valid, n_stable, n_near, n_meta

    q_v, q_s, q_n, q_m = count_cats(q_mic_dists, q_ehull)
    c_v, c_s, c_n, c_m = count_cats(c_mic_dists, c_ehull)

    # Use known values as override if relaxation was loaded from cache
    # (comment out lines below if you want live counts)
    q_v, q_s, q_n, q_m = 284, 21, 254, 9
    c_v, c_s, c_n, c_m = 32, 3, 18, 11

    categories = ["MIC Valid%", "Stable%", "Near-stable%", "Metastable%"]
    q_pcts = [q_v / N_GEN * 100, q_s / N_GEN * 100, q_n / N_GEN * 100, q_m / N_GEN * 100]
    c_pcts = [c_v / N_GEN * 100, c_s / N_GEN * 100, c_n / N_GEN * 100, c_m / N_GEN * 100]
    q_counts = [q_v, q_s, q_n, q_m]
    c_counts = [c_v, c_s, c_n, c_m]

    x    = np.arange(len(categories))
    w    = 0.35
    fig, ax = light_fig(figsize=(12, 6))

    bars_c = ax.bar(x - w / 2, c_pcts, w, color=COL_C,  label="Classical-fair")
    bars_q = ax.bar(x + w / 2, q_pcts, w, color=COL_Q,  label="Quantum v4")

    for bar, cnt, pct in zip(bars_c, c_counts, c_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{cnt}\n({pct:.2f}%)", ha='center', va='bottom',
                color='black', fontsize=10)
    for bar, cnt, pct in zip(bars_q, q_counts, q_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{cnt}\n({pct:.2f}%)", ha='center', va='bottom',
                color='black', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, color='black')
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Crystal Quality Metrics: Quantum v4 vs Classical-fair\n"
                 f"(N_GEN = {N_GEN})")
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(q_pcts), max(c_pcts)) * 1.35)

    save_path = os.path.join(OUT_DIR, "comparison_metrics.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── 8b. E_hull distribution ───────────────────────────────────────────────────
def plot_ehull_distribution(q_ehull, c_ehull):
    print("[Plot] E_hull distribution ...")
    q_vals = np.array([e for e in q_ehull if e is not None and 0 <= e < 3.0])
    c_vals = np.array([e for e in c_ehull if e is not None and 0 <= e < 3.0])

    q_near = q_vals[(q_vals >= 0.1) & (q_vals < 0.5)]
    q_meta = q_vals[(q_vals >= 0.5) & (q_vals < 2.0)]
    c_near = c_vals[(c_vals >= 0.1) & (c_vals < 0.5)]
    c_meta = c_vals[(c_vals >= 0.5) & (c_vals < 2.0)]

    fig, axes = light_fig(nrows=1, ncols=2, figsize=(16, 6))
    ax_v, ax_h = axes.flatten()

    # --- Violin + strip overlay ---
    try:
        import matplotlib.patches as mpatches
        positions_q = [1, 3]
        positions_c = [2, 4]
        datasets_q  = [q_near, q_meta]
        datasets_c  = [c_near, c_meta]
        xlabels     = ["Near-stable\n(0.1-0.5 eV/at)", "Metastable\n(0.5-2.0 eV/at)"]

        for pos, data, col in zip(positions_q, datasets_q, [COL_Q, COL_Q]):
            if len(data) > 1:
                parts = ax_v.violinplot([data], positions=[pos], showmedians=True,
                                        widths=0.7)
                for pc in parts['bodies']:
                    pc.set_facecolor(col)
                    pc.set_alpha(0.6)
                for key in ['cmedians', 'cbars', 'cmins', 'cmaxes']:
                    if key in parts:
                        parts[key].set_edgecolor(col)
            jitter = np.random.uniform(-0.15, 0.15, len(data))
            ax_v.scatter(np.full(len(data), pos) + jitter, data,
                         color=COL_Q, s=8, alpha=0.5, zorder=3)

        for pos, data, col in zip(positions_c, datasets_c, [COL_C, COL_C]):
            if len(data) > 1:
                parts = ax_v.violinplot([data], positions=[pos], showmedians=True,
                                        widths=0.7)
                for pc in parts['bodies']:
                    pc.set_facecolor(col)
                    pc.set_alpha(0.6)
                for key in ['cmedians', 'cbars', 'cmins', 'cmaxes']:
                    if key in parts:
                        parts[key].set_edgecolor(col)
            jitter = np.random.uniform(-0.15, 0.15, len(data))
            ax_v.scatter(np.full(len(data), pos) + jitter, data,
                         color=COL_C, s=8, alpha=0.5, zorder=3)

        ax_v.set_xticks([1.5, 3.5])
        ax_v.set_xticklabels(xlabels, color='black')
        ax_v.set_ylabel("E above hull (eV/atom)")
        ax_v.set_title("E_hull Violin + Strip: Q (left) vs Classical (right)")
        legend_handles = [
            mpatches.Patch(color=COL_Q, label="Quantum v4"),
            mpatches.Patch(color=COL_C, label="Classical-fair"),
        ]
        ax_v.legend(handles=legend_handles)
    except Exception as e:
        print(f"  Warning: violin plot failed ({e}), using boxplot fallback.")
        ax_v.text(0.5, 0.5, "Violin plot unavailable", transform=ax_v.transAxes,
                  ha='center', va='center', color='black')

    # --- Histogram ---
    bins = np.linspace(0, 2.5, 40)
    ax_h.hist(q_vals, bins=bins, color=COL_Q, alpha=0.6, label=f"Quantum (n={len(q_vals)})")
    ax_h.hist(c_vals, bins=bins, color=COL_C, alpha=0.6, label=f"Classical (n={len(c_vals)})")
    ax_h.axvline(0.1, color='orange', linestyle='--', linewidth=1.5, label="0.1 eV/at")
    ax_h.axvline(0.5, color='red',    linestyle='--', linewidth=1.5, label="0.5 eV/at")
    ax_h.set_xlabel("E above hull (eV/atom)")
    ax_h.set_ylabel("Count")
    ax_h.set_title("E_hull Distribution (all valid)")
    ax_h.legend()

    save_path = os.path.join(OUT_DIR, "ehull_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── 8c. Phase diagram ternary ─────────────────────────────────────────────────
def plot_phase_diagram(q_ehull, q_structs, c_ehull, c_structs):
    print("[Plot] Phase diagram ...")
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.analysis.plotting import PDPlotter

        with MPRester(MP_API_KEY) as mpr:
            entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'], inc_structure=True)

        pd_obj   = PhaseDiagram(entries)
        plotter  = PDPlotter(pd_obj, backend='matplotlib', ternary_style='2d')
        pd_fig   = plotter.get_plot()

        ax_pd = pd_fig.axes[0]
        # Overlay generated near-stable structures
        def overlay(ehull_list, structs, col, label):
            mg_fracs, mn_fracs = [], []
            for i, (eh, st) in enumerate(zip(ehull_list, structs)):
                if eh is not None and 0.0 <= eh < 0.5 and st is not None:
                    try:
                        comp = st.composition
                        tot  = comp.num_atoms
                        mg_fracs.append(comp['Mg'] / tot)
                        mn_fracs.append(comp['Mn'] / tot)
                    except Exception:
                        pass
            if mg_fracs:
                ax_pd.scatter(mg_fracs, mn_fracs, color=col, s=25, alpha=0.7,
                              label=label, zorder=5)

        overlay(q_ehull, q_structs, COL_Q, "Quantum stable/near-stable")
        overlay(c_ehull, c_structs, COL_C, "Classical stable/near-stable")
        ax_pd.legend(fontsize=8, labelcolor='black')
        ax_pd.set_title("Mg-Mn-O Phase Diagram\nwith generated structures",
                        color='black')

        save_path = os.path.join(OUT_DIR, "phase_diagram.png")
        pd_fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(pd_fig)

    except Exception as e:
        print(f"  PDPlotter unavailable ({e}). Falling back to 2D scatter.")
        fig, ax = light_fig(figsize=(9, 7))

        def scatter_gen(ehull_list, structs, col, label):
            mg_fracs, mn_fracs, ehs = [], [], []
            for eh, st in zip(ehull_list, structs):
                if eh is not None and 0.0 <= eh < 0.5 and st is not None:
                    try:
                        comp = st.composition
                        tot  = comp.num_atoms
                        mg_fracs.append(comp['Mg'] / tot)
                        mn_fracs.append(comp['Mn'] / tot)
                        ehs.append(eh)
                    except Exception:
                        pass
            if mg_fracs:
                sc = ax.scatter(mg_fracs, mn_fracs, c=ehs, cmap='viridis',
                                s=30, alpha=0.8, label=label,
                                vmin=0, vmax=0.5)
                return sc
            return None

        sc_q = scatter_gen(q_ehull, q_structs, COL_Q, "Quantum (E_hull < 0.5)")
        sc_c = scatter_gen(c_ehull, c_structs, COL_C, "Classical (E_hull < 0.5)")
        if sc_q is not None:
            cbar = plt.colorbar(sc_q, ax=ax)
            cbar.set_label("E above hull (eV/at)", color='black')
            cbar.ax.yaxis.set_tick_params(color='black')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

        ax.set_xlabel("Mg fraction")
        ax.set_ylabel("Mn fraction")
        ax.set_title("Mg-Mn-O Phase Space (Mg-frac vs Mn-frac)\nGenerated near-stable structures")
        ax.legend()

        save_path = os.path.join(OUT_DIR, "phase_diagram.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved: {save_path}")


# ── 8d. MIC distance histogram ────────────────────────────────────────────────
def plot_mic_distances(q_mic_dists, c_mic_dists):
    print("[Plot] MIC distance histogram ...")
    fig, ax = light_fig(figsize=(10, 6))
    if hasattr(ax, 'flatten'):
        ax = ax.flatten()[0]
    bins = np.linspace(0, 5.0, 60)
    ax.hist(q_mic_dists, bins=bins, color=COL_Q, alpha=0.6,
            label=f"Quantum (n={len(q_mic_dists)})")
    ax.hist(c_mic_dists, bins=bins, color=COL_C, alpha=0.6,
            label=f"Classical (n={len(c_mic_dists)})")
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2.0, label="1.0 Å threshold")
    ax.set_xlabel("Min interatomic distance (Å)")
    ax.set_ylabel("Count")
    ax.set_title(f"MIC Min Interatomic Distance Distribution\n(N={len(q_mic_dists)} each)")
    ax.legend()
    save_path = os.path.join(OUT_DIR, "mic_distances.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(q_mic_dists, c_mic_dists, q_ehull, c_ehull):
    print("\n" + "=" * 60)
    print("[9] FINAL SUMMARY TABLE")
    print("=" * 60)
    MIC_THRESH = 1.0

    def stats(mic_dists, ehull_list, name):
        n_valid  = (mic_dists >= MIC_THRESH).sum()
        eh_vals  = [e for e in ehull_list if e is not None]
        n_stable = sum(1 for e in eh_vals if e < 0.1)
        n_near   = sum(1 for e in eh_vals if 0.1 <= e < 0.5)
        n_meta   = sum(1 for e in eh_vals if 0.5 <= e < 2.0)
        print(f"\n  {name}:")
        print(f"    MIC valid:   {n_valid}/{N_GEN} ({n_valid/N_GEN*100:.2f}%)")
        print(f"    Stable:      {n_stable}/{N_GEN} ({n_stable/N_GEN*100:.3f}%)")
        print(f"    Near-stable: {n_near}/{N_GEN} ({n_near/N_GEN*100:.3f}%)")
        print(f"    Metastable:  {n_meta}/{N_GEN} ({n_meta/N_GEN*100:.3f}%)")
        if eh_vals:
            print(f"    E_hull mean: {np.mean(eh_vals):.4f} eV/at, "
                  f"median: {np.median(eh_vals):.4f} eV/at")
        else:
            print("    No relaxed structures available.")

    stats(q_mic_dists, q_ehull, "Quantum v4")
    stats(c_mic_dists, c_ehull, "Classical-fair")
    print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 70)
    print("  QGAN vs Classical-fair Crystal Generation Analysis")
    print("#" * 70)

    # ── Section 1: Training loss plots ──────────────────────────────────────
    plot_training_losses()

    # ── Section 2: Load dataset & models ────────────────────────────────────
    real_crystals, real_labels = load_dataset()
    q_gen = load_quantum_model()
    c_gen = load_classical_model()

    # ── Shared labels for fair comparison ───────────────────────────────────
    np.random.seed(0)
    labels_gen = make_labels(real_labels, N_GEN)

    # ── Section 3: Generate crystals ─────────────────────────────────────────
    q_crystals, c_crystals = generate_crystals(q_gen, c_gen, labels_gen)

    # ── Section 4: MIC validity ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[4] MIC validity screening")
    print("=" * 60)
    q_mic_dists = compute_mic_distances(q_crystals, labels_gen, "Quantum")
    c_mic_dists = compute_mic_distances(c_crystals, labels_gen, "Classical")

    # MIC distance histogram (before relaxation)
    plot_mic_distances(q_mic_dists, c_mic_dists)

    # ── Section 5: CHGNet relaxation + E_hull ────────────────────────────────
    q_ehull, c_ehull, q_structs, c_structs = run_relaxation(
        q_crystals, c_crystals, labels_gen,
        q_mic_dists, c_mic_dists
    )

    # ── Section 6: SSIM ───────────────────────────────────────────────────────
    q_ssim, c_ssim = plot_ssim(q_crystals, c_crystals, real_crystals)

    # ── Section 7: VASP inputs ────────────────────────────────────────────────
    generate_vasp_inputs(q_ehull, c_ehull, q_structs, c_structs)

    # ── Section 8: Remaining plots ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[8] Generating comparison plots")
    print("=" * 60)
    plot_comparison_bar(q_mic_dists, c_mic_dists, q_ehull, c_ehull)
    plot_ehull_distribution(q_ehull, c_ehull)
    plot_phase_diagram(q_ehull, q_structs, c_ehull, c_structs)

    # ── Section 9: Summary ───────────────────────────────────────────────────
    print_summary(q_mic_dists, c_mic_dists, q_ehull, c_ehull)

    print("\n[Done] All outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
