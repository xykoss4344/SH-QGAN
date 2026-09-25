import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
run_final_comparison.py
=======================
Run after QGAN training completes to produce the full side-by-side comparison.

Usage:
    py -3.12 run_final_comparison.py
    py -3.12 run_final_comparison.py --qgan_checkpoint results_crystal_qgan_v2/checkpoint_490.pt

Steps:
    1. Evaluates the QGAN (results_crystal_qgan_v2/) → qgan_*.png + qgan_evaluation_summary.csv
    2. Reads classical GAN results from evaluation_summary.csv in the classical project
    3. Prints the complete side-by-side comparison table
"""

import os, sys, subprocess, argparse, glob, re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--qgan_checkpoint', type=str, default=None,
                    help='QGAN checkpoint .pt. Default: latest in results_crystal_qgan_v2/')
parser.add_argument('--n_gen', type=int, default=200)
parser.add_argument('--z_dim', type=int, default=16)
parser.add_argument('--hidden_features', type=int, default=6)
parser.add_argument('--hidden_layers',   type=int, default=2)
parser.add_argument('--spectrum_layer',  type=int, default=2)
args = parser.parse_args()

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_V2   = os.path.join(SCRIPT_DIR, "results_crystal_qgan_v2")
CLASSICAL_DIR = r"C:\Users\Adminb\OneDrive\Documents\Projects\crystalGan\Composition-Conditioned-Crystal-GAN\Composition_Conditioned_Crystal_GAN"

# ── Resolve QGAN checkpoint ───────────────────────────────────────────────────
if args.qgan_checkpoint:
    qgan_ckpt = args.qgan_checkpoint
else:
    ckpts = sorted(
        glob.glob(os.path.join(RESULTS_V2, "checkpoint_*.pt")),
        key=lambda p: int(re.search(r"checkpoint_(\d+)", os.path.basename(p)).group(1))
    )
    if not ckpts:
        print(f"ERROR: No checkpoints found in {RESULTS_V2}")
        print("Wait for QGAN training to complete and re-run.")
        sys.exit(1)
    qgan_ckpt = ckpts[-1]
    print(f"Using latest QGAN checkpoint: {os.path.basename(qgan_ckpt)}")

# ── Run QGAN evaluate.py ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running QGAN evaluation...")
print("="*60)
result = subprocess.run(
    [
        sys.executable, os.path.join(SCRIPT_DIR, "evaluate.py"),
        "--checkpoint", qgan_ckpt,
        "--n_gen", str(args.n_gen),
        "--z_dim", str(args.z_dim),
        "--hidden_features", str(args.hidden_features),
        "--hidden_layers",   str(args.hidden_layers),
        "--spectrum_layer",  str(args.spectrum_layer),
    ],
    cwd=SCRIPT_DIR,
    capture_output=False,
)
if result.returncode != 0:
    print("QGAN evaluation failed.")
    sys.exit(1)

# ── Load both CSVs and merge ──────────────────────────────────────────────────
print("\n" + "="*60)
print("Building comparison table...")
print("="*60)

qgan_csv     = os.path.join(SCRIPT_DIR, "qgan_evaluation_summary.csv")
classical_csv = os.path.join(CLASSICAL_DIR, "evaluation_summary.csv")

qgan_df     = pd.read_csv(qgan_csv,     index_col="Metric")
classical_df = pd.read_csv(classical_csv, index_col="Metric")

# Pull numbers from each
comparison = pd.DataFrame({
    "Classical GAN (retrained)": classical_df["Classical GAN"],
    "QINR-QGAN (fixed)":        qgan_df["QINR-QGAN"],
})

print("\n" + comparison.to_string())

out_path = os.path.join(SCRIPT_DIR, "final_comparison.csv")
comparison.to_csv(out_path)
print(f"\nSaved → {out_path}")

# ── Plain-text report ─────────────────────────────────────────────────────────
report = [
    "QINR-QGAN vs Classical GAN — Final Comparison",
    "="*60,
    f"QGAN checkpoint : {os.path.basename(qgan_ckpt)}",
    f"Dataset         : mgmno_100_aug.pickle (10,400 samples, 100/composition)",
    "",
    comparison.to_string(),
    "",
    "Notes:",
    "  - Both models trained 500 epochs on identical fresh-augmented dataset",
    "  - QGAN uses fixed architecture: Sigmoid output + 0.01×Q-Head in D-loss + LR decay",
    "  - Structural Validity threshold: min interatomic distance >= 1.0 A",
    "  - StructureMatcher: ltol=0.3, stol=0.5, angle_tol=10 (identical for both)",
]
with open(os.path.join(SCRIPT_DIR, "final_comparison.txt"), "w") as f:
    f.write("\n".join(report))
print("Saved → final_comparison.txt")
print("="*60)
