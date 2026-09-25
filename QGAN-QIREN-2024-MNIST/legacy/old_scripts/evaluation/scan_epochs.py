import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
scan_epochs.py  (QINR-QGAN version)
=====================================
Batch E_hull pre-screen across multiple QINR-QGAN checkpoints.
Mirrors crystalGAN/scan_epochs.py for 1:1 comparison.

By default scans all checkpoint_*.pt files found in results_crystal_qgan/
and reports the validity % for each, then identifies the best epoch.

Usage:
    py -3.12 scan_epochs.py
    py -3.12 scan_epochs.py --results_dir ./results_crystal_qgan
    py -3.12 scan_epochs.py --epochs 100 200 300 400 500

Output:
    Prints validity table to stdout.
    Writes best_epoch.txt with the epoch number that had highest validity.
"""

import subprocess, sys, os, glob, re, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='./results_crystal_qgan',
                    help='Directory containing checkpoint_N.pt files')
parser.add_argument('--epochs', type=int, nargs='*', default=None,
                    help='Specific epoch numbers to scan. '
                         'Default: auto-discover all checkpoints.')
parser.add_argument('--n_gen',   type=int, default=200,
                    help='Structures to generate per checkpoint (passed to quick_ehull_qgan.py)')
parser.add_argument('--z_dim',           type=int,   default=16)
parser.add_argument('--hidden_features', type=int,   default=6)
parser.add_argument('--hidden_layers',   type=int,   default=2)
parser.add_argument('--spectrum_layer',  type=int,   default=2)
parser.add_argument('--use_noise',       type=float, default=0.0)
args = parser.parse_args()

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.abspath(args.results_dir)
EHULL_SCRIPT = os.path.join(SCRIPT_DIR, "quick_ehull_qgan.py")

# ── Discover checkpoints ──────────────────────────────────────────────────────
if args.epochs:
    checkpoints = {}
    for ep in args.epochs:
        p = os.path.join(RESULTS_DIR, f"checkpoint_{ep}.pt")
        if os.path.isfile(p):
            checkpoints[ep] = p
        else:
            print(f"  WARNING: checkpoint_{ep}.pt not found, skipping.")
else:
    all_ckpts = sorted(
        glob.glob(os.path.join(RESULTS_DIR, "checkpoint_*.pt")),
        key=lambda p: int(re.search(r"checkpoint_(\d+)", os.path.basename(p)).group(1))
    )
    checkpoints = {}
    for p in all_ckpts:
        ep = int(re.search(r"checkpoint_(\d+)", os.path.basename(p)).group(1))
        checkpoints[ep] = p

if not checkpoints:
    print(f"ERROR: No checkpoints found in {RESULTS_DIR}")
    sys.exit(1)

print(f"Scanning {len(checkpoints)} checkpoints in {RESULTS_DIR}")
print(f"Using quick_ehull_qgan.py  (n_gen={args.n_gen})")
print()

# ── Run quick_ehull_qgan.py for each checkpoint ───────────────────────────────
results = {}
for ep in sorted(checkpoints.keys()):
    ckpt_path = checkpoints[ep]
    cmd = [
        sys.executable, EHULL_SCRIPT,
        "--checkpoint",      ckpt_path,
        "--n_gen",           str(args.n_gen),
        "--z_dim",           str(args.z_dim),
        "--hidden_features", str(args.hidden_features),
        "--hidden_layers",   str(args.hidden_layers),
        "--spectrum_layer",  str(args.spectrum_layer),
        "--use_noise",       str(args.use_noise),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        validity_pct = None
        mean_dist    = None
        for line in r.stdout.splitlines():
            if "Valid" in line and "%" in line:
                validity_pct = float(line.split("(")[1].split("%")[0])
            if "Mean min" in line:
                mean_dist = float(line.split(":")[1].strip().split()[0])
        results[ep] = {"validity": validity_pct, "mean_dist": mean_dist}
        if validity_pct is not None:
            print(f"  Epoch {ep:>4d}: {validity_pct:.1f}%  (mean dist {mean_dist:.3f} A)")
        else:
            print(f"  Epoch {ep:>4d}: ERROR — could not parse output")
            if r.stderr:
                print(f"    stderr: {r.stderr[:200]}")
    except subprocess.TimeoutExpired:
        results[ep] = {"validity": None, "mean_dist": None}
        print(f"  Epoch {ep:>4d}: TIMEOUT")
    except Exception as e:
        results[ep] = {"validity": None, "mean_dist": None}
        print(f"  Epoch {ep:>4d}: EXCEPTION — {e}")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("Epoch | Validity%  | Mean min dist (A)")
print("-" * 42)
for ep in sorted(results.keys()):
    v = results[ep]["validity"]
    d = results[ep]["mean_dist"]
    v_str = f"{v:.1f}%" if v is not None else "  --  "
    d_str = f"{d:.3f}"  if d is not None else "  --  "
    print(f"  {ep:>4d}  |   {v_str:<8} |  {d_str}")

valid_epochs = {ep: r["validity"] for ep, r in results.items() if r["validity"] is not None}
if valid_epochs:
    best_ep = max(valid_epochs, key=lambda k: valid_epochs[k])
    print(f"\nBest epoch: {best_ep}  ({valid_epochs[best_ep]:.1f}%  validity)")
    with open("best_epoch.txt", "w") as f:
        f.write(str(best_ep))
    print("Saved → best_epoch.txt")
else:
    print("\nNo valid results found.")
