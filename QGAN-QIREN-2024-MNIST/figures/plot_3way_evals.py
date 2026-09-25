import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14, 'font.family': 'serif', 'axes.labelsize': 15,
    'axes.titlesize': 16, 'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'axes.linewidth': 2.0, 'lines.linewidth': 2.5, 'legend.fontsize': 14,
    'legend.frameon': True, 'legend.edgecolor': 'black', 'legend.facecolor': 'white',
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'xtick.major.width': 2.0,
    'ytick.major.width': 2.0, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'xtick.direction': 'in', 'ytick.direction': 'in'
})

def light_fig(nrows=1, ncols=1, figsize=(12, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in axes_flat:
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return fig, axes

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch, io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

with open('results_eval_ablation/ablation_report.pkl', 'rb') as f:
    cache = SafeUnpickler(f).load()

q_ehull = cache['q_ehull']
a_ehull = cache['abl_ehull']
c_ehull = cache['cls_ehull']

COL_Q = "red"
COL_A = "magenta"
COL_C = "green"

def valid_ehull(eh): return np.array([e for e in eh if e is not None and not np.isnan(e) and 0 <= e < 3.0])

# ── E_HULL DISTRIBUTION ──
q_vals = valid_ehull(q_ehull)
a_vals = valid_ehull(a_ehull)
c_vals = valid_ehull(c_ehull)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax_v, ax_h = axes
for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Histogram
bins = np.linspace(0, 2.5, 40)
ax_h.hist(q_vals, bins=bins, color=COL_Q, alpha=0.4, label=f"Quantum GAN-v4 (n={len(q_vals)})")
ax_h.hist(a_vals, bins=bins, color=COL_A, alpha=0.5, label=f"Classical Ablation (n={len(a_vals)})")
ax_h.hist(c_vals, bins=bins, color=COL_C, alpha=0.6, label=f"Classical CNN Baseline (n={len(c_vals)})")
ax_h.axvline(0.08, color='red', linestyle='--', linewidth=1.5, label="0.08 eV/at (Stable)")
ax_h.axvline(0.12, color='darkorange', linestyle='--', linewidth=1.5, label="0.12 eV/at (Metastable)")
ax_h.set_xlabel("Energy Above Hull (eV/atom)")
ax_h.set_ylabel("Count")
ax_h.set_title("E_hull Distribution (All Valid)")
ax_h.legend()

# Violin
def get_cats(vals): return vals[(vals >= 0.08) & (vals < 0.12)], vals[(vals >= 0.12) & (vals < 2.0)]
q_near, q_meta = get_cats(q_vals)
a_near, a_meta = get_cats(a_vals)
c_near, c_meta = get_cats(c_vals)

positions_c = [1, 5]
positions_a = [2, 6]
positions_q = [3, 7]

import matplotlib.patches as mpatches
for pos, data, col in zip(positions_c, [c_near, c_meta], [COL_C, COL_C]):
    if len(data) > 1:
        parts = ax_v.violinplot([data], positions=[pos], showmedians=True, widths=0.8)
        for pc in parts['bodies']: pc.set_facecolor(col); pc.set_alpha(0.5)
    ax_v.scatter(np.full(len(data), pos) + np.random.uniform(-0.15, 0.15, len(data)), data, color=col, s=8, alpha=0.5)

for pos, data, col in zip(positions_a, [a_near, a_meta], [COL_A, COL_A]):
    if len(data) > 1:
        parts = ax_v.violinplot([data], positions=[pos], showmedians=True, widths=0.8)
        for pc in parts['bodies']: pc.set_facecolor(col); pc.set_alpha(0.5)
    ax_v.scatter(np.full(len(data), pos) + np.random.uniform(-0.15, 0.15, len(data)), data, color=col, s=8, alpha=0.5)

for pos, data, col in zip(positions_q, [q_near, q_meta], [COL_Q, COL_Q]):
    if len(data) > 1:
        parts = ax_v.violinplot([data], positions=[pos], showmedians=True, widths=0.8)
        for pc in parts['bodies']: pc.set_facecolor(col); pc.set_alpha(0.5)
    ax_v.scatter(np.full(len(data), pos) + np.random.uniform(-0.15, 0.15, len(data)), data, color=col, s=8, alpha=0.5)

ax_v.set_xticks([2, 6])
ax_v.set_xticklabels(["Near-stable\n(0.08-0.12 eV/at)", "Metastable\n(0.12-2.0 eV/at)"], color='black', fontsize=14)
ax_v.set_ylabel("Energy Above Hull (eV/atom)")
ax_v.set_title("E_hull Violin & Strip: Cls-CNN (L), Ablation (M), QGAN-v4 (R)")
ax_v.legend(handles=[mpatches.Patch(color=COL_C, label="Classical CNN Baseline"), mpatches.Patch(color=COL_A, label="Classical Ablation"), mpatches.Patch(color=COL_Q, label="Quantum GAN-v4")])

fig.savefig("results_analysis/3way_ehull_distribution.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("Saved 3way_ehull_distribution.png")
