import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

OUT_DIR = 'results_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# Canvas Setup
FIG_WIDTH, FIG_HEIGHT = 18, 22
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(-10, 100)
ax.axis('off')
fig.patch.set_facecolor('#fafafa')

def draw_box(x, y, w, h, text, fc='#ffffff', ec='black', lw=1.5, fs=11, tc='black', ls='-'):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=1', 
                         facecolor=fc, edgecolor=ec, linewidth=lw, alpha=0.9, zorder=2, linestyle=ls)
    ax.add_patch(box)
    if text:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color=tc, zorder=5)
    return x+w/2, y+h

def draw_arrow(x0, y0, x1, y1, ls='-', lw=1.5, color='black'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=lw, ls=ls, color=color, mutation_scale=15), zorder=4)

def draw_annotation(x, y, text, title="Code Mapping"):
    abox = FancyBboxPatch((x, y), 22, 6, boxstyle='round,pad=0.2,rounding_size=1', 
                         facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=2, zorder=10)
    ax.add_patch(abox)
    ax.text(x+11, y+5, title, ha='center', va='center', fontsize=12, fontweight='bold', color='#f57f17')
    ax.text(x+11, y+2, text, ha='center', va='center', fontsize=10, color='black', family='monospace')

# ── 1. Generator Block ──
gen_bg = FancyBboxPatch((15, 45), 70, 25, boxstyle='round,pad=0.2,rounding_size=2', 
                        facecolor='#e2f0d9', edgecolor='#93a681', linewidth=1.5, zorder=1)
ax.add_patch(gen_bg)
ax.text(25, 68, 'Generator (Hybridren)', ha='center', va='center', fontsize=16, fontweight='bold', bbox=dict(facecolor='#ffca28', edgecolor='black'))

# Annotate Generator Level
draw_annotation(3, 73, "class Hybridren(nn.Module):\n  Sequential of HybridLayer objects", "CODE OVERVIEW")

# Inputs
ax.plot(5, 55, marker='o', markersize=35, color='#b2ebf2', markeredgecolor='black', markeredgewidth=2)
ax.text(5, 55, 'Pz', ha='center', va='center', fontsize=14)
draw_box(10, 51, 2, 8, '', fc='#ffccbc')
draw_arrow(7, 55, 10, 55); draw_arrow(12, 55, 17, 55)

# Classical Projection inside Generator
draw_box(17, 51, 4, 8, 'Linear', fc='#dcedc8', fs=10)
draw_arrow(21, 55, 23, 55)
draw_box(23, 49, 4, 12, 'Batch\nNorm', fc='#ffcdd2', fs=10)

# Annotate HybridLayer wrapping
draw_annotation(10, 36, "class HybridLayer(nn.Module):\n  self.clayer = nn.Linear(...)\n  self.norm = nn.BatchNorm1d(...)\n  self.qlayer = QuantumLayer(...)", "ACTUAL MODULE WIRING")
ax.plot([21, 21], [42, 49], 'r--', lw=2) # Pointing from annotation to clayer/bn

# ── 2. Quantum Circuit ──
qc_bg = FancyBboxPatch((32, 47), 35, 16, boxstyle='round,pad=0.2,rounding_size=1', 
                        facecolor='#f8f9f9', edgecolor='red', linewidth=1.5, linestyle='--', zorder=2)
ax.add_patch(qc_bg)
ax.text(49, 45, 'Data Re-uploading Quantum Circuit', ha='center', va='center', fontsize=12, fontweight='bold')

for y_os in [59, 57, 51]: ax.plot([27, 30, 30, 33], [55, 55, y_os, y_os], 'k-', lw=1.5)
ax.text(34, 59, '|0>', ha='center', va='center')
ax.text(34, 57, '|0>', ha='center', va='center')
ax.text(34, 51, '|0>', ha='center', va='center')
for y in [59, 57, 51]: ax.plot([35, 60], [y, y], 'k-', lw=1, zorder=1)
for i in [36, 44, 52]:
    draw_box(i, 50, 4, 10, f"$W$", fc='#d1c4e9', fs=11)
    if i < 52: draw_box(i+4.5, 50, 3, 10, "$S(h)$", fc='#c8e6c9', fs=11)
for y in [59, 57, 51]: draw_box(57, y-1.5, 3, 3, '$\lambda$', fc='#ffe082', fs=12)

# Annotate QuantumLayer internals
draw_annotation(40, 36, "class QuantumLayer(nn.Module):\n  qml.StronglyEntanglingLayers(W1)\n  qml.RZ(inputs)\n  qml.StronglyEntanglingLayers(W2)", "CIRCUIT INNARDS")
ax.plot([49, 49], [42, 47], 'r--', lw=2)

# ── 3. Reference Layers (Top Callouts) ──
param_bg = FancyBboxPatch((15, 80), 30, 16, boxstyle='round', fc='#e8eaf6', ec='black', zorder=1)
ax.add_patch(param_bg)
ax.text(30, 78, 'Parameter layer  [ qml.StronglyEntanglingLayers ]', ha='center')
enc_bg = FancyBboxPatch((50, 80), 25, 16, boxstyle='round', fc='#f1f8e9', ec='black', zorder=1)
ax.add_patch(enc_bg)
ax.text(62.5, 78, 'Encoding layer  [ qml.RZ ]', ha='center')

# Highlight Error/Deviation 1: Repetition Mapping
dev_box = FancyBboxPatch((30, 66), 40, 7, boxstyle='round', facecolor='#ffebee', edgecolor='#d32f2f', linewidth=2, zorder=10)
ax.add_patch(dev_box)
ax.text(50, 69.5, "DIFFERENCE #1: NOT CONTINUOUS RE-UPLOADING", ha='center', fontsize=12, fontweight='bold', color='#d32f2f')
ax.text(50, 67.5, "The reference image shows multiple W, S(h) blocks in ONE circuit.\nThe code actually loops multiple HybridLayers (Linear -> BN -> Circuit), \nmeaning the classical layers break up the quantum flow.", ha='center', fontsize=10)

# ── 4. Split-Head Modification ──
draw_arrow(67, 55, 71, 55)
draw_box(71, 52, 5, 6, 'Linear', fc='#dcedc8')
draw_arrow(76, 55, 78, 62)
draw_arrow(76, 55, 78, 48)
draw_box(78, 59, 14, 6, 'Atom Head (84-d)', fc='#dcfce7', ec='#1a7f37', ls='--')
draw_box(78, 45, 14, 6, 'Cell Head (6-d)', fc='#ffedd5', ec='#bc4c00', ls='--')
draw_arrow(92, 62, 95, 55)
draw_arrow(92, 48, 95, 55)
draw_box(95, 52, 3, 6, '$\\oplus$', fc='white', fs=16)

# Highlight Error/Deviation 2: Linear vs SplitHead
dev2_box = FancyBboxPatch((70, 70), 28, 8, boxstyle='round', facecolor='#ffebee', edgecolor='#d32f2f', linewidth=2, zorder=10)
ax.add_patch(dev2_box)
ax.text(84, 75, "DIFFERENCE #2: NO SPLIT-HEAD MODULE", ha='center', fontsize=11, fontweight='bold', color='#d32f2f')
ax.text(84, 72.5, "Hybridren simply outputs a flat linear layer:\n`final_linear = nn.Linear(..., out_features)`\nThere is no physical branch for Atom vs Cell inside \nmodules.py! Both are clumped into `out_features` (90D).", ha='center', fontsize=10)
ax.plot([78, 84], [65, 70], 'r--', lw=2)

draw_arrow(96.5, 52, 96.5, 30); draw_arrow(96.5, 30, 85, 30)
draw_box(65, 23, 20, 12, 'Generated Crystal', fc='none', ec='none', fs=14)
draw_box(15, 23, 20, 12, 'Real Crystal', fc='none', ec='none', fs=14)
draw_arrow(35, 28, 45, 16); draw_arrow(65, 28, 55, 16)

# ── 6. Classical Critic ──
crit_bg = FancyBboxPatch((35, -2), 30, 18, boxstyle='round', fc='#f3e5f5', ec='black', zorder=1)
ax.add_patch(crit_bg)
ax.text(50, -4, 'Classical Critic / D', ha='center', fontsize=18)
draw_arrow(35, 8, 30, 8)
draw_box(10, 5, 20, 6, 'Wasserstein distance', fc='#ffcc80', fs=14)

draw_annotation(40, -12, "class ClassicalCritic(nn.Module):\n  Sequential(Linear(118->512), LeakyReLU, ...)", "CRITIC INNARDS")

# ── Save ──
plt.savefig(os.path.join(OUT_DIR, 'architecture_qgan_annotated.png'), dpi=300, bbox_inches='tight')
print("Successfully rendered Annotated QGAN diagram.")
