import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = 'results_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# Canvas Setup
FIG_WIDTH, FIG_HEIGHT = 18, 24
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
ax.set_xlim(0, 105)
ax.set_ylim(-30, 105)
ax.axis('off')
fig.patch.set_facecolor('white')

def draw_box(x, y, w, h, text, fc='#ffffff', ec='black', lw=1.5, fs=12, fw='normal', tc='black', ls='-'):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=1', 
                         facecolor=fc, edgecolor=ec, linewidth=lw, alpha=1.0, zorder=2, linestyle=ls)
    ax.add_patch(box)
    if text:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=tc, zorder=5)
    return x+w/2, y+h

def draw_arrow(x0, y0, x1, y1, ls='-', lw=1.5, color='black'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=lw, ls=ls, color=color, mutation_scale=15), zorder=4)

def draw_annotation(x, y, text, title=""):
    abox = FancyBboxPatch((x, y), 22, 6, boxstyle='round,pad=0.2,rounding_size=1', 
                         facecolor='#e3f2fd', edgecolor='#3f51b5', linewidth=2, zorder=10)
    ax.add_patch(abox)
    ax.text(x+11, y+5, title, ha='center', va='center', fontsize=13, fontweight='bold', color='#283593')
    ax.text(x+11, y+2, text, ha='center', va='center', fontsize=11, color='black')

# ── 1. Generator (Hybridren) Background ──
gen_bg = FancyBboxPatch((5, 55), 95, 45, boxstyle='round,pad=0.2,rounding_size=2', 
                        facecolor='#f1f8e9', edgecolor='#7cb342', linewidth=2, zorder=1)
ax.add_patch(gen_bg)
ax.text(52.5, 96, 'QGAN-v4 Generator (Hybridren)', ha='center', va='center', fontsize=20, fontweight='bold', bbox=dict(facecolor='#c5e1a5', edgecolor='black', boxstyle='round,pad=0.5'))

# Inputs
ax.plot(10, 85, marker='o', markersize=40, color='#b2ebf2', markeredgecolor='black', markeredgewidth=2)
ax.text(10, 85, '$Z$', ha='center', va='center', fontsize=18, fontweight='bold')
ax.text(10, 91, 'Latent Noise\n(16-dim)', ha='center', va='center', fontsize=12)

draw_arrow(13, 85, 18, 85)

# Shared Quantum Trunk
draw_box(18, 75, 30, 20, "Shared Quantum Trunk\n\n(Hybrid Layers + PennyLane QC)\n$| \psi \\rangle = \dots W^2 S(h) W^1 |0\\rangle$", fc='#ede7f6', ec='#5e35b1', fs=14, fw='bold')
draw_annotation(22, 65, "Extracts deep quantum entanglement\nacross coordinates before splitting.", "Shared Embedding")

draw_arrow(48, 85, 52, 85)

# Shared Feature Vector
draw_box(52, 82, 10, 6, "256-dim\nFeatures", fc='#fff3e0', ec='#e65100', fs=12, fw='bold')

# Branch to Split-Heads
draw_arrow(62, 85, 68, 92) # Up to Cell Head
draw_arrow(62, 85, 68, 78) # Down to Atom Head

# The Split Heads
draw_box(68, 88, 14, 8, "Cell Head\nLinear $\\rightarrow$ Sigmoid\nOutput: [6]", fc='#fff8e1', ec='#e65100', fs=13, fw='bold', ls='--')
draw_annotation(71, 98, "Predicts Lattice Metrics:\n(a, b, c, $\\alpha, \\beta, \gamma$)", "Cell Parameters")

draw_box(68, 74, 14, 8, "Atom Head\nLinear $\\rightarrow$ Sigmoid\nOutput: [84]", fc='#e8f5e9', ec='#2e7d32', fs=13, fw='bold', ls='--')
draw_annotation(65, 64, "Predicts Fractional Coords:\n(28 atoms $\\times$ XYZ)", "Atomic Coordinates")

# Explicit Concatenation Block (Question 1)
draw_arrow(82, 92, 86, 88)
draw_arrow(82, 78, 86, 82)
draw_box(86, 82, 10, 6, "Concat Block\n$\\oplus$ torch.cat", fc='#e0f7fa', ec='#1565c0', fs=12, fw='bold')

draw_arrow(91, 82, 91, 50) # Feed generated sample down to Critic area

# ── 2. Samples ──
# Fake Crystal
draw_box(84, 42, 14, 8, "Generated Crystal\nFake Sample $\\tilde{x}$", fc='#ffcdd2', ec='#c62828', fs=12, fw='bold')

# Real Crystal Database
draw_box(16, 42, 14, 8, "Training Dataset\nReal Crystal $x$", fc='#c8e6c9', ec='#2e7d32', fs=12, fw='bold')

# ── 3. WGAN-GP Interpolation Block (Question 2) ──
draw_arrow(23, 42, 45, 34) # Real going diagonal to interpolation
draw_arrow(91, 42, 65, 34) # Fake going diagonal to interpolation

draw_box(45, 28, 20, 8, "WGAN Interpolation Block\n$\\epsilon \sim U[0,1]$\n$\\hat{x} = \epsilon x + (1 - \epsilon) \\tilde{x}$", fc='#bbdefb', ec='#0277bd', fs=13, fw='bold')
draw_annotation(44, 40, "Linearly blends Real & Fake crystals\nto enforce 1-Lipschitz continuity.", "Gradient Penalty Blend")

# ── 4. Classical Critic ──
crit_bg = FancyBboxPatch((20, -25), 70, 48, boxstyle='round,pad=0.2,rounding_size=2', 
                         facecolor='#fce4ec', edgecolor='#ad1457', linewidth=2, zorder=1)
ax.add_patch(crit_bg)
ax.text(55, 19, 'Classical Critic ($D$)', ha='center', va='center', fontsize=20, fontweight='bold', bbox=dict(facecolor='#f8bbd0', edgecolor='black', boxstyle='round,pad=0.5'))

# Critic Inputs
draw_arrow(23, 42, 35, 12) # Real -> Critic
draw_arrow(91, 42, 75, 12) # Fake -> Critic
draw_arrow(55, 28, 55, 12) # Interpolated -> Critic

draw_box(30, 2, 50, 10, "Deep MLP (512 $\\rightarrow$ 256 $\\rightarrow$ 1)", fc='#ffffff', ec='#ad1457', fs=14, fw='bold')

draw_arrow(35, 2, 35, -8) # D(Real)
draw_arrow(75, 2, 75, -8) # D(Fake)
draw_arrow(55, 2, 55, -8) # D(Interpolated)

draw_box(28, -14, 14, 6, "$D(x)$", fc='#c8e6c9', ec='black', fs=14)
draw_box(68, -14, 14, 6, "$D(\\tilde{x})$", fc='#ffcdd2', ec='black', fs=14)
draw_box(48, -14, 14, 6, "$D(\\hat{x})$\nGradient Penalty", fc='#bbdefb', ec='black', fs=14)

draw_arrow(35, -14, 55, -20)
draw_arrow(75, -14, 55, -20)
draw_box(40, -24, 30, 6, "Wasserstein Loss + Physics Penalty", fc='#fff9c4', ec='#fbc02d', fs=14, fw='bold')

plt.savefig(os.path.join(OUT_DIR, 'architecture_qgan_v4_COMBINED.png'), dpi=300, bbox_inches='tight')
print("Successfully generated detailed V4 architecture diagram with Interpolation block.")
