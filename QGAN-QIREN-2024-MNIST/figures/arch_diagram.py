"""
Architecture diagrams for Classical CWGAN, Classical Ablation, and Quantum PQWGAN.
Saves to results_analysis/
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette (white background) ──
BG      = '#ffffff'
PANEL   = '#f6f8fa'
BORDER  = '#d0d7de'
WHITE   = '#1a1a2e'   # text colour (dark on white)
DIM     = '#57606a'
ACCENT1 = '#0969da'   # blue  – classical
ACCENT2 = '#1a7f37'   # green – quantum
PURPLE  = '#8250df'   # quantum circuit
ORANGE  = '#bc4c00'   # output
TEAL    = '#0a7d4b'
YELLOW  = '#9a6700'
RED     = '#cf222e'
GRAY    = '#8c959f'

def setup_axes(fig):
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 26)  # Expanded Y-axis for spacing
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(1.5)
    return ax

def draw_box(ax, x, y, w, h, label, sublabel=None,
             fc='#1f2937', ec=ACCENT1, lw=1.5, fontsize=10, radius=0.4):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f'round,pad=0.05,rounding_size={radius}',
                         fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.25, label, ha='center', va='center',
                color=WHITE, fontsize=fontsize, fontweight='bold', zorder=4)
        ax.text(x, y - 0.25, sublabel, ha='center', va='center',
                color=DIM, fontsize=8, zorder=4)
    else:
        ax.text(x, y, label, ha='center', va='center',
                color=WHITE, fontsize=fontsize, fontweight='bold', zorder=4)

def arrow(ax, x0, y0, x1, y1, color=DIM, lw=1.5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=5)

def dim_text(ax, x, y, txt, color=DIM, fontsize=8, ha='center'):
    ax.text(x, y, txt, ha=ha, va='center', color=color, fontsize=fontsize, zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING LOGIC: 1. CLASSICAL CWGAN
# ══════════════════════════════════════════════════════════════════════════════
def draw_classical_cnn():
    print("Drawing architecture_classical.png...")
    fig = plt.figure(figsize=(11, 28), facecolor=BG)
    ax = setup_axes(fig)
    CX = 5.0
    
    # Title
    ax.text(CX, 25.2, 'Classical CWGAN', ha='center', va='center', color=ACCENT1, fontsize=16, fontweight='bold')
    ax.text(CX, 24.6, 'Composition-Conditioned Crystal Generator (No Split-Head)', ha='center', va='center', color=DIM, fontsize=10)

    # Inputs
    draw_box(ax, 2.7, 23.0, 2.5, 0.9, 'Noise z', 'latent dim = 512', fc='#dbeafe', ec=ACCENT1, fontsize=10)
    draw_box(ax, 7.3, 23.0, 2.5, 0.9, 'Composition Labels', 'c₁(8) c₂(8) c₃(12) cₙ(1)', fc='#dbeafe', ec=YELLOW, fontsize=9.5)
    
    arrow(ax, 2.7, 22.55, 4.0, 21.6, ACCENT1)
    arrow(ax, 7.3, 22.55, 6.0, 21.6, YELLOW)
    dim_text(ax, CX, 21.9, 'concat  [512 + 8 + 8 + 12 + 1 = 541]')

    draw_box(ax, CX, 21.1, 4.5, 0.8, 'Linear Mapping', '541 → 128 × 28', fc='#dbeafe', ec=ACCENT1)
    arrow(ax, CX, 20.7, CX, 19.6)
    dim_text(ax, CX + 0.15, 20.3, 'Reshape  →  (B, 128, 28, 1)', ha='left')

    # ConvTranspose tower
    conv_steps = [
        ('ConvTranspose2d', '128 ch → 256 ch', 'kernel (1,3)  stride (1,1)',  19.2),
        ('BatchNorm2d + ReLU', '256 ch', '',                                    18.0),
        ('ConvTranspose2d', '256 ch → 512 ch', 'kernel (1,1)',                 16.8),
        ('BatchNorm2d + ReLU', '512 ch', '',                                    15.6),
        ('ConvTranspose2d', '512 ch → 256 ch', 'kernel (1,1)',                 14.4),
        ('BatchNorm2d + ReLU', '256 ch', '',                                    13.2),
        ('ConvTranspose2d', '256 ch → 1 ch',   'kernel (1,1)',                  12.0),
        ('Sigmoid Activation', 'Feature map  28 × 3', '',                           10.8),
    ]
    for label, sub1, sub2, cy in conv_steps:
        full_sub = sub1 + ('   ' + sub2 if sub2 else '')
        draw_box(ax, CX, cy, 5.6, 0.8, label, full_sub, fc='#dbeafe', ec=ACCENT1, fontsize=10)
        if cy < 19.2:
            arrow(ax, CX, cy + 0.8, CX, cy + 0.4)

    arrow(ax, CX, 10.4, CX, 9.4)
    # Split
    split_y = 9.4
    dim_text(ax, CX + 0.15, 9.7, 'Flatten  →  84-dim (atom positions)', ha='left')
    
    draw_box(ax, 2.7, 8.4, 3.2, 0.8, 'Atom Positions', '84-dim  (28 atoms × 3)', fc='#dcfce7', ec=ACCENT2, fontsize=10)
    draw_box(ax, 7.3, 8.4, 3.2, 0.8, 'Cell Map', 'Linear 84→30 + BN + ReLU → 6 + Sigmoid', fc='#ffedd5', ec=ORANGE, fontsize=8)
    
    arrow(ax, CX, split_y, 2.7, 8.8, ACCENT2)
    arrow(ax, CX, split_y, 7.3, 8.8, ORANGE)
    
    draw_box(ax, 2.7, 7.0, 3.2, 0.8, 'Frac. Coords', '(B, 28, 3)', fc='#dcfce7', ec=ACCENT2, fontsize=10)
    draw_box(ax, 7.3, 7.0, 3.2, 0.8, 'Lattice Params', '(B, 6)  a,b,c,α,β,γ', fc='#ffedd5', ec=ORANGE, fontsize=10)
    
    arrow(ax, 2.7, 8.0, 2.7, 7.4)
    arrow(ax, 7.3, 8.0, 7.3, 7.4)
    
    arrow(ax, 2.7, 6.6, 4.3, 5.6, ACCENT2)
    arrow(ax, 7.3, 6.6, 5.7, 5.6, ORANGE)
    dim_text(ax, CX, 5.9, 'concat')
    
    draw_box(ax, CX, 5.0, 5.0, 0.9, 'Crystal Output', '90-dim  (30 × 3)', fc='#f3e8ff', ec=ORANGE, fontsize=11, lw=2)

    # Discriminator
    ax.text(CX, 3.9, '─── Discriminator ───', ha='center', color=RED, fontsize=11, fontweight='bold')
    arrow(ax, CX, 4.55, CX, 3.35, ORANGE)
    
    draw_box(ax, CX, 3.0, 5.8, 0.7, 'Conv2d + LeakyReLU', '1 ch → 512 ch  kernel(1,3)', fc='#fee2e2', ec=RED, fontsize=9.5)
    arrow(ax, CX, 2.65, CX, 2.25)
    draw_box(ax, CX, 1.9, 5.8, 0.7, 'Conv2d × 2 + LeakyReLU', '512 → 512 → 256 ch  kernel(1,1)', fc='#fee2e2', ec=RED, fontsize=9.5)
    arrow(ax, CX, 1.55, CX, 1.15)
    draw_box(ax, CX, 0.8, 5.8, 0.7, 'Linear Layers', '1280 → 1000 → 200 → 10 (Output Logits)', fc='#fee2e2', ec=RED, fontsize=9.5)

    legend_items = [
        mpatches.Patch(fc='#dbeafe', ec=ACCENT1, label='Generator CNN layers'),
        mpatches.Patch(fc='#dcfce7', ec=ACCENT2, label='Atom processing branch'),
        mpatches.Patch(fc='#ffedd5', ec=ORANGE,  label='Cell processing branch'),
        mpatches.Patch(fc='#fee2e2', ec=RED,     label='Discriminator blocks'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=8,
              facecolor='#ffffff', edgecolor=BORDER, labelcolor=WHITE, bbox_to_anchor=(0.98, 0.98))

    fig.savefig(os.path.join(OUT_DIR, 'architecture_classical.png'), dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED DRAWING: Trunk for Quantum & Ablation
# ══════════════════════════════════════════════════════════════════════════════
def draw_trunk_and_split(ax, CX, trunk_y):
    # Deep trunk
    draw_box(ax, CX, trunk_y, 4.8, 0.7, 'Linear + LeakyReLU', '8 → 128', fc='#dcfce7', ec=ACCENT2, fontsize=10)
    arrow(ax, CX, trunk_y - 0.35, CX, trunk_y - 0.85, ACCENT2)
    draw_box(ax, CX, trunk_y - 1.2, 4.8, 0.7, 'Linear + LeakyReLU', '128 → 256', fc='#dcfce7', ec=ACCENT2, fontsize=10)
    
    split2_y = trunk_y - 2.0
    arrow(ax, CX, trunk_y - 1.55, CX, split2_y, ACCENT2)
    dim_text(ax, CX + 0.15, split2_y + 0.25, '256-dim trunk features', ha='left')
    
    # Heads
    bx1, bx2 = 2.6, 7.4
    draw_box(ax, bx1, split2_y - 0.9, 4.2, 0.9, 'Atom Head', '256 → 512 → 256 → 84  +  Sigmoid', fc='#dcfce7', ec=ACCENT2, fontsize=9.5)
    draw_box(ax, bx2, split2_y - 0.9, 4.2, 0.9, 'Cell Head', '256 → 64 → 6  +  Sigmoid', fc='#ffedd5', ec=ORANGE, fontsize=9.5)
    arrow(ax, CX, split2_y, bx1, split2_y - 0.45, ACCENT2)
    arrow(ax, CX, split2_y, bx2, split2_y - 0.45, ORANGE)
    
    out_y = split2_y - 2.3
    draw_box(ax, bx1, out_y, 4.2, 0.8, 'Frac. Coords', '84-dim (28 atoms × 3)', fc='#dcfce7', ec=ACCENT2, fontsize=10)
    draw_box(ax, bx2, out_y, 4.2, 0.8, 'Lattice Params', '6-dim (a,b,c,α,β,γ)', fc='#ffedd5', ec=ORANGE, fontsize=10)
    arrow(ax, bx1, split2_y - 1.35, bx1, out_y + 0.4, ACCENT2)
    arrow(ax, bx2, split2_y - 1.35, bx2, out_y + 0.4, ORANGE)
    
    concat2_y = out_y - 1.2
    arrow(ax, bx1, out_y - 0.4, 4.2, concat2_y + 0.45, ACCENT2)
    arrow(ax, bx2, out_y - 0.4, 5.8, concat2_y + 0.45, ORANGE)
    dim_text(ax, CX, concat2_y + 0.65, 'concat')
    
    draw_box(ax, CX, concat2_y, 5.0, 0.9, 'Crystal Output', '90-dim (30 × 3)', fc='#f3e8ff', ec=ORANGE, fontsize=11, lw=2)

    # Critic
    critic_top = concat2_y - 1.2
    ax.text(CX, critic_top, '─── Critic (Wasserstein) ───', ha='center', color=RED, fontsize=11, fontweight='bold')
    arrow(ax, CX, concat2_y - 0.45, CX, critic_top - 0.45, ORANGE)
    
    draw_box(ax, CX, critic_top - 0.8, 5.6, 0.7, 'Concat Input', '90 + 28 = 118-dim', fc='#fee2e2', ec=RED, fontsize=9.5)
    arrow(ax, CX, critic_top - 1.15, CX, critic_top - 1.5)
    draw_box(ax, CX, critic_top - 1.85, 5.6, 0.7, 'Linear + LeakyReLU', '118 → 512 → 256', fc='#fee2e2', ec=RED, fontsize=9.5)
    arrow(ax, CX, critic_top - 2.2, CX, critic_top - 2.5)
    draw_box(ax, CX, critic_top - 2.85, 5.6, 0.7, 'Linear', 'Wasserstein score (B,)', fc='#fee2e2', ec=RED, fontsize=9.5)


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING LOGIC: 2. QUANTUM PQWGAN
# ══════════════════════════════════════════════════════════════════════════════
def draw_quantum_qgan():
    print("Drawing architecture_quantum.png...")
    fig = plt.figure(figsize=(11, 28), facecolor=BG)
    ax = setup_axes(fig)
    QX = 5.0

    ax.text(QX, 25.2, 'Quantum PQWGAN (v4)', ha='center', va='center', color=ACCENT2, fontsize=16, fontweight='bold')
    ax.text(QX, 24.6, 'Hybrid Quantum-Classical Split-Head Crystal Generator (8 qubits)', ha='center', va='center', color=DIM, fontsize=10)

    draw_box(ax, 3.0, 23.0, 2.5, 0.9, 'Noise z', 'dim = 64', fc='#dbeafe', ec=ACCENT2, fontsize=10)
    draw_box(ax, 7.0, 23.0, 2.5, 0.9, 'Composition Labels', '28-dim (one-hot)', fc='#dbeafe', ec=YELLOW, fontsize=9.5)
    arrow(ax, 3.0, 22.55, 4.2, 21.6, ACCENT2)
    arrow(ax, 7.0, 22.55, 5.8, 21.6, YELLOW)
    dim_text(ax, QX, 21.9, 'concat  [64 + 28 = 92-dim]')

    draw_box(ax, QX, 21.1, 4.5, 0.8, 'Linear Mapping', '92 → 8', fc='#dcfce7', ec=ACCENT2)
    arrow(ax, QX, 20.7, QX, 19.6)

    # Hybrid Layers
    hybrid_top = 18.7
    for i in range(4):
        cy = hybrid_top - i * 2.2
        outer = FancyBboxPatch((QX - 3.8, cy - 0.95), 7.6, 1.85,
                               boxstyle='round,pad=0.05,rounding_size=0.3',
                               fc='#f0fdf4', ec=ACCENT2, lw=2.0, zorder=2,
                               linestyle='-')
        ax.add_patch(outer)
        ax.text(QX - 3.0, cy + 0.7, f'HybridLayer {i+1}', color=ACCENT2, fontsize=9, fontweight='bold', zorder=4)

        draw_box(ax, QX - 1.25, cy + 0.40, 1.8, 0.65, 'Linear', '8 → 8', fc='#dbeafe', ec=ACCENT1, fontsize=9)
        draw_box(ax, QX + 1.25, cy + 0.40, 1.8, 0.65, 'BatchNorm1d', '+ Tanh', fc='#dbeafe', ec=ACCENT1, fontsize=9)
        draw_box(ax, QX, cy - 0.45, 5.6, 0.7, 'QuantumLayer (PennyLane)', '8 qubits · RZ re-upload · StronglyEntangling(L=2) · PauliZ', fc='#ede9fe', ec=PURPLE, fontsize=9)

        arrow(ax, QX - 0.35, cy + 0.40, QX + 0.35, cy + 0.40, ACCENT1)
        arrow(ax, QX + 1.25, cy + 0.075, QX + 1.25, cy - 0.10, ACCENT1)
        arrow(ax, QX - 1.25, cy + 0.075, QX - 1.25, cy - 0.10, ACCENT1)

        if i > 0:
            arrow(ax, QX, cy + 1.25, QX, cy + 0.90, ACCENT2)

    arrow(ax, QX, hybrid_top - 3*2.2 - 0.95, QX, hybrid_top - 3*2.2 - 1.65, ACCENT2)
    dim_text(ax, QX + 0.15, hybrid_top - 3*2.2 - 1.3, '8-dim PauliZ outputs', ha='left')

    trunk_y = hybrid_top - 3*2.2 - 2.0
    draw_trunk_and_split(ax, QX, trunk_y)
    
    # Legend
    legend_items = [
        mpatches.Patch(fc='#dcfce7', ec=ACCENT2, label='Quantum/Hybrid Generator layers'),
        mpatches.Patch(fc='#ede9fe', ec=PURPLE,  label='Quantum circuit (PennyLane)'),
        mpatches.Patch(fc='#ffedd5', ec=ORANGE,  label='Cell output branch'),
        mpatches.Patch(fc='#fee2e2', ec=RED,     label='Critic (Wasserstein) layers'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=8, facecolor='#ffffff', edgecolor=BORDER, labelcolor=WHITE, bbox_to_anchor=(0.98, 0.98))
    
    fig.savefig(os.path.join(OUT_DIR, 'architecture_quantum.png'), dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING LOGIC: 3. CLASSICAL ABLATION
# ══════════════════════════════════════════════════════════════════════════════
def draw_classical_ablation():
    print("Drawing architecture_ablation.png...")
    fig = plt.figure(figsize=(11, 28), facecolor=BG)
    ax = setup_axes(fig)
    QX = 5.0

    ax.text(QX, 25.2, 'Classical Ablation (Split-Head)', ha='center', va='center', color=GRAY, fontsize=16, fontweight='bold')
    ax.text(QX, 24.6, 'Classical Variant of PQWGAN (No Quantum Components)', ha='center', va='center', color=DIM, fontsize=10)

    draw_box(ax, 3.0, 23.0, 2.5, 0.9, 'Noise z', 'dim = 64', fc='#dbeafe', ec=GRAY, fontsize=10)
    draw_box(ax, 7.0, 23.0, 2.5, 0.9, 'Composition Labels', '28-dim (one-hot)', fc='#dbeafe', ec=YELLOW, fontsize=9.5)
    arrow(ax, 3.0, 22.55, 4.2, 21.6, GRAY)
    arrow(ax, 7.0, 22.55, 5.8, 21.6, YELLOW)
    dim_text(ax, QX, 21.9, 'concat  [64 + 28 = 92-dim]')

    # Dense Layers Replacing Quantum Layers
    # First layer is 92->8, then three layers of 8->8.
    hybrid_top = 21.0
    for i in range(4):
        cy = hybrid_top - i * 1.8
        outer = FancyBboxPatch((QX - 3.5, cy - 0.75), 7.0, 1.5,
                               boxstyle='round,pad=0.05,rounding_size=0.3',
                               fc='#f8fafc', ec=GRAY, lw=2.0, zorder=2,
                               linestyle='-')
        ax.add_patch(outer)
        ax.text(QX - 2.8, cy + 0.55, f'Classical Trunk {i+1}', color=GRAY, fontsize=9, fontweight='bold', zorder=4)

        if i == 0:
            in_d, out_d = 92, 8
        else:
            in_d, out_d = 8, 8

        draw_box(ax, QX - 1.25, cy, 1.8, 0.7, 'Linear', f'{in_d} → {out_d}', fc='#dbeafe', ec=ACCENT1, fontsize=9)
        draw_box(ax, QX + 1.25, cy, 2.0, 0.7, '2-Layer MLP', 'Lin → LeakyReLU → Lin', fc='#dbeafe', ec=ACCENT1, fontsize=8)
        arrow(ax, QX - 0.35, cy, QX + 0.25, cy, ACCENT1)

        if i == 0:
            arrow(ax, QX, 21.6, QX, cy + 0.75, GRAY)
        elif i > 0:
            arrow(ax, QX, cy + 1.8 - 0.75, QX, cy + 0.75, GRAY)

    arrow(ax, QX, hybrid_top - 3*1.8 - 0.75, QX, hybrid_top - 3*1.8 - 1.55, GRAY)
    dim_text(ax, QX + 0.15, hybrid_top - 3*1.8 - 1.0, '8-dim linear representations', ha='left')

    trunk_y = hybrid_top - 3*1.8 - 1.9
    draw_trunk_and_split(ax, QX, trunk_y)
    
    # Legend
    legend_items = [
        mpatches.Patch(fc='#f1f5f9', ec=GRAY, label='Classical Dense Sequence'),
        mpatches.Patch(fc='#dcfce7', ec=ACCENT2, label='Atom processing branch'),
        mpatches.Patch(fc='#ffedd5', ec=ORANGE,  label='Cell processing branch'),
        mpatches.Patch(fc='#fee2e2', ec=RED,     label='Critic (Wasserstein) layers'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=8, facecolor='#ffffff', edgecolor=BORDER, labelcolor=WHITE, bbox_to_anchor=(0.98, 0.98))
    
    fig.savefig(os.path.join(OUT_DIR, 'architecture_ablation.png'), dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)

def run():
    draw_classical_cnn()
    draw_quantum_qgan()
    draw_classical_ablation()
    print("All architecture diagrams created successfully.")

if __name__ == "__main__":
    run()
