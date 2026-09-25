import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = 'results_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Canvas ──
FIG_WIDTH, FIG_HEIGHT = 16, 12
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
ax.set_xlim(0, 160)
ax.set_ylim(-10, 110)
ax.axis('off')
fig.patch.set_facecolor('#ffffff')

# ── Helpers ──
def draw_box(x, y, w, h, text, fc='#ffffff', ec='none', lw=1.5, fs=10, fw='normal', tc='w', ls='-', z=3):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.1,rounding_size=1.5',
                         facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(box)
    if text: ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=tc, zorder=z+2)

def arrow(x0, y0, x1, y1, color='#2c3e50', lw=2.0, ls='-', z=2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=lw, ls=ls, color=color, mutation_scale=15), zorder=z)

def line(pts, col='#2c3e50', lw=2.0, ls='-', z=2):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=z, solid_capstyle='round')

# ── 1. Generator ──
gx, gy, gw, gh = 52, 65, 80, 24
draw_box(gx, gy, gw, gh, '', fc='#f5fafe', ec='#cbd5e1', lw=2, z=1)
ax.text(gx - gw/2 + 2, gy + gh/2 - 2, 'Generator (Hybrid Layer block)', ha='left', va='top', fontsize=16, fontweight='bold', color='#1e293b', zorder=5)

# Input
ax.text(5, 65, 'z', ha='center', va='center', fontsize=16, fontweight='bold', color='#1e293b')
draw_box(5, 59, 6, 3, 'P(z)', fc='#dbeafe', tc='#1e293b')
arrow(7, 65, 14, 65)

# Classical Layers
draw_box(18, 65, 6, 8, 'Linear', fc='#818cf8')
arrow(21, 65, 23, 65)
draw_box(26, 65, 6, 12, 'Batch\nNorm', fc='#fb7185')

# Quantum Circuit
qcx, qcy, qcw, qch = 56, 62, 50, 15
draw_box(qcx, qcy, qcw, qch, '', fc='#faf5ff', ec='#c084fc', lw=2, ls='--', z=2)
ax.text(qcx, qcy + qch/2 - 2, 'Quantum Layer (spectrum_layer=1)', ha='center', va='top', fontsize=12, fontweight='bold', color='#7e22ce', zorder=5)

# QC innards (wires)
for y in [63, 60, 57]: line([(32, y), (80, y)], col='#94a3b8', lw=1.5)
ax.text(32, 63, '|0⟩ ', ha='right', va='center', fontsize=10, color='#333333')
ax.text(32, 60, '|0⟩ ', ha='right', va='center', fontsize=10, color='#333333')
ax.text(32, 57, '|0⟩ ', ha='right', va='center', fontsize=10, color='#333333')

# QC Boxes (just W1, S(h), W2)
draw_box(42, 60, 6, 10, '$W^1$', fc='#a78bfa', fs=12)
draw_box(56, 60, 6, 10, '$S(h)$', fc='#34d399', fs=12)
draw_box(70, 60, 6, 10, '$W^2$', fc='#a78bfa', fs=12)

# Explicit Re-uploading routing into S(h)
line([(29, 65), (56, 65)], col='#64748b', lw=1.5)
arrow(56, 65, 56, 65+0.1, color='#64748b', lw=1.5) # Arrow pointing down into S(h) top
# Let's make it more explicit: from BN, up out of generator block then down into S(h)
line([(29, 68), (32, 68), (32, 75), (56, 75)], col='#10b981', lw=2)
arrow(56, 75, 56, 65, color='#10b981', lw=2)
ax.text(44, 76.5, 'Data Re-uploading (h)', fontsize=10, color='#10b981', fontweight='bold', ha='center',
            bbox=dict(facecolor='#f5fafe', edgecolor='#10b981', boxstyle='round,pad=0.3'))

# Measurement
for y in [63, 60, 57]: draw_box(77, y, 3, 2.5, r'$\lambda$', fc='#fbbf24', tc='#593d04')

# ── 2. Top Callouts ──
# Parameter Layer
px, py = 40, 95
draw_box(px, py, 26, 14, '', fc='#fefce8', ec='#fde047', lw=1.5, z=1)
ax.text(px, py+5, 'Parameter Layer', fontweight='bold', color='#a16207')
for y, i in zip([94, 91, 88], [1, 2, 'n']): 
    line([(30, y), (49, y)], col='#94a3b8', lw=1.2)
    draw_box(40, y, 14, 2.5, f'$Rot(\\alpha_{i}, \\beta_{i}, \\gamma_{i})$', fc='white', ec='#fde047', tc='#3f3f46', fs=9, lw=1)
    ax.text(29, y, f'$|\\phi_{i}⟩$', ha='right', va='center', fontsize=9)
line([(45, 94), (45, 88)], col='#94a3b8', lw=1.5)
ax.plot(45, 94, 'o', color='#94a3b8'); ax.plot(45, 91, 'o', color='#94a3b8'); ax.plot(45, 88, 'o', color='#94a3b8')
ax.text(51, 91, '×K', fontweight='bold', fontsize=12, color='#a16207')

# Encoding Layer
ex, ey = 75, 95
draw_box(ex, ey, 24, 14, '', fc='#f0fdf4', ec='#86efac', lw=1.5, z=1)
ax.text(ex, ey+5, 'Encoding Layer', fontweight='bold', color='#166534')
for y, i in zip([94, 91, 88], [1, 2, 'n']): 
    line([(65, y), (85, y)], col='#94a3b8', lw=1.2)
    draw_box(75, y, 10, 2.5, f'$R_z(h_{i})$', fc='white', ec='#86efac', tc='#3f3f46', fs=9, lw=1)
    ax.text(64, y, f'$|\\phi_{i}⟩$', ha='right', va='center', fontsize=9)

# Connect callouts
line([(px, 88), (42, 65)], col='#d1d5db', ls='--', lw=1.5)
line([(ex, 88), (56, 65)], col='#d1d5db', ls='--', lw=1.5)

# ── 3. Split Head ──
arrow(81, 65, 85, 65)
draw_box(88, 65, 6, 8, 'Linear', fc='#818cf8')
line([(91, 65), (94, 65)], col='#2c3e50', lw=2)
line([(94, 72), (94, 58)], col='#2c3e50', lw=2)
arrow(94, 72, 98, 72); arrow(94, 58, 98, 58)

draw_box(108, 72, 20, 7, 'Atom Head (84)', fc='#dcfce7', tc='#14532d', ec='#16a34a', lw=1.5, fw='bold')
draw_box(108, 58, 20, 7, 'Cell Head (6)', fc='#ffedd5', tc='#7c2d12', ec='#ea580c', lw=1.5, fw='bold')

arrow(118, 72, 122, 72); arrow(118, 58, 122, 58)
line([(122, 72), (122, 58)], col='#2c3e50', lw=2)
arrow(122, 65, 126, 65)

# ── 4. Outputs & Data Elements ──
draw_box(134, 65, 16, 12, 'Generated\nCrystal Sample\n$(Mg, Mn, O)$', fc='#f1f5f9', tc='#0f172a', ec='#94a3b8', lw=1.5)
draw_box(134, 30, 16, 12, 'Real\nCrystal Sample\n(Database)', fc='#f1f5f9', tc='#0f172a', ec='#94a3b8', lw=1.5)

line([(142, 65), (148, 65)], col='#2c3e50')
line([(148, 65), (148, 50)], col='#2c3e50')
line([(142, 30), (148, 30)], col='#2c3e50')
line([(148, 30), (148, 45)], col='#2c3e50')

arrow(148, 50, 148, 47); arrow(148, 45, 148, 47)

# ── 5. Classical Critic ──
cx, cy, cw, ch = 148, 15, 20, 60
draw_box(120, 10, 56, 18, '', fc='#fdf4ff', ec='#d946ef', lw=2, z=1)
ax.text(120, 15, 'Classical Critic / D', fontweight='bold', fontsize=14, color='#701a75', zorder=5)

arrow(148, 22, 148, 18)
draw_box(140, 6, 4, 12, 'FC', fc='#e879f9', fs=9)
draw_box(130, 6, 4, 10, 'FC', fc='#c084fc', fs=9)
draw_box(120, 6, 4, 8, 'FC', fc='#a78bfa', fs=9)
draw_box(110, 6, 4, 4, 'FC', fc='#818cf8', fs=9)

arrow(148, 12, 142, 12)
line([(148, 6), (142, 6)]); line([(148, 0), (142, 0)])

arrow(138, 6, 132, 6); arrow(128, 6, 122, 6); arrow(118, 6, 112, 6)
arrow(108, 6, 92, 6)
draw_box(72, 6, 40, 8, 'Wasserstein Distance Loss', fc='#ffedd5', tc='#9a3412', ec='#f97316', lw=1.5, fs=14, fw='bold')

# ── 6. Updates Loop ──
line([(52, 6), (5, 6)], ls='--', col='#ef4444', lw=2)
line([(5, 6), (5, 65)], ls='--', col='#ef4444', lw=2)
arrow(5, 65, 15, 65, ls='--', color='#ef4444', lw=2)
ax.text(25, 10, 'Generator Updates (Backprop)', color='#ef4444', fontweight='bold', fontsize=12)

# Critic Updates
line([(92, 3), (95, 3)], ls='--', col='#ef4444', lw=2)
line([(95, 3), (95, -5)], ls='--', col='#ef4444', lw=2)
line([(95, -5), (148, -5)], ls='--', col='#ef4444', lw=2)
arrow(148, -5, 148, -1, ls='--', color='#ef4444', lw=2)
ax.text(120, -8, 'Critic Updates', color='#ef4444', fontweight='bold', fontsize=12, ha='center')

# Header
ax.text(160, 105, 'QINR-QGAN Minimal PPT Schematic (V4 corrected)', ha='right', va='top', fontsize=16, fontweight='bold', color='#1e293b')

plt.savefig(os.path.join(OUT_DIR, 'architecture_qgan_ppt.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Successfully generated corrected PPT schematic.")
