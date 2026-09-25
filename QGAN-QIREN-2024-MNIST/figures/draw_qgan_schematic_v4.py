import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = 'results_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# Canvas Setup
FIG_WIDTH, FIG_HEIGHT = 16, 22
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
ax.set_xlim(0, 105)
ax.set_ylim(-25, 100) # Shifted down to make space
ax.axis('off')
fig.patch.set_facecolor('white')

def draw_box(x, y, w, h, text, fc='#ffffff', ec='black', lw=1.5, fs=11, fw='normal', tc='black', alpha=1.0, ls='-'):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=1', 
                         facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=2, linestyle=ls)
    ax.add_patch(box)
    if text:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=tc, zorder=5)
    return x+w/2, y+h

def draw_arrow(x0, y0, x1, y1, ls='-', lw=1.5, color='black'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=lw, ls=ls, color=color, mutation_scale=15), zorder=4)

# ── 1. Generator Box ──
gen_bg = FancyBboxPatch((15, 45), 75, 25, boxstyle='round,pad=0.2,rounding_size=2', 
                        facecolor='#e2f0d9', edgecolor='#93a681', linewidth=1.5, zorder=1)
ax.add_patch(gen_bg)
ax.text(25, 68, 'Generator', ha='center', va='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='#ffca28', edgecolor='black'))

# Inputs
ax.plot(5, 55, marker='o', markersize=35, color='#b2ebf2', markeredgecolor='black', markeredgewidth=2)
ax.text(5, 55, 'Pz', ha='center', va='center', fontsize=14)
ax.text(5, 60, 'Latent\nspace', ha='center', va='center', fontsize=14)

draw_box(10, 51, 2, 8, '', fc='#ffccbc')
ax.text(11, 48, 'z ~ Pz', ha='center', va='center')
ax.text(11, 61, 'Latent\nvector', ha='center', va='center')
draw_arrow(7, 55, 10, 55)
draw_arrow(12, 55, 17, 55)

# Classical Projection inside Generator
draw_box(17, 51, 4, 8, 'Linear', fc='#dcedc8', fs=10)
draw_arrow(21, 55, 23, 55)
draw_box(23, 49, 4, 12, 'Batch\nNorm', fc='#ffcdd2', fs=10)

# ── 2. Quantum Circuit ──
qc_bg = FancyBboxPatch((32, 47), 44, 16, boxstyle='round,pad=0.2,rounding_size=1', 
                        facecolor='#f8f9f9', edgecolor='red', linewidth=1.5, linestyle='--', zorder=2)
ax.add_patch(qc_bg)
ax.text(54, 45, 'Data Re-uploading Quantum Circuit', ha='center', va='center', fontsize=14, fontweight='bold')

ax.text(34, 59, '|0>', ha='center', va='center', fontsize=12)
ax.text(34, 57, '|0>', ha='center', va='center', fontsize=12)
ax.text(34, 54, '...', ha='center', va='center', fontsize=14)
ax.text(34, 51, '|0>', ha='center', va='center', fontsize=12)
for y in [59, 57, 51]: ax.plot([35, 76], [y, y], 'k-', lw=1, zorder=1)

# Sequence
x_pos = [36, 41.5, 47, 52.5, 58, 63, 68.5, 74]
labels = ["$W^1$", "$S(h)$", "$W^2$", "$S(h)$", "...", "$W^{L-1}$", "$S(h)$", "$W^L$"]
is_w = [True, False, True, False, False, True, False, True]

for x, lbl, w in zip(x_pos, labels, is_w):
    if lbl == "...":
        ax.text(x, 55, '$\dots$', fontsize=20, ha='center', va='center')
        continue
    if w:
        draw_box(x, 50, 4, 10, lbl, fc='#d1c4e9', fs=11)
    else:
        draw_box(x, 50, 3.5, 10, lbl, fc='#c8e6c9', fs=11)

# Wiring BN to QC
ax.plot([27, 30, 30, 70.25], [55, 55, 66, 66], 'k-', lw=1.5)
ax.text(28.5, 60, 'Re-uploading', ha='center', va='center', fontsize=9, rotation=90, bbox=dict(facecolor='white', edgecolor='none'))
for i in [1, 3, 6]:
    sh_x = x_pos[i] + 1.75 
    draw_arrow(sh_x, 66, sh_x, 60.1, color='black', lw=1.5)

for y in [59, 57, 51]: draw_box(79, y-1.5, 3, 3, '$\lambda$', fc='#ffe082', fs=12)

# ── 3. Reference Layers (Top Callouts) ──
param_bg = FancyBboxPatch((12, 80), 36, 16, boxstyle='round,pad=0.2,rounding_size=1', fc='#e8eaf6', ec='black', zorder=1)
ax.add_patch(param_bg)
ax.text(30, 78, 'Parameter layer', ha='center', va='center', fontweight='bold', fontsize=16)
for idx, y in enumerate([93, 88.5, 84]):
    if idx == 2: ax.text(15, y, f'$|\\phi_{{n}}>$', ha='center', va='center', fontsize=14)
    else: ax.text(15, y, f'$|\\phi_{idx+1}>$', ha='center', va='center', fontsize=14)
    num = 'n' if idx == 2 else str(idx+1)
    draw_box(22, y-2.25, 12, 4.5, rf'$Rot(\alpha_{num}, \beta_{num}, \dots)$', fc='white', fs=13)
    ax.plot([16, 22], [y, y], 'k-'); ax.plot([34, 45], [y, y], 'k-')
ax.plot([36, 36], [84, 93], 'k-')
ax.plot([36], [93], 'ko', markersize=8); ax.plot([36], [88.5], 'ko', markersize=8)
ax.plot([40, 40], [84, 88.5], 'k-'); ax.plot([40], [88.5], 'ko', markersize=8)
ax.text(44, 88.5, r'$\times K$', fontsize=18)

enc_bg = FancyBboxPatch((55, 80), 28, 16, boxstyle='round,pad=0.2,rounding_size=1', fc='#f1f8e9', ec='black', zorder=1)
ax.add_patch(enc_bg)
ax.text(69, 78, 'Encoding layer', ha='center', va='center', fontweight='bold', fontsize=16)
for idx, y in enumerate([93, 88.5, 84]):
    if idx == 2: ax.text(58, y, f'$|\\phi_{{n}}>$', ha='center', va='center', fontsize=14)
    else: ax.text(58, y, f'$|\\phi_{idx+1}>$', ha='center', va='center', fontsize=14)
    num = 'n' if idx == 2 else str(idx+1)
    draw_box(63.5, y-2.25, 10, 4.5, rf'$R_z(h_{num})$', fc='white', fs=16)
    ax.plot([59, 63.5], [y, y], 'k-'); ax.plot([73.5, 80], [y, y], 'k-')

ax.plot([30, 40], [80, 60], 'k--', lw=1.5); ax.plot([69, 56], [80, 60], 'k--', lw=1.5)

# ── 4. Split-Head Modification ──
draw_arrow(82, 55, 84, 55)
draw_box(84, 52, 4, 6, 'Linear', fc='#dcedc8', fs=9)
draw_arrow(88, 55, 89, 62) 
draw_arrow(88, 55, 89, 48) 

draw_box(89, 59, 11, 6, 'Atom Head (84)', fc='#dcfce7', ec='#1a7f37', ls='--')
draw_box(89, 45, 11, 6, 'Cell Head (6)', fc='#ffedd5', ec='#bc4c00', ls='--')

draw_arrow(100, 62, 102, 55)
draw_arrow(100, 48, 102, 55)
draw_box(102, 52, 2.5, 6, '$\\oplus$', fc='white', fs=16)

draw_arrow(103.25, 52, 103.25, 30)
draw_arrow(103.25, 30, 95, 30)

# ── 5. Samples and Data Flow ──
draw_box(75, 23, 20, 12, 'Generated Crystal Sample\n'+r'$\tilde{x} = (Atom_{84}, Cell_{6})$', fc='none', ec='none', fs=14)
img_path = os.path.join(OUT_DIR, 'publication_ternary_contour.png')
if os.path.exists(img_path):
    ax.imshow(plt.imread(img_path), extent=[77, 95, 20, 36], zorder=1, alpha=0.3)

draw_box(20, 23, 20, 12, 'Real Crystal Sample\n'+r'$x$ database', fc='none', ec='none', fs=14)
if os.path.exists(img_path):
    ax.imshow(plt.imread(img_path), extent=[22, 40, 20, 36], zorder=1, alpha=0.3)

# ── NEW: WGAN-GP Interpolation Block ──
# Drew directly beneath Real and Fake crystals
draw_box(41, 10, 33, 8, "WGAN Interpolation (Gradient Penalty)\n" + r"$\hat{x} = \alpha x + (1 - \alpha) \tilde{x}$", fc='#e3f2fd', ec='#1565c0', fs=13, lw=2)

draw_arrow(35, 20, 45, 18, ls='--') # Real into Interpolation
draw_arrow(80, 20, 70, 18, ls='--') # Fake into Interpolation

# Arrows from Real, Fake, and Interpolation into Critic
draw_arrow(35, 20, 45, -2, lw=1.5) # Real to Critic
draw_arrow(80, 20, 65, -2, lw=1.5) # Fake to Critic
draw_arrow(56.5, 10, 55, -2, lw=1.5, ls='--' , color='#1565c0') # Interpolation to Critic

# ── 6. Classical Critic (Shifted Down!) ──
crit_bg = FancyBboxPatch((40, -18), 30, 16, boxstyle='round,pad=0.2,rounding_size=1', fc='#f3e5f5', ec='black', zorder=1)
ax.add_patch(crit_bg)
ax.text(55, -20, 'Classical Critic', ha='center', va='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='#ffca28', edgecolor='black'))

# Adjust Neural Net nodes inside Critic down by 16 units
y_1 = [-4, -6, -8, -10, -12]; y_2 = [-5, -7, -9, -11]; y_3 = [-6, -8, -10]; y_4 = [-8]
x_n = [45, 52, 59, 65]
for n1 in y_1:
    for n2 in y_2: ax.plot([x_n[0], x_n[1]], [n1, n2], 'k-', zorder=2, alpha=0.5)
for n2 in y_2:
    for n3 in y_3: ax.plot([x_n[1], x_n[2]], [n2, n3], 'k-', zorder=2, alpha=0.5)
for n3 in y_3:
    for n4 in y_4: ax.plot([x_n[2], x_n[3]], [n3, n4], 'k-', zorder=2, alpha=0.5)

for y in y_1: ax.plot(x_n[0], y, 'o', color='#81c784', markersize=14, markeredgecolor='k', zorder=3)
for y in y_2: ax.plot(x_n[1], y, 'o', color='#ffb74d', markersize=14, markeredgecolor='k', zorder=3)
for y in y_3: ax.plot(x_n[2], y, 'o', color='#64b5f6', markersize=14, markeredgecolor='k', zorder=3)
ax.plot(x_n[3], y_4[0], 'o', color='#ba68c8', markersize=14, markeredgecolor='k', zorder=3)

# ── 7. Feedback Loops (Shifted Down!) ──
draw_arrow(40, -8, 35, -8)
draw_box(15, -11, 20, 6, 'Wasserstein distance\n& Gradient Penalty', fc='#ffcc80', fs=12)

ax.plot([10, 15], [-8, -8], 'k--', lw=2.5)
ax.plot([10, 10], [-8, 77], 'k--', lw=2.5)
draw_arrow(10, 77, 45, 77, ls='--', lw=2.5) 
ax.text(15, 79, 'Updates', fontsize=14, fontweight='bold')

ax.plot([25, 25], [-11, -24], 'k--', lw=2.5)
ax.plot([25, 85], [-24, -24], 'k--', lw=2.5)
ax.plot([85, 85], [-24, -8], 'k--', lw=2.5)
draw_arrow(85, -8, 70, -8, ls='--', lw=2.5)
ax.text(55, -26, 'Updates', fontsize=14, fontweight='bold', ha='center')

plt.savefig(os.path.join(OUT_DIR, 'architecture_qgan_schematic_v4_FINAL.png'), dpi=300, bbox_inches='tight')
print("Successfully rendered precise styled QGAN diagram matching the requested reference format.")
