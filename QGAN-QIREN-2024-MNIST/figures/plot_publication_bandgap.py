import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ── Global Styling to Match Previous Publication Assets ──
plt.rcParams.update({
    'font.size': 14, 
    'font.family': 'sans-serif', 
    'axes.labelsize': 18,
    'axes.titlesize': 20, 
    'axes.titleweight': 'bold', 
    'axes.labelweight': 'bold',
    'legend.fontsize': 13, 
    'legend.frameon': True,
    'legend.edgecolor': 'black', 
    'figure.facecolor': 'white', 
    'axes.facecolor': 'white',
    # Match the thick border from the user's reference
    'axes.linewidth': 1.5  
})

OUT_DIR = 'results_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Create the Figure and Axes ──
fig, ax = plt.subplots(figsize=(10, 7.5))

# ── 2. Add the Target Dotted Blue Box ──
# Region: Bandgap ~1.6 to 3.0 eV, Stability ~0.0 to 0.6 eV/atom
target_box = Rectangle((1.6, 0.0), 1.4, 0.6, linewidth=2, edgecolor='steelblue', facecolor='none', linestyle=':')
ax.add_patch(target_box)

# ── 3. Plot Known Benchmarks (From Papers) ──
# Shinde et al: MgMn2O4
ax.plot(1.82, 0.60, marker='*', markersize=14, color='mediumvioletred', markeredgecolor='m', linestyle='None')
ax.annotate('Shinde et al\n(MgMn$_{2}$O$_{4}$)', xy=(1.8, 0.61), xytext=(1.2, 0.68),
            arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center', va='center', fontweight='bold', fontsize=13)

# Noh et al: Mg2MnO4, Mg2Mn3O8, Mg2MnO4 (The image points to three stars)
# Experimental: solid star
ax.plot(2.70, 0.06, marker='*', markersize=14, color='mediumvioletred', markeredgecolor='m', linestyle='None')
# Computational/Other: empty stars
ax.plot(2.90, 0.15, marker='*', markersize=14, color='none', markeredgecolor='black', linestyle='None')
ax.plot(2.95, 0.11, marker='*', markersize=14, color='none', markeredgecolor='black', linestyle='None')
ax.annotate('Noh et al\n(Mg$_{2}$MnO$_{4}$)\n(Mg$_{2}$Mn$_{3}$O$_{8}$)\n(Mg$_{2}$MnO$_{4}$)', 
            xy=(2.8, 0.13), xytext=(1.6, 0.14),
            arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center', va='center', fontweight='bold', fontsize=13)

# Zhou et al: Mg6MnO8
ax.plot(3.76, 0.03, marker='*', markersize=14, color='mediumvioletred', markeredgecolor='m', linestyle='None')
ax.annotate('Zhou et al\n(Mg$_{6}$MnO$_{8}$)', xy=(3.77, 0.04), xytext=(3.55, 0.28),
            arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center', va='center', fontweight='bold', fontsize=13)


# ── 4. Plot 'Discovered in this work' (QGAN MOCK DATA) ──
# MOCK DATA INJECTION: Replace this with df['HSE_Bandgap'] and df['Delta_G_pbx'] once DFT is finished!
np.random.seed(42)
mock_eg = np.random.uniform(0.4, 3.9, 50)
mock_pbx = np.random.uniform(0.04, 0.8, 50)

# The user's image uses red empty circles for their discoveries
ax.plot(mock_eg, mock_pbx, marker='o', markersize=8, color='none', markeredgecolor='red', markeredgewidth=2, linestyle='None')


# ── 5. Axis Formatting ──
ax.set_xlim(0.0, 4.0)
ax.set_ylim(0.0, 0.9)

# Standardize ticks
ax.set_xticks(np.arange(0.0, 4.5, 0.5))
ax.set_yticks(np.arange(0.0, 1.0, 0.1))

# Labels using LaTeX math rendering for perfect match
ax.set_xlabel('$E_g^{HSE}$ (eV)', fontsize=20, fontweight='bold', labelpad=10)
ax.set_ylabel('$\Delta G_{pbx}^{min} @ 1.5V$ (eV/atom)', fontsize=20, fontweight='bold', labelpad=10)

# ── 6. Legend ──
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markeredgecolor='red', markeredgewidth=2, markersize=10, label='Discovered in this work'),
    Line2D([0], [0], marker='*', color='w', markeredgecolor='black', markersize=14, label='Discovered in other paper'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='mediumvioletred', markeredgecolor='m', markersize=14, label='Experimentally synthesized')
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=1.0)

# Minor tick configuration to perfectly match the user's reference image
ax.minorticks_on()
ax.tick_params(which='major', length=7, width=1.5, direction='in')
ax.tick_params(which='minor', length=4, width=1, direction='in')
ax.tick_params(axis='both', which='both', top=True, right=True)

# Save the plot
final_path = os.path.join(OUT_DIR, 'publication_bandgap_pourbaix.png')
plt.tight_layout()
fig.savefig(final_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ Generated Mock HSE Bandgap vs Pourbaix plot: {final_path}")
print("❗ NOTE: The red circles currently use random placeholder values since pw.x (HSE) has not been run. Replace 'mock_eg' and 'mock_pbx' arrays with real DFT metrics when ready.")
