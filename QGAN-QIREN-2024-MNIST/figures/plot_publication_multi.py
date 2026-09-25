import os
import sys
import pickle
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.core import Composition
from mp_api.client import MPRester
from PIL import Image

plt.rcParams.update({
    'font.size': 16, 'font.family': 'serif', 'axes.labelsize': 18,
    'axes.titlesize': 24, 'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'legend.fontsize': 14, 'legend.title_fontsize': 16, 'legend.frameon': True,
    'legend.edgecolor': 'black', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.linewidth': 2.0, 'patch.linewidth': 2.0, 'lines.linewidth': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 2.0, 'ytick.major.width': 2.0
})

CACHE_FILE = 'results_eval_ablation/ablation_report.pkl'
OUT_DIR = 'results_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch, io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

with open(CACHE_FILE, 'rb') as f:
    cache = SafeUnpickler(f).load()
q_ehull = cache['q_ehull']
q_structs = cache['q_structs']
a_ehull = cache['abl_ehull']
a_structs = cache['abl_structs']
c_ehull = cache['cls_ehull']
c_structs = cache['cls_structs']

def get_native_coords(comp, pd_obj):
    els = pd_obj.elements
    fracs = [comp.get_atomic_fraction(el) for el in els]
    x = 0.5 * (2 * fracs[2] + fracs[1])
    y = (math.sqrt(3) / 2) * fracs[1]
    return x, y

print("Fetching MP Phase Diagram...")
with MPRester('hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA') as mpr:
    all_entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'], inc_structure=False)

# ── 1. Phase Diagram (Holistic) ──
dft_pd_b = PhaseDiagram(all_entries)
plotter_b = PDPlotter(dft_pd_b, show_unstable=0)
ax_b = plotter_b.get_contour_pd_plot()

for txt in ax_b.texts[:]:
    t = txt.get_text().strip()
    if t in ['Mg', 'Mn', 'O', 'O$_{2}$']:
        txt.set_fontsize(36)
        txt.set_fontweight('bold')
        txt.set_clip_on(False)
    else:
        txt.set_visible(False)

for coll in ax_b.collections:
    try:
        coll.set_cmap('YlGnBu_r')
    except Exception: pass

for line in ax_b.lines:
    if line.get_marker() not in ['None', '', ' ']:
        line.set_markerfacecolor('white')
        line.set_markeredgecolor('black')
        line.set_markersize(8)
        line.set_zorder(5)

fig_b = ax_b.figure
fig_b.set_size_inches(10, 8)
ax_b.set_title('')

def plot_pd_points(structs, col, z):
    xs, ys = [], []
    for st in structs:
        if st is not None:
            x, y = get_native_coords(st.composition, dft_pd_b)
            xs.append(x); ys.append(y)
    ax_b.plot(xs, ys, marker='o', color=col, markersize=5, alpha=0.8, markeredgecolor='black', markeredgewidth=0.5, linestyle='None', zorder=z)

plot_pd_points(c_structs, 'green', 6)
plot_pd_points(a_structs, 'magenta', 7)
plot_pd_points(q_structs, 'red', 8)

ax_b.text(0.5, -0.05, 'Mg-Mn-O Phase Composition Space', ha='center', va='center', transform=ax_b.transAxes, fontsize=16, fontweight='bold')

try:
    cbar_ax_b = fig_b.axes[-1]
    cbar_ax_b.set_ylabel('Formation Energy ($E_f$) [eV/atom]', fontsize=14, fontweight='bold')
except Exception: pass

from matplotlib.lines import Line2D
custom_b = [
    Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='white', markersize=10, label='Stable MP Phases'),
    Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='red', markersize=8, label='QGAN-v4'),
    Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='magenta', markersize=8, label='Classical Ablation'),
    Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='green', markersize=8, label='Classical CNN')
]
ax_b.legend(handles=custom_b, loc='lower center', bbox_to_anchor=(0.5, 1.05), title='Phase Diagram Legend', title_fontsize=14, fontsize=12, frameon=True, framealpha=0.9, edgecolor='black')
# Force Matplotlib's tight_layout bounding box to include significant left-margin by plotting a completely transparent boundary anchor
ax_b.plot([-0.15], [-0.05], marker='.', color='white', alpha=0.0)
ax_b.text(-0.08, 1.15, '(a)', transform=ax_b.transAxes, fontsize=36, fontweight='bold', va='top', ha='right')

fig_b.savefig(os.path.join(OUT_DIR, 'panel_pd.png'), dpi=300, bbox_inches='tight', pad_inches=0.4)
plt.close(fig_b)

# ── 2. Data Munging for Scatter ──
data = []
def append_data(ehull, structs, tname):
    for eh, st in zip(ehull, structs):
        if eh is not None and st is not None:
            data.append({'Formula': st.composition.reduced_formula, 'Energy (meV/atom)': float(eh) * 1000, 'Type': tname})

append_data(c_ehull, c_structs, 'Classical CNN Baseline')
append_data(a_ehull, a_structs, 'Classical Ablation')
append_data(q_ehull, q_structs, 'Quantum GAN-v4')

df = pd.DataFrame(data)
top_formulas = df['Formula'].value_counts().index[:12]
df_top = df[df['Formula'].isin(top_formulas)]

df_top = df_top[df_top['Energy (meV/atom)'] >= -10]
df_top = df_top[df_top['Energy (meV/atom)'] <= 300]

# ── 3. Quantum-Only Scatter Plot ──
print("Generating Scatter Plot...")
fig_c, ax_c = plt.subplots(figsize=(14, 6.5))
sns.stripplot(data=df_top, x='Formula', y='Energy (meV/atom)', hue='Type',
              palette={'Quantum GAN-v4': 'red', 'Classical Ablation': 'magenta', 'Classical CNN Baseline': 'green'},
              marker='o', size=7, alpha=0.7, ax=ax_c, order=top_formulas, jitter=0.25, dodge=True)

ax_c.set_title('')
ax_c.axhline(0, color='red', linestyle='-', linewidth=1.5)
ax_c.axhline(80, color='red', linestyle='--', linewidth=1.5)
ax_c.axhline(120, color='darkorange', linestyle='--', linewidth=1.5)
ax_c.set_ylim(-10, 250)

# Safer alignment for thresholds
ax_c.text(0.1, 5, 'Convex Hull ($E_{hull} = 0$)', color='red', fontsize=14, fontweight='bold', ha='left')
ax_c.text(0.1, 84, r'Threshold ($E_{hull} \leq 80$)', color='red', fontsize=14, fontweight='bold', ha='left')
ax_c.text(0.1, 124, r'Metastable threshold ($E_{hull} \leq 120$)', color='darkorange', fontsize=14, fontweight='bold', ha='left')

ax_c.set_ylabel('Energy Above Convex Hull\n$E_{hull}$ (meV/atom)', fontsize=20, fontweight='bold', labelpad=15)
ax_c.set_xlabel('', fontsize=20, fontweight='bold', labelpad=15)
plt.setp(ax_c.get_xticklabels(), rotation=45, ha='right', fontsize=16, fontweight='bold')
plt.setp(ax_c.get_yticklabels(), fontsize=16)

legend_elements_c = [
    Line2D([0], [0], marker='o', color='w', label='Quantum GAN-v4', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Classical Ablation', markerfacecolor='magenta', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Classical CNN Baseline', markerfacecolor='green', markersize=10)
]
ax_c.legend(handles=legend_elements_c, loc='upper right', title="Data Legend", fontsize=14, title_fontsize=16, frameon=True, framealpha=0.9, edgecolor='black')
ax_c.text(-0.08, 1.15, '(b)', transform=ax_c.transAxes, fontsize=36, fontweight='bold', va='top', ha='right')
fig_c.tight_layout()
fig_c.savefig(os.path.join(OUT_DIR, 'panel_scatter.png'), dpi=300, facecolor='white')
plt.close(fig_c)

# ── Stitching Panels (Vertical 2-Panel) ──
print("Stitching panels together...")
img_pd = Image.open(os.path.join(OUT_DIR, 'panel_pd.png'))
img_sc = Image.open(os.path.join(OUT_DIR, 'panel_scatter.png'))

# Ensure identical widths conceptually
target_width = 1600
img_pd = img_pd.resize((target_width, int(img_pd.height * target_width / img_pd.width)))
img_sc = img_sc.resize((target_width, int(img_sc.height * target_width / img_sc.width)))

final_h = img_pd.height + img_sc.height
final_img = Image.new('RGB', (target_width, final_h), 'white')
final_img.paste(img_pd, (0, 0))
final_img.paste(img_sc, (0, img_pd.height))

final_path = os.path.join(OUT_DIR, 'publication_multipanel_figure_FINAL.png')
final_img.save(final_path, quality=100)
print(f"Successfully generated clean 2-panel figure: {final_path}")
