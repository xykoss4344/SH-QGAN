import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter
from mp_api.client import MPRester

# IEEE Styling
plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif', 'axes.labelsize': 13,
    'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'legend.fontsize': 10, 'legend.frameon': True, 'legend.edgecolor': 'black',
    'figure.facecolor': 'white', 'axes.facecolor': 'white'
})

# Path definitions
CACHE_FILE = 'results_analysis/relaxed_structures.pkl'
OUT_DIR = 'results_analysis'

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch, io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

print("Loading cached structures...")
with open(CACHE_FILE, 'rb') as f:
    cache = SafeUnpickler(f).load()

q_ehull, c_ehull = cache['q_ehull'], cache['c_ehull']
q_structs, c_structs = cache['q_structs'], cache['c_structs']

# --- PART 1: TERNARY PHASE DIAGRAM (Background + Contours) ---
print("Fetching Materials Project Phase Diagram...")
with MPRester('hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA') as mpr:
    all_entries = mpr.get_entries_in_chemsys(['Mg', 'Mn', 'O'], inc_structure=False)

dft_pd = PhaseDiagram(all_entries)
plotter = PDPlotter(dft_pd, show_unstable=0)

try:
    ax = plotter.get_contour_pd_plot()
    fig = ax.figure
    fig.set_size_inches(8, 7)
    fig.savefig(os.path.join(OUT_DIR, 'publication_ternary_contour.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved ternary background contour diagram.")
except Exception as e:
    print(f"Warning: Could not create contour pd plot due to: {e}")

# --- PART 2: STOICHIOMETRY SCATTER/STRIP PLOT ---
print("Extracting data for stoichiometry energy plots...")

data = []
# Process Classical
for eh, st in zip(c_ehull, c_structs):
    if eh is not None and st is not None:
        try:
            val = float(eh) * 1000  # Convert to meV
            if -10 <= val <= 300: # Limit Y-axis scope for clear visualization
                data.append({
                    'Formula': st.composition.reduced_formula,
                    'Energy (meV/atom)': val,
                    'Type': 'Classical'
                })
        except: pass

# Process Quantum
for eh, st in zip(q_ehull, q_structs):
    if eh is not None and st is not None:
        try:
            val = float(eh) * 1000  # Convert to meV
            if -10 <= val <= 300:
                data.append({
                    'Formula': st.composition.reduced_formula,
                    'Energy (meV/atom)': val,
                    'Type': 'Quantum'
                })
        except: pass

df = pd.DataFrame(data)

# Sort formulas by number of occurrences so top formulas appear first
top_formulas = df['Formula'].value_counts().index[:15]
df_top = df[df['Formula'].isin(top_formulas)]

fig2, (ax_c, ax_q) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Scatter properties matched exactly to requested reference
# (c) Quantum
sns.stripplot(data=df_top[df_top['Type']=='Quantum'], x='Formula', y='Energy (meV/atom)', 
              color='blue', marker='o', size=5, alpha=0.7, ax=ax_q, order=top_formulas, jitter=False)

# (d) Classical
sns.stripplot(data=df_top[df_top['Type']=='Classical'], x='Formula', y='Energy (meV/atom)', 
              color='red', marker='x', size=6, alpha=0.7, ax=ax_c, order=top_formulas, jitter=False)

for ax, title in zip([ax_q, ax_c], ['(c) Quantum GAN Generated', '(d) Classical GAN Generated']):
    ax.set_title(title, loc='left', fontsize=16)
    ax.axhline(0, color='red', linestyle='-', linewidth=0.8)
    ax.axhline(80, color='red', linestyle='--', linewidth=0.8) # Reference hull line
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_ylim(-10, 250)
    ax.set_ylabel('Energy above convex hull\n(meV/atom)', fontsize=12)

ax_q.set_xlabel('')
ax_c.set_xlabel('')
plt.xticks(rotation=45, ha='right')

fig2.tight_layout()
out_scatter = os.path.join(OUT_DIR, 'publication_ehull_scatter.png')
fig2.savefig(out_scatter, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved formula scatter plots: {out_scatter}")

print("All plots generated successfully.")
