import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IEEE Styling
plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif', 'axes.labelsize': 13,
    'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'legend.fontsize': 10, 'legend.frameon': True, 'legend.edgecolor': 'black',
    'figure.facecolor': 'white', 'axes.facecolor': 'white'
})

COL_Q = '#1f77b4' # Blue
COL_C = '#d62728' # Red

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch, io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

with open('results_analysis/relaxed_structures.pkl', 'rb') as f:
    cache = SafeUnpickler(f).load()
q_ehull, c_ehull = cache['q_ehull'], cache['c_ehull']
q_structs, c_structs = cache['q_structs'], cache['c_structs']

fig, ax = plt.subplots(figsize=(9, 7))
ax.grid(True, linestyle=':', alpha=0.6, color='gray')

for eh_list, structs, col, lbl, mkr, sz, zr in [
    (c_ehull, c_structs, COL_C, 'Classical (E_hull < 0.5)', '^', 60, 4), # c below q
    (q_ehull, q_structs, COL_Q, 'Quantum (E_hull < 0.5)', 'o', 40, 5),   # q on top
]:
    mg_f, mn_f = [], []
    for eh, st in zip(eh_list, structs):
        if eh is not None and 0 <= eh < 0.5 and st is not None:
            comp = st.composition
            tot = comp.num_atoms
            mg_f.append(comp['Mg']/tot)
            mn_f.append(comp['Mn']/tot)
    if mg_f:
        ax.scatter(mg_f, mn_f, color=col, s=sz, alpha=0.85, label=lbl, zorder=zr, marker=mkr, edgecolors='black', linewidths=0.6)

ax.set_xlabel('Mg fraction')
ax.set_ylabel('Mn fraction')
ax.set_title('Mg-Mn-O Phase Space\nGenerated near-stable structures', color='black', fontweight='bold')
ax.legend(fontsize=10, labelcolor='black', facecolor='white', edgecolor='black', loc='best')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for sp in ax.spines.values():
    if sp.get_visible():
        sp.set_edgecolor('black')
        sp.set_linewidth(1.5)

fig.tight_layout()
fig.savefig('results_analysis/phase_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print('Saved native white phase diagram as a 2D scatter!')
