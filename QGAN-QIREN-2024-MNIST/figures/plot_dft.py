import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,
    'lines.linewidth': 1.5,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

OUT = 'results_analysis/dft_mace'
json_path = os.path.join(OUT, 'mace_results.json')

with open(json_path, 'r') as f:
    results = json.load(f)

valid = [r for r in results if 'error' not in r and r['mace_ehull_meV'] is not None]
chg_vals  = [r['chgnet_ehull_meV'] for r in valid]
mace_vals = [r['mace_ehull_meV']   for r in valid]
labels    = [r['formula']           for r in valid]

fig, ax = plt.subplots(figsize=(8, 7))
ax.grid(True, linestyle=':', alpha=0.6, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

sc = ax.scatter(chg_vals, mace_vals, c='#f78166', s=120, zorder=5, edgecolors='black', linewidths=0.8)
for x, y, lbl in zip(chg_vals, mace_vals, labels):
    ax.annotate(lbl, (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8, color='black', alpha=0.7)

all_v = chg_vals + mace_vals
lo, hi = min(all_v) - 10, max(all_v) + 10
ax.plot([lo, hi], [lo, hi], '--', color='blue', linewidth=1.5, label='Parity', zorder=3)
ax.axhline(100, color='gray', linewidth=0.8, linestyle=':')
ax.axvline(100, color='gray', linewidth=0.8, linestyle=':')
ax.fill_between([lo, 100], lo, 100, alpha=0.1, color='green')
ax.text(lo + 5, 95, 'stable\n< 100 meV/at', color='green', fontsize=8, va='top')

ax.set_xlabel('CHGNet  E_above_hull  (meV/at)')
ax.set_ylabel('MACE-MP-0  E_above_hull  (meV/at)')
ax.set_title('CHGNet vs MACE-MP-0 Cross-Validation\nTop-10 Quantum v4 Near-Stable Structures')
ax.legend(loc='lower right')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

plot_path = os.path.join(OUT, 'chgnet_vs_mace_scatter.png')
fig.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved natively styled plot to {plot_path}")
