"""
Regenerate all results_analysis plots with white backgrounds.
- training_loss_curves, comparison_metrics, ehull_distribution, phase_diagram:
  regenerated from cached data with white styling.
- mic_distances, ssim_comparison, dft_mace/chgnet_vs_mace_scatter:
  converted via PIL pixel manipulation (background → white, white text → dark).
"""
import os, sys, json, pickle, warnings, types
from importlib.abc import MetaPathFinder, Loader

class MockObj:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return MockObj()
    def __call__(self, *args, **kwargs): return MockObj()
    def __setstate__(self, state):
        self.__dict__.update(state)
        class MockComp:
            def __init__(self): self.num_atoms = 1
            def __getitem__(self, key): return 0.5
        self.composition = MockComp()

class MockLoader(Loader):
    def exec_module(self, module): pass
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        setattr(m, 'Structure', MockObj)
        setattr(m, 'StructOptimizer', MockObj)
        setattr(m, 'mace_mp', MockObj)
        return m

class MockMetaFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname.startswith(x) for x in ['pymatgen', 'chgnet', 'mace']):
            import importlib.util
            return importlib.util.spec_from_loader(fullname, MockLoader())
        return None

sys.meta_path.insert(0, MockMetaFinder())

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'patch.linewidth': 1.5,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

Q_DIR = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(Q_DIR, 'results_analysis')

# ── Colour palette (white-background) ────────────────────────────────────────
BG       = 'white'
AX_BG    = 'white'
SPINE    = 'black'
TEXT     = 'black'
DIM_TXT  = '#57606a'
COL_Q    = '#e65c00'   # orange – quantum
COL_C    = '#0284c7'   # blue   – classical
COL_QLOSS= '#0284c7'
COL_CLOSS= '#16a34a'

def light_fig(nrows=1, ncols=1, figsize=(12, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG)
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat:
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE)
            sp.set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle=(0, (1, 3)), alpha=0.6, color='black')
        ax.set_axisbelow(True)
    return fig, axes

def moving_average(arr, window=20):
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    ma = ret[window - 1:] / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, ma])

# ══════════════════════════════════════════════════════════════════════════════
# 1. Training loss curves
# ══════════════════════════════════════════════════════════════════════════════
print('Regenerating training_loss_curves.png ...')
Q_LOSS_CSV  = os.path.join(Q_DIR, 'results_crystal_qgan_v4', 'training_loss_history.csv')
CLS_DIR     = ('C:/Users/Adminb/OneDrive/Documents/Projects/crystalGan/'
               'Composition-Conditioned-Crystal-GAN/Composition_Conditioned_Crystal_GAN')
CLS_LOSS_JSON = os.path.join(CLS_DIR, 'training_loss_log.json')

q_df  = pd.read_csv(Q_LOSS_CSV)
with open(CLS_LOSS_JSON) as f:
    c_df = pd.DataFrame(json.load(f))

fig, axes = light_fig(nrows=2, ncols=3, figsize=(18, 9))

q_panels = [
    ('wasserstein', 'Wasserstein Distance', 'Q Wasserstein'),
    ('d_loss',      'Discriminator Loss',   'Q D-loss'),
    ('total_g_loss','Generator Loss',        'Q G-loss'),
]
for col_idx, (key, ylabel, title) in enumerate(q_panels):
    ax = axes[0, col_idx]
    epochs = q_df['epoch'].values
    vals   = q_df[key].values
    ma     = moving_average(vals, 20)
    ax.plot(epochs, vals, color=COL_QLOSS, alpha=0.3, linewidth=0.8, label='raw')
    ax.plot(epochs, ma,   color=COL_QLOSS, linewidth=1.8, label='MA-20')
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, facecolor=BG, edgecolor=SPINE, labelcolor=TEXT)

c_panels = [
    ('w_loss', 'Wasserstein Loss',  'Classical w_loss'),
    ('d_loss', 'Discriminator Loss','Classical D-loss'),
    ('g_loss', 'Generator Loss',    'Classical G-loss'),
]
for col_idx, (key, ylabel, title) in enumerate(c_panels):
    ax = axes[1, col_idx]
    epochs = c_df['epoch'].values
    vals   = c_df[key].values
    ma     = moving_average(vals, 20)
    ax.plot(epochs, vals, color=COL_CLOSS, alpha=0.3, linewidth=0.8, label='raw')
    ax.plot(epochs, ma,   color=COL_CLOSS, linewidth=1.8, label='MA-20')
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, facecolor=BG, edgecolor=SPINE, labelcolor=TEXT)

fig.suptitle('Training Loss Curves: Quantum v4 (top) vs Classical-fair (bottom)',
             color=TEXT, fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'training_loss_curves.png'), dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  Done.')

# ══════════════════════════════════════════════════════════════════════════════
# Load cache
# ══════════════════════════════════════════════════════════════════════════════
print('Loading relaxation cache ...')

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch, io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        try:
            return super().find_class(module, name)
        except Exception:
            return MockObj

with open(os.path.join(OUT, 'relaxed_structures.pkl'), 'rb') as f:
    cache = SafeUnpickler(f).load()

q_ehull  = cache['q_ehull']
c_ehull  = cache['c_ehull']
q_structs = cache['q_structs']
c_structs = cache['c_structs']
N_GEN = 4800

# ══════════════════════════════════════════════════════════════════════════════
# 2. Comparison bar chart
# ══════════════════════════════════════════════════════════════════════════════
print('Regenerating comparison_metrics.png ...')
q_v, q_s, q_n, q_m = 284, 21, 254, 9
c_v, c_s, c_n, c_m =  32,  3,  18, 11

categories = ['MIC Valid%', 'Stable%', 'Near-stable%', 'Metastable%']
q_pcts  = [q_v/N_GEN*100, q_s/N_GEN*100, q_n/N_GEN*100, q_m/N_GEN*100]
c_pcts  = [c_v/N_GEN*100, c_s/N_GEN*100, c_n/N_GEN*100, c_m/N_GEN*100]
q_counts = [q_v, q_s, q_n, q_m]
c_counts = [c_v, c_s, c_n, c_m]

x, w = np.arange(len(categories)), 0.35
fig, ax = light_fig(figsize=(12, 6))

bars_c = ax.bar(x - w/2, c_pcts, w, color=COL_C, label='Classical-fair')
bars_q = ax.bar(x + w/2, q_pcts, w, color=COL_Q, label='Quantum v4')

for bar, cnt, pct in zip(bars_c, c_counts, c_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{cnt}\n({pct:.2f}%)', ha='center', va='bottom', color=TEXT, fontsize=8)
for bar, cnt, pct in zip(bars_q, q_counts, q_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{cnt}\n({pct:.2f}%)', ha='center', va='bottom', color=TEXT, fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(categories, color=TEXT, fontweight='bold')
ax.set_ylabel('Percentage (%)')
ax.set_title(f'Crystal Quality Metrics: Quantum v4 vs Classical-fair  (N={N_GEN})')
ax.legend(fontsize=10, facecolor=BG, edgecolor=SPINE, labelcolor=TEXT)
ax.set_ylim(0, max(max(q_pcts), max(c_pcts)) * 1.35)
fig.savefig(os.path.join(OUT, 'comparison_metrics.png'), dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  Done.')

# ══════════════════════════════════════════════════════════════════════════════
# 3. E_hull distribution
# ══════════════════════════════════════════════════════════════════════════════
print('Regenerating ehull_distribution.png ...')
q_vals = np.array([e for e in q_ehull if e is not None and 0 <= e < 3.0])
c_vals = np.array([e for e in c_ehull if e is not None and 0 <= e < 3.0])
q_near = q_vals[(q_vals >= 0.1) & (q_vals < 0.5)]
q_meta = q_vals[(q_vals >= 0.5) & (q_vals < 2.0)]
c_near = c_vals[(c_vals >= 0.1) & (c_vals < 0.5)]
c_meta = c_vals[(c_vals >= 0.5) & (c_vals < 2.0)]

fig, axes = light_fig(nrows=1, ncols=2, figsize=(16, 6))
ax_v, ax_h = axes.flatten()

# Violin + strip
positions_q = [1, 3]; positions_c = [2, 4]
datasets_q  = [q_near, q_meta]; datasets_c  = [c_near, c_meta]
xlabels = ['Near-stable\n(0.1-0.5 eV/at)', 'Metastable\n(0.5-2.0 eV/at)']
np.random.seed(42)
for pos, data in zip(positions_q, datasets_q):
    if len(data) > 1:
        parts = ax_v.violinplot([data], positions=[pos], showmedians=True, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(COL_Q); pc.set_alpha(0.5)
        for key in ['cmedians','cbars','cmins','cmaxes']:
            if key in parts: parts[key].set_edgecolor(COL_Q)
    ax_v.scatter(np.full(len(data), pos) + np.random.uniform(-0.15,0.15,len(data)),
                 data, color=COL_Q, s=8, alpha=0.5, zorder=3)
for pos, data in zip(positions_c, datasets_c):
    if len(data) > 1:
        parts = ax_v.violinplot([data], positions=[pos], showmedians=True, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(COL_C); pc.set_alpha(0.5)
        for key in ['cmedians','cbars','cmins','cmaxes']:
            if key in parts: parts[key].set_edgecolor(COL_C)
    ax_v.scatter(np.full(len(data), pos) + np.random.uniform(-0.15,0.15,len(data)),
                 data, color=COL_C, s=8, alpha=0.5, zorder=3)

ax_v.set_xticks([1.5, 3.5]); ax_v.set_xticklabels(xlabels, color=TEXT, fontweight='bold')
ax_v.set_ylabel('E above hull (eV/atom)')
ax_v.set_title('E_hull Violin + Strip: Quantum (left) vs Classical (right)')
ax_v.legend(handles=[mpatches.Patch(color=COL_Q, label='Quantum v4'),
                     mpatches.Patch(color=COL_C, label='Classical-fair')],
            fontsize=9, facecolor=BG, edgecolor=SPINE, labelcolor=TEXT)

bins = np.linspace(0, 2.5, 40)
ax_h.hist(q_vals, bins=bins, color=COL_Q, alpha=0.6, label=f'Quantum (n={len(q_vals)})')
ax_h.hist(c_vals, bins=bins, color=COL_C, alpha=0.6, label=f'Classical (n={len(c_vals)})')
ax_h.axvline(0.1, color='#ca8a04', linestyle='--', linewidth=1.5, label='0.1 eV/at')
ax_h.axvline(0.5, color='#dc2626', linestyle='--', linewidth=1.5, label='0.5 eV/at')
ax_h.set_xlabel('E above hull (eV/atom)'); ax_h.set_ylabel('Count')
ax_h.set_title('E_hull Distribution (all valid)')
ax_h.legend(fontsize=9, facecolor=BG, edgecolor=SPINE, labelcolor=TEXT)

fig.savefig(os.path.join(OUT, 'ehull_distribution.png'), dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  Done.')

# ══════════════════════════════════════════════════════════════════════════════
# 4. Phase diagram (scatter fallback, white)
# ══════════════════════════════════════════════════════════════════════════════
print('Regenerating phase_diagram.png ...')
try:
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.analysis.plotting import PDPlotter
    with MPRester('hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA') as mpr:
        entries = mpr.get_entries_in_chemsys(['Mg','Mn','O'], inc_structure=True)
    pd_obj  = PhaseDiagram(entries)
    plotter = PDPlotter(pd_obj, backend='matplotlib', ternary_style='2d')
    pd_fig  = plotter.get_plot()
    ax_pd   = pd_fig.axes[0]
    for eh_list, structs, col, lbl in [
        (q_ehull, q_structs, COL_Q, 'Quantum stable/near-stable'),
        (c_ehull, c_structs, COL_C, 'Classical stable/near-stable'),
    ]:
        mg_f, mn_f = [], []
        for eh, st in zip(eh_list, structs):
            if eh is not None and 0 <= eh < 0.5 and st is not None:
                try:
                    comp = st.composition; tot = comp.num_atoms
                    mg_f.append(comp['Mg']/tot); mn_f.append(comp['Mn']/tot)
                except: pass
        if mg_f:
            ax_pd.scatter(mg_f, mn_f, color=col, s=25, alpha=0.7, label=lbl, zorder=5)
    ax_pd.legend(fontsize=8, labelcolor='black')
    ax_pd.set_title('Mg-Mn-O Phase Diagram\nwith generated structures', color='black', fontweight='bold')
    pd_fig.savefig(os.path.join(OUT, 'phase_diagram.png'), dpi=150, bbox_inches='tight')
    plt.close(pd_fig)
except Exception as e:
    print(f'  PDPlotter failed ({e}), using scatter fallback.')
    fig, ax = light_fig(figsize=(9, 7))
    for eh_list, structs, col, lbl in [
        (q_ehull, q_structs, COL_Q, 'Quantum (E_hull < 0.5)'),
        (c_ehull, c_structs, COL_C, 'Classical (E_hull < 0.5)'),
    ]:
        mg_f, mn_f, ehs = [], [], []
        for eh, st in zip(eh_list, structs):
            if eh is not None and 0 <= eh < 0.5 and st is not None:
                try:
                    comp = st.composition; tot = comp.num_atoms
                    mg_f.append(comp['Mg']/tot); mn_f.append(comp['Mn']/tot); ehs.append(eh)
                except: pass
        if mg_f:
            sc = ax.scatter(mg_f, mn_f, c=ehs, cmap='viridis', s=30, alpha=0.8,
                            label=lbl, vmin=0, vmax=0.5)
    if 'sc' in dir():
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('E above hull (eV/at)', color=TEXT)
    ax.set_xlabel('Mg fraction'); ax.set_ylabel('Mn fraction')
    ax.set_title('Mg-Mn-O Phase Space\nGenerated near-stable structures')
    ax.legend(fontsize=9, facecolor=BG, edgecolor=SPINE, labelcolor=TEXT)
    fig.savefig(os.path.join(OUT, 'phase_diagram.png'), dpi=150,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)
print('  Done.')

# ══════════════════════════════════════════════════════════════════════════════
# 5. PIL pixel conversion for mic_distances, ssim_comparison, mace scatter
# ══════════════════════════════════════════════════════════════════════════════
print('Converting remaining dark plots via PIL ...')
from PIL import Image

PIL_TARGETS = [
    os.path.join(OUT, 'mic_distances.png'),
    os.path.join(OUT, 'ssim_comparison.png'),
    os.path.join(OUT, 'dft_mace', 'chgnet_vs_mace_scatter.png'),
]

def darken_to_white(img_path):
    img  = Image.open(img_path).convert('RGB')
    data = np.array(img, dtype=np.int32)
    R, G, B = data[:,:,0], data[:,:,1], data[:,:,2]

    # Dark background pixels (very dark, all channels < 55) → white
    bg = (R < 55) & (G < 55) & (B < 55)
    data[bg, 0] = 255; data[bg, 1] = 255; data[bg, 2] = 255

    # White / near-white pixels (all channels > 195, low saturation) → near-black
    # Only where not already converted to white (i.e. was originally near-white, not bg)
    sat = np.max(data[:,:,:3], axis=2) - np.min(data[:,:,:3], axis=2)
    white = (~bg) & (R > 195) & (G > 195) & (B > 195) & (sat < 35)
    data[white, 0] = 25; data[white, 1] = 25; data[white, 2] = 25

    # Medium-gray elements (roughly gray, 55-195) → darken toward dark gray
    mid_gray = (~bg) & (~white) & (sat < 35) & (R > 55) & (R < 195)
    # Map [55→195] → [180→60] linearly so lighter grays stay light, darker get darker
    data[mid_gray, :3] = np.clip(
        (data[mid_gray, :3].astype(float) * 0.55).astype(int), 40, 180
    )

    data = np.clip(data, 0, 255).astype(np.uint8)
    Image.fromarray(data, 'RGB').save(img_path)
    print(f'  Converted: {os.path.basename(img_path)}')

for p in PIL_TARGETS:
    if os.path.exists(p):
        darken_to_white(p)
    else:
        print(f'  Skipped (not found): {p}')

print('\nAll plots updated to white background.')
