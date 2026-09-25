import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14, 'font.family': 'serif', 'axes.linewidth': 2.0, 
    'patch.linewidth': 2.0, 'lines.linewidth': 2.5, 'axes.labelweight': 'bold', 
    'axes.titleweight': 'bold', 'xtick.direction': 'in', 'ytick.direction': 'in', 
    'xtick.major.width': 2.0, 'ytick.major.width': 2.0,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'legend.edgecolor': 'black', 'legend.framealpha': 1.0
})

Q_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(Q_DIR, "results_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Light theme colors for research paper
BG = "white"
AXES_BG = "white"       # Clean white paper feel
SPINE = "black"         # Solid black borders
TEXT = "black"          # Standard black text

COL_C = "#0284c7"       # Muted blue for Cls-CNN (Classical benchmark)
COL_A = "#94a3b8"       # Slate gray for Cls-Ablation (Classical no-quantum)
COL_Q = "#e65c00"       # Professional orange for QGAN-v4

N_GEN = 4800

def light_fig(nrows=1, ncols=1, figsize=(12, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG)
    if not isinstance(axes, (list, np.ndarray)):
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.set_facecolor(AXES_BG)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)
            spine.set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle=(0, (1, 3)), alpha=0.6, color='black')  # Dotted grid
        ax.set_axisbelow(True)
    return fig, axes

def plot_ablation_comparison():
    print("Generating three-way ablation comparison chart (White theme)...")
    categories = ["MIC Valid%", "Stable%\n(<= 80 meV/at)", "Near-stable%\n(80-120 meV/at)", "Metastable%\n(120-2000 meV/at)"]
    
    # Counts from ablation_report.txt
    c_counts = [40, 2, 3, 21]
    a_counts = [123, 36, 41, 38]
    q_counts = [283, 18, 50, 215]

    c_pcts = [x / N_GEN * 100 for x in c_counts]
    a_pcts = [x / N_GEN * 100 for x in a_counts]
    q_pcts = [x / N_GEN * 100 for x in q_counts]

    x = np.arange(len(categories))
    w = 0.25
    fig, ax = light_fig(figsize=(14, 7))

    bars_c = ax.bar(x - w, c_pcts, w, color=COL_C, edgecolor=SPINE, linewidth=1.5, label="Cls-CNN (Baseline)")
    bars_a = ax.bar(x, a_pcts, w, color=COL_A, edgecolor=SPINE, linewidth=1.5, label="Cls-Ablation (Split-Head, No Quantum)")
    bars_q = ax.bar(x + w, q_pcts, w, color=COL_Q, edgecolor=SPINE, linewidth=1.5, label="QGAN-v4 (Split-Head + Quantum)")

    def add_labels(bars, counts, pcts):
        for bar, cnt, pct in zip(bars, counts, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{cnt}\n({pct:.2f}%)", ha='center', va='bottom',
                    color=TEXT, fontsize=14)

    add_labels(bars_c, c_counts, c_pcts)
    add_labels(bars_a, a_counts, a_pcts)
    add_labels(bars_q, q_counts, q_pcts)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=TEXT, fontsize=14, fontweight='bold')
    ax.set_ylabel("Percentage of generated structures (%)", fontsize=15, fontweight='bold')
    ax.set_title(f"Ablation Study: Crystal Quality Metrics (N_GEN = {N_GEN})", fontsize=18, color=TEXT, fontweight='bold', pad=15)
    ax.legend(fontsize=14, labelcolor=TEXT, facecolor=BG, edgecolor=SPINE, framealpha=1.0)
    ax.set_ylim(0, max(max(q_pcts), max(a_pcts), max(c_pcts)) * 1.35)

    save_path = os.path.join(OUT_DIR, "ablation_metrics.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_ablation_comparison()
