import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'patch.linewidth': 1.5,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

Q_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(Q_DIR, "results_analysis")

BG = "white"
TEXT = "black"
SPINE = "black"

def plot_ehull_hist():
    print("Generating E-hull grouped bar chart...")
    categories = ["Stable (<0.1)", "Near-stable (0.1-0.5)", "Metastable (0.5-2.0)"]
    
    # extracted from ablation_report.txt (counts per N_GEN=4800)
    # converting counts to percentages
    # Cls-CNN: Stable=5, Near=20, Meta=14
    c_counts = [5, 20, 14]
    # Cls-Ablation: Stable=56, Near=60, Meta=7
    a_counts = [56, 60, 7]
    # QGAN-v4: Stable=14, Near=260, Meta=9
    q_counts = [14, 260, 9]

    # Convert to log scale or display direct counts
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor("#f6f8fa")

    x = np.arange(len(categories))
    w = 0.25

    bars_c = ax.bar(x - w, c_counts, w, color="#0284c7", edgecolor=SPINE, linewidth=1.5, label="Cls-CNN (Baseline)")
    bars_a = ax.bar(x, a_counts, w, color="#94a3b8", edgecolor=SPINE, linewidth=1.5, label="Cls-Ablation (No Quantum)")
    bars_q = ax.bar(x + w, q_counts, w, color="#e65c00", edgecolor=SPINE, linewidth=1.5, label="QGAN-v4 (Quantum)")

    def add_labels(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + (max(q_counts)*0.02),
                    str(int(h)), ha='center', va='bottom', color=TEXT, fontsize=12)

    add_labels(bars_c)
    add_labels(bars_a)
    add_labels(bars_q)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=TEXT, fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Generated Structures", fontsize=12, color=TEXT, fontweight='bold')
    ax.set_title("E_hull Distribution (Stable, Near-stable, Metastable) per 4800 generations", fontsize=14, color=TEXT, fontweight='bold', pad=15)
    
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
        spine.set_linewidth(1.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=(0, (1, 3)), alpha=0.6, color='black')  # Dotted grid
    ax.set_axisbelow(True)

    ax.tick_params(colors=TEXT, width=1.5)

    ax.legend(fontsize=12, facecolor=BG, edgecolor=SPINE, labelcolor=TEXT, framealpha=1.0)
    
    # Make room for labels
    ax.set_ylim(0, max(max(c_counts), max(a_counts), max(q_counts)) * 1.2)

    path = os.path.join(OUT_DIR, "ablation_ehull_histogram.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")

if __name__ == "__main__":
    plot_ehull_hist()
