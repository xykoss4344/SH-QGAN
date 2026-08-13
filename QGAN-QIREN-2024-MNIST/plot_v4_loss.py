import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

plt.rcParams.update({
    'font.size': 14, 
    'font.family': 'serif', 
    'axes.labelsize': 16,
    'axes.titlesize': 18, 
    'axes.titleweight': 'bold', 
    'axes.labelweight': 'bold',
    'legend.fontsize': 12, 
    'legend.frameon': True,
    'legend.edgecolor': 'black', 
    'figure.facecolor': 'white', 
    'axes.facecolor': 'white',
    'axes.linewidth': 2.0,
    'patch.linewidth': 2.0,
    'lines.linewidth': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0
})

epochs, d_loss, wasserstein, q_real, q_fake, g_loss = [], [], [], [], [], []

csv_path = os.path.join('results_crystal_qgan_v4', 'training_loss_history.csv')
if not os.path.exists(csv_path):
    print("CSV path not found.")
    exit(1)

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['epoch']))
        d_loss.append(float(row['d_loss']))
        wasserstein.append(float(row['wasserstein']))
        q_real.append(float(row['q_real_loss']))
        q_fake.append(float(row['q_fake_loss']))
        g_loss.append(float(row['total_g_loss']))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Panel 1: Critic Adversarial & Wasserstein
ax1.plot(epochs, d_loss, label='Critic (Discriminator) Loss', color='darkblue', linewidth=2.0)
ax1.plot(epochs, wasserstein, label='Wasserstein Distance', color='dodgerblue', linestyle='--', linewidth=2.0)
ax1.set_title("Adversarial Critic Convergence")
ax1.set_ylabel("Loss Value")
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel 2: Total Generator
ax2.plot(epochs, g_loss, label='Total Generator Loss', color='firebrick', linewidth=2.0)
ax2.set_title("Total Generator Convergence")
ax2.set_ylabel("Loss Value")
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

# Panel 3: Quantum Head
ax3.plot(epochs, q_real, label='Quantum Head: Real Discrepancy (Q_real)', color='purple', linewidth=2.0)
ax3.plot(epochs, q_fake, label='Quantum Head: Fake Measurement (Q_fake)', color='magenta', linestyle='--', linewidth=2.0)
ax3.set_title("Quantum Head (Q-Loss) Stabilization")
ax3.set_xlabel("Training Epochs")
ax3.set_ylabel("Quantum Divergence")
ax3.legend(loc='upper right')
ax3.grid(True, linestyle='--', alpha=0.5)

for ax in [ax1, ax2, ax3]:
    ax.minorticks_on()
    ax.tick_params(which='major', length=7, width=1.5, direction='in')
    ax.tick_params(which='minor', length=4, width=1, direction='in')
    ax.tick_params(axis='both', which='both', top=True, right=True)

out_dir = 'results_analysis'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'v4_training_loss_FINAL.png')
plt.tight_layout()
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Generated multi-pane Quantum QGAN-V4 specific loss plot: {out_path}")
