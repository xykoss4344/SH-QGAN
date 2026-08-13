import os, sys, pickle, random
sys.path.insert(0,'.')
sys.path.insert(0,'datasets')
import torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.QINR_Crystal import PQWGAN_CC_Crystal
from view_atoms_mgmno import view_atoms
from skimage.metrics import structural_similarity as ssim

device = torch.device('cpu')
Z_DIM, LABEL_DIM, DATA_DIM, NUM = 64, 28, 90, 50

with open('datasets/mgmno_mp.pickle','rb') as f: raw_data = pickle.load(f)
labels, real_atoms = [], []
for idx in random.sample(range(len(raw_data)), len(raw_data)):
    c, l = raw_data[idx]
    labels.append(np.array(l).flatten())
    try:
        a, _ = view_atoms(np.array(c).flatten(), view=False)
        real_atoms.append(a)
    except: pass
while len(labels) < NUM: labels.append(labels[0])
labels_t = torch.tensor(np.array(labels[:NUM], dtype=np.float32)).to(device)

CKPT = 'results_crystal_qgan_v4/checkpoint_450.pt'
cd = torch.load(CKPT, map_location=device)
gan = PQWGAN_CC_Crystal(input_dim_g=92, output_dim=90, input_dim_d=118, hidden_features=8, hidden_layers=3, spectrum_layer=1, use_noise=0.0)
gan.generator.load_state_dict(cd['generator_state_dict'] if 'generator_state_dict' in cd else cd['generator'])
gan.generator.eval()

with torch.no_grad():
    z = torch.randn(NUM, Z_DIM).to(device)
    fake_flat = gan.generator(torch.cat([z, labels_t], dim=1)).cpu().numpy()

gen_atoms = []
for img in fake_flat:
    try:
        a, _ = view_atoms(img, view=False)
        gen_atoms.append(a)
    except: pass

def dm(atoms):
    d = atoms.get_all_distances(mic=True)
    m = np.zeros((30,30)); n=min(30,d.shape[0]); m[:n,:n]=d[:n,:n]; return m
scores = []
for g in gen_atoms:
    try:
        scores.append(ssim(dm(g), dm(random.choice(real_atoms)), data_range=25.0))
    except Exception:
        pass

plt.rcParams.update({
    'font.size': 15, 'font.family': 'serif', 'axes.linewidth': 2.0, 
    'patch.linewidth': 2.0, 'lines.linewidth': 2.5, 'axes.labelweight': 'bold', 
    'axes.titleweight': 'bold', 'xtick.direction': 'in', 'ytick.direction': 'in', 
    'xtick.major.width': 2.0, 'ytick.major.width': 2.0,
    'figure.facecolor': 'white', 'axes.facecolor': 'white'
})
fig, axes = plt.subplots(1,2,figsize=(16,6))
fig.suptitle('SSIM Analysis (MP Dataset Output)', fontsize=18, fontweight='bold')
axes[0].hist(scores, bins=15, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(np.mean(scores), color='red', linestyle='--', lw=2.5, label=f'Mean: {np.mean(scores):.3f}')
axes[0].axvline(0.65, color='orange', linestyle=':', lw=2.5, label='Classical GAN (~0.65)')
axes[0].set_title('SSIM Score Distribution', fontsize=18, fontweight='bold')
axes[0].set_xlabel('Structural Similarity Index (SSIM)', fontsize=16, fontweight='bold')
axes[0].legend(fontsize=15)
axes[1].imshow(np.hstack([dm(gen_atoms[0]), np.ones((30,2))*25, dm(real_atoms[0])]), cmap='viridis', vmin=0, vmax=25)
axes[1].set_title('Distance Matrix: Generated | Real', fontsize=18, fontweight='bold')
axes[1].axis('off')
plt.tight_layout()
os.makedirs('logs', exist_ok=True)
plt.savefig('logs/eval_ssim.png', dpi=300)
print("eval_ssim.png saved successfully with enlarged fonts.")
