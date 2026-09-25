"""Collapse probe for the split-head crystal generator.

The failure mode that wasted v4-v6 was invisible to every loss curve: the
generator became a deterministic function of the conditioning label and ignored
z entirely (std 8e-4 against 0.08 for the label), i.e. one crystal per
composition. Wasserstein loss looked fine throughout.

This measures it directly, in seconds, and is called from train_crystal.py at
every checkpoint.

    std_z     spread across 256 different z at ONE fixed label.
              Below 1e-2 the model is a lookup table -- stop the run.
    std_label spread across 256 different labels.
              Confirms conditioning still works.

Baseline (pre-fix checkpoint_490, 256 samples):
    v4  std_z 0.00082  std_label 0.08287  cell 4.2/5.3/5.2  25.0% valid  0.59 A
    v5  std_z 0.00288  std_label 0.07762  cell 4.4/5.0/4.9   0.0% valid  0.27 A
    v6  std_z 0.00033  std_label 0.11376  cell 3.8/4.2/3.3   0.0% valid  0.16 A
    real data                             cell 6.1/6.2/6.4 100.0% valid  1.92 A
"""
import numpy as np
import torch

from crystal_mic import lemat_valid, validity

N_PROBE = 256
Z_DEAD_THRESHOLD = 1e-2
# Real Mg-Mn-O is 11.77 +/- 1.84 A^3/atom (median 11.25). The old 9-18 band was
# wide enough that a checkpoint at 15.5 qualified as "best" -- and such a
# checkpoint scored 94.5% validity worth +6.5 eV/atom of CHGNet energy, because
# validity is a monotonic function of cell inflation. A gate that admits the
# artefact is not a gate. See research-vault/'The Validity Bar Is Too Low.md'.
VPA_OK = (9.5, 13.5)


def volume_per_atom(coords, labels):
    """Cell volume divided by atom count, in A^3. The cell-inflation detector."""
    arr = np.asarray(coords).reshape(-1, 30, 3)
    L = arr[:, 0] * 30.0
    A = np.deg2rad(np.clip(arr[:, 1] * 180.0, 30.0, 150.0))
    ca, cb, cg = np.cos(A[:, 0]), np.cos(A[:, 1]), np.cos(A[:, 2])
    vol = L[:, 0] * L[:, 1] * L[:, 2] * np.sqrt(
        np.clip(1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg, 1e-9, None))
    n = np.asarray(labels).reshape(len(arr), -1).sum(axis=1).clip(min=1)
    return vol / n


def _ordering(coords, labels):
    import torch
    from crystal_physics import NN_CLASSES, masked_pair_distances, nn_class_values
    c = torch.as_tensor(np.asarray(coords), dtype=torch.float32)
    l = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
    d = masked_pair_distances(c, l)
    out = {}
    for k in ('cc', 'oo'):
        v = nn_class_values(d, l, *NN_CLASSES[k])
        out[f'nn_{k}'] = float(v.median()) if v.numel() else float('nan')
    return out


def probe_generator(generator, labels_all, z_dim, device, n=N_PROBE):
    """Print diversity + validity for a generator. Returns the metrics dict."""
    was_training = generator.training
    generator.eval()

    fixed = np.repeat(labels_all[:1], n, axis=0)
    varied = labels_all[np.linspace(0, len(labels_all) - 1, n).astype(int)]

    with torch.no_grad():
        def gen(lbl):
            t = torch.from_numpy(np.asarray(lbl, dtype=np.float32)).to(device)
            z = torch.randn(len(lbl), z_dim, device=device)
            return generator(torch.cat([z, t], dim=1)).cpu().numpy()

        out_z = gen(fixed)
        out_l = gen(varied)

    if was_training:
        generator.train()

    frac_valid, dists = validity(out_l, varied)
    m = {
        'std_z': float(out_z[:, 6:].std(axis=0).mean()),
        'std_label': float(out_l[:, 6:].std(axis=0).mean()),
        'cell_a': out_z[0, :3] * 30.0,
        'valid': frac_valid,
        'mean_dist': float(dists.mean()),
        'vpa': volume_per_atom(out_l, varied).mean(),
        # The benchmark's species-aware distance rule (LeMat-GenBench).
        'lemat_valid': float(np.mean([lemat_valid(c, l) for c, l in zip(out_l, varied)])),
        # Ionic ordering: median cation -> nearest cation (real 3.00 A) and
        # O -> nearest O (real 2.74 A). Distance checks cannot see this.
        **_ordering(out_l, varied),
    }
    flag = '  <-- COLLAPSED (z is dead)' if m['std_z'] < Z_DEAD_THRESHOLD else ''
    # Validity is meaningless without vpa beside it: inflating the lattice
    # satisfies any minimum-distance criterion. Real Mg-Mn-O is 11.8 A^3/atom.
    if not (VPA_OK[0] <= m['vpa'] <= VPA_OK[1]):
        flag += f"  <-- CELL INFLATED (vpa {m['vpa']:.1f} vs real ~11.8)"
    print(f"  [probe] std_z={m['std_z']:.5f} std_label={m['std_label']:.5f} "
          f"cell={np.round(m['cell_a'], 2)}A vpa={m['vpa']:.1f} "
          f"valid={m['valid'] * 100:.1f}% lemat={m['lemat_valid'] * 100:.1f}% "
          f"ccNN={m['nn_cc']:.2f} ooNN={m['nn_oo']:.2f} meanD={m['mean_dist']:.2f}A{flag}", flush=True)
    return m


if __name__ == '__main__':
    import argparse, pickle, sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.QINR_Crystal import PQWGAN_CC_Crystal

    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--dataset', default='datasets/mgmno_100.pickle')
    p.add_argument('--z_dim', type=int, default=64)
    p.add_argument('--hidden_features', type=int, default=12)
    p.add_argument('--hidden_layers', type=int, default=1)
    p.add_argument('--spectrum_layer', type=int, default=1)
    a = p.parse_args()

    raw = pickle.load(open(a.dataset, 'rb'))
    labels = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = PQWGAN_CC_Crystal(input_dim_g=a.z_dim + 28, output_dim=90, input_dim_d=118,
                            hidden_features=a.hidden_features,
                            hidden_layers=a.hidden_layers,
                            spectrum_layer=a.spectrum_layer, use_noise=0.0)
    gan.generator.load_state_dict(torch.load(a.checkpoint, map_location=device)['generator'])
    print(a.checkpoint)
    probe_generator(gan.generator.to(device), labels, a.z_dim, device)
