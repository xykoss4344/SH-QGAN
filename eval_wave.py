"""Cheap-tier metric suite for a wave of runs, through one code path.

Produces the comparable table the Benchmark Plan asks for, for every run in
`runs/`, from each run's `checkpoint_best.pt`. Everything here is minutes, not
hours -- CHGNet relaxation and E_hull are the medium tier and belong only on the
winning config and the baselines (see research-vault/'Compute Plan.md').

    python eval_wave.py                      # every run in runs/
    python eval_wave.py --runs runs/w1_base_s0 runs/w1_floors_s0
    python eval_wave.py --n 2000             # more samples

Metrics, and why each is here:

  valid@0.5 / valid@1.0   CDVAE's threshold and ours. Both, always, because
                          silently switching thresholds is how numbers drift.
  vpa                     volume per atom. NEVER reported without it -- v9 hit
                          98.8% "valid" purely by inflating the box 6x.
  W(vpa)                  Wasserstein distance between generated and real
                          volume-per-atom distributions. This is the metric that
                          would have caught v9 publicly.
  neutral%                composition validity: cells whose formal charges sum
                          to zero.
  unique%                 distinct structures under StructureMatcher, not SSIM.
  novel%                  not matching any training structure.
  std_z                   diversity. A model that collapsed is not a model.

A row is marked UNRELIABLE if the cell is inflated or z has collapsed, because
its other numbers are then not comparable to anything.
"""
import argparse
import glob
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

from crystal_mic import SPECIES_MAP, decode, min_dist
from crystal_physics import charge_residual, lattice_matrix

VPA_OK = (9.5, 13.5)   # tightened: 9-18 admitted the cell-inflation artefact
Z_DEAD = 1e-2


def load_generator(ckpt_path, z_dim=64, hidden=12, layers=1, spectrum=1):
    """Rebuild the generator, inferring whether the run used the sf head."""
    from models.QINR_Crystal import PQWGAN_CC_Crystal
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # our own checkpoints; they carry probe metrics, not just tensors
    sd = ck['generator']
    # Infer the architecture from the weights rather than requiring the caller
    # to remember which flags a run used.
    sf = any('rho_proj' in k for k in sd)
    split = not any(k.startswith('joint_head') for k in sd)
    gan = PQWGAN_CC_Crystal(input_dim_g=z_dim + 28, output_dim=90, input_dim_d=126,
                            hidden_features=hidden, hidden_layers=layers,
                            spectrum_layer=spectrum, use_noise=0.0, sf_head=sf,
                            split_head=split)
    gan.generator.load_state_dict(sd)
    gan.generator.eval()
    return gan.generator, ck.get('epoch', -1), sf


def sample(gen, labels, n, z_dim=64, batch=64):
    """Generate n structures using labels drawn from the real label pool."""
    out = []
    idx = np.random.choice(len(labels), n)
    with torch.no_grad():
        for s in range(0, n, batch):
            lab = torch.tensor(labels[idx[s:s + batch]], dtype=torch.float32)
            z = torch.randn(lab.shape[0], z_dim)
            out.append(gen(torch.cat([z, lab], dim=1)).cpu().numpy())
    return np.concatenate(out), labels[idx]


def std_z(gen, labels, z_dim=64, n=256):
    """Spread across many z at ONE fixed label. Below 1e-2 the model is a table."""
    lab = torch.tensor(np.repeat(labels[:1], n, axis=0), dtype=torch.float32)
    with torch.no_grad():
        out = gen(torch.cat([torch.randn(n, z_dim), lab], dim=1)).cpu().numpy()
    return float(out[:, 6:].std(axis=0).mean())


def to_structures(coords, labels, cap=300):
    """Decode to pymatgen Structures, skipping undecodable ones."""
    from pymatgen.core import Lattice, Structure
    out = []
    for c, l in list(zip(coords, labels))[:cap]:
        try:
            sp, frac, cell = decode(c, l)
            if len(sp) < 2:
                continue
            out.append(Structure(Lattice.from_parameters(*cell), sp, frac))
        except Exception:
            continue
    return out


def uniqueness_novelty(gen_structs, real_structs):
    """(unique fraction, novel fraction) under StructureMatcher.

    Uniqueness is pairwise within the generated set; novelty is against the
    training set. StructureMatcher, never SSIM -- see Metric Problems item 1.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    if not gen_structs:
        return float('nan'), float('nan')
    groups = sm.group_structures(gen_structs)
    unique = len(groups) / len(gen_structs)
    reps = [g[0] for g in groups]
    novel = sum(not any(sm.fit(r, t) for t in real_structs) for r in reps)
    return unique, novel / max(len(reps), 1)


def wasserstein(a, b):
    """1-D Wasserstein distance without scipy: mean |quantile difference|."""
    q = np.linspace(0, 1, 101)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def evaluate(run, real_coords, real_labels, real_structs, real_vpa, n, cap):
    ck = os.path.join(run, 'checkpoint_best.pt')
    if not os.path.exists(ck):
        cks = sorted(glob.glob(os.path.join(run, 'checkpoint_*.pt')),
                     key=lambda p: int(''.join(filter(str.isdigit,
                                                      os.path.basename(p))) or 0))
        if not cks:
            return None
        ck, tag = cks[-1], 'last'
    else:
        tag = 'best'

    gen, epoch, sf = load_generator(ck)
    coords, labels = sample(gen, real_labels, n)

    dists = np.array([min_dist(c, l) for c, l in zip(coords, labels)])
    lat = lattice_matrix(torch.tensor(coords, dtype=torch.float32))
    vol = torch.linalg.det(lat).abs().numpy()
    natoms = labels.sum(axis=1).clip(min=1)
    vpa = vol / natoms
    neutral = (charge_residual(torch.tensor(labels, dtype=torch.float32))
               .abs().numpy() < 1e-3).mean()

    gs = to_structures(coords, labels, cap=cap)
    uniq, novel = uniqueness_novelty(gs, real_structs)

    return dict(run=os.path.basename(run), ckpt=tag, epoch=epoch, sf=sf,
                v05=float((dists >= 0.5).mean()), v10=float((dists >= 1.0).mean()),
                vpa=float(np.median(vpa)), wvpa=wasserstein(vpa, real_vpa),
                neutral=float(neutral), uniq=uniq, novel=novel,
                stdz=std_z(gen, real_labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--cap', type=int, default=300,
                    help='structures fed to StructureMatcher (it is O(n*m))')
    ap.add_argument('--dataset', default='datasets/mgmno_100.pickle')
    a = ap.parse_args()

    import pickle
    raw = pickle.load(open(a.dataset, 'rb'))
    real_coords = np.array([np.array(c).flatten() for c, l in raw], dtype=np.float32)
    real_labels = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)
    lat = lattice_matrix(torch.tensor(real_coords))
    real_vpa = (torch.linalg.det(lat).abs().numpy()
                / real_labels.sum(axis=1).clip(min=1))
    # Distinct structures only -- never the augmented row count.
    real_structs = to_structures(real_coords[::97], real_labels[::97], cap=120)
    print(f'real reference: {len(real_structs)} distinct structures, '
          f'median vpa {np.median(real_vpa):.2f} A^3/atom\n')

    # The training-free baseline goes in the table, always. It beats the model
    # on validity, density and novelty; omitting a baseline that wins on four
    # metrics is exactly what draws a reject.
    # See research-vault/'The Substitution Baseline Beats Us.md'.
    try:
        from baselines.substitution import generate as substitute
        idx = np.random.choice(len(real_labels), a.n)
        sub, kept = substitute(real_coords, real_labels, real_labels[idx])
        sub_lab = real_labels[idx][kept]
        d = np.array([min_dist(c, l) for c, l in zip(sub, sub_lab)])
        lat = lattice_matrix(torch.tensor(sub, dtype=torch.float32))
        v = (torch.linalg.det(lat).abs().numpy()
             / sub_lab.sum(axis=1).clip(min=1))
        gs = to_structures(sub, sub_lab, cap=a.cap)
        u, nv = uniqueness_novelty(gs, real_structs)
        print(f"{'substitution':<15}{'--':>5}{'--':>5}{(d >= 0.5).mean()*100:>7.1f}"
              f"{(d >= 1.0).mean()*100:>7.1f}{np.median(v):>7.1f}"
              f"{wasserstein(v, real_vpa):>8.2f}{100:>6.0f}{u*100:>6.0f}"
              f"{nv*100:>7.0f}{'--':>8}  TRAINING-FREE BASELINE")
    except Exception as e:
        print(f"{'substitution':<15}  baseline failed: {str(e)[:50]}")

    runs = a.runs or sorted(d for d in glob.glob('runs/w*') if os.path.isdir(d))
    hdr = (f"{'run':<15}{'ck':>5}{'ep':>5}{'v@0.5':>7}{'v@1.0':>7}{'vpa':>7}"
           f"{'W(vpa)':>8}{'neut':>6}{'uniq':>6}{'novel':>7}{'std_z':>8}  note")
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for r in runs:
        try:
            m = evaluate(r, real_coords, real_labels, real_structs, real_vpa,
                         a.n, a.cap)
        except Exception as e:
            print(f'{os.path.basename(r):<15}  ERROR: {str(e)[:60]}')
            continue
        if m is None:
            print(f'{os.path.basename(r):<15}  (no checkpoint yet)')
            continue
        bad = []
        if not (VPA_OK[0] <= m['vpa'] <= VPA_OK[1]):
            bad.append('CELL INFLATED')
        if m['stdz'] < Z_DEAD:
            bad.append('z COLLAPSED')
        note = 'UNRELIABLE: ' + ', '.join(bad) if bad else ''
        rows.append(m)
        print(f"{m['run']:<15}{m['ckpt']:>5}{m['epoch']:>5}{m['v05']*100:>7.1f}"
              f"{m['v10']*100:>7.1f}{m['vpa']:>7.1f}{m['wvpa']:>8.2f}"
              f"{m['neutral']*100:>6.0f}{m['uniq']*100:>6.0f}{m['novel']*100:>7.0f}"
              f"{m['stdz']:>8.4f}  {note}")

    print(f'\nreal median vpa {np.median(real_vpa):.2f}; W(vpa) is distance to the '
          f'real distribution, lower is better.')
    print('E_hull is NOT here: it is the medium tier, for the winning config and '
          'the baselines only.')
    print('Nothing in this table is a result until it holds across seeds.')


if __name__ == '__main__':
    main()
