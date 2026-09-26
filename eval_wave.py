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
  dE                      CHGNet energy/atom minus the REAL structure with the
                          identical label (paired: samples are drawn by picking a
                          real row and conditioning on its label). Median, eV/atom.
                          The one metric the substitution baseline does not win.
  dE<0.1%                 fraction within 0.1 eV/atom of its real partner -- the
                          usual "near the hull" tolerance, relative to real.
  unique%                 distinct structures under StructureMatcher, not SSIM.
  novel%                  not matching any training structure.
  std_z                   diversity, averaged over 8 labels. A model that
                          collapsed is not a model.

Both the best and the last checkpoint are reported. `best` is the maximum of the
training probe over ~50 checkpoints, so it is optimistically biased; `last`
is not. Fixed seed, so the table is reproducible.

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

from crystal_mic import SPECIES_MAP, decode, lemat_valid, min_dist
from crystal_physics import lattice_matrix

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
    rounds = len({k.split('.')[2] for k in sd if k.startswith('refiner.edge.')})
    net = [k for k in sd if '.qlayer.net.' in k]
    def _trunk():
        if not net:
            return 'quantum'
        w0 = next(sd[k] for k in net if k.endswith('net.0.weight'))
        if w0.shape[1] == 2 * w0.shape[0] or w0.shape[1] == 24:
            return 'classical_fourier'
        return 'classical_wide' if any('.qlayer.net.2.' in k for k in net) else 'classical_matched'
    trunk = _trunk()
    gan = PQWGAN_CC_Crystal(input_dim_g=z_dim + 28, output_dim=90, input_dim_d=126,
                            hidden_features=hidden, hidden_layers=layers,
                            spectrum_layer=spectrum, use_noise=0.0, sf_head=sf,
                            split_head=split, refine_rounds=rounds, trunk=trunk,
                            trunk_residual=not ck.get('args', {}).get('no_trunk_residual', False))
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
    return np.concatenate(out), labels[idx], idx


def std_z(gen, labels, z_dim=64, n=256, n_labels=8):
    """Spread across many z at a fixed label, averaged over n_labels labels.

    One label (the old version) is one draw; a model can be alive at one
    composition and dead at another. Below 1e-2 the model is a lookup table.
    """
    vals = []
    for j in np.linspace(0, len(labels) - 1, n_labels).astype(int):
        lab = torch.tensor(np.repeat(labels[j:j + 1], n, axis=0), dtype=torch.float32)
        with torch.no_grad():
            out = gen(torch.cat([torch.randn(n, z_dim), lab], dim=1)).cpu().numpy()
        vals.append(out[:, 6:].std(axis=0).mean())
    return float(np.mean(vals))


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
    by_formula = {}
    for t in real_structs:
        by_formula.setdefault(t.composition.reduced_formula, []).append(t)
    novel = sum(not any(sm.fit(r, t) for t in
                        by_formula.get(r.composition.reduced_formula, []))
                for r in reps)
    return unique, novel / max(len(reps), 1)


_CHGNET = []


def chgnet_energy(structs):
    """CHGNet energy per atom (eV), unrelaxed. NaN where CHGNet fails."""
    if not _CHGNET:
        from chgnet.model import CHGNet
        _CHGNET.append(CHGNet.load(verbose=False))
    out = []
    for st in structs:
        try:
            out.append(float(_CHGNET[0].predict_structure(st)['e']) if st else np.nan)
        except Exception:
            out.append(np.nan)
    return np.array(out)


def one_structure(c, l):
    s = to_structures(c[None], l[None], cap=1)
    return s[0] if s else None


def paired_energy(coords, labels, idx, real_coords, real_labels, n_e):
    """(median dE, frac dE<0.1) vs the real row each sample was conditioned on.

    Generated structures with a contact under 0.5 A are scored as failures
    (dE = +inf) rather than dropped, so a model cannot improve this number by
    emitting more broken structures.
    """
    k = min(n_e, len(coords))
    gen = [one_structure(coords[i], labels[i]) if min_dist(coords[i], labels[i]) >= 0.5
           else None for i in range(k)]
    real = [one_structure(real_coords[j], real_labels[j]) for j in idx[:k]]
    de = chgnet_energy(gen) - chgnet_energy(real)
    de = np.where(np.isnan(de), np.inf, de)
    return float(np.median(de)), float((de < 0.1).mean())


def relaxed_energy(structs, steps=300):
    """(energy/atom after CHGNet relaxation, RMSD in A moved by relaxation).

    Stability in CDVAE/DiffCSP/MatterGen is judged after relaxation, so this is
    the comparable number -- but a relaxer can turn garbage into a real crystal,
    which is the trap in Metric Problems ("the relaxer is doing the work"). The
    RMSD is reported beside it for that reason: a good generator lands close to
    a minimum (MatterGen's RMSD-to-relaxed), a bad one gets rescued from far away.
    """
    from chgnet.model import StructOptimizer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    opt = StructOptimizer()
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    e, rmsd, fins = [], [], []
    for st in structs:
        if st is None:
            e.append(np.nan); rmsd.append(np.nan); fins.append(None)
            continue
        try:
            r = opt.relax(st, steps=steps, verbose=False)
            fin = r['final_structure']
            fins.append(fin)
            e.append(float(r['trajectory'].energies[-1]) / len(fin))
            rm = sm.get_rms_dist(st, fin)
            rmsd.append(rm[0] if rm else np.inf)
        except Exception:
            e.append(np.nan); rmsd.append(np.nan); fins.append(None)
    return np.array(e), np.array(rmsd), fins


def paired_relaxed(coords, labels, idx, real_coords, real_labels, n_r, real_structs):
    """Relaxed metrics vs the real partner, and novelty AFTER relaxation.

    Returns (median dE, frac dE<0.1, median RMSD, frac novel after relaxation,
    S.U.N. rate). Novelty before relaxation is cheap to fake: a distorted copy
    of a training crystal matches nothing. A structure that relaxes back onto a
    training crystal is not new. S.U.N. (MatterGen) = stable (dE<0.1 vs real),
    unique among the relaxed set, and novel after relaxation, over all samples.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    by_formula = {}
    for t in real_structs:
        by_formula.setdefault(t.composition.reduced_formula, []).append(t)
    k = min(n_r, len(coords))
    gen = [one_structure(coords[i], labels[i]) if min_dist(coords[i], labels[i]) >= 0.5
           else None for i in range(k)]
    real = [one_structure(real_coords[j], real_labels[j]) for j in idx[:k]]
    eg, rmsd, fins = relaxed_energy(gen)
    er, _, _ = relaxed_energy(real)
    de = eg - er
    de = np.where(np.isnan(de), np.inf, de)
    novel = np.array([f is not None and not any(
        sm.fit(f, t) for t in by_formula.get(f.composition.reduced_formula, []))
        for f in fins])
    seen, uniq = [], np.zeros(len(fins), bool)
    for i, f in enumerate(fins):
        if f is not None and not any(sm.fit(f, g) for g in seen):
            uniq[i] = True
            seen.append(f)
    sun = (de < 0.1) & novel & uniq
    return (float(np.median(de)), float((de < 0.1).mean()), float(np.nanmedian(rmsd)),
            float(novel.mean()), float(sun.mean()))


def wasserstein(a, b):
    """1-D Wasserstein distance without scipy: mean |quantile difference|."""
    q = np.linspace(0, 1, 101)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def checkpoints(run):
    """[(tag, path)] for the best and the last checkpoint that exist."""
    out = []
    best = os.path.join(run, 'checkpoint_best.pt')
    if os.path.exists(best):
        out.append(('best', best))
    cks = sorted(glob.glob(os.path.join(run, 'checkpoint_[0-9]*.pt')),
                 key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p)))))
    if cks:
        out.append(('last', cks[-1]))
    return out


def evaluate(run, tag, ck, real_coords, real_labels, real_structs, real_vpa, n, cap, n_e,
             n_r=0):
    gen, epoch, sf = load_generator(ck)
    coords, labels, idx = sample(gen, real_labels, n)

    dists = np.array([min_dist(c, l) for c, l in zip(coords, labels)])
    lat = lattice_matrix(torch.tensor(coords, dtype=torch.float32))
    vol = torch.linalg.det(lat).abs().numpy()
    natoms = labels.sum(axis=1).clip(min=1)
    vpa = vol / natoms
    de, de_ok = (paired_energy(coords, labels, idx, real_coords, real_labels, n_e)
                 if n_e > 0 else (float('nan'), float('nan')))

    gs = to_structures(coords, labels, cap=cap)
    uniq, novel = uniqueness_novelty(gs, real_structs)

    return dict(run=os.path.basename(run), ckpt=tag, epoch=epoch, sf=sf,
                v05=float((dists >= 0.5).mean()), v10=float((dists >= 1.0).mean()),
                vlm=float(np.mean([lemat_valid(c, l) for c, l in zip(coords, labels)])),
                vpa=float(np.median(vpa)), wvpa=wasserstein(vpa, real_vpa),
                de=de, de_ok=de_ok, uniq=uniq, novel=novel,
                relaxed=(paired_relaxed(coords, labels, idx, real_coords, real_labels, n_r,
                                        real_structs) if n_r > 0 else None),
                stdz=std_z(gen, real_labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--cap', type=int, default=300,
                    help='structures fed to StructureMatcher (it is O(n*m))')
    ap.add_argument('--n_energy', type=int, default=200,
                    help='samples scored with CHGNet (~50 ms each, x2 for the real '
                         'partner). 0 = skip energy.')
    ap.add_argument('--n_relax', type=int, default=0,
                    help='samples relaxed with CHGNet (~seconds each, x2 for the '
                         'real partner). 0 = skip. Use on the winning config only.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dataset', default='datasets/mgmno_100.pickle')
    a = ap.parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    import pickle
    raw = pickle.load(open(a.dataset, 'rb'))
    real_coords = np.array([np.array(c).flatten() for c, l in raw], dtype=np.float32)
    real_labels = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)
    lat = lattice_matrix(torch.tensor(real_coords))
    real_vpa = (torch.linalg.det(lat).abs().numpy()
                / real_labels.sum(axis=1).clip(min=1))
    # Every distinct training structure (datasets/make_novelty_reference.py).
    # The old reference, every 97th row, was 108 of ~880 and overstated novelty.
    ref = np.load(os.path.join(os.path.dirname(os.path.abspath(a.dataset)),
                               'novelty_ref_idx.npy'))
    real_structs = to_structures(real_coords[ref], real_labels[ref], cap=len(ref))
    print(f'real reference: {len(real_structs)} distinct structures, '
          f'median vpa {np.median(real_vpa):.2f} A^3/atom\n')

    hdr = (f"{'run':<18}{'ck':>5}{'ep':>5}{'v@0.5':>7}{'v@1.0':>7}{'LeMat':>7}{'vpa':>7}"
           f"{'W(vpa)':>8}{'dE':>7}{'dE<.1':>7}{'uniq':>6}{'novel':>7}{'std_z':>8}  note")
    print(hdr)
    print('-' * len(hdr))

    def row(name, tag, ep, m, note=''):
        print(f"{name:<18}{tag:>5}{ep:>5}{m['v05']*100:>7.1f}{m['v10']*100:>7.1f}"
              f"{m.get('vlm', float('nan'))*100:>7.1f}"
              f"{m['vpa']:>7.1f}{m['wvpa']:>8.2f}{m['de']:>7.2f}{m['de_ok']*100:>7.0f}"
              f"{m['uniq']*100:>6.0f}{m['novel']*100:>7.0f}{m['stdz']:>8.4f}  {note}",
              flush=True)
        if m.get('relaxed'):
            rde, rok, rmsd, rnov, sun = m['relaxed']
            print(f"{'':<18}{'':>5}{'':>5}  relaxed: dE {rde:.2f}  dE<0.1 {rok*100:.0f}%"
                  f"  RMSD {rmsd:.2f} A  novel {rnov*100:.0f}%  S.U.N. {sun*100:.0f}%",
                  flush=True)

    # The training-free baseline goes in the table, always. It beats the model
    # on validity, density and novelty; omitting a baseline that wins on four
    # metrics is exactly what draws a reject.
    # See research-vault/'The Substitution Baseline Beats Us.md'.
    try:
        from baselines.substitution import generate as substitute
        idx = np.random.choice(len(real_labels), a.n)
        sub, kept = substitute(real_coords, real_labels, real_labels[idx])
        sub_idx = idx[kept]
        sub_lab = real_labels[sub_idx]
        d = np.array([min_dist(c, l) for c, l in zip(sub, sub_lab)])
        lat = lattice_matrix(torch.tensor(sub, dtype=torch.float32))
        v = (torch.linalg.det(lat).abs().numpy()
             / sub_lab.sum(axis=1).clip(min=1))
        u, nv = uniqueness_novelty(to_structures(sub, sub_lab, cap=a.cap), real_structs)
        de, de_ok = (paired_energy(sub, sub_lab, sub_idx, real_coords, real_labels,
                                   a.n_energy) if a.n_energy else (np.nan, np.nan))
        row('substitution', '--', '--',
            dict(v05=(d >= 0.5).mean(), v10=(d >= 1.0).mean(), vpa=np.median(v),
                 vlm=np.mean([lemat_valid(c, l) for c, l in zip(sub, sub_lab)]),
                 wvpa=wasserstein(v, real_vpa), de=de, de_ok=de_ok, uniq=u,
                 novel=nv, stdz=float('nan'),
                 relaxed=(paired_relaxed(sub, sub_lab, sub_idx, real_coords,
                                         real_labels, a.n_relax, real_structs)
                          if a.n_relax else None)), 'TRAINING-FREE BASELINE')
    except Exception as e:
        print(f"{'substitution':<18}  baseline failed: {str(e)[:50]}")

    runs = a.runs or sorted(d for d in glob.glob('runs/w*') if os.path.isdir(d))
    for r in runs:
        cks = checkpoints(r)
        if not cks:
            print(f'{os.path.basename(r):<18}  (no checkpoint yet)')
        for tag, ck in cks:
            try:
                m = evaluate(r, tag, ck, real_coords, real_labels, real_structs,
                             real_vpa, a.n, a.cap, a.n_energy, a.n_relax)
            except Exception as e:
                print(f'{os.path.basename(r):<18}{tag:>5}  ERROR: {str(e)[:60]}')
                continue
            bad = []
            if not (VPA_OK[0] <= m['vpa'] <= VPA_OK[1]):
                bad.append('CELL INFLATED')
            if m['stdz'] < Z_DEAD:
                bad.append('z COLLAPSED')
            row(m['run'], tag, m['epoch'], m,
                'UNRELIABLE: ' + ', '.join(bad) if bad else '')

    print(f'\nreal median vpa {np.median(real_vpa):.2f}; W(vpa) is distance to the '
          f'real distribution, lower is better.')
    print('dE: CHGNet eV/atom above the real structure with the same label, unrelaxed; '
          'contacts < 0.5 A count as failures. E_hull is the medium tier.')
    print("'best' is the max over training probes (biased up); 'last' is not.")
    print('Nothing in this table is a result until it holds across seeds.')


if __name__ == '__main__':
    main()
