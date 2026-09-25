"""CDVAE / DiffCSP standard generation metrics, and CIF export for LeMat-GenBench.

eval_wave.py is the cheap per-run table. This is the suite the field publishes
(Xie et al. 2022, CDVAE; Jiao et al. 2023, DiffCSP), computed the same way:

  valid_struct   every interatomic distance >= 0.5 A
  valid_comp     SMACT charge neutrality + Pauling electronegativity test
  valid          both
  COV-R / COV-P  coverage recall / precision on CrystalNN structure and Magpie
                 composition fingerprints, cutoffs 0.4 / 10 (the MP-20 values)
  W(density)     Wasserstein distance, g/cm^3
  W(#elements)   Wasserstein distance on the number of distinct elements

Two deviations from the papers, stated rather than hidden:
  * There is no held-out test split -- every Mg-Mn-O structure is in training --
    so coverage and W are against the distinct training structures.
  * CDVAE normalises fingerprints with MP-20 statistics; here the scaler is fit
    on the reference set, which is the equivalent for a single-chemistry dataset.

valid_comp is decided by the composition, and the model's composition is its
conditioning label, drawn from real rows -- so for the generator it equals the
real data's rate by construction. It is reported because the suite includes it.

    python benchmark.py --ckpt runs/w8_ms20_s2/checkpoint_best.pt --n 500
    python benchmark.py --ckpt ... --cif_dir bench_out   # also write CIFs

Then LeMat-GenBench (E_hull, S.U.N./M.S.U.N. vs every known material):
    uv run scripts/run_benchmarks.py --cifs bench_out/<method> \\
        --config comprehensive_multi_mlip_hull --name <method>
"""
import argparse
import itertools
import os
import pickle
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

from crystal_mic import min_dist
from eval_wave import load_generator, one_structure, sample, to_structures

STRUCT_CUTOFF, COMP_CUTOFF = 0.4, 10.0      # CDVAE's MP-20 thresholds


def smact_validity(structure):
    """CDVAE's smact_validity: neutral oxidation-state assignment that also
    passes the Pauling electronegativity test."""
    import smact
    from smact.screening import pauling_test
    # Reduced counts, as CDVAE does (counts / gcd); unreduced supercell counts
    # make neutral_ratios reject formulas it accepts in reduced form.
    comp = structure.composition.reduced_composition.get_el_amt_dict()
    elems = tuple(comp)
    counts = [int(round(comp[e])) for e in elems]
    space = smact.element_dictionary(elems)
    smact_elems = [space[e] for e in elems]
    if len(elems) == 1:
        return True
    enegs = [e.pauling_eneg for e in smact_elems]
    for ox in itertools.product(*[e.oxidation_states for e in smact_elems]):
        ok, _ = smact.neutral_ratios(ox, stoichs=[(c,) for c in counts],
                                     threshold=max(counts))
        if ok and pauling_test(ox, enegs):
            return True
    return False


def fingerprints(structs):
    """(structure fp, composition fp) as in CDVAE: mean CrystalNN site
    fingerprint, and Magpie elemental-property statistics."""
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.site import CrystalNNFingerprint
    cnn = CrystalNNFingerprint.from_preset('ops')
    mag = ElementProperty.from_preset('magpie')
    sf, cf = [], []
    for st in structs:
        try:
            sf.append(np.mean([cnn.featurize(st, i) for i in range(len(st))], axis=0))
            cf.append(np.array(mag.featurize(st.composition), dtype=float))
        except Exception:
            sf.append(None); cf.append(None)
    return sf, cf


def coverage(gen_sf, gen_cf, ref_sf, ref_cf):
    """CDVAE's COV-R (recall over the reference) and COV-P (precision over gen)."""
    from sklearn.preprocessing import StandardScaler
    keep = [i for i, (s, c) in enumerate(zip(gen_sf, gen_cf)) if s is not None]
    g_s = np.array([gen_sf[i] for i in keep]); g_c = np.array([gen_cf[i] for i in keep])
    r_s = np.array([s for s in ref_sf if s is not None])
    r_c = np.array([c for s, c in zip(ref_sf, ref_cf) if s is not None])
    sc = StandardScaler().fit(r_c)
    g_c, r_c = sc.transform(g_c), sc.transform(r_c)
    g_c, r_c = np.nan_to_num(g_c), np.nan_to_num(r_c)
    d_s = np.linalg.norm(g_s[:, None] - r_s[None], axis=-1)       # (G, R)
    d_c = np.linalg.norm(g_c[:, None] - r_c[None], axis=-1)
    hit = (d_s <= STRUCT_CUTOFF) & (d_c <= COMP_CUTOFF)
    return float(hit.any(axis=0).mean()), float(hit.any(axis=1).mean())


def wasserstein(a, b):
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b))


def suite(name, coords, labels, ref, cif_dir=None):
    # Row by row: to_structures drops undecodable rows, which would misalign
    # the distance mask. An undecodable row counts as structurally invalid.
    pairs = [(one_structure(c, l), min_dist(c, l)) for c, l in zip(coords, labels)]
    structs = [st for st, _ in pairs if st is not None]
    v_struct = np.array([d >= 0.5 for st, d in pairs if st is not None])
    v_comp = np.array([smact_validity(s) for s in structs])
    valid = v_struct & v_comp
    vs = [s for s, ok in zip(structs, valid) if ok]
    sf, cf = fingerprints(vs)
    cov_r, cov_p = coverage(sf, cf, ref['sf'], ref['cf'])
    dens = [s.density for s in vs]
    nel = [len(s.composition.elements) for s in vs]
    if cif_dir:
        out = os.path.join(cif_dir, name)
        os.makedirs(out, exist_ok=True)
        for i, s in enumerate(structs):
            s.to(filename=os.path.join(out, f'{name}_{i:04d}.cif'))
    return dict(name=name, n=len(coords), v_struct=v_struct.sum() / len(coords),
                v_comp=v_comp.mean(), valid=valid.sum() / len(coords), cov_r=cov_r, cov_p=cov_p,
                w_dens=wasserstein(dens, ref['dens']), w_nel=wasserstein(nel, ref['nel']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='*', default=[])
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cif_dir', default=None)
    ap.add_argument('--dataset', default='datasets/mgmno_100.pickle')
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)

    raw = pickle.load(open(a.dataset, 'rb'))
    rc = np.array([np.array(c).flatten() for c, l in raw], dtype=np.float32)
    rl = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)
    ridx = np.load(os.path.join(os.path.dirname(a.dataset), 'novelty_ref_idx.npy'))
    ref_structs = to_structures(rc[ridx], rl[ridx], cap=len(ridx))
    print(f'reference: {len(ref_structs)} distinct training structures', flush=True)
    sf, cf = fingerprints(ref_structs)
    ref = dict(sf=sf, cf=cf, dens=[s.density for s in ref_structs],
               nel=[len(s.composition.elements) for s in ref_structs])

    rows = []
    # Real data scored against itself: the ceiling each metric can reach here.
    pick = np.random.choice(len(ridx), min(a.n, len(ridx)), replace=False)
    rows.append(suite('real', rc[ridx][pick], rl[ridx][pick], ref, a.cif_dir))

    from baselines.substitution import generate as substitute
    idx = np.random.choice(len(rl), a.n)
    sub, kept = substitute(rc, rl, rl[idx])
    rows.append(suite('substitution', sub, rl[idx[kept]], ref, a.cif_dir))

    for ck in a.ckpt:
        gen = load_generator(ck)[0]
        coords, labels, _ = sample(gen, rl, a.n)
        name = f"{os.path.basename(os.path.dirname(ck))}_{os.path.basename(ck)[11:-3]}"
        rows.append(suite(name, coords, labels, ref, a.cif_dir))

    print(f"\n{'method':<28}{'n':>5}{'struct':>8}{'comp':>7}{'valid':>7}"
          f"{'COV-R':>7}{'COV-P':>7}{'W(dens)':>9}{'W(#el)':>8}")
    for r in rows:
        print(f"{r['name']:<28}{r['n']:>5}{r['v_struct']*100:>8.1f}{r['v_comp']*100:>7.1f}"
              f"{r['valid']*100:>7.1f}{r['cov_r']*100:>7.1f}{r['cov_p']*100:>7.1f}"
              f"{r['w_dens']:>9.3f}{r['w_nel']:>8.3f}")
    print('\nReference = distinct training structures (no held-out split exists). '
          'comp validity is set by the conditioning label for the generator.')


if __name__ == '__main__':
    main()
