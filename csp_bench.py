"""Crystal structure prediction (CSP) benchmark: match rate and RMSE.

The task DiffCSP, FlowMM, CrystalFlow, OMatG and CrysBFN report on MP-20: given
a target's composition, generate k structures; the target counts as matched if
ANY of them StructureMatcher-matches it (stol=0.5, angle_tol=10, ltol=0.3).
RMSE is the normalised RMS displacement of matched pairs (pymatgen's
get_rms_dist, normalised by (V/N)^(1/3)), averaged over matched targets.

Honest differences from the papers:
  * No held-out split exists for our data, so targets are the distinct
    TRAINING structures: this measures in-distribution recall, not
    generalisation to unseen compositions.
  * The substitution baseline is leave-one-out (never uses the target as its
    own template), which makes it a template-based CSP method.

    python csp_bench.py --ckpt runs/w20_force05_s1/checkpoint_90.pt --n_targets 300
"""
import argparse
import os
import pickle
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

from eval_wave import load_generator, one_structure


def match(sm, gen, target):
    """(matched?, rms) for the best of `gen` against `target`."""
    best = None
    for g in gen:
        if g is None:
            continue
        try:
            r = sm.get_rms_dist(g, target)
        except Exception:
            r = None
        if r is not None and (best is None or r[0] < best):
            best = r[0]
    return best is not None, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='*', default=[])
    ap.add_argument('--n_targets', type=int, default=300)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--substitution', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    raw = pickle.load(open('datasets/mgmno_100.pickle', 'rb'))
    rc = np.array([np.array(c).flatten() for c, _ in raw], dtype=np.float32)
    rl = np.array([np.array(l).flatten() for _, l in raw], dtype=np.float32)
    ref = np.load('datasets/novelty_ref_idx.npy')
    rng = np.random.default_rng(0)
    tgt = rng.choice(ref, size=min(a.n_targets, len(ref)), replace=False)
    targets = [one_structure(rc[i], rl[i]) for i in tgt]

    methods = []
    for ck in a.ckpt:
        gen = load_generator(ck)[0]

        def sampler(i, k, gen=gen):
            lab = torch.tensor(np.repeat(rl[i:i + 1], k, axis=0))
            torch.manual_seed(int(i))
            with torch.no_grad():
                out = gen(torch.cat([torch.randn(k, 64), lab], dim=1)).numpy()
            return [one_structure(o, rl[i]) for o in out]
        methods.append((os.path.basename(os.path.dirname(ck)) + '_' +
                        os.path.basename(ck)[11:-3], sampler))
    if a.substitution:
        from baselines.substitution import generate as substitute
        # Leave-one-out: remove every row of the target's own source block
        # (all 100 augmentations of a composition share one block) is too
        # strict; remove rows identical in geometry to the target instead.
        def sampler(i, k):
            keep = np.array([j for j in range(len(rc)) if not np.allclose(rc[j], rc[i], atol=1e-4)])
            sub, kept = substitute(rc[keep], rl[keep], np.repeat(rl[i:i + 1], k, axis=0),
                                   rng=np.random.default_rng(int(i)))
            return [one_structure(s, rl[i]) for s in sub]
        methods.append(('substitution_leave_one_out', sampler))

    rows = []
    for name, sampler in methods:
        hit1 = hitk = 0
        rms1, rmsk = [], []
        for t_i, target in zip(tgt, targets):
            gen = sampler(t_i, a.k)
            m1, r1 = match(sm, gen[:1], target)
            mk, rk = match(sm, gen, target)
            hit1 += m1; hitk += mk
            if m1: rms1.append(r1)
            if mk: rmsk.append(rk)
        n = len(targets)
        row = [name, n, hit1 / n, np.mean(rms1) if rms1 else float('nan'),
               hitk / n, np.mean(rmsk) if rmsk else float('nan')]
        rows.append(row)
        print(f'{name:36s} match@1 {row[2]:.1%}  RMSE@1 {row[3]:.4f}  '
              f'match@{a.k} {row[4]:.1%}  RMSE@{a.k} {row[5]:.4f}', flush=True)
    if a.out:
        import csv
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'n_targets', 'match_rate_k1', 'rmse_k1',
                        f'match_rate_k{a.k}', f'rmse_k{a.k}'])
            w.writerows(rows)


if __name__ == '__main__':
    main()
