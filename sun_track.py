"""Fast S.U.N. proxy for every checkpoint of a run, so training is steered by it.

LeMat-GenBench's S.U.N. (stable, unique, novel) is the target but costs ~1 h per
checkpoint. This scores a checkpoint in minutes, per sample, over ALL samples
(an invalid or duplicate sample counts against it):

  valid    LeMat-GenBench's species-aware distance rule (crystal_mic.lemat_valid)
  stable   unrelaxed CHGNet energy within TAU eV/atom of the real row with the
           same label (proxy -- confirm winners with LeMat, which relaxes and
           uses a real hull)
  unique   not StructureMatcher-equal to an earlier valid sample
  novel    not StructureMatcher-equal to any of the 882 distinct training
           structures of the same formula

    python sun_track.py runs/w19_ms02_s1 runs/w19_ms05_s1        # once
    python sun_track.py --watch runs/w19_*                        # keep scoring

Appends one row per checkpoint to <run>/sun_track.csv and prints it.
"""
import argparse
import glob
import os
import pickle
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

from crystal_mic import lemat_valid
from eval_wave import chgnet_energy, load_generator, one_structure, sample, std_z, to_structures

TAU = 0.3


def score(ck, rc, rl, by_formula, n):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    gen, epoch, _ = load_generator(ck)
    np.random.seed(0); torch.manual_seed(0)
    coords, labels, idx = sample(gen, rl, n)
    valid = np.array([lemat_valid(c, l) for c, l in zip(coords, labels)])
    gst = [one_structure(c, l) if v else None for c, l, v in zip(coords, labels, valid)]
    rst = [one_structure(rc[j], rl[j]) if v else None for j, v in zip(idx, valid)]
    de = chgnet_energy(gst) - chgnet_energy(rst)
    stable = valid & np.nan_to_num(de < TAU)
    seen, uniq, novel = [], np.zeros(n, bool), np.zeros(n, bool)
    for i, st in enumerate(gst):
        if st is None:
            continue
        if not any(sm.fit(st, s) for s in seen if s.composition == st.composition):
            uniq[i] = True
            seen.append(st)
        novel[i] = not any(sm.fit(st, t) for t in by_formula.get(st.composition.reduced_formula, []))
    sun = stable & uniq & novel
    return dict(epoch=epoch, valid=valid.mean(), stable=stable.mean(), unique=uniq.mean(),
                novel=novel.mean(), sun=sun.mean(), vun=(valid & uniq & novel).mean(),
                de_med=float(np.nanmedian(de[valid])) if valid.any() else float('nan'),
                std_z=std_z(gen, rl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--watch', action='store_true')
    a = ap.parse_args()
    raw = pickle.load(open('datasets/mgmno_100.pickle', 'rb'))
    rc = np.array([np.array(c).flatten() for c, _ in raw], dtype=np.float32)
    rl = np.array([np.array(l).flatten() for _, l in raw], dtype=np.float32)
    ref = np.load('datasets/novelty_ref_idx.npy')
    by_formula = {}
    for t in to_structures(rc[ref], rl[ref], cap=len(ref)):
        by_formula.setdefault(t.composition.reduced_formula, []).append(t)

    cols = ['epoch', 'valid', 'stable', 'unique', 'novel', 'sun', 'vun', 'de_med', 'std_z']
    while True:
        for run in a.runs:
            out = os.path.join(run, 'sun_track.csv')
            done = set()
            if os.path.exists(out):
                done = {int(l.split(',')[0]) for l in open(out).read().splitlines()[1:]}
            cks = sorted(glob.glob(os.path.join(run, 'checkpoint_[0-9]*.pt')),
                         key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p)))))
            for ck in cks:
                ep = int(''.join(filter(str.isdigit, os.path.basename(ck))))
                if ep in done or ep == 0:
                    continue
                m = score(ck, rc, rl, by_formula, a.n)
                new = not os.path.exists(out)
                with open(out, 'a') as f:
                    if new:
                        f.write(','.join(cols) + '\n')
                    f.write(','.join(f'{m[c]:.4f}' if c != 'epoch' else str(m[c]) for c in cols) + '\n')
                print(f"{os.path.basename(run)} ep{m['epoch']}: SUN {m['sun']*100:.1f}%  "
                      f"valid {m['valid']*100:.0f}  stable {m['stable']*100:.0f}  "
                      f"uniq {m['unique']*100:.0f}  novel {m['novel']*100:.0f}  "
                      f"dE {m['de_med']:.2f}  std_z {m['std_z']:.4f}", flush=True)
        if not a.watch:
            break
        time.sleep(300)


if __name__ == '__main__':
    main()
