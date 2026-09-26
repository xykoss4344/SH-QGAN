"""Relax a directory of CIFs with CHGNet: the generate-then-relax pipeline.

MatterGen and CDVAE-style pipelines relax generated structures before judging
stability; LeMat-GenBench itself relaxes only 50 steps. This relaxes fully and
records how far each structure moved, so the relaxer's share of the result
stays visible ("the relaxer is doing the work" -- Metric Problems.md). Apply it
to the baselines too, or the comparison is not fair.

    python relax_cifs.py bench_out/<method> bench_out/<method>_relaxed

Writes relaxed CIFs and relax_log.csv (file, n, E0, E, RMSD, steps).
"""
import csv
import glob
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def main(src, dst, steps=500, fmax=0.05):
    from chgnet.model import StructOptimizer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    os.makedirs(dst, exist_ok=True)
    opt = StructOptimizer()
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    rows = []
    for f in sorted(glob.glob(os.path.join(src, '*.cif'))):
        st = Structure.from_file(f)
        try:
            r = opt.relax(st, fmax=fmax, steps=steps, verbose=False)
        except Exception as e:
            rows.append([os.path.basename(f), len(st), '', '', '', f'fail: {e}'[:60]])
            continue
        fin, tr = r['final_structure'], r['trajectory']
        rm = sm.get_rms_dist(st, fin)
        fin.to(filename=os.path.join(dst, os.path.basename(f)))
        rows.append([os.path.basename(f), len(st), float(tr.energies[0]) / len(st),
                     float(tr.energies[-1]) / len(st), rm[0] if rm else 'unmatched',
                     len(tr.energies)])
    with open(os.path.join(dst, 'relax_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'n', 'E0', 'E', 'rmsd', 'steps'])
        w.writerows(rows)
    ok = [r for r in rows if r[2] != ""]
    drop = sorted(r[2] - r[3] for r in ok)
    matched = sum(1 for r in ok if r[4] != 'unmatched')
    print(f'{src}: {len(ok)}/{len(rows)} relaxed, median energy drop '
          f'{drop[len(drop) // 2]:.3f} eV/atom, {matched}/{len(ok)} still match their start')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
