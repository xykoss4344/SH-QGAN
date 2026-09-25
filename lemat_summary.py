"""One table from LeMat-GenBench result JSONs.

LeMat-GenBench (https://github.com/LeMaterial/lemat-genbench) stores each
benchmark family as the repr of a BenchmarkResult, not as JSON numbers, so this
pulls the headline metrics out with regexes -- the first occurrence of each key.

    python lemat_summary.py C:/Users/Adminb/lemat-genbench/results_final/*.json
"""
import json
import re
import sys

KEYS = [  # (column, family, key in the repr)
    ('valid', None, None),
    ('E_hull', 'stability', 'mean_e_above_hull'),
    ('E_orb', 'stability', 'mean_e_above_hull_orb'),
    ('E_mace', 'stability', 'mean_e_above_hull_mace'),
    ('stable', 'sun', 'stable_rate'),
    ('metast', 'sun', 'metastable_rate'),
    ('SUN', 'sun', 'sun_rate'),
    ('MSUN', 'sun', 'msun_rate'),
    ('novel', 'novelty', 'novelty_score'),
    ('uniq', 'uniqueness', 'Uniqueness'),
    ('JSD', 'distribution', 'Average_Jensen_Shannon_Distance'),
]


def grab(text, key):
    m = re.search(rf"'{re.escape(key)}': (?:np\.float64\()?([-0-9.eE]+|nan)", text)
    return float(m.group(1)) if m else float('nan')


def main(paths):
    print(f"{'run':<30}{'n':>5}" + ''.join(f'{c:>8}' for c, _, _ in KEYS))
    for p in paths:
        d = json.load(open(p))
        res = d['results']
        row = [d['run_info']['run_name'], d['run_info']['n_structures']]
        for col, fam, key in KEYS:
            if col == 'valid':
                row.append(d['validity_filtering']['validity_rate'])
            else:
                row.append(grab(str(res.get(fam, '')), key))
        print(f'{row[0]:<30}{row[1]:>5}' + ''.join(f'{v:>8.3f}' for v in row[2:]))
    print('\nE_hull in eV/atom (ensemble mean of per-MLIP hulls, after 50-step relaxation). '
          'Rates are fractions of VALID structures; novelty is vs LeMat-Bulk (5.3M).')


if __name__ == '__main__':
    main(sys.argv[1:])
