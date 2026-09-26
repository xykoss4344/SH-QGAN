"""Rebuild a per-epoch loss table from a run's stdout.log.

Runs launched before epoch_log.csv existed only have the batch lines printed
every 10 batches ([Epoch e/N] [Batch i/M] [D: ..] [W: ..] ...). This averages
every bracketed term per epoch.

    python parse_training_log.py runs/w20_force05_s1   # writes epoch_log_from_stdout.csv
"""
import os
import re
import sys
from collections import defaultdict

LINE = re.compile(r'^\[Epoch (\d+)/\d+\] \[Batch \d+/\d+\]')
TERM = re.compile(r'\[([A-Za-z_]+): (-?[0-9.]+(?:e-?\d+)?)')


def parse(run):
    sums, counts = defaultdict(lambda: defaultdict(float)), defaultdict(int)
    for line in open(os.path.join(run, 'stdout.log'), errors='ignore'):
        m = LINE.match(line)
        if not m:
            continue
        ep = int(m.group(1))
        counts[ep] += 1
        for k, v in TERM.findall(line):
            if k not in ('Epoch', 'Batch'):
                sums[ep][k] += float(v)
    keys = sorted({k for e in sums for k in sums[e]})
    rows = [[ep] + [sums[ep].get(k, 0.0) / counts[ep] for k in keys] for ep in sorted(sums)]
    return ['epoch'] + keys, rows


def main(runs):
    for run in runs:
        header, rows = parse(run)
        out = os.path.join(run, 'epoch_log_from_stdout.csv')
        with open(out, 'w') as f:
            f.write(','.join(header) + '\n')
            for r in rows:
                f.write(','.join(f'{v:.6g}' for v in r) + '\n')
        print(f'{run}: {len(rows)} epochs -> {out}')


if __name__ == '__main__':
    main(sys.argv[1:])
