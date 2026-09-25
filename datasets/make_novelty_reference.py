"""Distinct training structures under StructureMatcher, for novelty.

eval_wave.py used every 97th row of mgmno_100.pickle (108 rows) as the novelty
reference. Rows within a composition block are NOT all one structure (row 0 and
row 50 do not match), so that reference missed most of the training set and
overstated novelty. This dedups each block and saves the row indices of one
representative per distinct structure.

    python datasets/make_novelty_reference.py    # writes datasets/novelty_ref_idx.npy
"""
import os, pickle, sys, warnings
import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from eval_wave import to_structures
from pymatgen.analysis.structure_matcher import StructureMatcher

raw = pickle.load(open(os.path.join(HERE, 'mgmno_100.pickle'), 'rb'))
coords = np.array([np.array(c).flatten() for c, l in raw], dtype=np.float32)
labels = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)
sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

keep = []
for start in range(0, len(raw), 100):
    idx = np.arange(start, min(start + 100, len(raw)))
    structs = to_structures(coords[idx], labels[idx], cap=len(idx))
    # to_structures skips undecodable rows; real rows all decode (crystal_mic demo).
    assert len(structs) == len(idx), f'block {start}: {len(structs)}/{len(idx)} decoded'
    pos = {id(s): i for i, s in enumerate(structs)}
    for g in sm.group_structures(structs):
        keep.append(int(idx[pos[id(g[0])]]))
    print(f'block {start // 100}: {len(keep)} distinct so far', flush=True)

np.save(os.path.join(HERE, 'novelty_ref_idx.npy'), np.array(sorted(keep)))
print(f'{len(keep)} distinct structures out of {len(raw)} rows')
