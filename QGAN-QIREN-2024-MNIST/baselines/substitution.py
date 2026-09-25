"""Elemental-substitution baseline: the floor the whole thesis has to clear.

Reviewers asked whether a generative model is warranted at all. The cheapest
honest alternative is to take a known structure and swap the elements: keep a
real lattice and real fractional coordinates, reassign which cation sites are Mg
and which are Mn to hit the requested composition.

It needs no training, no GPU, and no quantum circuit. If it matches the GAN on
the metrics the paper reports, the paper has no thesis -- so this number belongs
in the results table, not in a drawer.

What it should show, and why that is the point:

  validity   near 100%. The geometry is real, so of course it is valid. This is
             exactly why validity alone proves nothing.
  novelty    near 0%. Every sample is a known structure with relabelled sites.

The interesting claim is therefore never "we beat substitution on validity" --
we will not. It is that the generator produces structures that are *both* valid
*and* novel, and the frontier between those two is where a generative model has
to earn its place.

    python baselines/substitution.py            # score it with the standard suite
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from crystal_mic import BOX_OFFSET, BOX_SCALE

MG, MN, O = slice(0, 8), slice(8, 16), slice(16, 28)


def _counts(label):
    return int(label[MG].sum()), int(label[MN].sum()), int(label[O].sum())


def generate(real_coords, real_labels, target_labels, rng=None):
    """((N, 90) structures, indices of the targets that could be filled).

    For each requested composition, pick a real structure with the same number
    of cations and the same number of anions, then reassign its cation sites to
    the requested Mg/Mn split. Geometry is untouched -- only the labels move.

    Falls back to the nearest-matching template when no exact match exists, which
    is the honest behaviour: a substitution baseline is limited by its library.
    """
    rng = rng or np.random.default_rng(0)
    donors = {}
    for i, lab in enumerate(real_labels):
        n_mg, n_mn, n_o = _counts(lab)
        donors.setdefault((n_mg + n_mn, n_o), []).append(i)
    keys = list(donors)

    out, kept = [], []
    for ti, tgt in enumerate(target_labels):
        t_mg, t_mn, t_o = _counts(tgt)
        key = (t_mg + t_mn, t_o)
        if key not in donors:
            # The template must have AT LEAST as many sites as the target needs.
            # A donor with fewer leaves target slots at 0.0, and every unfilled
            # slot decodes to the same coordinate -- which reads as coincident
            # atoms and a 0.00 A minimum distance, not as a substitution result.
            ok = [k for k in keys if k[0] >= t_mg + t_mn and k[1] >= t_o]
            key = (min(ok, key=lambda k: (k[0] - (t_mg + t_mn)) + (k[1] - t_o))
                   if ok else max(keys, key=lambda k: k[0] + k[1]))

        # Coordinates and label must come from the SAME donor, or the occupancy
        # mask selects sites the coordinates do not describe.
        di = int(rng.choice(donors[key]))
        d = real_coords[di].reshape(30, 3).copy()
        lab = real_labels[di]

        # Collect the donor's occupied cation and anion coordinates.
        cations = [d[2 + j] for j in range(16) if lab[j] > 0.5]
        anions = [d[2 + j] for j in range(16, 28) if lab[j] > 0.5]
        rng.shuffle(cations)
        if len(cations) < t_mg + t_mn or len(anions) < t_o:
            continue                      # cannot fill the target honestly
        kept.append(ti)

        new = np.zeros((30, 3), dtype=np.float32)
        new[0], new[1] = d[0], d[1]                      # keep the real cell
        # Fill exactly the slots the TARGET marks occupied. Filling the first
        # t_mg slots instead is wrong whenever the target's occupancy is not
        # contiguous: the mask then selects slots that were never written, which
        # all decode to the same coordinate and read as coincident atoms.
        cat_slots = ([j for j in range(8) if tgt[j] > 0.5]
                     + [j for j in range(8, 16) if tgt[j] > 0.5])
        an_slots = [j for j in range(16, 28) if tgt[j] > 0.5]
        for slot, xyz in zip(cat_slots, cations):
            new[2 + slot] = xyz
        for slot, xyz in zip(an_slots, anions):
            new[2 + slot] = xyz
        out.append(new.reshape(-1))
    return np.array(out, dtype=np.float32), np.array(kept, dtype=int)


def _demo():
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    from crystal_mic import min_dist, validity

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = pickle.load(open(os.path.join(here, 'datasets', 'mgmno_100.pickle'), 'rb'))
    coords = np.array([np.array(c).flatten() for c, l in raw], dtype=np.float32)
    labels = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(labels), 500)
    tgt = labels[idx]
    gen, kept = generate(coords, labels, tgt, rng)
    tgt = tgt[kept]
    print(f'filled {len(gen)} of {len(idx)} requested compositions')

    frac_valid, dists = validity(gen, tgt)
    lat = gen.reshape(-1, 30, 3)[:, 0] * 30.0
    ang = np.deg2rad(np.clip(gen.reshape(-1, 30, 3)[:, 1] * 180.0, 30, 150))
    ca, cb, cg = np.cos(ang[:, 0]), np.cos(ang[:, 1]), np.cos(ang[:, 2])
    vol = lat.prod(axis=1) * np.sqrt(np.clip(
        1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg, 1e-9, None))
    vpa = vol / tgt.sum(axis=1).clip(min=1)

    print(f'elemental substitution, {len(gen)} samples')
    print(f'  MIC-valid @1.0 A : {frac_valid*100:.1f}%')
    print(f'  valid @0.5 A     : {(dists >= 0.5).mean()*100:.1f}%')
    print(f'  volume/atom      : {np.median(vpa):.2f} A^3 (real 11.8)')
    print(f'  median min-dist  : {np.median(dists):.2f} A')
    # The baseline reuses real geometry, so it must be highly valid. If it is
    # not, the substitution logic is broken rather than the idea.
    assert frac_valid > 0.5, (
        f'substitution baseline only {frac_valid*100:.0f}% valid -- the site '
        f'reassignment is losing geometry, not the method')
    print('\nThis is the number the generator has to be compared against on '
          'validity,\nand the reason validity alone is not the claim: every '
          'sample here is a\nknown structure with relabelled sites, so novelty '
          'is ~0 by construction.')
    print('OK')


if __name__ == '__main__':
    _demo()
