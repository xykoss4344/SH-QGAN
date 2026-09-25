"""Differentiable physics terms for the crystal generator.

Everything here is torch, batched, and differentiable end-to-end back into the
generator's cell head and atom head. Three quantities, each with a job:

  madelung_energy   the classical (Hartree + ion-ion) part of the Kohn-Sham
                    total energy. This is what decides whether an ionic
                    arrangement is electrostatically sane, which no term in the
                    model previously touched.
  structure_factor  S(G) = sum_j f_j exp(2 pi i G . s_j). Exactly what a
                    diffraction experiment measures, and exactly the basis a
                    data-reupload quantum circuit natively represents.
  formal_charges    oxidation states derived from the composition label by
                    neutrality, so charge neutrality holds by construction.

Convention follows crystal_mic.py: one definition, used by training and eval
alike, with a `_demo()` that asserts against real data. Six divergent copies of
the distance check is how the 47% figure happened -- `lattice_matrix` lives here
for the same reason.
"""
import math

import numpy as np
import torch

from crystal_mic import BOX_OFFSET, BOX_SCALE

# e^2 / (4 pi eps_0) in eV.Angstrom.
COULOMB_EV_A = 14.399645

# Slot layout, shared with crystal_mic: Mg 0:8, Mn 8:16, O 16:28.
_MG, _MN, _O = slice(0, 8), slice(8, 16), slice(16, 28)

# Crude X-ray form factors: atomic number. Adequate here because the structure
# factor is used as a *relative* fingerprint, never as an absolute intensity.
_Z = torch.tensor([12.0] * 8 + [25.0] * 8 + [8.0] * 12)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
def lattice_matrix(fake):
    """(B, 3, 3) lattice in Angstrom from the generated cell head; rows are vectors.

    Single definition, imported by train_crystal.py. Row 0 of the 30x3 layout is
    lengths/30, row 1 is angles/180.
    """
    arr = fake.view(fake.shape[0], 30, 3)
    lengths = arr[:, 0] * 30.0
    angles = torch.deg2rad(torch.clamp(arr[:, 1] * 180.0, 30.0, 150.0))
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    al, be, ga = angles[:, 0], angles[:, 1], angles[:, 2]

    zero = torch.zeros_like(a)
    v1 = torch.stack([a, zero, zero], dim=-1)
    v2 = torch.stack([b * torch.cos(ga), b * torch.sin(ga), zero], dim=-1)
    cx = c * torch.cos(be)
    cy = c * (torch.cos(al) - torch.cos(be) * torch.cos(ga)) / (torch.sin(ga) + 1e-9)
    # Floor at 0.25 A^2 (cz >= 0.5 A), not 1e-6. This sqrt is inside the WGAN-GP
    # double backward via geometry_features, and d2(sqrt)/dx2 = -1/(4 x^1.5) is
    # -2.5e8 at x=1e-6 -- which is the gp=nan that killed every run at epoch 111.
    # Real cells have c^2 - cx^2 - cy^2 of 16-36, so this never binds physically.
    cz = torch.sqrt(torch.clamp(c ** 2 - cx ** 2 - cy ** 2, min=0.25))
    v3 = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([v1, v2, v3], dim=1)


def fractional(fake):
    """(B, 28, 3) true fractional coordinates, undoing the 15A re-boxing."""
    return (fake.view(fake.shape[0], 30, 3)[:, 2:] - BOX_OFFSET) / BOX_SCALE


# ─────────────────────────────────────────────────────────────────────────────
# Composition -> oxidation states
# ─────────────────────────────────────────────────────────────────────────────
def formal_charges(labels):
    """(B, 28) formal charge per slot, zero on empty slots.

    Mg is always +2 and O always -2; Mn is mixed-valence in Mg-Mn-O, so its
    charge is whatever makes the cell neutral:

        q_Mn = (2 n_O - 2 n_Mg) / n_Mn

    clamped to the physical [2, 4] range. Deriving it this way means the
    generated composition is charge-neutral by construction rather than by a
    penalty -- the SMACT composition-validity metric in the benchmark plan,
    satisfied structurally instead of scored after the fact.
    """
    n_mg = labels[:, _MG].sum(dim=1)
    n_mn = labels[:, _MN].sum(dim=1)
    n_o = labels[:, _O].sum(dim=1)
    q_mn = ((2.0 * n_o - 2.0 * n_mg) / n_mn.clamp(min=1.0)).clamp(2.0, 4.0)

    q = torch.zeros_like(labels)
    q[:, _MG] = 2.0
    q[:, _MN] = q_mn.unsqueeze(1)
    q[:, _O] = -2.0
    return q * labels


def charge_residual(labels):
    """(B,) net cell charge left over after clamping q_Mn. ~0 on real data."""
    return formal_charges(labels).sum(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Electrostatics
# ─────────────────────────────────────────────────────────────────────────────
def _images(device, n, dtype=torch.float32):
    """(P, 3) integer lattice translations with |n_i| <= n."""
    r = torch.arange(-n, n + 1, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1).reshape(-1, 3)


# Ewald convergence parameters, chosen together so both sums are converged on
# the real cells (smallest dimension measured at 2.80 A, a quarter under 4 A):
#   real space  erfc(alpha * R_CUT) = 4e-9, reached with n_img=3 (3.5 x 2.80 A)
#   reciprocal  exp(-k^2 / 4 alpha^2) < 1e-6 by |k| = 3.4, reached with k_max=4
# R_CUT is the parameter that actually sets real-space accuracy: at 6.0 A the
# neglected erfc tail is 1.3e-4 per pair, which summed over neighbours costs
# 0.03% on cubic NaCl and 0.5% on a triclinic cell -- and raising n_img alone
# does nothing, because everything past R_CUT is masked out regardless.
#
# A damped-shifted-force sum was tried first and is not usable here: it needs a
# charge-neutral cutoff sphere, and at these cell sizes the sphere holds partial
# shells, which put 9% of *real* structures at positive electrostatic energy.
N_IMG, R_CUT, ALPHA, K_MAX = 3, 9.0, 0.45, 4


def madelung_energy(fake, labels, alpha=ALPHA, r_cut=R_CUT, n_img=N_IMG,
                    k_max=K_MAX, chunk=27):
    """(B,) electrostatic energy per atom in eV, by Ewald summation.

    The classical Hartree + ion-ion part of the Kohn-Sham total energy, and the
    term that decides whether an ionic arrangement is electrostatically sane.
    Fully differentiable through the generated fractional coordinates *and* the
    lattice, so it reaches both heads.

        E = E_real + E_recip + E_self

    Real-space images are summed in chunks so peak memory does not grow with
    n_img. `_demo` validates the whole thing against pymatgen's EwaldSummation.
    """
    q = formal_charges(labels)                                    # (B, 28)
    frac = fractional(fake)                                       # (B, 28, 3)
    lattice = lattice_matrix(fake)                                # (B, 3, 3)
    vol = torch.linalg.det(lattice).abs().clamp(min=1e-3)         # (B,)

    # ── Real space: erfc-screened, short-ranged ──────────────────────────────
    pair = q.unsqueeze(2) * q.unsqueeze(1)                        # (B, 28, 28)
    eye = torch.eye(28, device=fake.device).view(1, 28, 28, 1)
    df0 = frac.unsqueeze(2) - frac.unsqueeze(1)                   # (B, 28, 28, 3)

    img = _images(fake.device, n_img, fake.dtype)                             # (P, 3)
    e_real = torch.zeros(fake.shape[0], device=fake.device)
    for s in range(0, img.shape[0], chunk):
        blk = img[s:s + chunk]
        df = df0.unsqueeze(3) + blk.view(1, 1, 1, -1, 3)          # (B, 28, 28, p, 3)
        r = torch.linalg.norm(torch.matmul(df, lattice.view(-1, 1, 1, 3, 3)),
                              dim=-1)                             # (B, 28, 28, p)
        # Floor at 0.5 A. An untrained generator puts every atom near the same
        # fractional position, so r -> 0 and q_i q_j / r reaches ~1e6 eV/atom,
        # which detonates the loss before training starts. No real pair is below
        # 1.43 A (measured), so this never activates on plausible input, and the
        # sub-Angstrom regime is min_dist_penalty's job, not electrostatics'.
        r = r.clamp(min=0.5)
        # Exclude i==j in the home image only. i==j in a *shifted* image is a
        # real interaction with the atom's own periodic replica and must count.
        home = (blk.abs().sum(dim=1) == 0).view(1, 1, 1, -1).float()
        mask = (1.0 - eye * home) * (r < r_cut).float()
        e_real = e_real + 0.5 * (pair.unsqueeze(3)
                                 * torch.erfc(alpha * r) / r * mask).sum(dim=(1, 2, 3))

    # ── Reciprocal space: the long-range tail, as a structure factor ─────────
    # Reciprocal lattice B = 2 pi (A^-1)^T, so k = 2 pi (A^-1)^T h and
    # k . r = 2 pi h . s with s the fractional coordinate -- no inverse needed
    # in the phase, only in |k|.
    h = _images(fake.device, k_max, fake.dtype)                               # (K, 3)
    h = h[h.abs().sum(dim=1) > 0]                                 # drop k = 0
    # Rows of `recip` are the reciprocal vectors b_j, so k = sum_j h_j b_j is
    # h @ recip -- NOT h @ recip.T, which is only equal for a diagonal (cubic)
    # cell and silently wrong for every triclinic one.
    recip = 2 * math.pi * torch.linalg.inv(lattice).transpose(1, 2)   # (B, 3, 3)
    k = torch.matmul(h.unsqueeze(0), recip)                        # (B, K, 3)
    k2 = (k ** 2).sum(dim=-1).clamp(min=1e-9)                      # (B, K)

    phase = 2 * math.pi * torch.matmul(frac, h.t())                # (B, 28, K)
    sf_re = (q.unsqueeze(2) * torch.cos(phase)).sum(dim=1)         # (B, K)
    sf_im = (q.unsqueeze(2) * torch.sin(phase)).sum(dim=1)
    struct = sf_re ** 2 + sf_im ** 2

    e_recip = (2 * math.pi / vol).unsqueeze(1) * (
        torch.exp(-k2 / (4 * alpha ** 2)) / k2 * struct)
    e_recip = e_recip.sum(dim=1)

    # ── Self-interaction introduced by the Gaussian screening ────────────────
    e_self = -alpha / math.sqrt(math.pi) * (q ** 2).sum(dim=1)

    n = labels.sum(dim=1).clamp(min=1.0)
    return COULOMB_EV_A * (e_real + e_recip + e_self) / n


# Measured on datasets/mgmno_100.pickle by `python crystal_physics.py`:
# electrostatic energy per atom p1 -43.37, median -26.32, p90 -5.39 eV/atom.
#
# The bounds are deliberately asymmetric. The upper one does the work: a
# positive or near-zero Madelung energy means like charges are sitting on top of
# each other, which is the failure this term exists to catch. The lower one is a
# loose safety rail at p1, not an active constraint -- it exists only so the
# generator cannot drive the energy arbitrarily negative, which it would satisfy
# by emitting one maximally-ionic arrangement for every sample.
#
# 9.3% of real rows are ABOVE the upper bound, and ~8% have positive energy even
# after the contact floors are satisfied (O-O contacts down to 1.43 A, a
# peroxide-like distance). Real data is not uniformly clean here; see the O-O
# row of the floor table in the vault.
MADELUNG_LO, MADELUNG_HI = -43.37, -5.39


def madelung_penalty(fake, labels, lo=None, hi=None):
    """Hinge holding electrostatic energy per atom inside the real range.

    Deliberately NOT a minimisation. The cheapest way to minimise a Madelung
    energy is to find one maximally-ionic arrangement and emit it for every
    sample -- exactly the mode collapse this project spent v4-v8 escaping. A
    hinge into the measured real range constrains without selecting a winner,
    the same shape as `volume_penalty`.
    """
    lo = MADELUNG_LO if lo is None else lo
    hi = MADELUNG_HI if hi is None else hi
    if lo is None or hi is None:
        raise RuntimeError("MADELUNG_LO/HI unmeasured -- run `python crystal_physics.py`")
    e = madelung_energy(fake, labels)
    return _huber(torch.relu(e - hi)).mean() + _huber(torch.relu(lo - e)).mean()


# Violations are quadratic near the bound and linear beyond it. A pure square is
# unusable here: at initialisation the electrostatic energy is ~1e5 eV/atom, and
# squaring that reaches 1e10 and takes the whole G loss to NaN on the first
# batch. Huber keeps the gradient bounded at 2*delta without ever letting it
# reach zero, so a badly wrong structure is still pushed in the right direction.
_HUBER_DELTA = 10.0


def _huber(v, delta=_HUBER_DELTA):
    return torch.where(v < delta, v ** 2, delta * (2 * v - delta))


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal space
# ─────────────────────────────────────────────────────────────────────────────
def miller_set(n_g, h_max=2):
    """(n_g, 3) smallest non-zero Miller indices, one per Friedel pair.

    S(-G) = conj(S(G)), so both halves of a pair carry the same |S|. Keep the one
    whose first non-zero component is positive, ordered by |G|^2.
    """
    out = []
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(-h_max, h_max + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if next(v for v in (h, k, l) if v != 0) < 0:
                    continue
                out.append((h * h + k * k + l * l, (h, k, l)))
    out.sort(key=lambda t: (t[0], t[1]))
    assert len(out) >= n_g, f"h_max={h_max} yields only {len(out)} G-vectors, need {n_g}"
    return torch.tensor([g for _, g in out[:n_g]], dtype=torch.float32)


def structure_factor(fake, labels, g):
    """(re, im), each (B, n_g), of the normalised structure factor.

        S(G) = sum_j f_j exp(2 pi i G . s_j) / sum_j f_j

    Normalised so S(0) would be 1, making it scale-free and comparable across
    compositions. |S(G)| is invariant under a rigid translation of all atoms and
    under any permutation of equal-species slots -- both symmetries of the
    training data, so this fingerprints the structure rather than its arbitrary
    description. `_demo` asserts both.
    """
    frac = fractional(fake)                                       # (B, 28, 3)
    f = _Z.to(fake.device, fake.dtype) * labels                               # (B, 28)

    phase = 2 * np.pi * torch.matmul(frac, g.to(fake.device, fake.dtype).t())  # (B, 28, n_g)
    norm = f.sum(dim=1).clamp(min=1e-6).unsqueeze(1)
    re = (f.unsqueeze(2) * torch.cos(phase)).sum(dim=1) / norm
    im = (f.unsqueeze(2) * torch.sin(phase)).sum(dim=1) / norm
    return re, im


def structure_factor_features(fake, labels, g):
    """(B, n_g) diffraction intensities |S(G)|^2, for the critic.

    The critic previously had to infer periodicity from 90 raw numbers, which it
    can no more do than it could infer contact distances -- the same reason
    geometry_features exists. These are the intensities an XRD measurement
    returns.

    ponytail: fixed-setting intensities, not powder-binned by |G|, so they are
    tied to the axis convention the dataset fixes. Bin by |G| in reciprocal
    Angstroms if axis-independent comparison is ever needed.
    """
    re, im = structure_factor(fake, labels, g)
    return re ** 2 + im ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Distribution matching
# ─────────────────────────────────────────────────────────────────────────────
# Every one-sided hinge in this project became a target the model saturated:
# contacts sat at the floor, volume sat at the ceiling, and validity turned out
# to be a monotonic function of cell inflation worth +6.5 eV/atom of CHGNet
# energy. See research-vault/'The Validity Bar Is Too Low.md'.
#
# A distribution loss has no bottom to sit on. Being at 1.1 A is penalised even
# though it clears every floor, and exceeding the real distance is not rewarded.
# It also does not shrink to irrelevance as it succeeds, which is what let the
# WGAN term take back over in 'Why Validity Decays.md'.
_QUANTILE_CACHE = {}


def real_quantiles(name, device=None, dtype=torch.float32):
    """Load the measured real quantile targets ('nn' or 'vpa'), cached.

    Regenerate with the measurement in `_demo` if the dataset changes.
    """
    if name not in _QUANTILE_CACHE:
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'datasets', f'real_{name}_quantiles.npy')
        _QUANTILE_CACHE[name] = torch.from_numpy(np.load(path))
    q = _QUANTILE_CACHE[name]
    return q.to(device=device, dtype=dtype) if device is not None else q.to(dtype)


def masked_pair_distances(fake, labels, pad=1e3):
    """(B, 28, 28) MIC distances with non-pairs (empty slot, self) pushed to `pad`."""
    lattice = lattice_matrix(fake)
    frac = fractional(fake)
    df = frac.unsqueeze(2) - frac.unsqueeze(1)
    df = df - torch.round(df)                              # minimum image
    d = torch.sqrt((torch.matmul(df, lattice.unsqueeze(1)) ** 2).sum(-1) + 1e-6)
    occ = labels.unsqueeze(2) * labels.unsqueeze(1)
    pair = occ * (1.0 - torch.eye(28, device=fake.device).unsqueeze(0))
    return d + (1.0 - pair) * pad


def nn_distances(fake, labels, pad=1e3):
    """(B, 28) nearest-neighbour distance per atom under the minimum image.

    Empty slots and unoccupied neighbours are pushed to `pad` so they cannot be
    anyone's nearest neighbour. Mask with `labels` before pooling.
    """
    return masked_pair_distances(fake, labels, pad).min(dim=2).values


# Species-resolved nearest neighbours: (source class, destination class).
# Slots 0-15 are cations (Mg, Mn), 16-27 are O.
_CATION = torch.arange(28) < 16
NN_CLASSES = {'cc': (_CATION, _CATION), 'co': (_CATION, ~_CATION),
              'oo': (~_CATION, ~_CATION), 'oc': (~_CATION, _CATION)}


def nn_class_values(d, labels, src, dst, pad=1e3):
    """Per-atom distance to the nearest `dst`-class atom, for `src`-class atoms."""
    dst = dst.to(d.device)
    nn = (d + (~dst).float().view(1, 1, 28) * pad).min(dim=2).values
    v = nn[(labels > 0.5) & src.to(d.device).view(1, 28)]
    return v[v < 50.0]


# Extreme quantiles are dominated by outliers -- real volume per atom runs to
# 30 A^3 at the maximum against a median of 11.25 -- so matching them adds noise
# without adding signal, and leaves a floor the loss cannot go below on a finite
# batch. Trim both tails.
_Q_TRIM = 2


def quantile_loss(values, target, trim=_Q_TRIM):
    """1-D Wasserstein-1 between `values` and a quantile target.

    For one dimension W1 is the mean absolute difference of matched quantiles,
    and torch.quantile interpolates between sorted values, so this is
    differentiable end-to-end. Two-sided by construction: too close is penalised
    and too far is penalised.
    """
    if values.numel() < 2:
        return torch.zeros((), device=values.device)
    q = torch.linspace(0, 1, target.shape[0], device=values.device,
                       dtype=values.dtype)
    if trim:
        q, target = q[trim:-trim], target[trim:-trim]
    return (torch.quantile(values, q) - target).abs().mean()


def nn_distribution_loss(fake, labels):
    """Match the per-atom nearest-neighbour distance distribution to real."""
    nn = nn_distances(fake, labels)
    vals = nn[labels > 0.5]
    vals = vals[vals < 50.0]                               # drop padded slots
    return quantile_loss(vals, real_quantiles('nn', fake.device, fake.dtype))


def nn_class_distribution_loss(fake, labels):
    """Match four species-resolved nearest-neighbour distributions to real.

    The pooled NN loss is species-blind and a cation's nearest neighbour is
    almost always O, so it never sees cation-cation or O-O contacts. With it
    and the contact floors, wave 12 passed LeMat's distance check at 83% but
    had 3.3 cation neighbours per cation within 2.9 A (real 0.65) and 2.5 O
    per O within 2.4 A (real 0.06): no ionic ordering, +2.5 eV/atom E_hull.
    The floors became targets; these are two-sided, so there is no floor to
    sit on -- cation->cation must look like real cation->cation.
    """
    d = masked_pair_distances(fake, labels)
    losses = [quantile_loss(nn_class_values(d, labels, s, t),
                            real_quantiles(f'nn_{k}', fake.device, fake.dtype))
              for k, (s, t) in NN_CLASSES.items()]
    return sum(losses) / len(losses)


def save_nn_class_quantiles(coords, labels, out_dir, n_q=65):
    """Measure and save the real species-resolved NN quantile targets."""
    import os
    vals = {k: [] for k in NN_CLASSES}
    for i in range(0, len(coords), 512):
        d = masked_pair_distances(coords[i:i + 512], labels[i:i + 512])
        for k, (s, t) in NN_CLASSES.items():
            vals[k].append(nn_class_values(d, labels[i:i + 512], s, t))
    q = torch.linspace(0, 1, n_q)
    for k, v in vals.items():
        v = torch.cat(v)
        np.save(os.path.join(out_dir, f'real_nn_{k}_quantiles.npy'), torch.quantile(v, q).numpy())
        print(f'nn {k}: median {v.median():.3f} A, p5 {torch.quantile(v, 0.05):.3f}, n={len(v)}')


def vpa_distribution_loss(fake, labels):
    """Match the volume-per-atom distribution to real.

    Replaces the [10.0, 15.6] hinge, whose ceiling the model saturated at 15.5
    while real sits at 11.25.
    """
    vol = torch.linalg.det(lattice_matrix(fake)).abs()
    vpa = vol / labels.sum(dim=1).clamp(min=1.0)
    return quantile_loss(vpa, real_quantiles('vpa', fake.device, fake.dtype))


# ─────────────────────────────────────────────────────────────────────────────
def _demo():
    """Assert the physics on real structures. These are the ground truth."""
    import os
    import pickle

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'datasets', 'mgmno_100.pickle')
    raw = pickle.load(open(path, 'rb'))
    coords = torch.tensor(np.array([np.array(c).flatten() for c, l in raw]),
                          dtype=torch.float32)
    labels = torch.tensor(np.array([np.array(l).flatten() for c, l in raw]),
                          dtype=torch.float32)

    # 1. Charge neutrality must hold on real compositions, or q_Mn is being
    #    clamped away from the value neutrality demands.
    res = charge_residual(labels).abs()
    print(f'charge residual: max {res.max():.3f} e, mean {res.mean():.3f} e')
    assert res.max() < 1e-3, f'compositions are not charge-neutral (max {res.max():.3f})'

    # 2. The image block must reach r_cut on the *smallest* real cell, or a
    #    quarter of the dataset silently loses its outer neighbour shells.
    lengths = coords.view(-1, 30, 3)[:, 0] * 30.0
    smallest = lengths.min(dim=1).values.min()
    print(f'cell lengths: smallest dimension {smallest:.2f} A, mean {lengths.mean():.2f} A')
    assert (N_IMG + 0.5) * smallest > R_CUT, (
        f'n_img={N_IMG} reaches {(N_IMG + 0.5) * smallest:.2f} A, '
        f'short of r_cut={R_CUT} on the smallest cell ({smallest:.2f} A)')

    # 3. Both Ewald sums must be converged: another shell in either must not
    #    move the energy. Underconvergence would make the hinge range an
    #    artefact of the cutoffs rather than of the chemistry.
    sub = slice(0, 64)
    e_base = madelung_energy(coords[sub], labels[sub])
    for kw in ({'n_img': N_IMG + 1}, {'k_max': K_MAX + 1}):
        shift = (e_base - madelung_energy(coords[sub], labels[sub], **kw)).abs().max()
        print(f'ewald convergence, {kw}: max shift {shift:.4f} eV/atom')
        assert shift < 0.05, f'ewald not converged in {kw} ({shift:.3f} eV/atom)'

    # 3b. The Ewald total must not depend on the real/reciprocal splitting
    #     parameter alpha. This is the test that catches errors the cutoff
    #     sweeps cannot: it found k-vectors built as h @ B.T instead of h @ B,
    #     which is identical for a cubic cell (so NaCl validated clean) and
    #     wrong for every triclinic one.
    alphas = [0.35, 0.45, 0.60]
    e_alpha = [madelung_energy(coords[sub], labels[sub], alpha=a, r_cut=14.0,
                               n_img=5, k_max=8) for a in alphas]
    spread = torch.stack(e_alpha).max(dim=0).values - torch.stack(e_alpha).min(dim=0).values
    print(f'alpha-independence over {alphas}: max spread {spread.max():.4f} eV/atom')
    assert spread.max() < 0.01, (
        f'ewald total depends on the splitting parameter ({spread.max():.3f} eV/atom) '
        f'-- one of the three terms is wrong')

    # 4. Validate against pymatgen's exact Ewald on real structures. This is the
    #    check that matters: a damped-shifted-force sum passed every internal
    #    consistency test and still put 9% of real structures at positive
    #    energy. Only an independent implementation catches that.
    from pymatgen.analysis.ewald import EwaldSummation
    from pymatgen.core import Lattice, Structure
    from crystal_mic import decode

    ours, theirs = [], []
    for i in range(24):
        species, frac, cell = decode(coords[i].numpy(), labels[i].numpy())
        q = formal_charges(labels[i:i + 1])[0]
        q = q[labels[i].bool()].numpy()
        st = Structure(Lattice.from_parameters(*cell), species, frac,
                       site_properties={'charge': q.tolist()})
        st.add_oxidation_state_by_site(q.tolist())
        theirs.append(EwaldSummation(st).total_energy / len(st))
        ours.append(madelung_energy(coords[i:i + 1], labels[i:i + 1]).item())
    ours, theirs = np.array(ours), np.array(theirs)
    err = np.abs(ours - theirs).max()
    print(f'vs pymatgen EwaldSummation: max |diff| {err:.4f} eV/atom '
          f'(ours mean {ours.mean():.2f}, theirs {theirs.mean():.2f})')
    assert err < 0.05, f'disagrees with pymatgen Ewald by {err:.3f} eV/atom'

    # 5. Ionic oxides are bound, so the bulk of real data must be negative.
    #    Not *all* of it: 9.3% of rows sit above the upper bound, driven by
    #    cation-cation contacts near 1.93 A (median 2.65 A elsewhere). That is a
    #    property of the dataset, not of this estimator -- which is why the
    #    hinge bounds are percentiles and the assertion is on the median.
    e = madelung_energy(coords, labels)
    n_pos = int((e > 0).sum())
    print(f'madelung: median {e.median():.2f} eV/atom, p1 {torch.quantile(e, 0.01):.2f}, '
          f'p90 {torch.quantile(e, 0.90):.2f}, {n_pos} positive '
          f'({100 * n_pos / len(e):.1f}%)')
    assert e.median() < -15.0, f'median electrostatic energy {e.median():.2f} is not bound'
    assert n_pos / len(e) < 0.15, (
        f'{100 * n_pos / len(e):.1f}% of real structures are electrostatically '
        f'unbound -- expected ~9%, so either the species map or the data changed')
    frac_hi = (e > MADELUNG_HI).float().mean()
    print(f'  real rows above MADELUNG_HI={MADELUNG_HI}: {100 * frac_hi:.1f}%')

    # 6. |S(G)| is translation-invariant: shifting every atom by the same vector
    #    changes only a global phase. If this fails, the head is being asked to
    #    match a quantity that depends on where the origin happens to sit.
    g = miller_set(12)
    i0 = structure_factor_features(coords, labels, g)
    shifted = coords.clone().view(-1, 30, 3)
    shifted[:, 2:] += 0.1 * BOX_SCALE * labels.unsqueeze(-1)
    i1 = structure_factor_features(shifted.view(-1, 90), labels, g)
    drift = (i0 - i1).abs().max()
    print(f'|S(G)|^2 translation drift: {drift:.2e}')
    assert drift < 1e-4, f'structure factor is not translation invariant ({drift:.2e})'

    # 7. The feature must separate real from scrambled, or the critic gains
    #    nothing. The signal is *systematic absences*, not magnitude: an ordered
    #    crystal extinguishes most low-index reflections by destructive
    #    interference and concentrates intensity in a few, while a disordered
    #    arrangement spreads intensity across all of them. Real |S|^2 is
    #    therefore LOWER on average than scrambled -- the discriminative
    #    quantity is the fraction of reflections near zero.
    rand = coords.clone().view(-1, 30, 3)
    rand[:, 2:] = BOX_OFFSET + BOX_SCALE * torch.rand_like(rand[:, 2:])
    i_rand = structure_factor_features(rand.view(-1, 90), labels, g)
    absent, absent_rand = (i0 < 0.01).float().mean(), (i_rand < 0.01).float().mean()
    print(f'|S(G)|^2 mean: real {i0.mean():.4f}, scrambled {i_rand.mean():.4f}')
    print(f'reflections extinguished (<0.01): real {absent:.2f}, scrambled {absent_rand:.2f}')
    assert absent > 2 * absent_rand, (
        f'no crystalline signal: real absences {absent:.2f} vs '
        f'scrambled {absent_rand:.2f}')

    # 8. Distribution losses must be ~0 on real data and large on scrambled.
    #    A loss that does not separate them is not a constraint.
    l_nn_real = nn_distribution_loss(coords[:512], labels[:512])
    l_vpa_real = vpa_distribution_loss(coords[:512], labels[:512])
    scram = coords[:512].clone().view(-1, 30, 3)
    scram[:, 2:] = BOX_OFFSET + BOX_SCALE * torch.rand_like(scram[:, 2:])
    l_nn_scram = nn_distribution_loss(scram.view(-1, 90), labels[:512])
    inflated = coords[:512].clone().view(-1, 30, 3)
    inflated[:, 0] *= 1.35                       # the cell-inflation exploit
    l_vpa_infl = vpa_distribution_loss(inflated.view(-1, 90), labels[:512])
    print(f'nn-distribution loss : real {l_nn_real:.4f}, scrambled {l_nn_scram:.4f}')
    print(f'vpa-distribution loss: real {l_vpa_real:.4f}, inflated x1.35 {l_vpa_infl:.4f}')
    assert l_nn_real < 0.10, f'real data scores {l_nn_real:.3f} on its own distribution'
    assert l_nn_scram > 5 * l_nn_real, 'nn-distribution loss does not separate scrambled'
    assert l_vpa_infl > 5 * max(l_vpa_real, 1e-3), (
        'vpa-distribution loss does not punish cell inflation -- which is the '
        'exploit it exists to close')

    print('OK')


if __name__ == '__main__':
    _demo()
