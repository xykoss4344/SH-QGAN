"""DFT-surrogate force distillation into the generator.

CHGNet is trained on Materials Project DFT (GGA/GGA+U) energies and forces, so
its forces are a learned stand-in for the Kohn-Sham energy gradient. This module
feeds that gradient to the generator *without differentiating through CHGNet*.

The trick is that a force is already a gradient. Since

    F_i = -dE/dr_i

the surrogate objective

    L = -sum_i F_i.detach() . r_i

has, by construction,

    dL/dr_i = -F_i = dE/dr_i

so minimising L performs exact first-order descent on the DFT-surrogate energy
while costing one CHGNet *forward* per call -- no double backward, no autograd
through a 400k-parameter GNN. `_demo` validates this against finite-difference
CHGNet energies, which is the only check that catches a sign or scale error.

Kept in its own module so `crystal_physics` and `train_crystal` remain importable
without chgnet installed; it is only imported when --lambda_force is set.
"""
import numpy as np
import torch

from crystal_mic import SPECIES_MAP, min_dist
from crystal_physics import fractional, lattice_matrix

# Structures with a contact below this are not passed to CHGNet. Its neighbour
# list explodes on overlapping atoms -- a batch of untrained-generator output
# exhausted memory outright -- and the energy is meaningless there anyway. The
# sub-Angstrom regime is min_dist_penalty's job.
MIN_DIST_FOR_CHGNET = 0.7

# Forces are clamped before use. A 0.3 A contact produces hundreds of eV/A and
# would dominate every other term in the objective.
MAX_FORCE = 50.0


class ForceDistiller:
    """Feeds DFT-surrogate energy gradients to the generator.

    Args:
        k:         structures sampled per call. CHGNet costs ~24 ms each on CPU.
        every:     apply once every this many generator steps.
        gate:      per-atom force norm (eV/A) below which a structure is left
                   alone. Set from real data by `measure_force_gate` -- the point
                   is to pull bad geometry toward the DFT basin, not to drag
                   every sample toward one minimum, which is how an energy term
                   causes mode collapse.
    """

    def __init__(self, k=4, every=5, gate=1.0, max_force=MAX_FORCE, device='cpu'):
        from chgnet.model import CHGNet
        # CHGNet must be told the device explicitly -- it defaults to CPU, so
        # passing a device here and not using it would leave the single most
        # expensive term in the objective running on the wrong hardware.
        self.model = CHGNet.load(verbose=False, use_device=str(device))
        self.k, self.every, self.gate, self.max_force = k, every, gate, max_force
        self.device = device
        self._step = 0
        self.last_n = 0          # structures that actually contributed
        self.last_fmax = float('nan')

    def _structures(self, fake, labels, idx):
        """Decode selected rows to pymatgen Structures, skipping unusable ones."""
        from pymatgen.core import Lattice, Structure
        out = []
        fake_np, lab_np = fake.detach().cpu().numpy(), labels.detach().cpu().numpy()
        for i in idx:
            if min_dist(fake_np[i], lab_np[i]) < MIN_DIST_FOR_CHGNET:
                continue
            occ = lab_np[i].astype(bool)
            arr = fake_np[i].reshape(30, 3)
            frac = (arr[2:][occ] - 1.0 / 6.0) / (2.0 / 3.0)
            species = [SPECIES_MAP[j] for j in range(28) if occ[j]]
            if len(species) < 2:
                continue
            lengths = np.clip(arr[0] * 30.0, 2.0, 30.0)
            angles = np.clip(arr[1] * 180.0, 30.0, 150.0)
            try:
                st = Structure(Lattice.from_parameters(*lengths, *angles),
                               species, frac)
            except Exception:
                continue
            out.append((i, st))
        return out

    def loss(self, fake, labels):
        """Scalar surrogate whose gradient is the DFT-surrogate energy gradient.

        Returns an exact zero (detached from the graph) on skipped steps, so the
        caller can add it unconditionally.
        """
        self._step += 1
        self.last_n = 0
        if self._step % self.every != 0:
            return torch.zeros((), device=fake.device)

        b = fake.shape[0]
        idx = np.random.choice(b, size=min(self.k, b), replace=False)
        picked = self._structures(fake, labels, idx)
        if not picked:
            return torch.zeros((), device=fake.device)

        try:
            preds = self.model.predict_structure([st for _, st in picked],
                                                 batch_size=1)
        except Exception:
            # OOM or a degenerate cell: skip rather than kill a long run.
            return torch.zeros((), device=fake.device)
        if isinstance(preds, dict):
            preds = [preds]

        # Cartesian positions, differentiable w.r.t. both heads: r = frac @ A.
        frac = fractional(fake)                                  # (B, 28, 3)
        cart = torch.matmul(frac, lattice_matrix(fake))          # (B, 28, 3)

        total, n_used, fmax_seen = torch.zeros((), device=fake.device), 0, 0.0
        for (i, st), pred in zip(picked, preds):
            f = torch.as_tensor(np.asarray(pred['f']), dtype=fake.dtype,
                                device=fake.device)              # (n_occ, 3)
            fmax = float(f.norm(dim=-1).max())
            fmax_seen = max(fmax_seen, fmax)
            # Structures already as relaxed as real data are left alone.
            if fmax < self.gate:
                continue
            f = f.clamp(-self.max_force, self.max_force)
            occ = labels[i].bool()
            # -F . r : gradient is +dE/dr, i.e. descent on the DFT energy.
            total = total - (f * cart[i][occ]).sum() / f.shape[0]
            n_used += 1

        self.last_n, self.last_fmax = n_used, fmax_seen
        return total / max(n_used, 1)


class RelaxDistiller(ForceDistiller):
    """Pull generated structures toward their own short CHGNet relaxation.

    A force is one step's direction; a short relaxation says where the nearby
    minimum IS. Measured on the epoch-50 refiner model: raw structures drop
    1.75 eV/atom on relaxation and 81% relax into a different structure
    (median RMSD 0.42 A) -- the relaxer, not the generator, was doing the work.
    This trains the generator to emit the relaxed geometry directly, which is
    what RMSD-to-relaxed (MatterGen's quality measure) rewards.

        L = mean_atoms || MIC(frac_gen - frac_relaxed) @ A ||^2   (A^2)

    The relaxed target is detached; the cell is held fixed during relaxation
    so target and generated fractional coordinates share a lattice.
    """

    def __init__(self, k=4, every=5, steps=20, fmax=0.1, device='cpu'):
        super().__init__(k=k, every=every, gate=0.0, device=device)
        from chgnet.model import StructOptimizer
        self.opt = StructOptimizer(model=self.model)
        self.steps, self.fmax = steps, fmax

    def loss(self, fake, labels):
        self._step += 1
        self.last_n = 0
        if self._step % self.every != 0:
            return torch.zeros((), device=fake.device)
        b = fake.shape[0]
        idx = np.random.choice(b, size=min(self.k, b), replace=False)
        picked = self._structures(fake, labels, idx)
        if not picked:
            return torch.zeros((), device=fake.device)

        frac = fractional(fake)                                  # (B, 28, 3)
        lat = lattice_matrix(fake)                               # (B, 3, 3)
        total, n_used = torch.zeros((), device=fake.device), 0
        for i, st in picked:
            try:
                r = self.opt.relax(st, fmax=self.fmax, steps=self.steps,
                                   relax_cell=False, verbose=False)
            except Exception:
                continue
            target = torch.as_tensor(r['final_structure'].frac_coords,
                                     dtype=fake.dtype, device=fake.device)
            occ = labels[i].bool()
            d = frac[i][occ] - target
            d = d - torch.round(d)                               # minimum image
            total = total + (torch.matmul(d, lat[i]) ** 2).sum(-1).mean()
            n_used += 1
        self.last_n = n_used
        return total / max(n_used, 1)


def measure_force_gate(coords, labels, model=None, n=64, percentile=90):
    """Per-atom force norm at `percentile` of real structures, in eV/A.

    Real Mg-Mn-O cells here are unrelaxed, so their CHGNet forces are not zero.
    Gating at their p90 means the term fires only on structures worse than the
    training data rather than on everything.
    """
    from chgnet.model import CHGNet
    from pymatgen.core import Lattice, Structure
    from crystal_mic import decode

    model = model or CHGNet.load(verbose=False)
    out = []
    for i in range(min(n, len(coords))):
        species, frac, cell = decode(np.asarray(coords[i]), np.asarray(labels[i]))
        if len(species) < 2:
            continue
        try:
            st = Structure(Lattice.from_parameters(*cell), species, frac)
            out.append(float(np.linalg.norm(model.predict_structure(st)['f'],
                                            axis=-1).max()))
        except Exception:
            continue
    return float(np.percentile(out, percentile)) if out else 1.0


def _demo():
    """Validate the surrogate gradient against finite-difference CHGNet energy.

    If the sign or scale is wrong the term silently pushes structures *up* the
    energy surface, which no loss curve would reveal.
    """
    import pickle
    import os
    import warnings
    warnings.filterwarnings('ignore')
    from chgnet.model import CHGNet
    from pymatgen.core import Lattice, Structure
    from crystal_mic import decode

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'datasets', 'mgmno_100.pickle')
    raw = pickle.load(open(path, 'rb'))
    coords = np.array([np.array(c).flatten() for c, l in raw], dtype=np.float32)
    labels = np.array([np.array(l).flatten() for c, l in raw], dtype=np.float32)

    model = CHGNet.load(verbose=False)
    gate = measure_force_gate(coords, labels, model, n=48)
    print(f'real-data force gate (p90 of max |F|): {gate:.3f} eV/A')
    assert gate > 0, 'gate must be positive'

    # Finite difference: E(r + eps*d) - E(r) should equal -sum_i F_i . (eps*d_i).
    rng = np.random.default_rng(0)
    ok = 0
    # Spread across the file: consecutive rows are augmented copies of the same
    # structure, and several sit at configurations where a random displacement
    # produces no first-order energy change, so the check has nothing to measure.
    for i in np.linspace(0, len(coords) - 1, 8, dtype=int):
        species, frac, cell = decode(coords[i], labels[i])
        if len(species) < 2:
            continue
        lat = Lattice.from_parameters(*cell)
        st = Structure(lat, species, frac)
        p0 = model.predict_structure(st)
        e0, f0 = float(p0['e']) * len(st), np.asarray(p0['f'])

        # Central difference. A one-sided step at 1e-3 A moves the energy by
        # ~1e-5 eV, which is CHGNet's own numerical noise -- the check then
        # measures rounding, not physics. Central differencing also cancels the
        # second-order term, so 0.01 A is both resolvable and still linear.
        eps = 0.01
        d = rng.normal(size=f0.shape)
        d /= np.abs(d).max()                      # max per-atom displacement = eps
        r0 = lat.get_cartesian_coords(frac)
        e_plus = float(model.predict_structure(
            Structure(lat, species, lat.get_fractional_coords(r0 + eps * d))
        )['e']) * len(st)
        e_minus = float(model.predict_structure(
            Structure(lat, species, lat.get_fractional_coords(r0 - eps * d))
        )['e']) * len(st)

        actual = 0.5 * (e_plus - e_minus)
        predicted = -(f0 * (eps * d)).sum()
        print(f'  struct {i}: dE actual {actual:+.6f} eV, '
              f'from forces {predicted:+.6f} eV')
        # Sign agreement is the thing that matters; magnitudes agree to the
        # accuracy of a first-order expansion.
        if abs(actual) > 1e-4:
            assert np.sign(actual) == np.sign(predicted), (
                f'force sign disagrees with energy change on structure {i} -- '
                f'the surrogate would push uphill')
            assert abs(actual - predicted) < 0.3 * abs(actual) + 1e-4, (
                f'force-predicted dE {predicted:.6f} does not match '
                f'actual {actual:.6f}')
            ok += 1
    assert ok >= 3, f'only {ok} structures gave a usable finite-difference check'
    print(f'OK ({ok} structures validated)')


if __name__ == '__main__':
    _demo()
