
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.autograd as autograd
from models.QINR_Crystal import PQWGAN_CC_Crystal
from probe_collapse import probe_generator, VPA_OK, Z_DEAD_THRESHOLD
import crystal_mic
from crystal_mic import min_separation_matrix
from crystal_physics import (lattice_matrix, madelung_energy, madelung_penalty,
                             miller_set, nn_distribution_loss, structure_factor,
                             structure_factor_features, vpa_distribution_loss)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Penalty (WGAN-GP)
# ─────────────────────────────────────────────────────────────────────────────
def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """
    Gradient penalty from Gulrajani et al. (2017).
    Enforces the 1-Lipschitz constraint on the classical critic.
    lambda_gp = 10 is applied at the call site.
    """
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(critic_input(interpolates, labels))

    fake = torch.ones(real_samples.shape[0], 1).to(device).requires_grad_(False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ─────────────────────────────────────────────────────────────────────────────
# Q-Head — Slot + Count branches (InfoGAN auxiliary classifier)
# ─────────────────────────────────────────────────────────────────────────────
class QHead(nn.Module):
    """
    InfoGAN-style auxiliary classifier attached to the critic.

    Input  : raw crystal coordinate vector (90-dim).
    Outputs:
      Slot predictions  — binary occupancy per slot (28 logits, BCE)
        mg_slot  : 8 logits   (indices 0:8)
        mn_slot  : 8 logits   (indices 8:16)
        o_slot   : 12 logits  (indices 16:28)
      Count predictions — total atom count per element (multi-class, CE)
        mg_count : 9 logits   (n_Mg  ∈ {0…8})
        mn_count : 9 logits   (n_Mn  ∈ {0…8})
        o_count  : 13 logits  (n_O   ∈ {0…12})
    """
    MG_SLOTS = 8;  MN_SLOTS = 8;  O_SLOTS = 12
    MG_CLASSES = 9; MN_CLASSES = 9; O_CLASSES = 13   # 0…8 and 0…12

    def __init__(self, data_dim=90):
        super().__init__()
        hidden = 256
        self.shared = nn.Sequential(
            nn.Linear(data_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
        )
        # Slot branches (binary occupancy)
        self.mg_slot  = nn.Linear(hidden, self.MG_SLOTS)
        self.mn_slot  = nn.Linear(hidden, self.MN_SLOTS)
        self.o_slot   = nn.Linear(hidden, self.O_SLOTS)
        # Count branches (multi-class)
        self.mg_count = nn.Linear(hidden, self.MG_CLASSES)
        self.mn_count = nn.Linear(hidden, self.MN_CLASSES)
        self.o_count  = nn.Linear(hidden, self.O_CLASSES)

    def forward(self, x):
        h = self.shared(x)
        return (
            self.mg_slot(h),   self.mn_slot(h),   self.o_slot(h),
            self.mg_count(h),  self.mn_count(h),  self.o_count(h),
        )

    @staticmethod
    def _count_targets(labels_28):
        """Derive integer count targets from 28-dim binary slot labels."""
        n_mg = labels_28[:, 0:8].sum(dim=1).long().clamp(0, 8)
        n_mn = labels_28[:, 8:16].sum(dim=1).long().clamp(0, 8)
        n_o  = labels_28[:, 16:28].sum(dim=1).long().clamp(0, 12)
        return n_mg, n_mn, n_o

    def q_real_loss(self, coords, labels_28):
        """
        L_Q_real = CE_slot(Mg) + CE_slot(Mn) + CE_slot(O)
                 + 0.3 * [CE_count(Mg) + CE_count(Mn) + CE_count(O)]

        Used inside D loss:  L_D = L_critic + L_gp - L_Q_real
        Subtracting rewards D (and Q-Head) for accurately predicting
        composition from real crystals.
        """
        mg_s, mn_s, o_s, mg_c, mn_c, o_c = self.forward(coords)
        bce = nn.BCEWithLogitsLoss()
        ce  = nn.CrossEntropyLoss()
        n_mg, n_mn, n_o = self._count_targets(labels_28)

        slot_loss  = (bce(mg_s, labels_28[:, 0:8])
                    + bce(mn_s, labels_28[:, 8:16])
                    + bce(o_s,  labels_28[:, 16:28]))
        count_loss = (ce(mg_c, n_mg) + ce(mn_c, n_mn) + ce(o_c, n_o))
        return slot_loss + 0.3 * count_loss

    def q_fake_loss(self, coords, labels_28):
        """
        L_Q_fake = CE_count(Mg) + CE_count(Mn) + CE_count(O)

        Used inside G loss:  L_G = -E[D(G(z))] + lambda_q * L_Q_fake
        Penalises the quantum generator if the Q-Head cannot recover
        the correct atom-count composition from the generated crystal.

        The weight lives at the call site now. This used to apply a hidden 0.01
        on top of the caller's weight, which is why conditioning was effectively
        switched off.
        """
        _, _, _, mg_c, mn_c, o_c = self.forward(coords)
        ce = nn.CrossEntropyLoss()
        n_mg, n_mn, n_o = self._count_targets(labels_28)
        return ce(mg_c, n_mg) + ce(mn_c, n_mn) + ce(o_c, n_o)


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable minimum-image distance penalty
# ─────────────────────────────────────────────────────────────────────────────
_SEP = torch.from_numpy(min_separation_matrix())


def set_floors(name):
    """Select the contact-floor set and rebuild the cached matrix.

    _SEP is built at import time, so changing floors without this is a no-op --
    the flag would appear to work and change nothing.
    """
    global _SEP
    crystal_mic.set_floors(name)
    _SEP = torch.from_numpy(min_separation_matrix())


# One lattice definition, in crystal_physics. The local copy that used to live
# here is exactly the pattern that produced six divergent distance checks.
_lattice = lattice_matrix


# Measured on datasets/mgmno_100.pickle: volume per atom 11.77 +/- 1.84 A^3,
# 5th-95th percentile 10.0-15.6. Oxides are close-packed; this is a tight,
# physically meaningful constraint.
VPA_LO, VPA_HI = 10.0, 15.6


def volume_penalty(fake, labels, lo=None, hi=None):
    """Keep volume per atom inside the range real Mg-Mn-O oxides occupy.

    Without this, inflating the lattice is the cheapest way to satisfy the
    minimum-distance penalty, and the model takes it: v9 reached 98.8% "valid"
    at an ~11A cell against a real 6.1-6.4A, i.e. ~6x the true volume per atom.
    Validity bought that way is worthless -- density is a standard benchmark
    metric and such structures would score terribly on it and on E_hull.

    Log-space hinge so it is scale-free, with a dead zone across the real
    percentile range so the model is constrained but not pinned to the mean.
    """
    lo = VPA_LO if lo is None else lo
    hi = VPA_HI if hi is None else hi
    lat = _lattice(fake)
    vol = torch.linalg.det(lat).abs()                     # (B,)
    n   = labels.sum(dim=1).clamp(min=1.0)
    vpa = (vol / n).clamp(min=1e-3)
    logv = torch.log(vpa)
    return (torch.relu(logv - np.log(hi)) ** 2
            + torch.relu(np.log(lo) - logv) ** 2).mean()


def _pair_distances(fake, labels):
    """(B, 28, 28) interatomic distances in Angstrom under the minimum image.

    ponytail: single-image MIC wrap, exact for near-orthogonal cells and a good
    approximation otherwise. Swap for the full 27-image sum if strongly
    triclinic cells start mattering.
    """
    arr     = fake.view(fake.shape[0], 30, 3)
    lattice = _lattice(fake)

    # Undo the 15A re-boxing to get true fractional coords, then pair them up.
    frac = (arr[:, 2:] - 1.0 / 6.0) / (2.0 / 3.0)         # (B, 28, 3)
    df   = frac.unsqueeze(2) - frac.unsqueeze(1)          # (B, 28, 28, 3)
    df   = df - torch.round(df)                           # minimum image
    cart = torch.matmul(df, lattice.unsqueeze(1))         # (B, 28, 28, 3)
    # sqrt(sum^2 + eps), NOT norm(cart + 1e-12). The gradient of a norm at zero
    # is cart/norm, which reaches ~1e12 when two atoms coincide and takes the
    # whole G loss to NaN -- this run NaN'd at epoch 2 and then trained on NaN
    # for 40 epochs without a single warning. The epsilon bounds the gradient at
    # 1e3, and distances below 1e-3 A are physically meaningless anyway.
    return torch.sqrt((cart ** 2).sum(dim=-1) + 1e-6)     # (B, 28, 28)


def min_dist_penalty(fake, labels, threshold=1.0, species_aware=True):
    """Penalty on interatomic distances that are too close.

    Two things this gets right that the obvious version does not:

    1. Violations are summed PER STRUCTURE, not averaged over pairs. With 28
       atoms there are up to 378 pairs, so averaging dilutes a single clashing
       pair ~378x -- the v8 run showed Dist=0.0028 (apparently satisfied) while
       only 17% of structures were valid. One bad pair invalidates a whole
       crystal, so it must carry the weight of one bad pair.
    2. `species_aware` uses per-pair chemical floors (Mg-O 1.7A, O-O 2.0A, ...)
       rather than a flat bar. An Mg-O contact at 1.1A clears the 1.0A validity
       threshold and is still chemically nonsense.

    """
    dist = _pair_distances(fake, labels)

    # Only real atom pairs count: both slots occupied, and i != j.
    occ  = labels.unsqueeze(2) * labels.unsqueeze(1)      # (B, 28, 28)
    pair = occ * (1.0 - torch.eye(28, device=fake.device)).unsqueeze(0)

    if species_aware:
        floor = _SEP.to(fake.device).unsqueeze(0)         # (1, 28, 28)
    else:
        floor = torch.full_like(dist, threshold)

    viol = torch.relu(floor - dist) ** 2 * pair
    # Sum within a structure, mean across the batch -- see docstring point 1.
    return viol.sum(dim=(1, 2)).mean()


def geometry_features(x, labels, k=8):
    """k smallest interatomic distances per structure, for the critic.

    The critic saw 90 raw numbers and had to infer geometry from them, which it
    cannot really do: whether two fractional coordinates are close depends on the
    lattice, which is six other numbers away. Handing it the actual sorted
    contact distances makes "this structure has atoms on top of each other" a
    feature rather than something it must learn to compute.
    """
    # Clamp before the critic sees it. These features are inside the gradient
    # penalty, which takes a SECOND derivative: sqrt(x + 1e-6) has d2/dx2 of
    # -1/(4(x+eps)^1.5) ~ -2.5e8 near zero, so bounding the first derivative is
    # not enough and the GP goes NaN (observed at epoch 111). 0.3 A is far below
    # any physical contact, so this never binds on a plausible structure.
    d = _pair_distances(x, labels).clamp(min=0.3)                    # (B, 28, 28)
    occ = labels.unsqueeze(2) * labels.unsqueeze(1)
    pair = occ * (1.0 - torch.eye(28, device=x.device)).unsqueeze(0)
    # Non-pairs pushed out of the way so they never enter the k smallest.
    d = d + (1.0 - pair) * 1e3
    flat = d.reshape(d.shape[0], -1)
    kk = min(k, flat.shape[1])
    smallest = torch.topk(flat, kk, dim=1, largest=False).values
    return torch.clamp(smallest, max=20.0) / 20.0


# Set False to reproduce the flat-critic ablation.
USE_GEOMETRY_FEATURES = True
GEOM_K = 8

# Diffraction intensities |S(G)|^2 over the 12 smallest Miller indices. The
# critic can no more infer periodicity from 90 raw numbers than it could infer
# contact distances -- and periodicity is what separates a crystal from a bag of
# atoms. Measured on real data: 47% of these reflections are extinguished by
# destructive interference against 10% for scrambled coordinates, so this is a
# strong signal rather than a decorative one.
#
# 12 matches the qubit count deliberately: the quantum structure-factor readout
# emits an amplitude per qubit over this same Miller set.
# Default OFF. Left on by default this silently changed the critic for every
# run, including the v10 reference (input dim 138 against 126), so nothing could
# be compared to the published v10 numbers and the feature's own effect could not
# be isolated. Set by --sf_features.
USE_SF_FEATURES = False
N_G = 12
G_VECTORS = miller_set(N_G)


def mode_seeking_loss(generator, z, labels, z_dim, device):
    """Reward the generator for making z matter (Mao et al. 2019, MSGAN).

        L = - mean( ||G(z1,l) - G(z2,l)||_1 / ||z1 - z2||_1 )

    Every other term in this objective is satisfiable by a deterministic map
    from label to structure: the WGAN critic sees a batch, and the distribution
    losses are pooled over one, so label variety alone reproduces the real
    distribution and z contributes nothing. The trunk then zeroes its own z
    weights -- measured as std_z decaying to 0.002 in every wave 5 run, and to
    0.005 in wave 6 at a fifth of the weight.

    This is the only term that is *unsatisfiable* without z, because it compares
    two outputs at the SAME label that differ only in z.

    Costs one extra generator forward per G step (~110 ms, and G steps are one
    batch in five).
    """
    z2 = torch.randn_like(z)
    out1 = generator(torch.cat([z, labels], dim=1))
    out2 = generator(torch.cat([z2, labels], dim=1))
    d_out = (out1 - out2).abs().mean(dim=1)
    d_z = (z - z2).abs().mean(dim=1)
    return -(d_out / (d_z + 1e-5)).mean()


def sf_consistency_loss(rho, fake, labels, g=None):
    """Tie the circuit's emitted amplitudes to the crystal that was produced.

        L = || rho_circuit(G) - S_analytic(G) ||^2

    This is what stops the structure-factor readout being decorative. Without
    it the circuit could emit anything and the residual path around it would
    absorb the difference; with it the circuit has its own target and must
    genuinely encode the density it is claiming to describe.

    Both sides are bounded: expectation values lie in [-1, 1] and S is
    normalised by sum_j f_j, so no rescaling is needed between them.
    """
    g = G_VECTORS if g is None else g
    re, im = structure_factor(fake, labels, g)
    target = torch.stack([re, im], dim=-1)                # (B, n_g, 2)
    return ((rho - target) ** 2).mean()


SKIPS = {'total': 0, 'streak': 0}
MAX_SKIP_STREAK = 20


def step_if_finite(module, optimizer, name, epoch, i, clip=10.0):
    """Clip and step, unless the gradient is non-finite; then drop the step.

    Checking only the loss is not enough. w8_ms20_s4 died with d_loss=nan at
    epoch 7 batch 0 after a finite loss on the previous batch: the GP's double
    backward produced a NaN *gradient*, Adam wrote it into the critic, and the
    next forward was NaN. Same "batch 0" signature as the epoch-111 deaths.
    Dropping a rare bad step is what AMP's GradScaler does; a run of them means
    something is really broken, so that still aborts.
    """
    params = [p for p in module.parameters() if p.grad is not None]
    norm = torch.nn.utils.clip_grad_norm_(params, clip if clip else float('inf'))
    if torch.isfinite(norm):
        optimizer.step()
        SKIPS['streak'] = 0
        return True
    optimizer.zero_grad(set_to_none=True)
    SKIPS['total'] += 1
    SKIPS['streak'] += 1
    print(f"  [skip] non-finite {name} gradient at epoch {epoch} batch {i} "
          f"(skipped {SKIPS['total']} total)", flush=True)
    if SKIPS['streak'] > MAX_SKIP_STREAK:
        raise RuntimeError(f"{SKIPS['streak']} consecutive non-finite gradients; "
                           "aborting rather than training on nothing.")
    return False


def critic_input(x, labels):
    """Everything the critic sees: structure, label, geometry, diffraction."""
    parts = [x, labels]
    if USE_GEOMETRY_FEATURES:
        parts.append(geometry_features(x, labels, k=GEOM_K))
    if USE_SF_FEATURES:
        parts.append(structure_factor_features(x, labels, G_VECTORS))
    return torch.cat(parts, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # The circuit simulation does not thread: measured 257.7 ms/batch at one
    # thread against 281.6 ms at sixteen, i.e. contention makes it slower. One
    # thread per process and many processes in parallel is ~14x the throughput
    # of a single wide run. See bench_device.py.
    torch.set_num_threads(args.num_threads)
    global USE_SF_FEATURES, VPA_LO, VPA_HI
    VPA_LO, VPA_HI = args.vpa_lo, args.vpa_hi
    print(f"Volume prior: {VPA_LO}-{VPA_HI} A^3/atom (real median 11.25)")
    USE_SF_FEATURES = args.sf_features
    set_floors(args.floors)
    print(f"Contact floors: {args.floors}  |  Mg-Mg {_SEP[0, 0]:.2f} A, "
          f"O-O {_SEP[-1, -1]:.2f} A")
    # Default cpu, deliberately. bench_device.py measures this workload as
    # SLOWER on the RTX 5080 than on CPU -- 119 s/epoch against 93 -- because at
    # 12 qubits the state vector is 4096 complex numbers and the circuit
    # simulation is kernel-launch bound. Auto-selecting cuda also breaks
    # parallel runs outright: eight processes each claiming a GPU context
    # exhausted VRAM and took five of eight down with allocation failures.
    # Revisit above ~20 qubits, where the state vector gets big enough to win.
    device = torch.device(args.device)
    print(f"Using device: {device}  |  seed: {args.seed}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, 'rb') as f:
        raw_data = pickle.load(f)

    train_data_coords, train_data_labels = [], []
    for c, l in raw_data:
        train_data_coords.append(np.array(c).flatten())
        train_data_labels.append(np.array(l).flatten())

    train_data_coords = np.array(train_data_coords, dtype=np.float32)
    train_data_labels = np.array(train_data_labels, dtype=np.float32)

    dataset    = TensorDataset(torch.from_numpy(train_data_coords),
                               torch.from_numpy(train_data_labels))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    data_dim  = 90
    label_dim = 28
    z_dim     = args.z_dim

    # ── Model ─────────────────────────────────────────────────────────────────
    gen_input_dim    = z_dim + label_dim        # generator: noise + label
    critic_input_dim = (data_dim + label_dim
                        + (GEOM_K if USE_GEOMETRY_FEATURES else 0)
                        + (N_G if USE_SF_FEATURES else 0))
    print("Initializing QINR Crystal Model...")
    gan = PQWGAN_CC_Crystal(
        input_dim_g   = gen_input_dim,
        output_dim    = data_dim,
        input_dim_d   = critic_input_dim,
        hidden_features = args.hidden_features,
        hidden_layers   = args.hidden_layers,
        spectrum_layer  = args.spectrum_layer,
        use_noise       = args.use_noise,
        sf_head         = args.sf_head,
        split_head      = not args.no_split_head,
    )

    generator = gan.generator.to(device)
    critic    = gan.critic.to(device)
    q_head    = QHead(data_dim=data_dim).to(device)

    # EMA of the generator weights (as in StyleGAN / diffusion models). The
    # probe, best-checkpoint selection and saved 'generator' all use the EMA
    # copy, so a reported number is not one noisy step of an oscillating GAN.
    # Training itself is unchanged: the critic still plays the live generator.
    ema = None
    if args.ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(generator, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
                            use_buffers=True)
    eval_gen = ema.module if ema is not None else generator

    # ── DFT-surrogate force distillation (optional, needs chgnet) ─────────────
    distiller = None
    if args.lambda_force > 0:
        from dft_distill import ForceDistiller, measure_force_gate
        gate = (args.force_gate if args.force_gate > 0 else
                measure_force_gate(train_data_coords, train_data_labels, n=48))
        distiller = ForceDistiller(k=args.force_k, every=args.force_every,
                                   gate=gate, device=device)
        print(f"Force distillation on: k={args.force_k}, every={args.force_every} "
              f"G-steps, gate={gate:.3f} eV/A (real-data p90)")

    # ── Optimizers ────────────────────────────────────────────────────────────
    # Per specification: lr_critic=0.00005, lr_generator=0.000025
    optimizer_C = Adam(critic.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    optimizer_Q = Adam(q_head.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    optimizer_G = Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.9))

    lambda_gp   = 10
    n_critic    = 5     # critic must stay ahead of G in WGAN-GP
    lambda_q    = args.lambda_q      # composition conditioning weight on G
    # Min-distance penalty ramps in so the WGAN signal stabilises first.
    lambda_dist = args.lambda_dist
    lambda_vol  = args.lambda_vol
    warmup_dist = args.warmup_dist
    lambda_mad  = args.lambda_madelung
    lambda_sf   = args.lambda_sf
    lambda_nn   = args.lambda_nn
    lambda_vd   = args.lambda_vpa_dist
    lambda_ms   = args.lambda_mode_seek

    # ── Loss tracking ─────────────────────────────────────────────────────────
    epoch_losses = {
        'epoch': [], 'd_loss': [], 'wasserstein': [],
        'q_real_loss': [], 'q_fake_loss': [], 'dist_loss': [], 'total_g_loss': []
    }

    # Best-checkpoint tracking. A peak is not a result, but losing the peak
    # entirely is worse -- v10's 78.1% at epoch 20 was only recoverable because
    # save_interval happened to land on it.
    best = {'valid': -1.0, 'epoch': -1, 'vpa': float('nan'), 'std_z': float('nan')}

    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Resuming from {args.resume_checkpoint}...")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        generator.load_state_dict(ckpt.get('generator_live', ckpt['generator']))
        if ema is not None:
            ema.module.load_state_dict(ckpt['generator'])
        critic.load_state_dict(ckpt['critic'])
        q_head.load_state_dict(ckpt['q_head'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")

    for epoch in range(start_epoch, args.n_epochs):
        ep_d = ep_w = ep_qr = ep_qf = ep_dist = ep_g = 0.0
        dist_w = min(1.0, epoch / max(warmup_dist, 1)) * lambda_dist
        n_d = n_g = 0

        for i, (real_imgs, labels) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            bs        = real_imgs.shape[0]

            # ── Critic step ───────────────────────────────────────────────
            optimizer_C.zero_grad()
            optimizer_Q.zero_grad()

            z         = torch.randn(bs, z_dim).to(device)
            gen_input = torch.cat([z, labels], dim=1)
            # no_grad, not .detach(): detach builds the full autograd graph
            # through both quantum circuits and then discards it. The critic
            # step never backpropagates into G, so that graph is pure waste --
            # measured 110.4 ms with grad against 76.8 ms without, on 325
            # batches an epoch. BatchNorm running stats update either way, so
            # this is numerically identical.
            with torch.no_grad():
                fake_imgs = generator(gen_input)

            d_real    = critic(critic_input(real_imgs, labels))
            d_fake    = critic(critic_input(fake_imgs, labels))

            gp         = compute_gradient_penalty(critic, real_imgs, fake_imgs, labels, device)
            l_critic   = torch.mean(d_fake) - torch.mean(d_real)
            wasserstein = torch.mean(d_real) - torch.mean(d_fake)   # logged

            # Critic loss is pure WGAN-GP now.
            d_loss = l_critic + lambda_gp * gp
            if not torch.isfinite(d_loss):
                raise RuntimeError(
                    f"d_loss is {d_loss.item()} at epoch {epoch} batch {i} "
                    f"(gp={gp.item():.4g}). Aborting rather than training on NaN.")
            d_loss.backward()
            step_if_finite(critic, optimizer_C, 'critic', epoch, i)

            # Q-Head trains on its own full-weight objective. Folding it into
            # d_loss at 0.001 meant it barely learned, so the composition signal
            # it fed back to G was noise.
            l_q_real = q_head.q_real_loss(real_imgs, labels)
            l_q_real.backward()
            step_if_finite(q_head, optimizer_Q, 'q_head', epoch, i, clip=None)

            ep_d  += d_loss.item()
            ep_w  += wasserstein.item()
            ep_qr += l_q_real.item()
            n_d   += 1

            # ── Generator step (every n_critic critic steps) ───────────────
            if i % n_critic == 0:
                optimizer_G.zero_grad()

                z         = torch.randn(bs, z_dim).to(device)
                gen_input = torch.cat([z, labels], dim=1)
                fake_imgs = generator(gen_input)

                d_fake_for_g = critic(critic_input(fake_imgs, labels))
                g_wgan       = -torch.mean(d_fake_for_g)

                # Q-Head composition loss on FAKE structures
                l_q_fake = q_head.q_fake_loss(fake_imgs, labels)

                # Penalise atoms closer than 1.0A — the validity metric itself.
                l_dist = min_dist_penalty(fake_imgs, labels)
                # Stops the model buying validity by inflating the lattice.
                l_vol  = volume_penalty(fake_imgs, labels)
                # Electrostatics: the classical part of the Kohn-Sham energy.
                # Hinged into the real range, never minimised -- minimising an
                # energy is satisfied by emitting one maximally-ionic
                # arrangement every time, which is the collapse mode this
                # project spent v4-v8 escaping.
                l_mad  = (madelung_penalty(fake_imgs, labels) if lambda_mad > 0
                          else torch.zeros((), device=device))
                # Force the quantum readout to be an actual structure factor.
                l_sf   = (sf_consistency_loss(generator.last_rho, fake_imgs, labels)
                          if (lambda_sf > 0 and generator.last_rho is not None)
                          else torch.zeros((), device=device))

                # DFT-surrogate energy descent. Costs one CHGNet forward, no
                # backward through it -- see dft_distill for why that is exact.
                l_force = (distiller.loss(fake_imgs, labels) if distiller is not None
                           else torch.zeros((), device=device))
                # Distribution matching. Unlike the hinges these are two-sided:
                # a contact at 1.1 A is penalised even though it clears every
                # floor, an inflated cell is penalised even though it makes
                # validity easier, and neither goes quiet until the generated
                # distribution actually matches real.
                l_ms = (mode_seeking_loss(generator, z, labels, z_dim, device)
                        if lambda_ms > 0 else torch.zeros((), device=device))
                l_nn = (nn_distribution_loss(fake_imgs, labels) if lambda_nn > 0
                        else torch.zeros((), device=device))
                l_vd = (vpa_distribution_loss(fake_imgs, labels) if lambda_vd > 0
                        else torch.zeros((), device=device))

                # Total G loss: WGAN + composition + geometry + electrostatics
                #               + reciprocal-space consistency + DFT forces
                g_loss = (g_wgan + lambda_q * l_q_fake + dist_w * l_dist
                          + lambda_vol * l_vol + lambda_mad * l_mad
                          + lambda_sf * l_sf + args.lambda_force * l_force
                          + lambda_nn * l_nn + lambda_vd * l_vd
                          + lambda_ms * l_ms)

                if not torch.isfinite(g_loss):
                    raise RuntimeError(
                        f"g_loss is {g_loss.item()} at epoch {epoch} batch {i}. "
                        f"Components: wgan={g_wgan.item():.4g} q={l_q_fake.item():.4g} "
                        f"dist={l_dist.item():.4g} vol={l_vol.item():.4g} "
                        f"mad={l_mad.item():.4g} sf={l_sf.item():.4g}. "
                        f"Training on NaN silently wastes the entire run.")
                g_loss.backward()
                if step_if_finite(generator, optimizer_G, 'generator', epoch, i) and ema is not None:
                    ema.update_parameters(generator)

                ep_qf   += l_q_fake.item()
                ep_dist += l_dist.item()
                ep_g    += g_loss.item()
                n_g     += 1

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D: {d_loss.item():.3f}] [W: {wasserstein.item():.3f}] "
                          f"[Q_real: {l_q_real.item():.3f}] [Q_fake: {l_q_fake.item():.3f}] "
                          f"[Dist: {l_dist.item():.4f}] [Vol: {l_vol.item():.4f}] "
                          f"[Mad: {l_mad.item():.4f}] [SF: {l_sf.item():.4f}] "
                          f"[NN: {l_nn.item():.4f}] [VD: {l_vd.item():.4f}] "
                          f"[MS: {l_ms.item():.4f}] "
                          f"[F: {l_force.item():.4f}"
                          f"{f'/{distiller.last_n}' if distiller else ''}] "
                          f"[dist_w: {dist_w:.2f}]")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.out_folder, f"checkpoint_{epoch}.pt")
            # 'generator' is what eval loads: the EMA weights when EMA is on.
            # The live weights are kept separately so training can resume.
            ck = {
                'generator': eval_gen.state_dict(),
                'critic':    critic.state_dict(),
                'q_head':    q_head.state_dict(),
                'epoch':     epoch,
                'args':      vars(args),
            }
            if ema is not None:
                ck['generator_live'] = generator.state_dict()
            torch.save(ck, save_path)
            print(f"Saved checkpoint to {save_path}")
            # Collapse probe: dead z shows up here at epoch 10 instead of after
            # 500 epochs and a CHGNet run.
            m = probe_generator(eval_gen, train_data_labels, z_dim, device)

            # Keep the best checkpoint, not the last. v10's best point was epoch
            # 20 and the run degraded to less than half that by epoch 40, so
            # training to completion silently discards the result.
            #
            # Selection is gated, not on validity alone: a checkpoint only
            # qualifies if the cell is not inflated and z is not collapsed.
            # Validity on its own is exactly the metric v9 scored 98.8% on with
            # a 6x-oversized box.
            ok = (VPA_OK[0] <= m['vpa'] <= VPA_OK[1]
                  and m['std_z'] >= Z_DEAD_THRESHOLD)
            if ok and m['valid'] > best['valid']:
                best = {'valid': m['valid'], 'epoch': epoch,
                        'vpa': m['vpa'], 'std_z': m['std_z']}
                torch.save(dict(ck, probe=m),
                           os.path.join(args.out_folder, "checkpoint_best.pt"))
                print(f"  [best] new best: {m['valid'] * 100:.1f}% valid at "
                      f"{m['vpa']:.1f} A^3/atom, epoch {epoch}")
            elif not ok:
                print(f"  [best] epoch {epoch} not eligible "
                      f"(vpa {m['vpa']:.1f}, std_z {m['std_z']:.4f})")

        # ── LR decay — matches classical GAN: lr *= 0.99 every 10 epochs ────────
        if (epoch + 1) % 10 == 0:
            for opt_x in [optimizer_C, optimizer_Q, optimizer_G]:
                for pg in opt_x.param_groups:
                    pg['lr'] *= 0.99

        epoch_losses['epoch'].append(epoch)
        epoch_losses['d_loss'].append(ep_d    / max(1, n_d))
        epoch_losses['wasserstein'].append(ep_w    / max(1, n_d))
        epoch_losses['q_real_loss'].append(ep_qr   / max(1, n_d))
        epoch_losses['q_fake_loss'].append(ep_qf   / max(1, n_g))
        epoch_losses['dist_loss'].append(ep_dist / max(1, n_g))
        epoch_losses['total_g_loss'].append(ep_g    / max(1, n_g))

        # ── Plateau detection ─────────────────────────────────────────────────
        if args.plateau_window > 0 and len(epoch_losses['wasserstein']) >= args.plateau_window:
            recent_w = epoch_losses['wasserstein'][-args.plateau_window:]
            w_range  = max(recent_w) - min(recent_w)
            if w_range < args.plateau_tol:
                flag_path = os.path.join(args.out_folder, "PLATEAU_FLAG.txt")
                msg = (f"[PLATEAU] Epoch {epoch}: W-loss range={w_range:.5f} "
                       f"over last {args.plateau_window} epochs (tol={args.plateau_tol}). "
                       f"W_mean={sum(recent_w)/len(recent_w):.4f}")
                print(f"\n{'!'*60}", flush=True)
                print(msg, flush=True)
                print(f"{'!'*60}\n", flush=True)
                with open(flag_path, 'w') as _fp:
                    _fp.write(msg + '\n')
            else:
                # Clear flag if loss is moving again
                flag_path = os.path.join(args.out_folder, "PLATEAU_FLAG.txt")
                if os.path.exists(flag_path):
                    os.remove(flag_path)

    # ── Save loss CSV ─────────────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(args.out_folder, "training_loss_history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'd_loss', 'wasserstein', 'q_real_loss', 'q_fake_loss', 'dist_loss', 'total_g_loss'])
        for i in range(len(epoch_losses['epoch'])):
            writer.writerow([epoch_losses['epoch'][i],
                             epoch_losses['d_loss'][i],
                             epoch_losses['wasserstein'][i],
                             epoch_losses['q_real_loss'][i],
                             epoch_losses['q_fake_loss'][i],
                             epoch_losses['dist_loss'][i],
                             epoch_losses['total_g_loss'][i]])
    print(f"Saved training loss history to {csv_path}")

    if best['epoch'] >= 0:
        print(f"\nBest checkpoint: epoch {best['epoch']}, "
              f"{best['valid'] * 100:.1f}% valid at {best['vpa']:.1f} A^3/atom, "
              f"std_z {best['std_z']:.4f}  ->  checkpoint_best.pt")
        print("Report this next to volume per atom, never alone, and confirm it "
              "across seeds before it goes in a table.")
    else:
        print("\nNo checkpoint qualified: every probe was cell-inflated or "
              "z-collapsed. Nothing here is reportable.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",    type=str,   default=r"datasets\mgmno_100_aug.pickle")
    parser.add_argument("--n_epochs",        type=int,   default=500)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--z_dim",           type=int,   default=64)
    parser.add_argument("--hidden_features", type=int,   default=12)  # qubits
    parser.add_argument("--hidden_layers",   type=int,   default=1)   # + 1 = 2 quantum layers
    parser.add_argument("--spectrum_layer",  type=int,   default=1)
    parser.add_argument("--use_noise",       type=float, default=0.0)
    # TTUR (Heusel et al. 2017): the critic runs on the faster time scale.
    # Distinct learning rates AND n_critic=5 together implement the separation.
    parser.add_argument("--lr_g",            type=float, default=0.00005)
    parser.add_argument("--lr_d",            type=float, default=0.0001)
    parser.add_argument("--out_folder",      type=str,   default="./results_crystal_qgan_v2")
    parser.add_argument("--save_interval",   type=int,   default=10)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--seed",            type=int,   default=0,
                        help="RNG seed. Reviewers asked for multi-seed error bars; "
                             "run the same config across several seeds and aggregate.")
    parser.add_argument("--lambda_dist",     type=float, default=0.02,
                        help="Weight on the interatomic distance penalty. Rescaled from "
                             "1.0: the penalty now sums violations per structure instead "
                             "of averaging over ~378 pairs, so raw values start ~190 "
                             "rather than ~0.003. 0.02 puts it on par with the WGAN term.")
    parser.add_argument("--lambda_vol",      type=float, default=5.0,
                        help="Weight on the volume-per-atom prior. Applied from epoch 0 "
                             "(no warmup): it must be in force before the distance "
                             "penalty starts rewarding lattice inflation.")
    parser.add_argument("--lambda_q",        type=float, default=0.3,
                        help="Weight on the Q-Head composition loss in the G objective.")
    parser.add_argument("--floors",          type=str,   default="literature",
                        choices=["literature", "data", "bond"],
                        help="Species-aware contact floor set. 'literature' is "
                             "the v9/v10 0.8x-bond-length heuristic, which 16.3%% "
                             "of the real training rows violate -- putting the "
                             "distance penalty and the WGAN critic in conflict on "
                             "a sixth of the data. 'data' derives them from the "
                             "0.5th percentile of the measured distribution (1.5%% "
                             "violated). Default stays 'literature' so prior runs "
                             "reproduce; use 'data' for new ones.")
    parser.add_argument("--warmup_dist",     type=int,   default=20,
                        help="Epochs over which the distance penalty ramps to full "
                             "strength. v10 peaks at exactly this epoch and then "
                             "degrades; see the regression section of the "
                             "Experiment Log.")
    parser.add_argument("--lambda_madelung", type=float, default=0.0,
                        help="Weight on the Ewald electrostatic hinge. Defaults to "
                             "0 (off) so each new term is attributable in the "
                             "ablation. Raw values are O(1-100) eV^2/atom^2.")
    parser.add_argument("--sf_head",         action="store_true",
                        help="Read the trunk's last circuit in reciprocal space: "
                             "qubit i carries the density amplitude rho(G_i) for "
                             "the i-th smallest Miller index, via <Z_i> and <X_i> "
                             "from one circuit evaluation. Requires "
                             "hidden_features == N_G (12).")
    parser.add_argument("--lambda_sf",       type=float, default=0.0,
                        help="Weight on the structure-factor consistency loss, "
                             "which forces the circuit's amplitudes to match the "
                             "analytic S(G) of the emitted crystal. Without it the "
                             "readout is decorative -- the residual path routes "
                             "around the circuit. Needs --sf_head.")
    parser.add_argument("--lambda_force",    type=float, default=0.0,
                        help="Weight on CHGNet force distillation: exact "
                             "first-order descent on the DFT-surrogate energy, "
                             "at the cost of one CHGNet forward (no backward "
                             "through it). The only term that touches E_hull "
                             "directly. 0 = off.")
    parser.add_argument("--force_k",         type=int,   default=4,
                        help="Structures per force-distillation call (~24 ms each).")
    parser.add_argument("--force_every",     type=int,   default=5,
                        help="Apply force distillation every N generator steps.")
    parser.add_argument("--force_gate",      type=float, default=0.0,
                        help="Force norm (eV/A) below which a structure is left "
                             "alone. 0 = measure the real-data p90 at startup, "
                             "which is what makes this pull bad geometry toward "
                             "the DFT basin rather than dragging every sample "
                             "toward a single minimum.")
    parser.add_argument("--lambda_mode_seek", type=float, default=0.0,
                        help="Weight on the mode-seeking term (MSGAN). Rewards "
                             "the generator for producing different structures "
                             "from different z at the SAME label. Every other "
                             "term here is satisfiable without z, which is why "
                             "std_z decays to 0.002; this one is not.")
    parser.add_argument("--lambda_nn",       type=float, default=0.0,
                        help="Weight on nearest-neighbour DISTANCE DISTRIBUTION "
                             "matching (1-D Wasserstein to the measured real "
                             "distribution, median 1.99 A). Two-sided, so unlike "
                             "min_dist_penalty it penalises 1.1 A contacts that "
                             "clear every floor, and does not go quiet once a "
                             "floor is cleared. Use with --lambda_dist 0.")
    parser.add_argument("--lambda_vpa_dist",  type=float, default=0.0,
                        help="Weight on volume-per-atom DISTRIBUTION matching. "
                             "Replaces the [10.0, 15.6] hinge whose ceiling the "
                             "model saturated at 15.5 against a real 11.25. "
                             "Use with --lambda_vol 0.")
    parser.add_argument("--vpa_lo",          type=float, default=VPA_LO,
                        help="Lower bound of the volume-per-atom prior.")
    parser.add_argument("--vpa_hi",          type=float, default=VPA_HI,
                        help="Upper bound of the volume-per-atom prior. The "
                             "default 15.6 is the real p95, and the model "
                             "saturates it: raising lambda_dist drove vpa to "
                             "15.5 and bought 94.5%% validity worth +6.5 eV/atom. "
                             "A one-sided hinge is a target, so this ceiling is "
                             "where the model will sit.")
    parser.add_argument("--sf_features",     action="store_true",
                        help="Give the critic |S(G)|^2 diffraction intensities "
                             "over the 12 smallest Miller indices, alongside the "
                             "contact geometry. Off by default so the baseline "
                             "reproduces v10 exactly (critic input dim 126).")
    parser.add_argument("--no_split_head", action="store_true",
                        help="Quantum trunk with a SINGLE head emitting all 90 "
                             "outputs. This is the ablation reviewers asked for: "
                             "without a quantum-but-not-split model the split "
                             "head's contribution cannot be isolated. Output "
                             "ranges are mapped identically to the split version, "
                             "so the head structure is the only difference.")
    parser.add_argument("--device",          type=str,   default="cpu",
                        choices=["cpu", "cuda"],
                        help="Compute device. Default cpu: measured faster than "
                             "cuda for this qubit count (see bench_device.py), "
                             "and required for running seeds in parallel, since "
                             "concurrent CUDA contexts exhaust VRAM.")
    parser.add_argument("--num_threads",     type=int,   default=1,
                        help="torch intra-op threads. Default 1: the quantum "
                             "circuit does not parallelise, so the throughput "
                             "win comes from running many single-threaded seeds "
                             "at once, not from one wide run.")
    parser.add_argument("--ema_decay",        type=float, default=0.0,
                        help="EMA decay for the evaluated generator weights, "
                             "e.g. 0.999 (~1000 G steps, ~30 epochs). 0 = off, "
                             "which reproduces runs before wave 9.")
    parser.add_argument("--plateau_window",   type=int,   default=30,
                        help="Epochs to look back for plateau detection. 0 = disabled.")
    parser.add_argument("--plateau_tol",      type=float, default=0.02,
                        help="W-loss range threshold below which plateau is flagged.")
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    # Waves 7-8 cannot be reproduced: nothing recorded their flags.
    import json
    with open(os.path.join(args.out_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    train(args)
