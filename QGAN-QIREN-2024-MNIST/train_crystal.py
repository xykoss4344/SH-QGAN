
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
from probe_collapse import probe_generator
from crystal_mic import min_separation_matrix


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


def _lattice(fake):
    """(B, 3, 3) lattice matrix in Angstrom from the generated cell head."""
    arr     = fake.view(fake.shape[0], 30, 3)
    lengths = arr[:, 0] * 30.0
    angles  = torch.deg2rad(torch.clamp(arr[:, 1] * 180.0, 30.0, 150.0))
    a, b, c    = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    al, be, ga = angles[:, 0], angles[:, 1], angles[:, 2]

    # Rows are lattice vectors (same construction as eval_v4:build_lattice_matrix).
    zero = torch.zeros_like(a)
    v1 = torch.stack([a, zero, zero], dim=-1)
    v2 = torch.stack([b * torch.cos(ga), b * torch.sin(ga), zero], dim=-1)
    cx = c * torch.cos(be)
    cy = c * (torch.cos(al) - torch.cos(be) * torch.cos(ga)) / (torch.sin(ga) + 1e-9)
    cz = torch.sqrt(torch.clamp(c ** 2 - cx ** 2 - cy ** 2, min=1e-6))
    v3 = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([v1, v2, v3], dim=1)               # (B, 3, 3)


# Measured on datasets/mgmno_100.pickle: volume per atom 11.77 +/- 1.84 A^3,
# 5th-95th percentile 10.0-15.6. Oxides are close-packed; this is a tight,
# physically meaningful constraint.
VPA_LO, VPA_HI = 10.0, 15.6


def volume_penalty(fake, labels, lo=VPA_LO, hi=VPA_HI):
    """Keep volume per atom inside the range real Mg-Mn-O oxides occupy.

    Without this, inflating the lattice is the cheapest way to satisfy the
    minimum-distance penalty, and the model takes it: v9 reached 98.8% "valid"
    at an ~11A cell against a real 6.1-6.4A, i.e. ~6x the true volume per atom.
    Validity bought that way is worthless -- density is a standard benchmark
    metric and such structures would score terribly on it and on E_hull.

    Log-space hinge so it is scale-free, with a dead zone across the real
    percentile range so the model is constrained but not pinned to the mean.
    """
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
    return torch.linalg.norm(cart + 1e-12, dim=-1)        # (B, 28, 28)


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
    d = _pair_distances(x, labels)                                   # (B, 28, 28)
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


def critic_input(x, labels):
    """Everything the critic sees: structure, label, and contact geometry."""
    parts = [x, labels]
    if USE_GEOMETRY_FEATURES:
        parts.append(geometry_features(x, labels, k=GEOM_K))
    return torch.cat(parts, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    critic_input_dim = data_dim + label_dim + (GEOM_K if USE_GEOMETRY_FEATURES else 0)
    print("Initializing QINR Crystal Model...")
    gan = PQWGAN_CC_Crystal(
        input_dim_g   = gen_input_dim,
        output_dim    = data_dim,
        input_dim_d   = critic_input_dim,
        hidden_features = args.hidden_features,
        hidden_layers   = args.hidden_layers,
        spectrum_layer  = args.spectrum_layer,
        use_noise       = args.use_noise,
    )

    generator = gan.generator.to(device)
    critic    = gan.critic.to(device)
    q_head    = QHead(data_dim=data_dim).to(device)

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
    warmup_dist = 20

    # ── Loss tracking ─────────────────────────────────────────────────────────
    epoch_losses = {
        'epoch': [], 'd_loss': [], 'wasserstein': [],
        'q_real_loss': [], 'q_fake_loss': [], 'dist_loss': [], 'total_g_loss': []
    }

    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Resuming from {args.resume_checkpoint}...")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        generator.load_state_dict(ckpt['generator'])
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
            fake_imgs = generator(gen_input).detach()   # no G gradients here

            d_real    = critic(critic_input(real_imgs, labels))
            d_fake    = critic(critic_input(fake_imgs, labels))

            gp         = compute_gradient_penalty(critic, real_imgs, fake_imgs, labels, device)
            l_critic   = torch.mean(d_fake) - torch.mean(d_real)
            wasserstein = torch.mean(d_real) - torch.mean(d_fake)   # logged

            # Critic loss is pure WGAN-GP now.
            d_loss = l_critic + lambda_gp * gp
            d_loss.backward()
            optimizer_C.step()

            # Q-Head trains on its own full-weight objective. Folding it into
            # d_loss at 0.001 meant it barely learned, so the composition signal
            # it fed back to G was noise.
            l_q_real = q_head.q_real_loss(real_imgs, labels)
            l_q_real.backward()
            optimizer_Q.step()

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

                # Total G loss: WGAN + composition + geometry validity
                g_loss = g_wgan + lambda_q * l_q_fake + dist_w * l_dist + lambda_vol * l_vol

                g_loss.backward()
                optimizer_G.step()

                ep_qf   += l_q_fake.item()
                ep_dist += l_dist.item()
                ep_g    += g_loss.item()
                n_g     += 1

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D: {d_loss.item():.3f}] [W: {wasserstein.item():.3f}] "
                          f"[Q_real: {l_q_real.item():.3f}] [Q_fake: {l_q_fake.item():.3f}] "
                          f"[Dist: {l_dist.item():.4f}] [Vol: {l_vol.item():.4f}] [dist_w: {dist_w:.2f}]")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.out_folder, f"checkpoint_{epoch}.pt")
            torch.save({
                'generator': generator.state_dict(),
                'critic':    critic.state_dict(),
                'q_head':    q_head.state_dict(),
                'epoch':     epoch,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
            # Collapse probe: dead z shows up here at epoch 10 instead of after
            # 500 epochs and a CHGNet run.
            probe_generator(generator, train_data_labels, z_dim, device)

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
    parser.add_argument("--plateau_window",   type=int,   default=30,
                        help="Epochs to look back for plateau detection. 0 = disabled.")
    parser.add_argument("--plateau_tol",      type=float, default=0.02,
                        help="W-loss range threshold below which plateau is flagged.")
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    train(args)
