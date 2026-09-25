"""
Classical ablation training script.

Same training loop, same hyperparameters, same QHead as train_crystal.py.
Only difference: uses Classical_Crystal (split-head, no quantum circuits)
instead of PQWGAN_CC_Crystal.

Run to match QGAN v4:
    python train_crystal_classical_ablation.py \
        --dataset_path datasets/mgmno_100_aug.pickle \
        --z_dim 64 \
        --hidden_features 8 \
        --hidden_layers 3 \
        --out_folder ./results_crystal_classical_ablation
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.autograd as autograd
from models.Classical_Crystal import Classical_CC_Crystal


def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    interpolates_cond = torch.cat([interpolates, labels], dim=1)
    d_interpolates = critic(interpolates_cond)
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
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class QHead(nn.Module):
    """Identical to train_crystal.py QHead — composition auxiliary classifier."""
    MG_SLOTS = 8;  MN_SLOTS = 8;  O_SLOTS = 12
    MG_CLASSES = 9; MN_CLASSES = 9; O_CLASSES = 13

    def __init__(self, data_dim=90):
        super().__init__()
        hidden = 256
        self.shared = nn.Sequential(
            nn.Linear(data_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
        )
        self.mg_slot  = nn.Linear(hidden, self.MG_SLOTS)
        self.mn_slot  = nn.Linear(hidden, self.MN_SLOTS)
        self.o_slot   = nn.Linear(hidden, self.O_SLOTS)
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
        n_mg = labels_28[:, 0:8].sum(dim=1).long().clamp(0, 8)
        n_mn = labels_28[:, 8:16].sum(dim=1).long().clamp(0, 8)
        n_o  = labels_28[:, 16:28].sum(dim=1).long().clamp(0, 12)
        return n_mg, n_mn, n_o

    def q_real_loss(self, coords, labels_28):
        mg_s, mn_s, o_s, mg_c, mn_c, o_c = self.forward(coords)
        bce = nn.BCEWithLogitsLoss()
        ce  = nn.CrossEntropyLoss()
        n_mg, n_mn, n_o = self._count_targets(labels_28)
        slot_loss  = bce(mg_s, labels_28[:, 0:8]) + bce(mn_s, labels_28[:, 8:16]) + bce(o_s, labels_28[:, 16:28])
        count_loss = ce(mg_c, n_mg) + ce(mn_c, n_mn) + ce(o_c, n_o)
        return slot_loss + 0.3 * count_loss

    def q_fake_loss(self, coords, labels_28):
        _, _, _, mg_c, mn_c, o_c = self.forward(coords)
        ce = nn.CrossEntropyLoss()
        n_mg, n_mn, n_o = self._count_targets(labels_28)
        return 0.01 * (ce(mg_c, n_mg) + ce(mn_c, n_mn) + ce(o_c, n_o))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    gen_input_dim    = z_dim + label_dim
    critic_input_dim = data_dim + label_dim

    print("Initializing Classical Ablation Model (split-head, no quantum)...")
    gan = Classical_CC_Crystal(
        input_dim_g     = gen_input_dim,
        output_dim      = data_dim,
        input_dim_d     = critic_input_dim,
        hidden_features = args.hidden_features,
        hidden_layers   = args.hidden_layers,
    )

    generator = gan.generator.to(device)
    critic    = gan.critic.to(device)
    q_head    = QHead(data_dim=data_dim).to(device)

    optimizer_C = Adam(critic.parameters(),    lr=args.lr_d, betas=(0.0, 0.9))
    optimizer_Q = Adam(q_head.parameters(),    lr=args.lr_d, betas=(0.0, 0.9))
    optimizer_G = Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.9))

    lambda_gp   = 10
    n_critic    = 3
    lambda_cell = args.lambda_cell
    warmup_cell = 30

    epoch_losses = {
        'epoch': [], 'd_loss': [], 'wasserstein': [],
        'q_real_loss': [], 'q_fake_loss': [], 'cell_loss': [], 'total_g_loss': []
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
        ep_d = ep_w = ep_qr = ep_qf = ep_cell = ep_g = 0.0
        cell_w = min(1.0, epoch / max(warmup_cell, 1)) * lambda_cell
        n_d = n_g = 0

        for i, (real_imgs, labels) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            bs        = real_imgs.shape[0]

            # ── Critic step ───────────────────────────────────────────────────
            optimizer_C.zero_grad()
            optimizer_Q.zero_grad()

            z         = torch.randn(bs, z_dim).to(device)
            gen_input = torch.cat([z, labels], dim=1)
            fake_imgs = generator(gen_input).detach()

            real_cond = torch.cat([real_imgs, labels], dim=1)
            fake_cond = torch.cat([fake_imgs, labels], dim=1)
            d_real    = critic(real_cond)
            d_fake    = critic(fake_cond)

            gp          = compute_gradient_penalty(critic, real_imgs, fake_imgs, labels, device)
            l_critic    = torch.mean(d_fake) - torch.mean(d_real)
            wasserstein = torch.mean(d_real) - torch.mean(d_fake)

            l_q_real = q_head.q_real_loss(real_imgs, labels)
            d_loss   = l_critic + lambda_gp * gp + 0.001 * l_q_real

            d_loss.backward()
            optimizer_C.step()
            optimizer_Q.step()

            ep_d  += d_loss.item()
            ep_w  += wasserstein.item()
            ep_qr += l_q_real.item()
            n_d   += 1

            # ── Generator step ────────────────────────────────────────────────
            if i % n_critic == 0:
                optimizer_G.zero_grad()

                z         = torch.randn(bs, z_dim).to(device)
                gen_input = torch.cat([z, labels], dim=1)
                fake_imgs = generator(gen_input)

                fake_cond    = torch.cat([fake_imgs, labels], dim=1)
                d_fake_for_g = critic(fake_cond)
                g_wgan       = -torch.mean(d_fake_for_g)

                l_q_fake = q_head.q_fake_loss(fake_imgs, labels)
                l_cell   = nn.functional.mse_loss(fake_imgs[:, :6], real_imgs[:, :6])
                g_loss   = g_wgan + l_q_fake + cell_w * l_cell

                g_loss.backward()
                optimizer_G.step()

                ep_qf   += l_q_fake.item()
                ep_cell += l_cell.item()
                ep_g    += g_loss.item()
                n_g     += 1

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D: {d_loss.item():.3f}] [W: {wasserstein.item():.3f}] "
                          f"[Q_real: {l_q_real.item():.3f}] [Q_fake: {l_q_fake.item():.3f}] "
                          f"[Cell: {l_cell.item():.4f}] [cell_w: {cell_w:.2f}]")

        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.out_folder, f"checkpoint_{epoch}.pt")
            torch.save({
                'generator': generator.state_dict(),
                'critic':    critic.state_dict(),
                'q_head':    q_head.state_dict(),
                'epoch':     epoch,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

        if (epoch + 1) % 10 == 0:
            for opt_x in [optimizer_C, optimizer_Q, optimizer_G]:
                for pg in opt_x.param_groups:
                    pg['lr'] *= 0.99

        epoch_losses['epoch'].append(epoch)
        epoch_losses['d_loss'].append(ep_d    / max(1, n_d))
        epoch_losses['wasserstein'].append(ep_w    / max(1, n_d))
        epoch_losses['q_real_loss'].append(ep_qr   / max(1, n_d))
        epoch_losses['q_fake_loss'].append(ep_qf   / max(1, n_g))
        epoch_losses['cell_loss'].append(ep_cell / max(1, n_g))
        epoch_losses['total_g_loss'].append(ep_g    / max(1, n_g))

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
                flag_path = os.path.join(args.out_folder, "PLATEAU_FLAG.txt")
                if os.path.exists(flag_path):
                    os.remove(flag_path)

    import csv
    csv_path = os.path.join(args.out_folder, "training_loss_history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'd_loss', 'wasserstein', 'q_real_loss', 'q_fake_loss', 'cell_loss', 'total_g_loss'])
        for i in range(len(epoch_losses['epoch'])):
            writer.writerow([epoch_losses['epoch'][i],
                             epoch_losses['d_loss'][i],
                             epoch_losses['wasserstein'][i],
                             epoch_losses['q_real_loss'][i],
                             epoch_losses['q_fake_loss'][i],
                             epoch_losses['cell_loss'][i],
                             epoch_losses['total_g_loss'][i]])
    print(f"Saved training loss history to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",    type=str,   default=r"datasets\mgmno_100_aug.pickle")
    parser.add_argument("--n_epochs",        type=int,   default=500)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--z_dim",           type=int,   default=64)
    parser.add_argument("--hidden_features", type=int,   default=8)
    parser.add_argument("--hidden_layers",   type=int,   default=3)
    parser.add_argument("--lr_g",            type=float, default=0.000025)
    parser.add_argument("--lr_d",            type=float, default=0.00005)
    parser.add_argument("--out_folder",      type=str,   default="./results_crystal_classical_ablation")
    parser.add_argument("--save_interval",   type=int,   default=10)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--lambda_cell",     type=float, default=5.0)
    parser.add_argument("--plateau_window",  type=int,   default=30)
    parser.add_argument("--plateau_tol",     type=float, default=0.02)
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    train(args)
