"""
Classical ablation of PQWGAN_CC_Crystal.

Identical architecture to QINR_Crystal:
  - Same split-head generator (shared trunk → cell_head + atom_head)
  - Same classical FC critic
  - Same input/output dimensions

The ONLY difference: QuantumLayer is replaced by a 2-layer MLP
with matched in/out dimensions (in_features → in_features).
This isolates the contribution of quantum circuits from the
contribution of the split-head architectural prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from crystal_mic import BOX_OFFSET, BOX_SCALE


class ClassicalTrunkLayer(nn.Module):
    """Drop-in replacement for HybridLayer with no quantum circuit."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.clayer = nn.Linear(in_features, out_features, bias=bias)
        self.norm   = nn.BatchNorm1d(out_features)
        # 2-layer MLP: same in/out as QuantumLayer, comparable parameter count
        self.net = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LeakyReLU(0.2),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x):
        # Mirrors HybridLayer exactly (norm + residual around the sub-block) so
        # the only difference from the quantum trunk stays the circuit itself.
        x = self.norm(self.clayer(x))
        return self.net(x) + x


class Classical_CC_Crystal():
    def __init__(self, input_dim_g, output_dim, input_dim_d,
                 hidden_features, hidden_layers, outermost_linear=True):
        self.output_dim = output_dim
        self.critic    = self.ClassicalCritic(input_dim_d)
        self.generator = self.SplitHeadGenerator(
            input_dim_g, hidden_features, hidden_layers, output_dim
        )

    class ClassicalCritic(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim + 1, 512)   # +1 minibatch-std feature
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            mbstd = x.std(dim=0, unbiased=False).mean().expand(x.shape[0], 1)
            x = torch.cat([x, mbstd], dim=1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)

    class SplitHeadGenerator(nn.Module):
        """
        Same split-head structure as QINR_Crystal Hybridren:
          Shared trunk  → 256-dim representation
          Cell head     → 6 values  (3 lengths + 3 angles, Sigmoid)
          Atom head     → 84 values (28 atoms × 3 coords, Sigmoid)
          Output: cat([cell, atom]) → 90-dim
        """
        def __init__(self, in_features, hidden_features, hidden_layers, out_features, label_dim=28):
            super().__init__()
            self.label_dim = label_dim

            # ── Shared classical trunk ────────────────────────────────────────
            trunk = [ClassicalTrunkLayer(in_features, hidden_features)]
            for _ in range(hidden_layers):
                trunk.append(ClassicalTrunkLayer(hidden_features, hidden_features))
            trunk += [
                nn.Linear(hidden_features, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
            ]
            self.trunk = nn.Sequential(*trunk)

            # ── Cell head (6 outputs) ─────────────────────────────────────────
            self.cell_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 6),
                nn.Sigmoid(),
            )
            self.register_buffer('cell_lo', torch.tensor([0.05] * 3 + [0.20] * 3))
            self.register_buffer('cell_hi', torch.tensor([0.45] * 3 + [0.80] * 3))
            with torch.no_grad():
                self.cell_head[-2].weight.mul_(0.1)
                self.cell_head[-2].bias.copy_(
                    torch.tensor([-0.4895] * 3 + [-0.1481] * 3))

            # ── Atom head (84 outputs) ────────────────────────────────────────
            self.atom_head = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 84),
                nn.Sigmoid(),
            )

        def forward(self, x):
            label  = x[:, -self.label_dim:]
            shared = self.trunk(x)
            cell   = self.cell_lo + (self.cell_hi - self.cell_lo) * self.cell_head(shared)
            atoms  = BOX_OFFSET + BOX_SCALE * self.atom_head(shared)
            atoms  = atoms * label.repeat_interleave(3, dim=1)
            return torch.cat([cell, atoms], dim=1)
