
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crystal_mic import BOX_OFFSET, BOX_SCALE

class HybridLayer(nn.Module):
    def __init__(self, in_features, out_features, spectrum_layer, use_noise, bias=True, idx=0):
        super().__init__()
        self.idx = idx
        self.clayer = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        self.qlayer = QuantumLayer(out_features, spectrum_layer, use_noise)

    def forward(self, x):
        # BatchNorm was constructed here but never applied, which left the RZ
        # data-reupload angles unbounded: they wrapped far past 2pi, d<Z>/dx
        # oscillated and averaged to ~0 over a batch, so clayer's weights on the
        # high-variance z block collapsed to zero. That is what killed z in
        # v4-v6 (std 8e-4 against 0.08 for the label).
        x1 = self.norm(self.clayer(x))
        # Residual: the circuit still contributes at every layer, but gradients
        # keep a path home so the trunk cannot starve its own input.
        return self.qlayer(x1) + x1


class QuantumLayer(nn.Module):
    def __init__(self, in_features, spectrum_layer, use_noise):
        super().__init__()

        self.in_features = in_features
        self.n_layer = spectrum_layer
        self.use_noise = use_noise

        def _circuit(inputs, weights1, weights2):
            for i in range(self.n_layer):
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=qml.ops.CZ)
                for j in range(self.in_features):
                    qml.RZ(inputs[..., j], wires=j)
            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=qml.ops.CZ)

            if self.use_noise != 0:
                for i in range(self.in_features):
                    rand_angle = np.pi + self.use_noise * np.random.rand()
                    qml.RX(rand_angle, wires=i)

            res = []
            for i in range(self.in_features):
                res.append(qml.expval(qml.PauliZ(i)))
            return res

        # PL 0.38+ unified default.qubit auto-detects PyTorch/CUDA interface.
        # backprop vectorises over the full batch — faster than adjoint for small qubit counts.
        ql_device = qml.device('default.qubit', wires=in_features)
        weight_shape = {"weights1": (self.n_layer, 2, in_features, 3), "weights2": (2, in_features, 3)}

        self.qnode = qml.QNode(_circuit, ql_device, diff_method="backprop", interface="torch")

        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        orgin_shape = list(x.shape[0:-1]) + [-1]
        if len(orgin_shape) > 2:
            x = x.reshape((-1, self.in_features))
        # Bound the reupload angles to [-pi, pi] so the spectrum stays in one
        # period and the input gradient does not oscillate.
        out = self.qnn(torch.tanh(x) * np.pi)
        return out.reshape(orgin_shape)

class PQWGAN_CC_Crystal():
    def __init__(self, input_dim_g, output_dim, input_dim_d, hidden_features, hidden_layers, spectrum_layer, use_noise, outermost_linear=True):
        self.output_dim = output_dim
        self.critic = self.ClassicalCritic(input_dim_d)
        self.generator = self.Hybridren(input_dim_g, hidden_features, hidden_layers, output_dim, spectrum_layer, use_noise, outermost_linear=True)

    class ClassicalCritic(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim
            # +1 for the minibatch-std feature appended in forward.
            self.fc1 = nn.Linear(input_dim + 1, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            # Minibatch std (StyleGAN trick): one scalar telling the critic how
            # varied the batch is. Without it, "one crystal per label" is a Nash
            # point the critic literally cannot see, let alone punish -- which is
            # the collapse mode this model fell into.
            # ponytail: couples samples so the GP is approximate here; standard
            # practice, revisit only if the GP term misbehaves.
            mbstd = x.std(dim=0, unbiased=False).mean().expand(x.shape[0], 1)
            x = torch.cat([x, mbstd], dim=1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)

    class Hybridren(nn.Module):
        """
        Split-head generator:
          Shared quantum trunk  → 256-dim representation
          Cell head  (6 values) → 3 lengths (Sigmoid) + 3 angles (Sigmoid)
          Atom head (84 values) → 28 × 3 fractional positions (Sigmoid)
          Output: cat([cell, atom]) → 90-dim, matching the (30×3) crystal format.

        Keeping the heads separate ensures WGAN gradients can independently
        steer cell geometry vs atom positions, preventing cell-param collapse.
        """
        def __init__(self, in_features, hidden_features, hidden_layers, out_features, spectrum_layer, use_noise, outermost_linear=True, label_dim=28):
            super().__init__()
            self.label_dim = label_dim

            # ── Shared quantum trunk ──────────────────────────────────────────
            trunk = [HybridLayer(in_features, hidden_features, spectrum_layer, use_noise, idx=1)]
            for i in range(hidden_layers):
                trunk.append(HybridLayer(hidden_features, hidden_features, spectrum_layer, use_noise, idx=i + 2))
            # Project quantum output to shared representation
            trunk += [
                nn.Linear(hidden_features, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
            ]
            self.trunk = nn.Sequential(*trunk)

            # ── Cell parameter head (6 outputs: 3 lengths + 3 angles) ────────
            # Real data occupies a narrow band (lengths mean .202 sd .065,
            # angles mean .478 sd .116). A full-range sigmoid has to find that
            # band unaided and lands ~30% low, giving 3.7A cells that cannot
            # hold 28 atoms. Squash into the real support instead and start at
            # the dataset mean.
            self.cell_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 6),
                nn.Sigmoid(),
            )
            self.register_buffer('cell_lo', torch.tensor([0.05] * 3 + [0.20] * 3))
            self.register_buffer('cell_hi', torch.tensor([0.45] * 3 + [0.80] * 3))
            # logit((mean - lo) / (hi - lo)) for lengths .202 and angles .478
            with torch.no_grad():
                self.cell_head[-2].weight.mul_(0.1)
                self.cell_head[-2].bias.copy_(
                    torch.tensor([-0.4895] * 3 + [-0.1481] * 3))

            # ── Atom position head (84 outputs: 28 atoms × 3 coords) ─────────
            # Fractional coordinates in [0,1]
            self.atom_head = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 84),
                nn.Sigmoid(),
            )

        def forward(self, coords):
            label  = coords[:, -self.label_dim:]      # input is cat([z, label])
            shared = self.trunk(coords)               # (batch, 256)

            cell   = self.cell_lo + (self.cell_hi - self.cell_lo) * self.cell_head(shared)

            # Real occupied coords live in [1/6, 5/6] (the 15A re-boxing in
            # datasets/make_representation.py:go_to_15_cell); empty slots are
            # exactly 0.0, which a sigmoid can never emit. Map onto the real
            # support, then mask empty slots to exact zero -- that makes the
            # conditioning structural instead of something the critic has to teach.
            atoms  = BOX_OFFSET + BOX_SCALE * self.atom_head(shared)   # (batch, 84)
            atoms  = atoms * label.repeat_interleave(3, dim=1)

            return torch.cat([cell, atoms], dim=1)   # (batch, 90)
