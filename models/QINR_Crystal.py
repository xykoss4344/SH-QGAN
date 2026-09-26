
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crystal_mic import BOX_OFFSET, BOX_SCALE

class ClassicalTrunkLayer(nn.Module):
    """Classical stand-in for QuantumLayer -- the quantum-vs-classical ablation.

    Same interface and same bounded input/output: angles tanh(x)*pi in, values
    in [-1, 1] out (like Pauli-Z expectations), so HybridLayer's residual and
    BatchNorm around it are untouched and the circuit is the ONLY difference.

      'matched': one Linear(n, n) + tanh -- 156 params for n=12, vs 144 circuit
                 weights. Tests "is the circuit better at the same size?"
      'wide':    n -> 64 -> n MLP -- ~1.6k params, 11x the circuit. If this
                 wins, the circuit is a small nonlinearity, not an advantage.
    """

    def __init__(self, in_features, kind='matched'):
        super().__init__()
        self.in_features = in_features
        self.kind = kind
        if kind == 'matched':
            self.net = nn.Sequential(nn.Linear(in_features, in_features), nn.Tanh())
        elif kind == 'wide':
            self.net = nn.Sequential(nn.Linear(in_features, 64), nn.Tanh(),
                                     nn.Linear(64, in_features), nn.Tanh())
        else:
            # 'fourier': the classical twin of a one-layer data-reupload circuit.
            # That circuit outputs a trigonometric polynomial with frequencies
            # {-1, 0, 1} per input and cross terms from entanglement. Here the
            # same sin/cos features feed a small MLP that can form products.
            # If quantum only ties this, its benefit is the Fourier structure,
            # which a classical layer gets too, not anything quantum.
            # 24*6+6 + 6*12+12 = 234 params for n=12.
            self.net = nn.Sequential(nn.Linear(2 * in_features, 6), nn.Tanh(),
                                     nn.Linear(6, in_features), nn.Tanh())

    def forward(self, x):
        theta = torch.tanh(x) * np.pi
        if self.kind == 'fourier':
            theta = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        return self.net(theta)


class HybridLayer(nn.Module):
    def __init__(self, in_features, out_features, spectrum_layer, use_noise, bias=True,
                 idx=0, readout='z', trunk='quantum'):
        super().__init__()
        self.idx = idx
        self.readout = readout
        self.clayer = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        if trunk == 'quantum':
            self.qlayer = QuantumLayer(out_features, spectrum_layer, use_noise, readout=readout)
        else:
            assert readout == 'z', 'the sf_head readout needs the quantum circuit'
            self.qlayer = ClassicalTrunkLayer(out_features, kind=trunk.split('_')[1])
        # Set by forward when readout='zx': (B, n_qubits, 2), the complex
        # amplitude rho(G_i) read off qubit i. Stashed rather than returned so
        # the generator's 90-dim output signature stays what ten eval_*.py
        # scripts already expect.
        self.rho = None

    def forward(self, x):
        # BatchNorm was constructed here but never applied, which left the RZ
        # data-reupload angles unbounded: they wrapped far past 2pi, d<Z>/dx
        # oscillated and averaged to ~0 over a batch, so clayer's weights on the
        # high-variance z block collapsed to zero. That is what killed z in
        # v4-v6 (std 8e-4 against 0.08 for the label).
        x1 = self.norm(self.clayer(x))
        out = self.qlayer(x1)
        if self.readout == 'zx':
            # <Z_i> and <X_i> come out of ONE circuit evaluation -- a second
            # observable costs 14% in simulation, not another forward pass. The
            # Z half is bit-for-bit what the 'z' readout returns, so the trunk
            # signal below is unchanged and the ablation stays clean.
            z, x_obs = out.chunk(2, dim=-1)
            self.rho = torch.stack([z, x_obs], dim=-1)
            out = z
        # Residual: the circuit still contributes at every layer, but gradients
        # keep a path home so the trunk cannot starve its own input.
        return out + x1


class QuantumLayer(nn.Module):
    """Data-reupload circuit.

    With readout='zx' the circuit is additionally interpreted as a *structure
    factor generator*: qubit i is bound to reciprocal lattice vector G_i, and
    (<Z_i>, <X_i>) are the real and imaginary parts of the density amplitude
    rho(G_i).

    That interpretation is not decorative. A data-reupload circuit represents
    exactly a truncated Fourier series in its inputs (Schuld, Sweke & Meyer
    2021), and a plane-wave expansion of a crystal's density is exactly a
    truncated Fourier series on the same 3-torus. Reading the circuit in
    reciprocal space therefore puts it in the one basis where it is natively
    expressive, and turns the qubit count into a physical quantity: n qubits is
    n plane waves, i.e. the resolution of the density description. That makes
    the queued {8, 12, 16} qubit sweep a measurement of representational
    capacity rather than a bottleneck check.
    """

    def __init__(self, in_features, spectrum_layer, use_noise, readout='z'):
        super().__init__()

        self.in_features = in_features
        self.n_layer = spectrum_layer
        self.use_noise = use_noise
        self.readout = readout

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
            if self.readout == 'zx':
                # Im rho(G_i). Measured in the same circuit evaluation; only the
                # observable differs, so this is ~14% not 100% extra cost.
                for i in range(self.in_features):
                    res.append(qml.expval(qml.PauliX(i)))
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

class SetAtomHead(nn.Module):
    """Permutation-equivariant head that places 28 atoms as a *set*.

    The flat MLP head it replaces emitted 84 independent coordinates: no atom
    could see where any other atom was going, so overlaps were only discouraged
    afterwards by a soft loss. Here the slots attend to each other, so "don't sit
    on top of that one" is expressible in the architecture rather than left to
    the penalty.

    Two further wins that matter for this project:
      - Equivariance is by construction. The training data is augmented by
        permuting slots (datasets/6.data_augmentation_mgmno.py), so the old head
        burned capacity relearning a symmetry this one cannot violate. That is
        real data efficiency on a 107-structure dataset.
      - Each token carries its element identity and the predicted cell, so an
        atom knows both what it is and how big the box is before choosing a
        position -- the old head knew neither.

    The quantum trunk conditions every token through FiLM, so the circuit
    modulates all 28 placements rather than being a bottleneck they pass through.
    """

    def __init__(self, d_model=128, n_heads=4, n_blocks=2, n_slots=28, trunk_dim=256,
                 n_g=0):
        super().__init__()
        self.n_slots = n_slots
        # Learned per-slot identity: encodes both which element the slot holds
        # (fixed layout Mg 0:8, Mn 8:16, O 16:28) and slot index.
        self.slot_embed = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.cell_proj = nn.Linear(6, d_model)
        # FiLM: trunk representation -> per-token scale and shift.
        self.film = nn.Linear(trunk_dim, 2 * d_model)
        # Plane-wave conditioning: every atom sees the density amplitudes the
        # circuit emitted before choosing where to sit, so placement is decoded
        # from a reciprocal-space description rather than invented in real space.
        self.rho_proj = nn.Linear(2 * n_g, d_model) if n_g else None

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'n1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                    nn.Linear(d_model * 2, d_model)),
                'n2': nn.LayerNorm(d_model),
            }) for _ in range(n_blocks)
        ])
        self.out = nn.Linear(d_model, 3)

    def forward(self, shared, cell, label, rho=None, return_tokens=False):
        b = shared.shape[0]
        x = self.slot_embed.unsqueeze(0).expand(b, -1, -1)          # (B, 28, d)
        x = x + self.cell_proj(cell).unsqueeze(1)                   # every atom sees the box
        scale, shift = self.film(shared).chunk(2, dim=-1)
        x = x * (1 + scale).unsqueeze(1) + shift.unsqueeze(1)       # quantum conditioning
        if self.rho_proj is not None and rho is not None:
            x = x + self.rho_proj(rho.flatten(1)).unsqueeze(1)      # plane-wave conditioning

        # Empty slots must not influence the occupied ones.
        pad = (label < 0.5)                                          # (B, 28) True = ignore
        # A structure with every slot empty would make attention produce NaN;
        # keep at least one key visible in that degenerate case.
        pad = pad & ~pad.all(dim=1, keepdim=True)

        for blk in self.blocks:
            h = blk['n1'](x)
            a, _ = blk['attn'](h, h, h, key_padding_mask=pad, need_weights=False)
            x = x + a
            x = x + blk['ff'](blk['n2'](x))

        pos = torch.sigmoid(self.out(x)).reshape(b, self.n_slots * 3)
        return (pos, x) if return_tokens else pos


def _lattice_from_cell(cell6):
    """(B, 3, 3) lattice, rows are vectors, from the normalised 6 cell outputs."""
    from crystal_physics import lattice_matrix
    fake = torch.zeros(cell6.shape[0], 90, device=cell6.device, dtype=cell6.dtype)
    fake[:, :6] = cell6
    return lattice_matrix(fake)


class PeriodicRefiner(nn.Module):
    """E(3)-equivariant refinement over the periodic neighbour graph (EGNN-style).

    The set head emits all 28 positions in one shot and no atom sees where the
    others actually landed. Measured on waves 12-15: generated same-species
    atoms sit at a mean fractional separation of 0.49-0.52, i.e. uniformly
    random (0.48), where real crystals are more ordered (0.55) -- so contacts
    were fixed by the penalties but cations still had 3.3 cation neighbours
    within 2.9 A (real 0.65) and E_hull stayed at +2.5 eV/atom. Every modern
    crystal generator (CDVAE, DiffCSP, MatterGen) builds geometry by message
    passing on the minimum-image graph; this adds that, after the split head.

    Each round: messages from species, RBF(distance); each atom moves along its
    neighbour vectors weighted by a learned scalar -- equivariant by
    construction. The last coordinate layer starts at zero, so an untrained
    refiner is the identity.
    """

    def __init__(self, d_tok=128, d=64, rounds=3, n_rbf=16, cutoff=6.0):
        super().__init__()
        self.rounds, self.cutoff = rounds, cutoff
        self.inp = nn.Linear(d_tok, d)
        self.species = nn.Embedding(3, d)
        self.register_buffer('species_idx', torch.tensor([0] * 8 + [1] * 8 + [2] * 12))
        self.register_buffer('mu', torch.linspace(0.5, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.edge = nn.ModuleList([nn.Sequential(nn.Linear(2 * d + n_rbf, d), nn.SiLU(),
                                                 nn.Linear(d, d), nn.SiLU())
                                   for _ in range(rounds)])
        self.node = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(),
                                                 nn.Linear(d, d)) for _ in range(rounds)])
        self.coord = nn.ModuleList()
        for _ in range(rounds):
            last = nn.Linear(d, 1)
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
            self.coord.append(nn.Sequential(nn.Linear(d, d), nn.SiLU(), last))

    def forward(self, frac, cell6, label, tokens):
        b, n = frac.shape[:2]
        h = self.inp(tokens) + self.species(self.species_idx).unsqueeze(0)
        occ = label > 0.5
        eye = torch.eye(n, dtype=torch.bool, device=frac.device)
        pair = (occ.unsqueeze(2) & occ.unsqueeze(1) & ~eye).float()
        lat = _lattice_from_cell(cell6)                              # (B, 3, 3)
        lat_inv = torch.linalg.inv(lat)
        for r in range(self.rounds):
            df = frac.unsqueeze(2) - frac.unsqueeze(1)
            df = df - torch.round(df)                                # minimum image
            vec = torch.matmul(df, lat.unsqueeze(1))                 # (B, n, n, 3), j -> i
            d = torch.sqrt((vec ** 2).sum(-1) + 1e-6)
            env = 0.5 * (torch.cos(torch.pi * d.clamp(max=self.cutoff) / self.cutoff) + 1)
            m = pair * env * (d < self.cutoff).float()
            rbf = torch.exp(-self.gamma * (d.unsqueeze(-1) - self.mu) ** 2)
            hi = h.unsqueeze(2).expand(b, n, n, -1)
            hj = h.unsqueeze(1).expand(b, n, n, -1)
            e = self.edge[r](torch.cat([hi, hj, rbf], dim=-1)) * m.unsqueeze(-1)
            norm = m.sum(2, keepdim=True) + 1.0
            h = h + self.node[r](torch.cat([h, e.sum(2) / norm], dim=-1))
            w = self.coord[r](e).squeeze(-1) * m                     # >0 pushes i from j
            dx = (vec * w.unsqueeze(-1)).sum(2) / norm               # (B, n, 3) Angstrom
            dx = dx.clamp(-0.5, 0.5)                                 # bounded step
            frac = frac + torch.matmul(dx.unsqueeze(2), lat_inv.unsqueeze(1)).squeeze(2)
        return frac - torch.floor(frac)                              # wrap into [0, 1)


class PQWGAN_CC_Crystal():
    def __init__(self, input_dim_g, output_dim, input_dim_d, hidden_features, hidden_layers, spectrum_layer, use_noise, outermost_linear=True, set_head=True, sf_head=False, split_head=True, refine_rounds=0, trunk='quantum'):
        self.output_dim = output_dim
        self.critic = self.ClassicalCritic(input_dim_d)
        self.generator = self.Hybridren(input_dim_g, hidden_features, hidden_layers, output_dim, spectrum_layer, use_noise, outermost_linear=True, set_head=set_head, sf_head=sf_head, split_head=split_head, refine_rounds=refine_rounds, trunk_type=trunk)

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
        def __init__(self, in_features, hidden_features, hidden_layers, out_features, spectrum_layer, use_noise, outermost_linear=True, label_dim=28, set_head=True, sf_head=False, split_head=True, refine_rounds=0, trunk_type='quantum'):
            super().__init__()
            self.label_dim = label_dim
            self.set_head = set_head
            self.split_head = split_head
            # Structure-factor readout requires one qubit per reciprocal lattice
            # vector, so the Miller set size is the qubit count. Not a free
            # parameter -- see train_crystal.N_G, which must agree.
            self.sf_head = sf_head
            self.n_g = hidden_features if sf_head else 0

            # ── Shared quantum trunk ──────────────────────────────────────────
            # The LAST quantum layer carries the structure-factor readout: it is
            # the one whose output the heads actually consume, so its amplitudes
            # are the ones that must describe the emitted crystal. Readout is
            # fixed at construction -- the QNode's output shape is baked into
            # the TorchLayer, so flipping it afterwards is not safe.
            n_trunk = 1 + hidden_layers
            trunk = []
            for i in range(n_trunk):
                last = (i == n_trunk - 1)
                trunk.append(HybridLayer(
                    in_features if i == 0 else hidden_features, hidden_features,
                    spectrum_layer, use_noise, idx=i + 1,
                    readout='zx' if (last and sf_head) else 'z', trunk=trunk_type))
            # Index, not a second reference. Assigning the module to an
            # attribute registers it twice, which duplicates every one of its
            # tensors under `sf_layer.*` in state_dict() and makes strict
            # loading of existing v10 checkpoints fail -- which is what every
            # eval_*.py does.
            self._sf_idx = n_trunk - 1
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
            self.cell_head = None if not split_head else nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 6),
                nn.Sigmoid(),
            )
            self.register_buffer('cell_lo', torch.tensor([0.05] * 3 + [0.20] * 3))
            self.register_buffer('cell_hi', torch.tensor([0.45] * 3 + [0.80] * 3))
            # logit((mean - lo) / (hi - lo)) for lengths .202 and angles .478
            if split_head:
                with torch.no_grad():
                    self.cell_head[-2].weight.mul_(0.1)
                    self.cell_head[-2].bias.copy_(
                        torch.tensor([-0.4895] * 3 + [-0.1481] * 3))

            # ── Atom position head (84 outputs: 28 atoms × 3 coords) ─────────
            # Fractional coordinates in [0,1]
            # ── Joint head: the "quantum, no split head" ablation ────────────
            # Reviewers cannot isolate the split head's benefit without a
            # quantum model that lacks it. One head emits all 90 outputs; the
            # cell/atom output ranges are mapped identically to the split
            # version, so the ONLY difference is whether the heads are separate.
            if not split_head:
                self.joint_head = nn.Sequential(
                    nn.Linear(256, 512), nn.LeakyReLU(0.2),
                    nn.Linear(512, 256), nn.LeakyReLU(0.2),
                    nn.Linear(256, 90), nn.Sigmoid(),
                )
            elif set_head:
                self.atom_head = SetAtomHead(trunk_dim=256, n_g=self.n_g)
            else:
                # Flat baseline, kept for the ablation: 84 independent outputs,
                # no atom aware of any other.
                self.atom_head = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 84),
                    nn.Sigmoid(),
                )
            # Periodic message-passing refinement after the set head (optional).
            self.refiner = (PeriodicRefiner(rounds=refine_rounds)
                            if (refine_rounds > 0 and split_head and set_head) else None)

        @property
        def sf_layer(self):
            """The trunk layer carrying the structure-factor readout."""
            return self.trunk[self._sf_idx]

        def forward(self, coords):
            label  = coords[:, -self.label_dim:]      # input is cat([z, label])
            shared = self.trunk(coords)               # (batch, 256)
            # rho(G): (B, n_g, 2) complex density amplitudes read off the trunk's
            # last circuit. Stashed on the module rather than returned, so the
            # 90-dim output signature every eval_*.py depends on is unchanged.
            # train_crystal reads generator.last_rho for the consistency loss.
            self.last_rho = self.sf_layer.rho if self.sf_head else None

            if not self.split_head:
                # One head, both quantities. Same output ranges as the split
                # version, so the ablation isolates the head structure alone.
                raw90 = self.joint_head(shared)
                cell  = self.cell_lo + (self.cell_hi - self.cell_lo) * raw90[:, :6]
                atoms = BOX_OFFSET + BOX_SCALE * raw90[:, 6:]
                atoms = atoms * label.repeat_interleave(3, dim=1)
                return torch.cat([cell, atoms], dim=1)

            cell   = self.cell_lo + (self.cell_hi - self.cell_lo) * self.cell_head(shared)

            # Real occupied coords live in [1/6, 5/6] (the 15A re-boxing in
            # datasets/make_representation.py:go_to_15_cell); empty slots are
            # exactly 0.0, which a sigmoid can never emit. Map onto the real
            # support, then mask empty slots to exact zero -- that makes the
            # conditioning structural instead of something the critic has to teach.
            if self.set_head and self.refiner is not None:
                raw, tok = self.atom_head(shared, cell, label, self.last_rho,
                                          return_tokens=True)
                raw = self.refiner(raw.view(-1, 28, 3), cell, label, tok).reshape(-1, 84)
            else:
                raw = (self.atom_head(shared, cell, label, self.last_rho) if self.set_head
                       else self.atom_head(shared))                    # (batch, 84)
            atoms  = BOX_OFFSET + BOX_SCALE * raw
            atoms  = atoms * label.repeat_interleave(3, dim=1)

            return torch.cat([cell, atoms], dim=1)   # (batch, 90)
