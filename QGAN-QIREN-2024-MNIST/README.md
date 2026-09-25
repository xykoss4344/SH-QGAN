# SH-QGAN — Split-Head Quantum GAN for Crystal Generation

Hybrid quantum-classical WGAN-GP that generates Mg-Mn-O crystal structures. A
shared quantum trunk (PennyLane data-reupload circuits) feeds a split head: one
for the unit cell, one that places 28 atoms as a permutation-equivariant set.

Backs submission 2002. **Full project memory, evidence, and the resubmission
checklist live in `../../research-vault/` — start at `Home.md`.**

## Layout

| path | what |
|---|---|
| `train_crystal.py` | training entry point; every addition is behind a flag |
| `crystal_mic.py` | decode + minimum-image distance. **One definition, used everywhere** |
| `crystal_physics.py` | Ewald electrostatics, structure factors, formal charges |
| `dft_distill.py` | CHGNet force distillation (DFT-surrogate energy gradients) |
| `probe_collapse.py` | collapse probe; runs at every checkpoint |
| `eval_wave.py` | the metric suite, one code path |
| `bench_device.py` | CPU vs CUDA benchmark |
| `models/` | generator, critic, quantum layers |
| `datasets/` | dataset build pipeline (7 numbered scripts) |
| `runs/` | live training runs + `status.py` |
| `evaluation/`, `figures/`, `dft/`, `tools/` | grouped scripts (import shim at top) |
| `experiments/` | archived results from previous waves |
| `legacy/` | superseded code: v4-v6 evals, SSIM plots, MNIST-era modules |

## Running

```bash
python train_crystal.py --dataset_path datasets/mgmno_100_aug.pickle --floors data
python runs/status.py            # live status of every run
python eval_wave.py              # metric table across runs
```

Every module with non-trivial logic has a `_demo()` self-check that asserts
against real data:

```bash
python crystal_mic.py        # decode must score real structures ~100% valid
python crystal_physics.py    # Ewald vs pymatgen, and the NaCl Madelung constant
python dft_distill.py        # force surrogate vs finite-difference energy
```

## Three rules this project learned the hard way

1. **Never report validity without volume per atom beside it.** A 98.8% figure
   came from inflating the unit cell ~6x. See `Gaming the Metric.md`.
2. **A peak is not a result.** Stable, multi-seed, or it does not go in a table.
3. **Report distinct source structures, never augmented row counts.** The 112k
   rows are 107 Materials Project entries.

## Compute

CPU, single-threaded per process, many processes in parallel. The GPU is
*slower* here — at 12 qubits the state vector is 4096 complex numbers and the
simulation is kernel-launch bound. Measured in `../../research-vault/Compute Plan.md`.
Concurrent CUDA contexts also exhaust VRAM, so `--device` defaults to `cpu`.

~60 s/epoch. Cap parallel runs at 8 on a 16-core box; 12 exhausted memory.

## Running experiments without killing them

Two waves have been lost to `DefaultCPUAllocator: not enough memory`, both times
because a **CHGNet evaluation was run alongside CHGNet-using training runs**.
CHGNet is ~1-2 GB per process and the spike is transient, so free RAM looks fine
minutes later and the cause is easy to misattribute to the training config.

- Cap training at **5-6 concurrent runs** on this 16-core / 64 GB box.
- **Never run `eval_wave.py` or a CHGNet script while a `--lambda_force` run is
  training.** Wait, or stop that run first.
- Check free RAM before launching, not after: `(Get-CimInstance
  Win32_OperatingSystem).FreePhysicalMemory/1MB`.
- Kill with PowerShell `Stop-Process -Id <id> -Force`. Bash `kill` takes the
  shell wrapper and leaves Python running (see the vault).
