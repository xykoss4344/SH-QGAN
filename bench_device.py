"""Benchmark every expensive term on CPU vs CUDA.

Answers one question: does the RTX 5080 make the ablation grid take hours
instead of days, and which term is actually the bottleneck?

The circuit is the interesting case. At 12 qubits the state vector is only 4096
complex numbers, so the simulation may well be kernel-launch bound and gain
little or nothing from a GPU -- which is why this measures rather than assumes.

    python bench_device.py
"""
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

BATCH = 32
REPEAT = 5


def timed(fn, repeat=REPEAT, warmup=2):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeat * 1000.0      # ms


def bench(device):
    import pickle
    from crystal_physics import madelung_energy, miller_set, structure_factor_features
    from models.QINR_Crystal import PQWGAN_CC_Crystal

    out = {}
    raw = pickle.load(open('datasets/mgmno_100.pickle', 'rb'))
    co = torch.tensor(np.array([np.array(c).flatten() for c, l in raw[:BATCH]]),
                      dtype=torch.float32, device=device)
    la = torch.tensor(np.array([np.array(l).flatten() for c, l in raw[:BATCH]]),
                      dtype=torch.float32, device=device)
    g = miller_set(12).to(device)

    out['ewald'] = timed(lambda: madelung_energy(co, la))
    out['structure_factor'] = timed(lambda: structure_factor_features(co, la, g))

    gan = PQWGAN_CC_Crystal(input_dim_g=92, output_dim=90, input_dim_d=126,
                            hidden_features=12, hidden_layers=1, spectrum_layer=1,
                            use_noise=0.0, sf_head=True)
    gen = gan.generator.to(device).train()
    x = torch.randn(BATCH, 92, device=device)

    def gen_fwd_bwd():
        gen.zero_grad(set_to_none=True)
        gen(x).sum().backward()

    out['generator fwd+bwd'] = timed(gen_fwd_bwd)

    # The circuit alone, isolated from the rest of the generator.
    ql = gen.trunk[0].qlayer

    def circuit():
        z = torch.randn(BATCH, 12, device=device, requires_grad=True)
        ql(torch.tanh(z) * np.pi).sum().backward()

    out['quantum circuit'] = timed(circuit)
    return out


def bench_chgnet(device):
    from chgnet.model import CHGNet
    from pymatgen.core import Lattice, Structure
    import pickle
    from crystal_mic import decode

    model = CHGNet.load(verbose=False, use_device=str(device))
    raw = pickle.load(open('datasets/mgmno_100.pickle', 'rb'))
    sts = []
    for c, l in raw[:8]:
        sp, frac, cell = decode(np.array(c).flatten(), np.array(l).flatten())
        if len(sp) >= 2:
            sts.append(Structure(Lattice.from_parameters(*cell), sp, frac))
    return timed(lambda: [model.predict_structure(s) for s in sts], repeat=3) / len(sts)


def main():
    print(f'torch {torch.__version__}   cuda available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'gpu: {torch.cuda.get_device_name(0)}')
    print(f'batch {BATCH}, mean of {REPEAT} runs\n')

    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    results = {}
    for d in devices:
        print(f'benchmarking {d} ...', flush=True)
        results[d] = bench(d)
        try:
            results[d]['chgnet (per structure)'] = bench_chgnet(d)
        except Exception as e:
            print(f'  chgnet on {d} failed: {str(e)[:60]}')

    keys = list(results[devices[0]])
    w = max(len(k) for k in keys)
    print(f'\n{"term".ljust(w)}  {"cpu (ms)":>10}  {"cuda (ms)":>10}  {"speedup":>8}')
    print('-' * (w + 34))
    for k in keys:
        c = results['cpu'].get(k, float('nan'))
        g = results.get('cuda', {}).get(k, float('nan'))
        sp = f'{c / g:.1f}x' if g == g and g > 0 else '--'
        gs = f'{g:10.2f}' if g == g else f'{"--":>10}'
        print(f'{k.ljust(w)}  {c:10.2f}  {gs}  {sp:>8}')

    print('\nEpoch estimate (325 batches, generator fwd+bwd per batch,')
    print('plus one G-step every 5 with all physics terms):')
    for d in devices:
        r = results[d]
        per_batch = r['generator fwd+bwd']
        per_gstep = (r['generator fwd+bwd'] + r['ewald'] + r['structure_factor']
                     + 4 * r.get('chgnet (per structure)', 0.0) / 5)
        est = (325 * per_batch + 65 * per_gstep) / 1000.0
        print(f'  {d:5s} ~{est:6.1f} s/epoch   ->  500 epochs = {est * 500 / 3600:5.2f} h')


if __name__ == '__main__':
    main()
