"""Wait for eval_v6.py to finish then launch 16-qubit quantum v7 training."""
import subprocess, psutil, time, os, sys

QDIR = os.path.dirname(os.path.abspath(__file__))

def eval_running():
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(p.info['cmdline'] or [])
            if 'eval_v6' in cmd:
                return True
        except:
            pass
    return False

print("Waiting for eval_v6.py to finish...", flush=True)
while eval_running():
    time.sleep(30)
    print(f"  eval still running... {time.strftime('%H:%M:%S')}", flush=True)

print("Eval finished. Launching quantum v7 (16 qubits)...", flush=True)
time.sleep(5)  # brief pause to let GPU memory free

cmd = [
    sys.executable, '-u', os.path.join(QDIR, 'train_crystal.py'),
    '--dataset_path',    'datasets/mgmno_1000.pickle',
    '--out_folder',      './results_crystal_qgan_v7',
    '--lambda_cell',     '0',
    '--z_dim',           '64',
    '--hidden_features', '16',
    '--hidden_layers',   '3',
    '--spectrum_layer',  '1',
    '--lr_g',            '0.0002',
    '--lr_d',            '0.00005',
    '--batch_size',      '128',
    '--plateau_window',  '30',
    '--plateau_tol',     '0.02',
]

log_path = os.path.join(QDIR, 'v7_log.txt')
print(f"Log: {log_path}", flush=True)
print(f"Cmd: {' '.join(cmd)}", flush=True)
with open(log_path, 'w') as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=QDIR)
print(f"v7 training started (PID {proc.pid})", flush=True)
proc.wait()
print("v7 training completed.", flush=True)
