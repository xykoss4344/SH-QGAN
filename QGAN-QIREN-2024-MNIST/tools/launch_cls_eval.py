"""
Polls for classical-fair generator_490, then runs eval_cls_fair.py.
Run this in background: py -3.12 -u launch_cls_eval.py
"""
import os, sys, time, subprocess

CLS_GEN  = ('C:/Users/Adminb/OneDrive/Documents/Projects/crystalGan/'
             'Composition-Conditioned-Crystal-GAN/Composition_Conditioned_Crystal_GAN/'
             'model_cwgan_mgmno_fair/generator_490')
QDIR     = os.path.dirname(os.path.abspath(__file__))
EVAL_SCR = os.path.join(QDIR, 'eval_cls_fair.py')
LOG      = os.path.join(QDIR, 'results_eval_fair', 'cls_fair_eval.log')

print(f'Waiting for {CLS_GEN} ...', flush=True)
while not os.path.exists(CLS_GEN):
    time.sleep(120)
    print(f'  [{time.strftime("%H:%M:%S")}] still training...', flush=True)

print(f'\nTraining done! Launching eval_cls_fair.py -> {LOG}', flush=True)
time.sleep(10)   # let the file flush fully

os.makedirs(os.path.dirname(LOG), exist_ok=True)
with open(LOG, 'w') as log:
    proc = subprocess.Popen(
        [sys.executable, '-u', EVAL_SCR],
        stdout=log, stderr=subprocess.STDOUT, cwd=QDIR
    )
print(f'eval_cls_fair PID: {proc.pid}', flush=True)
proc.wait()
print('Classical eval complete.', flush=True)
