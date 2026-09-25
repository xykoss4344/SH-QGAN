"""
Periodic monitor for Classical v3 + Quantum v5.
Every CHECK_INTERVAL seconds:
  - Reports latest epoch & W-loss for both models
  - Triggers a quick E_hull spot-check when classical advances ~50 epochs
  - Alerts if W-loss is stuck (flat for WINDOW lines over MIN_EPOCHS)
"""
import re, time, json, subprocess, os
from pathlib import Path

Q_DIR    = Path(__file__).parent
CLS_DIR  = Path('C:/Users/Adminb/OneDrive/Documents/Projects/crystalGan/'
                'Composition-Conditioned-Crystal-GAN/Composition_Conditioned_Crystal_GAN')
Q_LOG    = Q_DIR / 'v5_log.txt'
CLS_LOSS = CLS_DIR / 'training_loss_log.json'   # written to CLS_DIR root by train.py
CLS_CKPT = CLS_DIR / 'model_cwgan_mgmno_v3'
EVAL_SCR = Q_DIR / 'eval_v5.py'

CHECK_INTERVAL   = 180   # seconds
WINDOW           = 30
FLAT_TOL         = 0.01
MIN_EPOCHS_FLAT  = 3
SPOT_EVERY       = 50    # trigger quick E_hull check every N classical epochs

last_cls_spot  = -1      # last classical epoch we ran a spot-check on
last_alert_q   = -999
last_alert_c   = -999

Q_PAT = re.compile(r'\[Epoch (\d+)/\d+\].*?\[W: ([+-]?\d+\.\d+)\]')
q_history: list = []

def notify(msg: str):
    print(msg, flush=True)
    (Q_DIR / 'monitor_alert.txt').write_text(msg)
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Warning; "
        "$n.Visible = $true; "
        f"$n.ShowBalloonTip(10000, 'QGAN Monitor', '{msg}', "
        "[System.Windows.Forms.ToolTipIcon]::Warning); "
        "Start-Sleep -Seconds 12; $n.Dispose()"
    )
    subprocess.Popen(
        ['powershell', '-WindowStyle', 'Hidden', '-Command', ps],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def check_quantum():
    global last_alert_q
    if not Q_LOG.exists():
        return None, None
    text    = Q_LOG.read_text(errors='ignore')
    matches = Q_PAT.findall(text)
    if not matches:
        return None, None
    epoch, w = int(matches[-1][0]), float(matches[-1][1])
    # flat detection
    recent = [(int(e), float(w2)) for e, w2 in matches[-WINDOW:]]
    if len(recent) >= WINDOW:
        ws = [x[1] for x in recent]
        ep = [x[0] for x in recent]
        if (max(ws) - min(ws) < FLAT_TOL
                and len(set(ep)) >= MIN_EPOCHS_FLAT
                and epoch != last_alert_q):
            msg = f'[QUANTUM] STUCK at epoch {epoch}: W range={max(ws)-min(ws):.5f}'
            notify(msg)
            last_alert_q = epoch
    return epoch, w

def check_classical():
    global last_alert_c
    if not CLS_LOSS.exists():
        return None, None
    try:
        records = json.loads(CLS_LOSS.read_text())
    except:
        return None, None
    if not records:
        return None, None
    last = records[-1]
    epoch, w = last['epoch'], last['w_loss']
    # flat detection (last WINDOW records)
    recent = records[-WINDOW:]
    if len(recent) >= WINDOW:
        ws = [r['w_loss'] for r in recent]
        ep = [r['epoch'] for r in recent]
        if (max(ws) - min(ws) < FLAT_TOL
                and len(set(ep)) >= MIN_EPOCHS_FLAT
                and epoch != last_alert_c):
            msg = f'[CLASSICAL] STUCK at epoch {epoch}: W range={max(ws)-min(ws):.5f}'
            notify(msg)
            last_alert_c = epoch
    return epoch, w

def find_latest_cls_epoch():
    """Return highest generator_N epoch saved."""
    best = -1
    for p in CLS_CKPT.glob('generator_*'):
        try:
            n = int(p.name.split('_')[1])
            if n > best:
                best = n
        except:
            pass
    return best

def run_spot_check(cls_epoch: int):
    out_file = Q_DIR / f'results_eval_v5v3/quickcheck_cls{cls_epoch}.txt'
    if out_file.exists():
        return  # already done
    print(f'\n[SPOT-CHECK] Running quick E_hull for classical epoch {cls_epoch}...', flush=True)
    subprocess.Popen(
        ['py', '-3.12', '-u', str(EVAL_SCR), 'quick', str(cls_epoch)],
        stdout=open(Q_DIR / f'results_eval_v5v3/spot_{cls_epoch}.log', 'w'),
        stderr=subprocess.STDOUT,
        cwd=str(Q_DIR)
    )

def report_spot_results():
    """Print any completed quick-check results."""
    results_dir = Q_DIR / 'results_eval_v5v3'
    if not results_dir.exists():
        return
    for txt in sorted(results_dir.glob('quickcheck_cls*.txt')):
        print(f'\n--- {txt.name} ---', flush=True)
        print(txt.read_text(), flush=True)

print('=' * 60, flush=True)
print('Monitor started — Classical v3 + Quantum v5', flush=True)
print(f'Check interval: {CHECK_INTERVAL}s  Spot-check every {SPOT_EVERY} epochs', flush=True)
print('=' * 60, flush=True)

# Report any existing spot results immediately
report_spot_results()

while True:
    try:
        q_epoch, q_w = check_quantum()
        c_epoch, c_w = check_classical()

        print(f'\n[{time.strftime("%H:%M:%S")}]', flush=True)
        if q_epoch is not None:
            status = 'DONE' if q_epoch >= 498 else 'training'
            print(f'  Quantum  v5: epoch {q_epoch}/500  W={q_w:.4f}  [{status}]', flush=True)
        else:
            print('  Quantum  v5: no log yet', flush=True)

        if c_epoch is not None:
            status = 'DONE' if c_epoch >= 498 else 'training'
            print(f'  Classical v3: epoch {c_epoch}/500  W={c_w:.6f}  [{status}]', flush=True)
        else:
            print('  Classical v3: no log yet', flush=True)

        # Trigger spot-check every SPOT_EVERY classical epochs
        if c_epoch is not None:
            latest_ckpt = find_latest_cls_epoch()
            if latest_ckpt >= 0:
                bucket = (latest_ckpt // SPOT_EVERY) * SPOT_EVERY
                if bucket > last_cls_spot and latest_ckpt >= bucket:
                    last_cls_spot = bucket
                    run_spot_check(latest_ckpt)

        # Show any new spot results
        report_spot_results()

    except Exception as e:
        print(f'Monitor error: {e}', flush=True)

    time.sleep(CHECK_INTERVAL)
