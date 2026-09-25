"""
Monitor v3 training log for stagnation.
Checks every CHECK_INTERVAL seconds.
If the last WINDOW parsed W values all fall within FLAT_TOL of each other,
across at least MIN_EPOCHS distinct epochs, it fires a Windows toast notification
and writes to v3_alert.txt.
"""

import re
import time
import subprocess
from pathlib import Path
from collections import deque

LOG_FILE      = Path(__file__).parent / "v5_log.txt"
ALERT_FILE    = Path(__file__).parent / "v5_alert.txt"
CHECK_INTERVAL = 120        # seconds between checks
WINDOW         = 30         # number of recent log lines to inspect
FLAT_TOL       = 0.005      # W must change by more than this to be "alive"
MIN_EPOCHS     = 2          # stagnation must span at least this many distinct epochs

# Regex: [Epoch X/500] [Batch Y/Z] [D: d] [W: w] ...
PAT = re.compile(
    r'\[Epoch (\d+)/\d+\].*?\[W: ([+-]?\d+\.\d+)\]'
)

history: deque = deque(maxlen=WINDOW)
last_alert_epoch = -999


def notify(msg: str):
    """Windows toast + console print + write alert file."""
    print(msg, flush=True)
    ALERT_FILE.write_text(msg)
    # Windows balloon notification (works without extra packages)
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Warning; "
        "$n.Visible = $true; "
        f"$n.ShowBalloonTip(10000, 'QGAN v3 ALERT', '{msg}', "
        "[System.Windows.Forms.ToolTipIcon]::Warning); "
        "Start-Sleep -Seconds 12; $n.Dispose()"
    )
    subprocess.Popen(
        ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def check():
    global last_alert_epoch
    if not LOG_FILE.exists():
        return

    text = LOG_FILE.read_text(errors='ignore')
    matches = PAT.findall(text)
    if not matches:
        return

    # Keep last WINDOW entries
    for ep_str, w_str in matches:
        history.append((int(ep_str), float(w_str)))

    if len(history) < WINDOW:
        return  # not enough data yet

    epochs = [e for e, _ in history]
    ws     = [w for _, w in history]
    n_distinct_epochs = len(set(epochs))
    w_range = max(ws) - min(ws)
    current_epoch = epochs[-1]

    status = (f"Epoch {current_epoch} | last {WINDOW} lines: "
              f"W in [{min(ws):.4f}, {max(ws):.4f}]  range={w_range:.4f}  "
              f"distinct_epochs={n_distinct_epochs}")
    print(status, flush=True)

    if (w_range < FLAT_TOL
            and n_distinct_epochs >= MIN_EPOCHS
            and current_epoch != last_alert_epoch):
        msg = (f"STUCK at epoch {current_epoch}: W range={w_range:.5f} "
               f"over {n_distinct_epochs} epochs — training likely broken!")
        notify(msg)
        last_alert_epoch = current_epoch


print(f"Monitor started. Watching {LOG_FILE}", flush=True)
print(f"Alert fires if W range < {FLAT_TOL} over {WINDOW} lines spanning >= {MIN_EPOCHS} epochs.", flush=True)

while True:
    try:
        check()
    except Exception as e:
        print(f"Monitor error: {e}", flush=True)
    time.sleep(CHECK_INTERVAL)
