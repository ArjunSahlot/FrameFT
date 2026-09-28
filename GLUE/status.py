"""One-screen health check of the overnight queue.   python status.py"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STALE_MINUTES = 10  # an RTE epoch takes ~40 s and an MRPC epoch ~30 s, so 10 min without an update means trouble


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


print("time:            ", time.strftime("%Y-%m-%d %H:%M:%S"))
print("tmux 'frameft':  ", "alive" if subprocess.run(["tmux", "has-session", "-t", "frameft"], capture_output=True).returncode == 0 else "MISSING")
print("runner pid:      ", run(["pgrep", "-f", "run_queue.py"]).replace("\n", " ") or "NOT RUNNING")
print("training pid:    ", run(["pgrep", "-f", "NLU_GLUE.py"]).replace("\n", " ") or "none")
print("gpu (used, total, util, temp):", run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader"]))
print("disk free:       ", run(["df", "-h", "--output=avail", str(HERE)]).splitlines()[-1].strip())
print("STOP file:       ", "PRESENT" if (RESULTS / "STOP").exists() else "no")

for p in sorted(RESULTS.glob("*.json")):
    r = json.loads(p.read_text())
    if r["status"] == "running":
        age = (datetime.now() - datetime.strptime(r["updated_at"], "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
        best = f"{100 * r['best']:.2f}" if r["best"] is not None else "-"
        stale = "   <-- STALE" if age > STALE_MINUTES else ""
        print(f"running:          {r['exp_name']} epoch {r['epochs_done']}/{r['epochs_total']}, best so far {best}, "
              f"last update {age:.1f} min ago{stale}")

print("\nlast runner.log lines:")
print(run(["tail", "-n", "6", str(RESULTS / "runner.log")]))
print()
print(run([sys.executable, str(HERE / "summarize.py"), str(RESULTS)]))
