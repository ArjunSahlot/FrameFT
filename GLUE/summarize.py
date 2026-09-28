"""Print a results table from results/*.json.   python summarize.py [results_dir]"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
records = []
for p in sorted(results_dir.glob("*.json")):
    try:
        records.append(json.loads(p.read_text()))
    except json.JSONDecodeError:
        print(f"unreadable: {p}")

print(f"{'run':24s} {'status':10s} {'epochs':>9s} {'best':>7s} {'@ep':>4s} {'final':>7s} {'minutes':>8s}")
for r in records:
    best = f"{100 * r['best']:.2f}" if r.get("best") is not None else "-"
    final = f"{100 * r['final']:.2f}" if r.get("final") is not None else "-"
    print(f"{r['exp_name']:24s} {r['status']:10s} {r['epochs_done']:>4d}/{r['epochs_total']:<4d} {best:>7s} "
          f"{str(r.get('best_epoch', '-')):>4s} {final:>7s} {r.get('elapsed_s', 0) / 60:8.1f}")

groups = defaultdict(list)
for r in records:
    if r["status"] == "done":
        groups[(r["task"], r["basis"])].append(r)


def fmt(values):
    values = [100 * v for v in values]
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


print(f"\n{'task':6s} {'basis':9s} {'n':>2s}  {'best (mean ± sd)':>18s}  {'final (mean ± sd)':>18s}  seeds")
for (task, basis), rs in sorted(groups.items()):
    seeds = ",".join(str(r["seed"]) for r in rs)
    print(f"{task:6s} {basis:9s} {len(rs):>2d}  {fmt([r['best'] for r in rs]):>18s}  {fmt([r['final'] for r in rs]):>18s}  {seeds}")
