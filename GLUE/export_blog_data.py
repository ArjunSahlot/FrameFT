"""Export results/*.json as one compact JSON file for the blog charts.   python export_blog_data.py OUT.json

scores: every finished run's best-epoch and final-epoch accuracy (%)
curves: per task and basis, the 5-epoch trailing average of validation accuracy (%) and training loss,
        as the mean, min and max across seeds
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TASKS, BASES = ["rte", "mrpc"], ["frame", "random", "identity"]
PAPER = {"rte": 79.8, "mrpc": 92.3}  # FrameFT, RoBERTa-base, Table 1 of the paper (accuracy)
WINDOW = 5


def rolling(values):
    return [statistics.mean(values[max(0, i - WINDOW + 1):i + 1]) for i in range(len(values))]


def across_seeds(series, digits):
    per_epoch = list(zip(*series))
    return {name: [round(f(v), digits) for v in per_epoch]
            for name, f in (("mean", statistics.mean), ("lo", min), ("hi", max))}


runs = defaultdict(list)
for p in sorted(RESULTS.glob("*.json")):
    r = json.loads(p.read_text())
    if r["status"] == "done":
        runs[(r["task"], r["basis"])].append(r)

out = {"paper": PAPER, "window": WINDOW, "scores": {}, "curves": {}}
for task in TASKS:
    out["scores"][task], out["curves"][task] = {}, {}
    for basis in BASES:
        rs = sorted(runs[(task, basis)], key=lambda r: r["seed"])
        out["scores"][task][basis] = [
            {"seed": r["seed"], "best": round(100 * r["best"], 2), "final": round(100 * r["final"], 2)} for r in rs]
        out["curves"][task][basis] = {
            "n": len(rs),
            "acc": across_seeds([rolling([100 * e["metric"] for e in r["per_epoch"]]) for r in rs], 2),
            "loss": across_seeds([rolling([e["train_loss"] for e in r["per_epoch"]]) for r in rs], 4),
        }

Path(sys.argv[1]).write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {sys.argv[1]}: {sum(len(v) for t in out['scores'].values() for v in t.values())} runs")
