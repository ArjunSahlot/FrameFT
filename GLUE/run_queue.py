"""Work through queue.txt one run at a time, skipping runs that already finished.

    python run_queue.py            # the real runs, results in results/
    python run_queue.py --smoke    # 1 epoch per run, results in results_smoke/

Each run writes results/<task>-<basis>-s<seed>.json (updated after every epoch) and
results/logs/<task>-<basis>-s<seed>.log. queue.txt is re-read before every run, so lines can be
added while the runner is going. To stop after the current run finishes: touch results/STOP
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# paper hyperparameters, copied from scripts/RoBERTa-base-<task>.sh. micro_bs only changes memory use, not the math.
COMMON = {"model_name_or_path": "roberta-base", "n_ff_coeffs": 1000, "bs": 32}
TASKS = {
    "rte":  {"max_length": 512, "num_epochs": 60,  "head_lr": 0.006,   "fft_lr": 0.321, "scale": 10.0, "tff_l": 2, "tff_block_size": 2,   "micro_bs": 16},
    "mrpc": {"max_length": 512, "num_epochs": 100, "head_lr": 0.01028, "fft_lr": 0.078, "scale": 50.0, "tff_l": 2, "tff_block_size": 768, "micro_bs": 16},
    "cola": {"max_length": 512, "num_epochs": 100, "head_lr": 0.00714, "fft_lr": 0.186, "scale": 30.0, "tff_l": 2, "tff_block_size": 768, "micro_bs": 16},
    "stsb": {"max_length": 512, "num_epochs": 30,  "head_lr": 0.00657, "fft_lr": 0.864, "scale": 10.0, "tff_l": 4, "tff_block_size": 2,   "micro_bs": 16},
}
TERMINAL = {"done", "failed_nan"}  # statuses that are never retried
MAX_ATTEMPTS = 2


def run_name(task, basis, seed):
    return f"{task}-{basis}-s{seed}"


def read_queue(path):
    runs = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            task, basis, seed = line.split()
            if task not in TASKS:
                raise ValueError(f"unknown task {task!r} in {path}")
            runs.append((task, basis, int(seed)))
    return runs


def read_status(json_path):
    try:
        return json.loads(json_path.read_text()).get("status")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 epoch per run, results in results_smoke/")
    parser.add_argument("--queue", default=str(HERE / "queue.txt"))
    cli = parser.parse_args()

    results_dir = HERE / ("results_smoke" if cli.smoke else "results")
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    runner_log = results_dir / "runner.log"

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(runner_log, "a") as f:
            f.write(line + "\n")

    env = dict(os.environ, WANDB_MODE="disabled", HF_DATASETS_TRUST_REMOTE_CODE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    attempts = {}
    log(f"runner started (smoke={cli.smoke}, queue={cli.queue})")
    while True:
        if (results_dir / "STOP").exists():
            log("STOP file found, exiting")
            break
        pending = [r for r in read_queue(Path(cli.queue))
                   if read_status(results_dir / f"{run_name(*r)}.json") not in TERMINAL
                   and attempts.get(r, 0) < MAX_ATTEMPTS]
        if not pending:
            log("queue finished")
            break

        task, basis, seed = run = pending[0]
        name = run_name(*run)
        attempts[run] = attempts.get(run, 0) + 1
        hp = {**COMMON, **TASKS[task]}
        if cli.smoke:
            hp["num_epochs"] = 1
        cmd = [sys.executable, "NLU_GLUE.py", "--task", task, "--dataset", task, "--seed", str(seed),
               "--basis", basis, "--share_entry", "--no_save_ckpt", "--exp_name", name,
               "--output_dir", str(results_dir / "ckpt" / name),
               "--results_json", str(results_dir / f"{name}.json")]
        for k, v in hp.items():
            cmd += [f"--{k}", str(v)]

        log(f"start {name} (attempt {attempts[run]}/{MAX_ATTEMPTS})")
        t0 = time.time()
        with open(results_dir / "logs" / f"{name}.log", "w") as f:
            f.write(" ".join(cmd) + "\n\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        record = {}
        try:
            record = json.loads((results_dir / f"{name}.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        log(f"end   {name} rc={rc} status={record.get('status')} best={record.get('best')} "
            f"epochs={record.get('epochs_done')}/{record.get('epochs_total')} minutes={(time.time() - t0) / 60:.1f}")


if __name__ == "__main__":
    main()
