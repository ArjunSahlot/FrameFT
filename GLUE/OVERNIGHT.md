# Overnight run: instructions for the babysitting agent

## Goal
Keep the experiment queue running on this machine's GPU until **09:30 local time**, so that by
morning `GLUE/results/` holds valid, finished runs and `GLUE/results/MORNING_REPORT.md` explains
them. You are the mechanic, not the scientist: fix whatever stops the runs, never change what the
runs measure.

**You are done only when** the queue has finished (or it is 09:30), AND `MORNING_REPORT.md` is
written. Until then, keep looping. Most check-ins will find nothing wrong; that is the expected,
successful outcome. Ending the session early because "everything looks fine" is a failure: a crash
at 04:00 with nobody watching loses the rest of the night.

## What is running
FrameFT fine-tunes RoBERTa-base on GLUE tasks by training 1,000 coefficients per attention matrix.
The weight update is `ΔW = Bᵀ S B`, where `S` holds the trained coefficients and `B` is a fixed
768×768 orthogonal basis. The experiment asks whether the basis matters, comparing three `--basis`
values under otherwise identical settings:
- `frame`: the paper's tight fusion frame (at these settings, a real-valued Fourier matrix)
- `random`: a random orthogonal matrix
- `identity`: the identity matrix, so each coefficient changes one weight

- `GLUE/run_queue.py` is the autopilot. It runs `GLUE/queue.txt` top to bottom, one run at a time,
  skips runs whose JSON says `done` (or `failed_nan`), and retries a failed run once. It lives in
  the tmux session `frameft`.
- Each run writes `GLUE/results/<task>-<basis>-s<seed>.json`, rewritten after every epoch, and
  `GLUE/results/logs/<name>.log`, which holds the full stdout and traceback.
- `GLUE/results/runner.log` records start and end lines for each run.
- `python GLUE/status.py` shows everything on one screen: tmux, processes, GPU, the current run's
  progress with a STALE flag, and the results table.
- `python GLUE/summarize.py` prints the results table only.
- `python GLUE/check_bases.py` is a CPU-only sanity test of the basis code, safe to run anytime.

Environment: activate with `source /home/arjun/projects/FrameFT/.venv/bin/activate`. This is a uv
project with Python 3.11 and torch 2.7.1+cu128, running under WSL2 on an RTX 5070 Ti Laptop GPU
(12 GB). The display uses about 1 GB, and a training run peaks at about 5.5 GB total.

Expected timing: RTE is about 40 s per epoch × 60 epochs, roughly 40 min per run. MRPC is about
30 s per epoch × 100 epochs, roughly 50 min per run. GPU temperatures around 80–87 °C are normal
for this laptop.

## The loop
Repeat until done:
1. Run `python GLUE/status.py`.
2. Compare what you see against the table below. If something is wrong, diagnose it, fix it, and
   verify the fix.
3. Append one line to `GLUE/results/agent_journal.md`:
   `HH:MM | run, epoch x/y | OK` or `HH:MM | ... | PROBLEM: ... -> ACTION: ...`.
4. When a run has just finished, open its JSON and check it:
   - `status` is `done` and `epochs_done == epochs_total`.
   - `best` is plausible (see below).
   - `train_loss` in `per_epoch` isn't NaN or exploding.
5. Wait about 10 minutes, then go back to step 1. If your shell has a command timeout, use
   several `sleep 290` calls. Never wait more than 15 minutes between checks.

| Symptom | Likely cause | Action |
|---|---|---|
| tmux `MISSING` or runner `NOT RUNNING`, and the queue isn't finished | runner crashed, or WSL restarted | Restart with `tmux new-session -d -s frameft "cd /home/arjun/projects/FrameFT/GLUE && source ../.venv/bin/activate && python run_queue.py; exec bash"`. Finished runs are skipped. An interrupted run restarts from epoch 0, which is fine. |
| Current run `STALE` (no update for more than 10 min) | hung process | Read the tail of its log. If it's truly stuck, `kill` the NLU_GLUE.py pid. The runner then logs the failure and retries once. |
| Log shows `out of memory` | a long batch, or another GPU process | Check `nvidia-smi` for other processes. Lower that task's `micro_bs` in `TASKS` in `run_queue.py` (16 → 8). This only changes memory use, not the math, so it's allowed. Delete the failed run's JSON and let it rerun. |
| Run ends `failed_nan` | training diverged | Don't retry and don't touch learning rates. Record it for the report. |
| Network or Hugging Face download error | transient | The runner retries once. If both attempts fail, fix it, delete that JSON, and let it rerun. |
| Run failed twice (`rc != 0`) and the runner moved on | a real bug | Read the traceback and fix only infrastructure. Delete the JSON, and restart the runner if it has already finished. |
| Disk nearly full | checkpoints | Delete `GLUE/output/*/model_ckpt.pt` (old test checkpoints). Never delete anything in `GLUE/results/`. |

## Hard rules (these protect the validity of the results)
1. **Never change what a run measures.** Don't change hyperparameters in `TASKS` (other than
   `micro_bs`), seeds, `entry_seed`, `basis_seed`, epochs, the metric, the data, or any code in
   `peft/` or the training/eval math in `NLU_GLUE.py`. The three bases must differ *only* in
   `--basis`.
2. If a fix would change the numbers a run produces, don't make it. Stop the runner
   (`touch GLUE/results/STOP`), and write in the report what broke and what fix you propose.
   Mixing runs from two code versions ruins the comparison.
3. **Never edit or delete a finished run's JSON**, and never write a result by hand. Delete a JSON
   only as the rerun step in the table above, and only for a run that did not finish.
4. **Only one training process on the GPU at a time.** Don't launch GPU jobs of your own.
5. Don't install, upgrade, or remove packages unless a run can't proceed otherwise. If you do, log
   the exact command.
6. No `git commit`, `git push`, `git reset`, or `git checkout` of files. Leave any code changes
   uncommitted, and list them in the report (`git diff`).
7. Don't reorder the first six runs in `queue.txt`. If, at about 07:30, the queue will clearly
   finish before 09:30, you may append `mrpc frame 44`, `mrpc random 44`, and
   `mrpc identity 44`, in that order, and nothing else.

## What "plausible" looks like
The paper reports, for `frame` on RoBERTa-base (single numbers; the seed count isn't stated):
- RTE: 79.8
- MRPC: 92.3

Our script reports the best validation accuracy over all epochs.
- A `frame` result within about 1.5 points of the paper means the reproduction is fine.
- If `rte-frame-s42` or `mrpc-frame-s42` comes in more than 3 points below, don't tune anything.
  Check for infrastructure causes: did it finish all epochs, and are there warnings in the log?
  Note it prominently in the report, and keep the queue running, because the comparison between
  bases is still valid within our setup.
- `random` and `identity` can legitimately score higher or lower than `frame`. That is the
  experiment, not a bug. Only diverged or crashed runs are problems.

## MORNING_REPORT.md
Write it when the queue finishes or at 09:30, whichever comes first. Plain language, short:
1. **Headline:** how many runs finished, and any that failed or are still in progress.
2. **The results table:** paste the output of `python GLUE/summarize.py`.
3. **Reproduction check:** `frame` vs. the paper's numbers, per task.
4. **A first read of the comparison:**
   - `frame` vs. `random` vs. `identity`, per task.
   - Say whether the gaps are larger than the seed-to-seed spread. With 1–3 seeds, be modest.
     RTE has 277 validation examples, so one example is 0.36 points.
5. **Incidents:** every problem, what you did, and the time. Include `git diff --stat` if you
   changed any code.
6. **Anything the user must look at first.**
