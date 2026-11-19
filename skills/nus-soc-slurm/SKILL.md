---
name: nus-soc-slurm
description: Prepare, review, submit, monitor, and troubleshoot NUS School of Computing Slurm CPU or GPU jobs for Kuriyo. Use whenever work mentions the SoC cluster, xlogin, sbatch, srun, sinfo, sacct, GPU training, batch logs, allocations, quotas, node failures, or moving Kuriyo experiments from a local machine to SoC.
---

# NUS SoC Slurm for Kuriyo

Use this skill for cluster work; read the repository `AGENTS.md`, the relevant component README, and `scripts/run.sbatch` before acting. Partition names, GPU types, node health, and limits below are **observations with dates**, not current SoC policy: re-check them on the live cluster before relying on them.

## Access and live checks

1. If outside the NUS network, follow the current NUS VPN instructions. Connect to a SoC **login node** such as `xlogin.comp.nus.edu.sg` using the user's configured account; do not put an account name, password, token, or SSH key in a repository file.
2. Read the current [SoC Slurm guide](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm) and [Slurm Info](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm-info). If they are inaccessible, inspect the live cluster and state what could not be verified:
   - `sinfo -o '%P %l %a %G'` — partitions, time limits, GPU types.
   - `sinfo -N -o '%N %P %G %T' | sort -u` — every node with its partition, GPU type, and state.
   - `squeue -u "$USER"`, and when relevant `scontrol show partition <name>` or `scontrol show job <id>`.
3. Use the login node for `git`, `sbatch`, `squeue`, and `sacct` only. Do not import Torch or train there: in 2026-11 the login node's virtual-memory limit was 1 GiB and `import torch` failed while mapping libraries. Diagnose GPU code inside an allocation (`salloc` or `srun`).

## How code and logs move between the laptop and SoC

Kuriyo uses Git as the only channel; the agent never reads SoC directly.

- **Code to SoC:** commit and push to `main` locally (only when the user asks), then on SoC run `git pull --ff-only`. A job that is already running keeps the code it started with; the first log line records `commit=`.
- **Logs to the agent:** the user commits `slurm-*.out` on SoC and pushes; the agent runs `git fetch && git merge --ff-only origin/main` and reads the file. Old logs whose numbers are already recorded in an `INSIGHT.md` can be removed with `git rm`; record the retrieval command (`git show <commit>:<file>`) in `HANDOFF.md`.
- **Push rejected on SoC (`fetch first`) or `git pull` reports divergent branches:** inspect `git log --oneline origin/main..HEAD`, then `git pull --rebase` and `git push`. Never force-push.
- When giving the user a command, never leave placeholders such as `<JOBID>`: bash reads `<JOBID>` as an input redirection and fails with `JOBID: No such file or directory`. Give a command that finds the value itself, for example `f=$(ls -t slurm-kuriyo-sasrec-*.out | head -1)`.

## Kuriyo's batch entry point: `scripts/run.sbatch`

One stage-selectable script covers every Recall family. Submit from the repository root with `sbatch --export=ALL scripts/run.sbatch` and settings as environment variables:

| Setting | Values | Notes |
| --- | --- | --- |
| `MODEL` | `tiger` (default), `sasrec`, `fmrec` | SASRec/FMRec read the raw ASIN split and accept only `STAGE=prepare\|train\|test\|all` |
| `STAGE` | `prepare`, `sid`, `materialize`, `train`, `test`, `all` | `all` excludes the held-out `test` |
| `RUN_NAME` | new name per experiment | an existing run with `checkpoints/latest.pt` (or a flat pre-2026-11-19 `latest.pt`, moved into the new layout on resume) resumes; a run directory without a checkpoint is refused |
| `SID_RECIPE`, `SID_VERSION` | TIGER only | an existing SID version is reused only if its manifest records the same recipe |
| `CATEGORY` | `Beauty` (default), `Sports_and_Outdoors`, `Toys_and_Games` | the 2014 dataset; splits, SID versions and default run names are per category |
| `SEED` | integer, default `42` | passed to train and test; repeated seeds follow the README section "實驗設計與統計檢定" (name runs `-s<seed>`) |
| `TRAIN_ARGS` | trainer flags, e.g. `--loss bce` | one changed factor per `RUN_NAME`; passed to train and test alike |
| `ADAFACTOR_SCALE_PARAMETER` | `1` (default, T5X-style) or `0` | TIGER only; `0` is needed only to resume runs from before 2026-11-19; resume refuses a changed value |

Use `--job-name=kuriyo-<model>` so the log is `slurm-kuriyo-<model>-<jobid>.out`. The script's `#SBATCH` defaults (partition `gpu-long`, 2 days, 1 GPU, 8 CPUs, 32 GB) were chosen for TIGER; override with `sbatch` flags when the live cluster differs.

- **Several jobs at once:** submit each `MODEL` (or seed) as its own job with `STAGE=train` after the category's split exists. `STAGE=all` or `prepare` makes every job rewrite the same split concurrently. Each job requests its own GPU; check the user's GPU quota first.
- The first log line echoes `job=`, `commit=`, `model=`, `category=`, `seed=`, `train_args=`, `stage=`, and `run=`. Check it before reading any metric.
- `pip install` runs at the start of every job and should print only "Requirement already satisfied"; filter it with `grep -v "Requirement already"` when reading logs.

## Monitoring and reading results

- `squeue -u "$USER"` while queued or running.
- `sacct -u "$USER" -S today --format=JobID%12,JobName%15,State,ExitCode,Elapsed,NodeList%10` after it leaves the queue. `NodeList` shows which node each job used; compare failing and working jobs this way.
- A finished queue entry does not prove success. Read the log, distinguish `TIMEOUT` from `COMPLETED`, and verify the expected artifacts (`latest.pt`, `best-metrics.json`, validation files) before reporting results.
- Separate validation (model selection) from held-out test, and record job ID, commit, and run name for every number in `INSIGHT.md`/`HANDOFF.md`.

## Troubleshooting

### `CUDA unknown error` → `CUDA is unavailable`, job exits in seconds

Symptom in the log: a Torch warning `CUDA initialization: CUDA unknown error`, then `train.py: error: CUDA is unavailable` and `srun: error: <node>: task 0: Exited with exit code 2`. The training scripts check CUDA before creating the run directory, so the same `RUN_NAME` can be resubmitted.

- This is a node problem, not Kuriyo code. On 2026-11-19 it happened on `xgpe2`, `xgpe4`, and `xgpe6` (all `gpu:nv:1`), while jobs on `xgpf11` trained normally.
- **Trap:** broken nodes look `idle` in `sinfo` because every job on them dies at once, so Slurm keeps sending new jobs there. Excluding one node at a time just moves the job to the next broken sibling. Exclude the whole group.
- **Trap:** `--exclude` must name only existing nodes. `--exclude='xgpe[0-6]'` failed with `Invalid node name specified` because `xgpe5` does not exist. Build the list from `sinfo -N`, e.g. `--exclude='xgpe[0-4,6]'`.
- Alternatively request a GPU type from the `GRES` column, e.g. `--gpus=a100-40:1`, after checking that the partition offers it.

### `no kernel image is available` on older GPUs

`xgpd5` has a TITAN V (sm_70); other `gpu:nv` nodes may be similar (unverified). A Torch build for CUDA 13 dropped sm_70; `run.sbatch` therefore installs `torch==2.7.1+cu126` from the PyTorch cu126 index. Keep that pin aligned with the comment in the root `requirements.txt`.

### Other failures

- **Pending:** inspect `squeue`, `scontrol show job <id>`, requested resources, and partition availability before changing requests.
- **Failed or OOM:** use `sacct` state, exit code, and `MaxRSS` together with the log; distinguish memory failure from Python exceptions, missing data, time limits, and package errors.
- **`/tmp` quota during pip or Git:** set `TMPDIR` to a writable project directory before retrying (`run.sbatch` uses `.tmp/`); `--no-cache-dir` alone does not move temporary checkouts.
- **No log:** confirm the job started and where `--output` resolves; Slurm opens the log before the script runs, so its directory must already exist.
- **Wrong code or data:** compare the recorded commit, dataset SHA, SID version, and split manifest with the intended run before interpreting metrics.

## Rules

- Preparing or reviewing a script does not authorize a live job. Submit, cancel, or change jobs only when the user asks; the agent normally gives the user the exact command to run.
- Do not push code or publish data merely to make a submission possible without authorization.
- Keep source data, SID artifacts, checkpoints, and caches on a permitted persistent filesystem and out of Git; only small Slurm logs are committed.
- A single-process job gains nothing from extra GPUs. Do not pin a specific node unless the workload requires it; prefer excluding known-bad nodes.

## Provenance

Synthesized from the earlier `nus-soc-slurm` skill, the `rl-test` project's SoC notes, and Kuriyo's own SoC runs in 2026-11 (jobs 1153307–1165548; see `HANDOFF.md`). Refer to the [official Slurm `sbatch` manual](https://slurm.schedmd.com/sbatch.html) for command semantics.
