---
name: nus-soc-slurm
description: Prepare, review, submit, monitor, and troubleshoot NUS School of Computing Slurm CPU or GPU jobs for Kuriyo. Use whenever work mentions the SoC cluster, xlogin, sbatch, srun, sinfo, GPU training, batch logs, allocations, quotas, or moving Kuriyo experiments from a local machine to SoC.
---

# NUS SoC Slurm for Kuriyo

Use this skill for cluster work; read the repository `AGENTS.md`, the relevant component README, and the job script before acting. This skill combines the existing `nus-soc-slurm` workflow with cluster practices observed in the local `rl-test` project. The latter's partition limits, GPU names, account paths, and resource requests are historical examples, **not** current SoC policy.

## Access and live checks

1. If outside the NUS network, follow the current NUS VPN instructions. Connect to a SoC **login node** such as `xlogin.comp.nus.edu.sg` using the user's configured account; do not put an account name, password, token, or SSH key in a repository file.
2. Read the current [SoC Slurm guide](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm) and [Slurm Info](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm-info). If they are inaccessible, inspect the live cluster through the login node and state what could not be verified. Check `sinfo -o '%P %l %a %G'`, `squeue -u "$USER"`, and, when relevant, `scontrol show partition` or `sbatch --help` before selecting a partition, GPU type, wall time, CPUs, or memory.
3. Use the SoC login node for `sbatch`; use an allocated compute node for sustained CPU/GPU work. Do not run training on the login node. Prefer a short local or allocated smoke test before a full training run.

## Prepare a reproducible run

- Prepare the code and batch script locally. Inspect the Git diff, test the Python entry point, and record the intended Git revision. Sync code to SoC through the project's approved workflow; pushing or pulling is a separate action governed by `AGENTS.md` and the user's instructions. A job already running keeps the code and environment it started with.
- On SoC, enter the Kuriyo repository root, verify `README.md`, the expected Git revision, `.venv/bin/python`, input files, and output parent directory. Install `Representations/SID/requirements.txt` for SID generation and `Recall/TIGER/baseline/requirements.txt` for TIGER sequence training. Verify imports and `torch.cuda.is_available()` inside the intended environment.
- Keep Amazon source data, SID artifacts, checkpoints, caches, and logs on a permitted persistent filesystem. Confirm storage quota and free space. If a pip or Git install fails with `/tmp` quota errors, set `TMPDIR` to a writable project or home directory **before** retrying; `--no-cache-dir` alone does not relocate temporary checkouts. Do not assume a generic node-local scratch path persists after the job.
- Use a new SID version and model run name. For the current Beauty fixed-catalog workflow, SID generation covers all 12,101 catalog items at `data/amazon_reviews/2014/catalog/Beauty/items.jsonl`; the resulting SID catalog belongs under `data/amazon_reviews/2014/artifacts/sids/<method>/Beauty/<version>/`. Materialize split examples before TIGER training. Do not treat a limited-item smoke run as a full catalog.
- Keep validation for model selection and test for the final report. Record source file SHA, split manifest, SID catalog/version/SHA, config, seed, Git revision, metrics, and output path for each run.

## Batch script pattern

Put `#SBATCH` directives immediately after the shebang, before shell commands. Slurm reads their values literally; shell variables in these directives are not expanded. Choose resource values only after the live checks above. Create the log directory **before** `sbatch`, because Slurm opens the log before the script body runs.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=kuriyo-sid
#SBATCH --partition=<verified-partition>
#SBATCH --time=<verified-wall-time>
#SBATCH --cpus-per-task=<cpu-count>
#SBATCH --mem=<memory>
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the Kuriyo repository root}"
test -f README.md
test -x .venv/bin/python
test -f data/amazon_reviews/2014/catalog/Beauty/items.jsonl
echo "job=${SLURM_JOB_ID:?} commit=$(git rev-parse HEAD)"
.venv/bin/python -c 'import torch; assert torch.cuda.is_available()'
srun .venv/bin/python -m Representations.SID.scripts.generate_sids \
  --output-dir data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/<new-version> \
  --encoder sentence-t5 --device cuda
```

Adapt the command for TIGER using `python -m Recall.TIGER.baseline.train_sequence` and its documented `--sid-run`, `--sid-method`, and `--run-name`. Do not copy the `rl-test` project paths or model-specific environment variables into Kuriyo. Do not pin a specific node unless the workload requires it. A single-process job gains no speed from requesting extra GPUs without multi-GPU code.

## Submission and monitoring

- Preparing or reviewing a script does not authorize a live job. Submit, cancel, or change a job only when the user has requested that action. Do not push code or publish data merely to make a submission possible without authorization.
- Before submission, verify the script, new output path, environment, inputs, current resources, and `logs/` directory. Submit from the repository root with `sbatch --parsable <script>`, record the returned job ID, and inspect `squeue -j <job-id>`.
- Read the configured `logs/<job-name>-<job-id>.out` after the job starts; its location is relative to the submission directory unless an absolute output path was given. After it leaves the queue, use `sacct -j <job-id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS` and inspect the log and expected artifacts. A missing queue entry alone does not prove success.
- For dependent stages, use `--dependency=afterok:<job-id>` only after confirming the producer's output contract. For independent runs, an array job can be appropriate; set an explicit concurrency limit if needed. Do not schedule duplicate runs to compensate for a slow queue.
- Use `salloc` for interactive debugging or `srun` for a single allocated command. Exit idle allocations promptly. Use `scancel` only when stopping the job is intended.

## Troubleshooting

- **Pending:** inspect `squeue`, `scontrol show job <job-id>`, requested resources, and current partition availability before changing requests.
- **Failed or OOM:** use `sacct` state/exit code, `MaxRSS`, and the job log; distinguish memory failure from Python exceptions, missing data, time limits, and package errors.
- **No log:** confirm whether the job has started and where `--output` resolves. Create log directories before resubmission.
- **Wrong code or data:** compare recorded Git revision, dataset SHA, SID version, and split manifest with the intended run before interpreting metrics.

## Provenance

This repository skill was synthesized from the pre-existing `nus-soc-slurm` skill; `rl-test/README.md` (its “Running on NUS SoC Cluster” section); and its `verifier/train.sbatch`, `agent/eval.sbatch`, and `retrieval/build_fever_index.sbatch` examples. Verify live SoC rules at run time. Refer to the [official Slurm `sbatch` manual](https://slurm.schedmd.com/sbatch.html) for command semantics.
