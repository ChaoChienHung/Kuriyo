# Kuriyo repository conventions

Kuriyo is primarily a repository for recommendation models. Read `README.md` before making changes; consult `HANDOFF.md` for current experiment evidence and outstanding work. The repository is still being set up; do not assume a model architecture, dataset, training stack, or directory layout that has not been added yet.

Amazon Reviews data lives under `data/amazon_reviews/<year>/`: ignored downloads in `raw/<category>/`, a shared item catalog in `catalog/<category>/`, and prepared splits in `<train|validation|test>/raw/<category>/`. A versioned, model-independent item SID catalog lives in `artifacts/sids/<method>/<category>/<version>/`; converted examples live in `<split>/sids/<method>/<category>/<version>/`. The method names currently reserved are `sentence-t5_rqvae`, `sentence-t5_kmeans`, and `sentence-t5_lsh`. `validation/` is for model selection; `test/` is the held-out final evaluation. Preserve raw source files and keep years, categories, methods, and SID versions distinct. Document the source and splitting method when processed data is added.

The 2014 Beauty split is produced by `python3 -m scripts.prepare_amazon_reviews_2014`; `python3 -m scripts.prepare_amazon_reviews_2014_beauty` checks or creates the complete Beauty inputs idempotently. Keep its per-user chronological leave-one-out policy, source-row tie break, and 20-item history cap explicit. `Representations/SID/` owns model-independent SID generation, with its generator in `Representations/SID/scripts/generate_sids.py`; `scripts/materialize_amazon_sids.py` converts all three splits after a full item catalog is encoded. `Recall/core/` owns shared Recall metrics, batching, experiment helpers, and a basic Transformer, with tests in `Recall/tests/`. `Recall/TIGER/` is the TIGER model family: its root owns shared TIGER tokenization, decoding, and the smoke sequence trainer, and `Recall/TIGER/baseline/` owns the paper-configured T5 baseline (`python -m Recall.TIGER.baseline.train`). `Ranking/` is reserved for reranking work, and `pipelines/` for cross-stage or end-to-end flows. Root `requirements.txt` contains shared modeling dependencies; SID and TIGER extend it separately. Keep smoke-test results distinct from full datasets and paper reproduction. Keep each component README accurate about runnable commands and planned work.

`core/` is for cross-stage shared interfaces, `utils/` for reusable data helpers, and root `scripts/` for data preparation and compatibility entry points. Model-specific scripts and runs belong with their model; SID generation runs belong under `Representations/SID/`. Keep research findings and open questions in `INSIGHT.md`, separating facts from hypotheses and using General, Recall, Ranking, and TIGER subsections within each main section. Track prioritized implementation work in `TODO.md`. Keep `HANDOFF.md` concise, below 0.1M tokens (100,000 tokens), and update it when run evidence, blockers, or next steps change. Separate observed log output from artifact verification; distinguish validation from held-out test.

## Scope

- Keep recommendation model code, data preparation, training, evaluation, and experiment documentation here as those components are introduced.
- Keep data acquisition and preprocessing reproducible. Document each dataset's source, version or collection date, fields used, filtering rules, and any usage restrictions before relying on it.
- Keep raw datasets, downloaded images, model checkpoints, credentials, and large generated outputs out of Git. Commit the code and small configuration files needed to reproduce them.

## Experiments

- Record the dataset and split, preprocessing, model configuration, random seed, code revision, metrics, and artifacts for each run.
- Use the same held-out split and evaluation protocol when comparing models. Prevent user-item interaction leakage across time-based splits where applicable.
- Give each run a unique output location. Preserve previous run configurations and results rather than overwriting them.
- Add focused tests for data joins, split logic, and recommendation metrics when implementing those components.

## Model families, variants, and tests

- Group related variants under their model family (for example `Recall/TIGER/`). The family root holds code shared by its variants (tokenizers, decoding, data loading, shared tests). Each variant with a distinct implementation is a direct subdirectory of the family, sibling to the others: `Recall/TIGER/baseline/`, `Recall/TIGER/<variant-name>/`. Each variant directory contains its own `README.md`, entry point, `__init__.py`, and `tests/`.
- `baseline/` holds the implementation currently run and compared against; it denotes the comparison point, not proof of paper reproduction. TIGER's baseline entry point is `Recall/TIGER/baseline/train.py`.
- Keep configuration-only differences as named configurations of an existing variant; add a new variant directory only for a distinct implementation. Name variants by what differs (for example `constrained-decoding/`), not `variant1/` or `v2/`. Variants import shared family and `Recall/core/` modules; do not copy an entire training stack for each variant. Each variant writes its runs to its own Git-ignored `<family>/<variant>/runs/<RUN_NAME>/` (TIGER baseline: `Recall/TIGER/baseline/runs/`); the family-level `runs/` is only for family smoke tools. When a variant's runs directory changes, batch scripts must refuse to silently start fresh when a run exists only at the old path.
- When adding a variant, add it to the family README's variant table and document its differences from `baseline/`.
- Root `tests/` covers cross-stage utilities, data preparation, and pipeline contracts; `Recall/tests/` covers shared Recall modules; family-shared contracts belong in `<family>/tests/`, and variant-specific contracts and integration tests in `<family>/<variant>/tests/`. Keep `__init__.py` files so root unittest discovery includes nested suites.
- Test observable behavior: causal masking, padding invariance, valid SID decoding, gradients, checkpoint/resume, and held-out evaluation. For a variant, run both shared contracts and its own tests; passing a smoke test does not establish recommendation quality.
- Use `core/run_logging.py` for scheduler-independent run logging. Append training measurements to `train.jsonl` and lifecycle events to `events.log`; distinguish invocations with session IDs, preserve prior history on resume, and record the restored step. Never treat Slurm stdout as the only experiment record.

## Staged batch pipelines

- Prefer a single stage-selectable `run.sbatch` entry point for each training pipeline, following `scripts/run.sbatch`. Keep orchestration in the batch script and implementation in reusable Python entry points.
- Support individual stages and an `all` path that runs them in dependency order. The current TIGER pipeline uses `STAGE=prepare|sid|materialize|train|test|all`; `all` runs prepare → sid → materialize → train, while final held-out `test` is explicitly separate.
- Validate stage names and required input artifacts before expensive work. Stop on failure. Document any automatic prerequisites, such as `STAGE=sid` preparing or verifying Beauty inputs first.
- Reuse completed artifacts only after checking their required files and relevant manifests. Treat partial outputs as incomplete; do not silently overwrite or delete them. Resume only stages with implemented checkpoint support. Currently TIGER training can resume from `latest.pt`, but SID/RQ-VAE generation cannot resume mid-training.
- Keep dataset, SID version, and run identity consistent across stages. Expose settings such as `SID_VERSION` and `RUN_NAME`, validate values used in paths, and use fresh output locations for new experiments.
- Document Git-pull-to-run commands, environment installation, stage prerequisites and outputs, log locations, checkpoint behavior, and final evaluation in the relevant README. Keep SoC dependency overrides aligned with root `requirements.txt`.
- Use batch jobs for full runs and short interactive allocations for diagnostics. Record job ID and code revision in logs; distinguish time-limit termination from successful completion and verify expected artifacts before reporting results.
- Add focused orchestration tests when changing stage behavior, including prerequisite ordering, artifact reuse, and failure propagation. These tests should not require Slurm, downloads, or a GPU.

## Editing and running

- Update `README.md` when setup steps, data sources, commands, or the repository layout become concrete.
- Prefer small, reviewable changes and follow the conventions of code already present.
- Do not push, publish data, or submit remote training jobs unless the user requests it.
- For NUS SoC jobs, read `skills/nus-soc-slurm/SKILL.md` before preparing or operating Slurm scripts. Treat partition, GPU, quota, and access details copied from other projects as historical until verified against the live cluster.
