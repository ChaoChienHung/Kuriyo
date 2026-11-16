# Kuriyo repository conventions

Kuriyo is primarily a repository for recommendation models. Read `README.md` before making changes. The repository is still being set up; do not assume a model architecture, dataset, training stack, or directory layout that has not been added yet.

Amazon Reviews data lives under `data/amazon_reviews/<year>/`: ignored downloads in `raw/<category>/`, a shared item catalog in `catalog/<category>/`, and prepared splits in `<train|validation|test>/raw/<category>/`. A versioned, model-independent item SID catalog lives in `artifacts/sids/<method>/<category>/<version>/`; converted examples live in `<split>/sids/<method>/<category>/<version>/`. The method names currently reserved are `sentence-t5_rqvae`, `sentence-t5_kmeans`, and `sentence-t5_lsh`. `validation/` is for model selection; `test/` is the held-out final evaluation. Preserve raw source files and keep years, categories, methods, and SID versions distinct. Document the source and splitting method when processed data is added.

The 2014 Beauty split is produced by `python3 -m scripts.prepare_amazon_reviews_2014`; `python3 -m scripts.prepare_amazon_reviews_2014_beauty` checks or creates the complete Beauty inputs idempotently. Keep its per-user chronological leave-one-out policy, source-row tie break, and 20-item history cap explicit. `Representations/SID/` owns model-independent SID generation, with its generator in `Representations/SID/scripts/generate_sids.py`; `scripts/materialize_amazon_sids.py` converts all three splits after a full item catalog is encoded. `Recall/core/` owns shared Recall metrics, batching, experiment helpers, and a basic Transformer, with tests in `Recall/tests/`. `Recall/TIGER/` owns TIGER-specific tokenization, decoding, and sequence training, including a paper-configured T5 entry point. `Ranking/` is reserved for reranking work, and `pipelines/` for cross-stage or end-to-end flows. Root `requirements.txt` contains shared modeling dependencies; SID and TIGER extend it separately. Keep smoke-test results distinct from full datasets and paper reproduction. Keep each component README accurate about runnable commands and planned work.

`core/` is for cross-stage shared interfaces, `utils/` for reusable data helpers, and root `scripts/` for data preparation and compatibility entry points. Model-specific scripts and runs belong with their model; SID generation runs belong under `Representations/SID/`. Keep research findings and open questions in `INSIGHT.md`, separating facts from hypotheses and using General, Recall, Ranking, and TIGER subsections within each main section. Track implementation work in `TODO.md`.

## Scope

- Keep recommendation model code, data preparation, training, evaluation, and experiment documentation here as those components are introduced.
- Keep data acquisition and preprocessing reproducible. Document each dataset's source, version or collection date, fields used, filtering rules, and any usage restrictions before relying on it.
- Keep raw datasets, downloaded images, model checkpoints, credentials, and large generated outputs out of Git. Commit the code and small configuration files needed to reproduce them.

## Experiments

- Record the dataset and split, preprocessing, model configuration, random seed, code revision, metrics, and artifacts for each run.
- Use the same held-out split and evaluation protocol when comparing models. Prevent user-item interaction leakage across time-based splits where applicable.
- Give each run a unique output location. Preserve previous run configurations and results rather than overwriting them.
- Add focused tests for data joins, split logic, and recommendation metrics when implementing those components.

## Editing and running

- Update `README.md` when setup steps, data sources, commands, or the repository layout become concrete.
- Prefer small, reviewable changes and follow the conventions of code already present.
- Do not push, publish data, or submit remote training jobs unless the user requests it.
- For NUS SoC jobs, read `skills/nus-soc-slurm/SKILL.md` before preparing or operating Slurm scripts. Treat partition, GPU, quota, and access details copied from other projects as historical until verified against the live cluster.
