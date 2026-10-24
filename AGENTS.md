# Kuriyo repository conventions

Kuriyo is primarily a repository for recommendation models. Read `README.md` before making changes. The repository is still being set up; do not assume a model architecture, dataset, training stack, or directory layout that has not been added yet.

Amazon Reviews data lives under `data/amazon_reviews/`: ignored raw downloads in `raw/<year>/<category>/`, and future split outputs in `train/`, `eval/`, and `test/`. `eval/` means validation for model selection; `test/` is the held-out final evaluation. Preserve the raw source files and keep years and categories distinct. Document the source and splitting method when processed data is added.

`core/` is for recommendation models and shared interfaces, `utils/` for reusable data and evaluation helpers, and `scripts/` for runnable entry points. Keep research findings and open questions in `INSIGHT.md`, separating facts from hypotheses.

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
