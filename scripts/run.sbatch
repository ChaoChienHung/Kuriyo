#!/usr/bin/env bash
#SBATCH --job-name=kuriyo-tiger
#SBATCH --partition=gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

# Verify the live SoC partition, GPU and limits before submitting. Override
# these example directives with sbatch flags when the cluster configuration differs.
# STAGE=all runs prepare -> SID -> materialize -> TIGER; STAGE=test is separate.
# STAGE=sid also prepares/verifies the category's inputs before generating SIDs.
# CATEGORY (default Beauty; also Sports_and_Outdoors, Toys_and_Games) selects the 2014
# dataset; SID versions and run directories are kept apart by category. SEED (default
# 42) sets the trainer seed; repeat a comparison with several seeds (README, "Statistics").
# TRAIN_ARGS adds trainer flags for a single-factor comparison, e.g.
# TRAIN_ARGS="--loss bce"; they are passed to train and test alike, so test sees the
# run's settings. Each such setting needs its own RUN_NAME.
# Individual stages can be submitted separately when the full paper-budget run
# exceeds one allocation. SID_RECIPE (default stable) selects the RQ-VAE recipe
# in Representations/SID/scripts/generate_sids.py; an existing SID_VERSION is
# reused only when its manifest records the same recipe. ADAFACTOR_SCALE_PARAMETER
# (TIGER only, default 1) trains with T5X-style Adafactor (steps scaled by
# parameter RMS); 0 restores the absolute-step behaviour of runs before
# 2026-11-19 and is needed to resume them. Resume refuses a changed value.
# MODEL (default tiger) selects the Recall family. MODEL=sasrec|fmrec trains on
# the raw ASIN split and accepts STAGE=prepare|train|test|all only (all runs
# prepare -> train); SID settings do not apply. Submit several MODEL jobs at once
# with STAGE=train after the category's split exists, so they do not prepare it concurrently.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the Kuriyo repository root}"
test -f AGENTS.md && test -f README.md

model="${MODEL:-tiger}"
case "$model" in
  tiger|sasrec|fmrec) ;;
  *) echo "MODEL must be tiger, sasrec, or fmrec" >&2; exit 2 ;;
esac
stage="${STAGE:-all}"
case "$stage" in
  all|prepare|sid|materialize|train|test) ;;
  *) echo "STAGE must be all, prepare, sid, materialize, train, or test" >&2; exit 2 ;;
esac
category="${CATEGORY:-Beauty}"
case "$category" in
  Beauty|Sports_and_Outdoors|Toys_and_Games) ;;
  *) echo "CATEGORY must be Beauty, Sports_and_Outdoors, or Toys_and_Games" >&2; exit 2 ;;
esac
category_slug=$(printf '%s' "$category" | tr '[:upper:]' '[:lower:]')
sid_version="${SID_VERSION:-tiger-paper-v1}"
case "$model" in
  tiger) if [[ "$category" == Beauty ]]; then default_run=tiger-paper-v1; else default_run="tiger-paper-$category_slug-v1"; fi
         run_name="${RUN_NAME:-$default_run}" ;;
  sasrec) run_name="${RUN_NAME:-sasrec-$category_slug-v1}" ;;
  fmrec) run_name="${RUN_NAME:-fmrec-$category_slug-v1}" ;;
esac
seed="${SEED:-42}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 2; }
train_args_text="${TRAIN_ARGS:-}"
[[ "$train_args_text" =~ ^[A-Za-z0-9_.=\ -]*$ ]] || { echo "TRAIN_ARGS may contain only flags and plain values" >&2; exit 2; }
read -r -a train_args <<< "$train_args_text"
sid_recipe="${SID_RECIPE:-stable}"
adafactor_scale="${ADAFACTOR_SCALE_PARAMETER:-}"
[[ -z "$adafactor_scale" || "$adafactor_scale" == 0 || "$adafactor_scale" == 1 ]] || { echo "ADAFACTOR_SCALE_PARAMETER must be 0 or 1" >&2; exit 2; }
case "$sid_recipe" in
  paper-pytorch|paper-tf-adagrad|stable) ;;
  *) echo "SID_RECIPE must be paper-pytorch, paper-tf-adagrad, or stable" >&2; exit 2 ;;
esac
[[ "$sid_version" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid SID_VERSION" >&2; exit 2; }
if [[ "$model" != tiger ]]; then
  [[ "$stage" == all || "$stage" == prepare || "$stage" == train || "$stage" == test ]] || {
    echo "MODEL=$model supports STAGE=all, prepare, train, or test" >&2; exit 2;
  }
  [[ "$adafactor_scale" != 1 ]] || { echo "ADAFACTOR_SCALE_PARAMETER applies only to MODEL=tiger" >&2; exit 2; }
fi
if [[ -z "$adafactor_scale" ]]; then
  if [[ "$model" == tiger ]]; then adafactor_scale=1; else adafactor_scale=0; fi
fi
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid RUN_NAME" >&2; exit 2; }

export TMPDIR="${TMPDIR:-$PWD/.tmp}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$HF_HOME"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
# SoC override documented in root requirements.txt; keep both in sync.
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.7.1+cu126" \
  -r Representations/SID/requirements.txt -r Recall/TIGER/baseline/requirements.txt

echo "job=${SLURM_JOB_ID:-local} commit=$(git rev-parse HEAD) model=$model category=$category seed=$seed train_args=[$train_args_text] stage=$stage sid_version=$sid_version sid_recipe=$sid_recipe adafactor_scale_parameter=$adafactor_scale run=$run_name"
if [[ "$model" != tiger ]]; then
  case "$model" in
    sasrec) family=SASRec ;;
    fmrec) family=FMRec ;;
  esac
  module="Recall.$family.baseline.train"
  run_dir="Recall/$family/baseline/runs/$run_name"
  if [[ "$stage" == all || "$stage" == prepare ]]; then
    srun .venv/bin/python -m scripts.ensure_amazon_reviews_2014 --category "$category"
  fi
  common=(--category "$category" --seed "$seed" --run-name "$run_name" --device cuda ${train_args[@]+"${train_args[@]}"})
  if [[ "$stage" == all || "$stage" == train ]]; then
    for input in "catalog/$category/items.jsonl" "train/raw/$category/examples.jsonl" "validation/raw/$category/examples.jsonl"; do
      test -f "data/amazon_reviews/2014/$input" || {
        echo "$category input missing: $input; run STAGE=prepare first" >&2; exit 2;
      }
    done
    args=("${common[@]}")
    if [[ -e "$run_dir" ]]; then
      # checkpoints/latest.pt; runs before 2026-11-19 keep latest.pt at the top and are
      # moved into the current layout by the trainer (core/runs.py).
      test -f "$run_dir/checkpoints/latest.pt" || test -f "$run_dir/latest.pt" || {
        echo "Incomplete run without checkpoint: $run_name" >&2; exit 2;
      }
      args+=(--resume)
    fi
    srun .venv/bin/python -m "$module" "${args[@]}"
  fi
  if [[ "$stage" == test ]]; then
    srun .venv/bin/python -m "$module" "${common[@]}" --eval-test-only
  fi
  exit 0
fi

sid_dir="data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/$category/$sid_version"
split_dir="data/amazon_reviews/2014/train/sids/sentence-t5_rqvae/$category/$sid_version"
catalog="data/amazon_reviews/2014/catalog/$category/items.jsonl"
common=(--category "$category" --seed "$seed" --sid-run "$sid_dir" --sid-method sentence-t5_rqvae
        --run-name "$run_name" --device cuda ${train_args[@]+"${train_args[@]}"})

if [[ "$stage" == all || "$stage" == prepare || "$stage" == sid ]]; then
  srun .venv/bin/python -m scripts.ensure_amazon_reviews_2014 --category "$category"
fi

if [[ "$stage" == all || "$stage" == sid ]]; then
  test -f "$catalog" || {
    echo "$category catalog missing after preparation; inspect the preparation log" >&2; exit 2;
  }
  if [[ -e "$sid_dir" ]]; then
    test -f "$sid_dir/semantic_ids.jsonl" && test -f "$sid_dir/manifest.json" || {
      echo "Incomplete SID directory: $sid_dir" >&2; exit 2;
    }
    # Manifests written before recipes existed record only paper_baseline=true.
    recorded_recipe=$(python3 -c 'import json, sys
manifest = json.load(open(sys.argv[1]))
print(manifest.get("recipe") or ("paper-pytorch" if manifest.get("paper_baseline") else "none"))' "$sid_dir/manifest.json")
    [[ "$recorded_recipe" == "$sid_recipe" ]] || {
      echo "SID $sid_version was built with recipe $recorded_recipe, not SID_RECIPE=$sid_recipe; use a new SID_VERSION" >&2; exit 2;
    }
    echo "Reusing completed SID catalog: $sid_dir (recipe $sid_recipe)"
  else
    srun .venv/bin/python -m Representations.SID.scripts.generate_sids \
      --items "$catalog" --output-dir "$sid_dir" --encoder sentence-t5 --recipe "$sid_recipe" --device cuda
  fi
fi

if [[ "$stage" == all || "$stage" == materialize ]]; then
  test -f "$sid_dir/semantic_ids.jsonl" || { echo "SID catalog missing: $sid_dir" >&2; exit 2; }
  if [[ -f "$split_dir/split_manifest.json" ]]; then
    for split in train validation test; do
      test -f "data/amazon_reviews/2014/$split/sids/sentence-t5_rqvae/$category/$sid_version/examples.jsonl" || {
        echo "Incomplete SID split for $split" >&2; exit 2;
      }
    done
    echo "Reusing materialized SID splits: $sid_version"
  else
    srun .venv/bin/python -m scripts.materialize_amazon_sids \
      --year 2014 --category "$category" --method sentence-t5_rqvae \
      --sid-file "$sid_dir/semantic_ids.jsonl"
  fi
fi

if [[ "$stage" == all || "$stage" == train ]]; then
  test -f "$split_dir/split_manifest.json" || {
    echo "SID split missing; run STAGE=materialize first" >&2; exit 2;
  }
  args=("${common[@]}")
  if [[ "$adafactor_scale" == 1 ]]; then args+=(--adafactor-scale-parameter); else args+=(--no-adafactor-scale-parameter); fi
  run_dir="Recall/TIGER/baseline/runs/$run_name"
  if [[ ! -e "$run_dir" && -e "Recall/TIGER/runs/$run_name" ]]; then
    echo "Run $run_name is at the legacy path Recall/TIGER/runs/; move it to $run_dir first" >&2
    exit 2
  fi
  if [[ -e "$run_dir" ]]; then
    # checkpoints/latest.pt; runs before 2026-11-19 keep latest.pt at the top and are
    # moved into the current layout by the trainer (core/runs.py).
    test -f "$run_dir/checkpoints/latest.pt" || test -f "$run_dir/latest.pt" || {
      echo "Incomplete run without checkpoint: $run_name" >&2; exit 2;
    }
    args+=(--resume)
  fi
  srun .venv/bin/python -m Recall.TIGER.baseline.train "${args[@]}"
fi

if [[ "$stage" == test ]]; then
  args=("${common[@]}")
  if [[ "$adafactor_scale" == 0 ]]; then args+=(--no-adafactor-scale-parameter); fi
  srun .venv/bin/python -m Recall.TIGER.baseline.train "${args[@]}" --eval-test-only
fi
