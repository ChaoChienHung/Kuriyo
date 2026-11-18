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
# STAGE=sid also prepares/verifies Beauty inputs before generating SIDs.
# Individual stages can be submitted separately when the full paper-budget run
# exceeds one allocation. SID_RECIPE (default stable) selects the RQ-VAE recipe
# in Representations/SID/scripts/generate_sids.py; an existing SID_VERSION is
# reused only when its manifest records the same recipe.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the Kuriyo repository root}"
test -f AGENTS.md && test -f README.md

stage="${STAGE:-all}"
case "$stage" in
  all|prepare|sid|materialize|train|test) ;;
  *) echo "STAGE must be all, prepare, sid, materialize, train, or test" >&2; exit 2 ;;
esac
sid_version="${SID_VERSION:-tiger-paper-v1}"
run_name="${RUN_NAME:-tiger-paper-v1}"
sid_recipe="${SID_RECIPE:-stable}"
case "$sid_recipe" in
  paper-pytorch|paper-tf-adagrad|stable) ;;
  *) echo "SID_RECIPE must be paper-pytorch, paper-tf-adagrad, or stable" >&2; exit 2 ;;
esac
[[ "$sid_version" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid SID_VERSION" >&2; exit 2; }
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

echo "job=${SLURM_JOB_ID:-local} commit=$(git rev-parse HEAD) stage=$stage sid_version=$sid_version sid_recipe=$sid_recipe run=$run_name"
sid_dir="data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/$sid_version"
split_dir="data/amazon_reviews/2014/train/sids/sentence-t5_rqvae/Beauty/$sid_version"

if [[ "$stage" == all || "$stage" == prepare || "$stage" == sid ]]; then
  srun .venv/bin/python -m scripts.prepare_amazon_reviews_2014_beauty
fi

if [[ "$stage" == all || "$stage" == sid ]]; then
  test -f data/amazon_reviews/2014/catalog/Beauty/items.jsonl || {
    echo "Beauty catalog missing after preparation; inspect the preparation log" >&2; exit 2;
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
      --output-dir "$sid_dir" --encoder sentence-t5 --recipe "$sid_recipe" --device cuda
  fi
fi

if [[ "$stage" == all || "$stage" == materialize ]]; then
  test -f "$sid_dir/semantic_ids.jsonl" || { echo "SID catalog missing: $sid_dir" >&2; exit 2; }
  if [[ -f "$split_dir/split_manifest.json" ]]; then
    for split in train validation test; do
      test -f "data/amazon_reviews/2014/$split/sids/sentence-t5_rqvae/Beauty/$sid_version/examples.jsonl" || {
        echo "Incomplete SID split for $split" >&2; exit 2;
      }
    done
    echo "Reusing materialized SID splits: $sid_version"
  else
    srun .venv/bin/python -m scripts.materialize_amazon_sids \
      --year 2014 --category Beauty --method sentence-t5_rqvae \
      --sid-file "$sid_dir/semantic_ids.jsonl"
  fi
fi

if [[ "$stage" == all || "$stage" == train ]]; then
  test -f "$split_dir/split_manifest.json" || {
    echo "SID split missing; run STAGE=materialize first" >&2; exit 2;
  }
  args=(--sid-run "$sid_dir" --sid-method sentence-t5_rqvae
        --run-name "$run_name" --device cuda)
  run_dir="Recall/TIGER/baseline/runs/$run_name"
  if [[ ! -e "$run_dir" && -e "Recall/TIGER/runs/$run_name" ]]; then
    echo "Run $run_name is at the legacy path Recall/TIGER/runs/; move it to $run_dir first" >&2
    exit 2
  fi
  if [[ -e "$run_dir" ]]; then
    test -f "$run_dir/latest.pt" || {
      echo "Incomplete run without checkpoint: $run_name" >&2; exit 2;
    }
    args+=(--resume)
  fi
  srun .venv/bin/python -m Recall.TIGER.baseline.train "${args[@]}"
fi

if [[ "$stage" == test ]]; then
  srun .venv/bin/python -m Recall.TIGER.baseline.train \
    --sid-run "$sid_dir" --sid-method sentence-t5_rqvae \
    --run-name "$run_name" --device cuda --eval-test-only
fi
