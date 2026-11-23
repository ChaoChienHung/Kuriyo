#!/bin/bash
#SBATCH --job-name=kuriyo-external
#SBATCH --partition=gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out
#
# Run a community TIGER implementation unmodified, pinned to one commit, so its numbers
# can be set beside Kuriyo's (TIGER Experiment-log). Nothing here imports Kuriyo code and
# Kuriyo imports nothing from it. The clone, its virtualenv, data and outputs live under
# the Git-ignored third_party/<EXTERNAL>/; only the Slurm log is evidence.
#
# EXTERNAL=botta  EdoardoBotta/RQ-VAE-Recommender (MIT). Trains the retrieval decoder with
#                 configs/decoder_amazon.gin on P5 Beauty, using the RQ-VAE checkpoint the
#                 repository ships (trained_models/rqvae_amazon_beauty/checkpoint_399999.pt).
#                 Its item text is encoded with sentence-t5-xxl, so request a GPU with at
#                 least 24 GB (e.g. sbatch --gpus=a100-40:1). The only change is a copy of the
#                 config with wandb_logging=False. Note: it evaluates on the TEST split every
#                 full_eval_every steps (data/processed.py), not on validation.
#
# Submit from the repository root:
#   EXTERNAL=botta sbatch --export=ALL --gpus=a100-40:1 scripts/external_tiger.sbatch
set -euo pipefail

external="${EXTERNAL:-}"
case "$external" in
  botta)
    repo=https://github.com/EdoardoBotta/RQ-VAE-Recommender.git
    commit=957d32bda43958ba641ae919abb1092ac6738798 ;;
  *) echo "EXTERNAL must be botta" >&2; exit 2 ;;
esac

dir="third_party/$external"
if [[ ! -d "$dir/.git" ]]; then
  [[ ! -e "$dir" ]] || { echo "$dir exists but is not a clone; inspect it" >&2; exit 2; }
  git clone --quiet "$repo" "$dir"
fi
git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
[[ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ]] || {
  echo "$dir has local changes to tracked files; refusing to run modified code" >&2; exit 2;
}
git -C "$dir" checkout --quiet "$commit"

export TMPDIR="${TMPDIR:-$PWD/.tmp}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$HF_HOME"

if [[ ! -x "$dir/.venv/bin/python" ]]; then
  python3 -m venv "$dir/.venv"
fi
# Same torch pin as run.sbatch (cu126 keeps sm_70 GPUs working); the project's own
# dependencies are installed around it.
"$dir/.venv/bin/python" -m pip install --quiet \
  --extra-index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1+cu126"
"$dir/.venv/bin/python" -m pip install --quiet -e "$dir"

echo "job=${SLURM_JOB_ID:-local} kuriyo_commit=$(git rev-parse HEAD) external=$external repo=$repo commit=$(git -C "$dir" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$external" in
  botta)
    config=configs/kuriyo_decoder_amazon.gin
    sed 's/^train.wandb_logging=True$/train.wandb_logging=False/' "$dir/configs/decoder_amazon.gin" > "$dir/$config"
    grep -q '^train.wandb_logging=False$' "$dir/$config" || { echo "could not disable wandb in $config" >&2; exit 2; }
    echo "config (only wandb_logging changed):"
    diff "$dir/configs/decoder_amazon.gin" "$dir/$config" || true
    cd "$dir"
    srun .venv/bin/python train_decoder.py "$config"
    ;;
esac
