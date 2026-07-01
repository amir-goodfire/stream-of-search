#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --job-name=frost-train
#SBATCH --output=logs/slurm-%A_%a.log
#SBATCH --array=1-3%3

# Fine-tune Qwen2.5-7B (LoRA) on the DFS search traces, one model per dataset,
# as 3 concurrent array tasks (single H200 each):
#   1) randomize_backtrack only  (data/dfs_backtrack)
#   2) randomize_heuristic only  (data/dfs_heuristic)
#   3) both                      (data/dfs_both)
# Submit from the repository root so the relative log path resolves:
#     sbatch scripts/train_slurm.sh
# The logs/ directory must already exist (committed via logs/.gitkeep).
# Generate the datasets first with: sbatch scripts/gen_data_slurm.sh

date; hostname

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

# Run from src/ so train.py's relative paths (../configs, data/, ckpts/) resolve.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}/src"

# Weights & Biases is enabled by default (run `wandb login` beforehand). Set
# WANDB_DISABLED=true in the environment to turn it off for a given run.
WANDB_FLAG="--wandb"

# Each task trains on the dataset its matching gen_data_slurm.sh task produced.
# TAG must match the randomization tag countdown_generate.py derives from the
# flags (and the data dir gen_data_slurm.sh writes to).
case $SLURM_ARRAY_TASK_ID in
  1) TAG="backtrack" ;;
  2) TAG="heuristic" ;;
  3) TAG="backtrack_heuristic" ;;
  *)
    echo "Unexpected task id: $SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac

DATA_DIR="data/dfs_${TAG}"
NAME="sft-qwen2.5-7b-lora-dfs-${TAG}"
OUTPUT_DIR="ckpts/${NAME}"
# Data filenames carry the same tag, matching countdown_generate.py's output.
BASE="b4_t100_n500000_dfs_${TAG}"
TRAIN_FILE="train1_${BASE}.json"
VAL_FILE="val1_${BASE}.json"
VAL_TARGET_FILE="val_target1_${BASE}.json"

# A single H200 (141 GB) is ample for Qwen2.5-7B + LoRA, so plain `python`
# (HF Trainer handles the single GPU) is enough. The shared config is overridden
# per task via --data_dir/--output_dir/--name/--*_file. To use multiple GPUs on
# the node instead, bump `--gres=gpu:N` above and launch with:
#   accelerate launch --config_file ../configs/accelerate_lora.yaml \
#       --num_processes N train.py ...
# Train for a fixed 4000 optimizer steps (== W&B steps), saving a checkpoint
# every 1000 steps (-> checkpoints at 1000/2000/3000/4000). --max_steps overrides
# the config's num_train_epochs. save_steps stays a multiple of the config's
# eval_steps (500) so load_best_model_at_end works.
MAX_STEPS=4000
SAVE_STEPS=1000
uv run python train.py --config ../configs/sft-qwen-lora-cd.conf \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --name "$NAME" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --val_target_file "$VAL_TARGET_FILE" \
    --max_steps "$MAX_STEPS" \
    --save_steps "$SAVE_STEPS" \
    $WANDB_FLAG

date; hostname
