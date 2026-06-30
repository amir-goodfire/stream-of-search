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

case $SLURM_ARRAY_TASK_ID in
  1)
    DATA_DIR="data/dfs_backtrack"
    NAME="sft-qwen2.5-7b-lora-dfs-backtrack"
    ;;
  2)
    DATA_DIR="data/dfs_heuristic"
    NAME="sft-qwen2.5-7b-lora-dfs-heuristic"
    ;;
  3)
    DATA_DIR="data/dfs_both"
    NAME="sft-qwen2.5-7b-lora-dfs-both"
    ;;
  *)
    echo "Unexpected task id: $SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="ckpts/${NAME}"

# A single H200 (141 GB) is ample for Qwen2.5-7B + LoRA, so plain `python`
# (HF Trainer handles the single GPU) is enough. The shared config is overridden
# per task via --data_dir/--output_dir/--name. To use multiple GPUs on the node
# instead, bump `--gres=gpu:N` above and launch with:
#   accelerate launch --config_file ../configs/accelerate_lora.yaml \
#       --num_processes N train.py ...
uv run python train.py --config ../configs/sft-qwen-lora-cd.conf \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --name "$NAME" \
    $WANDB_FLAG

date; hostname
