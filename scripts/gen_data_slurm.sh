#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --job-name=frost-gen
#SBATCH --output=logs/gen-%A_%a.log
#SBATCH --array=1-3%3

# Generate three DFS datasets (CPU-only, no GPUs) as 3 concurrent array tasks:
#   1) randomize_backtrack only
#   2) randomize_heuristic only
#   3) both
# Submit from the repository root:  sbatch scripts/gen_data_slurm.sh
# The logs/ directory must already exist (committed via logs/.gitkeep).

date; hostname

set -euo pipefail

# Run from src/ so the data_dir paths land under src/data/.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}/src"

NUM_SAMPLES=500000
SEED=42

case $SLURM_ARRAY_TASK_ID in
  1)
    FLAGS="--randomize_backtrack"
    DATA_DIR="data/dfs_backtrack"
    ;;
  2)
    FLAGS="--randomize_heuristic"
    DATA_DIR="data/dfs_heuristic"
    ;;
  3)
    FLAGS="--randomize_backtrack --randomize_heuristic"
    DATA_DIR="data/dfs_both"
    ;;
  *)
    echo "Unexpected task id: $SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac

uv run python countdown_generate.py \
    --seed $SEED \
    --data_dir "$DATA_DIR" \
    --min_range 4 --start_range 4 \
    --max_target 100 \
    --num_samples $NUM_SAMPLES \
    --search dfs $FLAGS

date; hostname
