#!/bin/bash

cd src

# Deterministic DFS baseline (mult_heuristic).
uv run python countdown_generate.py --seed 42 --data_dir data/dfs/ --min_range 4 --start_range 4 --num_samples 500000 --search dfs

# Uncertainty variants. Each flag relaxes one source of determinism; combine freely.
# --randomize_op_order   : uniformly shuffle the order successors are explored
# --randomize_heuristic  : sample successor order weighted by the heuristic (prune still deterministic); takes precedence over --randomize_op_order
# --randomize_backtrack  : sample a random open state to revisit on backtrack
# --temperature T        : sharpness of the heuristic-weighted distribution (lower = sharper)
# The per-step candidate states and probability distributions are saved under "search_steps" in the output JSON.
#
# uv run python countdown_generate.py --seed 42 --data_dir data/dfs_stochastic/ --min_range 4 --start_range 4 --num_samples 500000 --search dfs \
#     --randomize_op_order --randomize_heuristic --randomize_backtrack --temperature 1.0
