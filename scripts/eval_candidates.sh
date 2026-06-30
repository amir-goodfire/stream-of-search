#!/bin/bash
# Verify that the fine-tuned model generates steps within the candidate set.
# Pass the LoRA adapter dir (the training output_dir) and a dfs data file that
# contains "search_steps". Add --prune_repeated_states if the data was
# generated with state dedup so the free-generation recompute matches.

cd src

python eval_candidates.py \
    --base_model Qwen/Qwen2.5-7B \
    --adapter ../ckpts/qwen2.5-7b-lora-dfs \
    --data_dir data/dfs \
    -d val1_b4_t100_n500000_dfs.json \
    --num 200 --max_decisions 2000 --batch_size 32 --ctx 4096 --mode both
