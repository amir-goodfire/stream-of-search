#!/bin/bash
# Fine-tune Qwen2.5-7B (LoRA) on the DFS search traces.
# Set --num_processes to the number of GPUs available.

cd src

accelerate launch --config_file ../configs/accelerate_lora.yaml --num_processes 1 \
    train.py --config ../configs/sft-qwen-lora-cd.conf
