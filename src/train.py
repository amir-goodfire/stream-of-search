import argparse
import json
import os
import random

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

import wandb


def _load_split(path, train_type):
    """Load a generated data file, keeping only the text fields needed for
    training. Loading via json (rather than datasets' arrow inference) keeps the
    heterogeneous ``search_steps`` column from breaking schema inference."""
    field = {"sft": "search_path", "oft": "optimal_path", "dt": "search_path"}[train_type]
    with open(path, "r") as f:
        raw = json.load(f)
    return Dataset.from_list(
        [{"text": r[field], "rating": r.get("rating", 0.0)} for r in raw]
    )


def main(args):
    with open(args.config, "r") as f:
        config = json.load(f)

    # CLI overrides (handy for array jobs that share one config file).
    for key in ("data_dir", "output_dir", "name"):
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # wandb is only initialised on the main process; under accelerate/torchrun
    # LOCAL_RANK is set for non-main ranks.
    is_main = os.environ.get("LOCAL_RANK", "0") == "0"
    if args.wandb and is_main:
        wandb_kwargs = config.get("wandb", {"project": "", "entity": "", "dir": ""})
        wandb.init(
            project=wandb_kwargs["project"],
            entity=wandb_kwargs["entity"],
            name=config["name"],
            config=config,
            dir=wandb_kwargs["dir"],
        )

    model_name = args.ckpt or config["model_name_or_path"]
    attn_impl = config.get("attn_implementation", "flash_attention_2")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_impl
        )
    except (ImportError, ValueError):
        # flash-attn not installed; fall back to PyTorch SDPA
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA fine-tuning (base weights frozen). Loading from a saved adapter ckpt
    # is handled by AutoModel above only for full models; for LoRA resume use
    # --ckpt pointing at a merged model or pass the adapter separately.
    use_lora = config.get("use_lora", True)
    if use_lora and not args.reset:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
            ),
        )
        model = get_peft_model(model, lora_config)
        # required so gradient checkpointing works with frozen base + LoRA
        model.enable_input_require_grads()
        if is_main:
            model.print_trainable_parameters()

    # data
    train_file = os.path.join(config["data_dir"], config["train_file"])
    val_file = os.path.join(config["data_dir"], config["val_file"])
    val_target_file = os.path.join(config["data_dir"], config["val_target_file"])
    train_type = config["train_type"]
    hf_datasets = DatasetDict(
        {
            "train": _load_split(train_file, train_type),
            "val": _load_split(val_file, train_type),
            "val_target": _load_split(val_target_file, train_type),
        }
    )
    num_train = int(config["num_train"])
    hf_datasets["train"] = hf_datasets["train"].select(
        range(min(num_train, len(hf_datasets["train"])))
    )

    context_length = config["context_length"]
    tokenizer.model_max_length = context_length
    mask_prompt = config.get("mask_prompt", True)
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token

    def tokenize(batch):
        input_ids_list, labels_list = [], []
        for text, rating in zip(batch["text"], batch["rating"]):
            text = text.strip()
            if train_type == "dt":
                # decision-transformer style: condition on the target rating
                prefix = f"{rating:0.2f}->"
            else:
                prefix = ""
            full = bos + prefix + text + eos
            full_ids = tokenizer(
                full, truncation=True, max_length=context_length,
                add_special_tokens=False,
            )["input_ids"]
            labels = list(full_ids)
            if mask_prompt:
                # mask the conditioning prefix + the first "Current State" line
                # (the given initial state); train only on the generated search.
                newline = text.find("\n")
                first_line = text[: newline + 1] if newline != -1 else text
                prompt_text = bos + prefix + first_line
                plen = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
                plen = min(plen, len(labels))
                labels[:plen] = [-100] * plen
            input_ids_list.append(full_ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized_datasets = hf_datasets.map(
        tokenize, batched=True, remove_columns=hf_datasets["train"].column_names
    )
    print("tokenized dataset", tokenized_datasets)

    # dynamically pads input_ids (pad token) and labels (-100), builds masks
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, padding="longest"
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config.get("eval_batch_size", config["batch_size"]),
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_steps=config["log_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        learning_rate=config["lr"],
        save_strategy="steps",
        save_total_limit=config["save_total_limit"],
        save_steps=config["save_steps"],
        seed=config["seed"],
        bf16=True,
        push_to_hub=False,
        report_to="wandb" if args.wandb else "none",
        run_name=config["name"],
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        torch_compile=config.get("torch_compile", False),
        metric_for_best_model="eval_valid_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset={
            "valid": tokenized_datasets["val"],
            "valid_target": tokenized_datasets["val_target"],
        },
    )

    if args.resume:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()

    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/sft-qwen-lora-cd.conf")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="model id/path to load instead of config's model_name_or_path")
    parser.add_argument("--reset", action="store_true",
                        help="skip LoRA wrapping (e.g. to resume an already-LoRA/merged ckpt)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="override config['data_dir']")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="override config['output_dir']")
    parser.add_argument("--name", type=str, default=None,
                        help="override config['name'] (run/output name)")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args)
