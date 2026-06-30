"""
Candidate-set verification for a fine-tuned model.

Two checks, both by exact string match against the candidate sets we recorded
while generating the search traces (see ``countdown_dfs.dfs`` -> ``steps``):

1. Teacher-forced (exact): for each recorded ``expand`` step, feed the model the
   ground-truth trace prefix up to that node's "Current State" line, greedily
   generate the next line, and check whether it is in that step's recorded
   candidate set. Measures, per decision, whether the model stays in-set.

2. Free-generation (recomputed): let the model generate a whole trace from the
   initial state, then check each "Exploring Operation" line against the
   candidate set recomputed for the state it was taken from. Measures true
   autoregressive behaviour.

Both report an *in-set* rate (the key metric: does the model only ever propose
states the search algorithm allows?). The teacher-forced check also reports an
*exact-gold* rate (did it reproduce the specific step the trace took).
"""
import os
import json
import argparse

import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from countdown_utils import metric_fn
from candidate_utils import (
    build_state_to_candidates,
    iter_decision_prefixes,
    evaluate_generated_trace,
)


def load_model(base_model, adapter=None, ckpt=None):
    name = ckpt or base_model
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(adapter or name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval().cuda()
    return model, tokenizer


def _first_line(text):
    nl = text.find("\n")
    return text if nl == -1 else text[:nl]


@torch.no_grad()
def generate_next_lines(model, tokenizer, prefixes, batch_size, max_new_tokens=48):
    """Greedily continue each prefix and return only the first generated line."""
    outs = []
    tokenizer.padding_side = "left"
    for b in tqdm.trange(0, len(prefixes), batch_size, desc="teacher-forced"):
        batch = prefixes[b:b + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        # decode only the newly generated tokens
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
            outs.append(_first_line(text.lstrip("\n")))
    return outs


@torch.no_grad()
def generate_traces(model, tokenizer, prompts, batch_size, context_len):
    outs = []
    tokenizer.padding_side = "left"
    for b in tqdm.trange(0, len(prompts), batch_size, desc="free-generation"):
        batch = prompts[b:b + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(
            **enc, max_length=context_len, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs


def teacher_forced(model, tokenizer, data, max_decisions, batch_size):
    """Exact in-set check at each recorded decision point."""
    prefixes, states, golds, candsets = [], [], [], []
    bos = tokenizer.bos_token or ""
    for sample in data:
        steps = sample.get("search_steps")
        if not steps:
            continue
        mapping = build_state_to_candidates(steps)
        for prefix, state, gold in iter_decision_prefixes(sample["search_path"]):
            if state not in mapping:
                continue
            prefixes.append(bos + prefix)
            states.append(state)
            golds.append(gold)
            candsets.append(mapping[state])
            if len(prefixes) >= max_decisions:
                break
        if len(prefixes) >= max_decisions:
            break

    if not prefixes:
        return {"decisions": 0}
    preds = generate_next_lines(model, tokenizer, prefixes, batch_size)
    in_set = [p in c for p, c in zip(preds, candsets)]
    exact = [p == g for p, g in zip(preds, golds)]
    return {
        "decisions": len(preds),
        "in_set_acc": float(np.mean(in_set)),
        "exact_gold_acc": float(np.mean(exact)),
        "examples": [
            {"state": s, "pred": p, "gold": g, "in_set": isin}
            for s, p, g, isin in list(zip(states, preds, golds, in_set))[:20]
        ],
    }


def free_generation(model, tokenizer, data, num, batch_size, context_len,
                    prune_repeated_states):
    """Generate full traces, check every explored step against recomputed sets."""
    bos = tokenizer.bos_token or ""
    samples = [s for s in data if len(s["nums"]) == 4][:num]
    prompts = [bos + f"Current State: {s['target']}:{s['nums']}, Operations: []"
               for s in samples]
    traces = generate_traces(model, tokenizer, prompts, batch_size, context_len)

    total = in_set = parse_fail = 0
    solved = []
    per_trace = []
    for sample, full in zip(samples, traces):
        trace = full.split(bos)[-1] if bos else full
        stats = evaluate_generated_trace(
            trace, sample["target"], threshold=sample["target"],
            prune_repeated_states=prune_repeated_states)
        total += stats["total"]
        in_set += stats["in_set"]
        parse_fail += stats["parse_fail"]
        solved.append(1.0 if "Goal Reached" in trace else 0.0)
        per_trace.append(stats["in_set_frac"])
    return {
        "traces": len(traces),
        "decisions": total,
        "in_set_frac": (in_set / total) if total else 0.0,
        "mean_in_set_frac_per_trace": float(np.mean(per_trace)) if per_trace else 0.0,
        "parse_fail": parse_fail,
        "solve_rate": float(np.mean(solved)) if solved else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--adapter", type=str, default=None,
                        help="path to a LoRA adapter to merge onto the base model")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="path to a full (merged) model to load instead of base")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("-n", "--num", type=int, default=200,
                        help="number of traces for free-generation")
    parser.add_argument("--max_decisions", type=int, default=2000,
                        help="number of decision points for teacher-forced check")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--prune_repeated_states", action="store_true",
                        help="set if the data was generated with state dedup (for recompute)")
    parser.add_argument("--mode", choices=["teacher_forced", "free", "both"],
                        default="both")
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.base_model, args.adapter, args.ckpt)

    with open(os.path.join(args.data_dir, args.data), "r") as f:
        data = json.load(f)

    results = {}
    if args.mode in ("teacher_forced", "both"):
        results["teacher_forced"] = teacher_forced(
            model, tokenizer, data, args.max_decisions, args.batch_size)
        print("\n=== Teacher-forced (exact recorded candidate sets) ===")
        tf = results["teacher_forced"]
        print(f"  decisions:       {tf.get('decisions')}")
        print(f"  in-set accuracy: {tf.get('in_set_acc'):.4f}")
        print(f"  exact-gold acc:  {tf.get('exact_gold_acc'):.4f}")

    if args.mode in ("free", "both"):
        results["free_generation"] = free_generation(
            model, tokenizer, data, args.num, args.batch_size, args.ctx,
            args.prune_repeated_states)
        print("\n=== Free-generation (recomputed candidate sets) ===")
        fg = results["free_generation"]
        print(f"  traces:                 {fg['traces']}")
        print(f"  explored decisions:     {fg['decisions']}")
        print(f"  in-set fraction:        {fg['in_set_frac']:.4f}")
        print(f"  mean in-set per trace:  {fg['mean_in_set_frac_per_trace']:.4f}")
        print(f"  solve rate:             {fg['solve_rate']:.4f}")
        print(f"  parse failures:         {fg['parse_fail']}")

    out_dir = args.adapter or args.ckpt or "."
    out_dir = out_dir if os.path.isdir(out_dir) else os.path.dirname(out_dir) or "."
    out_file = os.path.join(out_dir, f"candidate_eval_{args.data.replace('/', '_')}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_file}")


if __name__ == "__main__":
    main()
