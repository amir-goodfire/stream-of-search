"""
Script for  generating data for the countdown task.
"""
import json
import argparse
import random
import tiktoken
import os

import tqdm

from countdown import CountDown
from countdown_utils import *
from countdown_bfs import bfs
from countdown_dfs import dfs


parser = argparse.ArgumentParser()

# data args
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--data_dir", type=str, default=None, help="Directory to store data")

# countdown specific
parser.add_argument("--start_range", type=int, default=3, help="Max Range of starting numbers [M, N]")
parser.add_argument("--min_range", type=int, default=3, help="Min Range of starting numbers [M, N]")
parser.add_argument("--max_target", type=int, default=100, help="Maximum target number")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of data samples to generate")

# search args
parser.add_argument("--search", type=str, default="random", help="Search type")

# dfs uncertainty args (focus: dfs with mult_heuristic)
parser.add_argument("--randomize_op_order", action="store_true", help="DFS: uniformly shuffle the order successors are explored")
parser.add_argument("--randomize_heuristic", action="store_true", help="DFS: sample successor order weighted by the heuristic (prune still deterministic). Takes precedence over --randomize_op_order")
parser.add_argument("--randomize_backtrack", action="store_true", help="DFS: sample a random open state to revisit on backtrack instead of the last open state")
parser.add_argument("--prune_repeated_states", action="store_true", help="DFS: prune any successor whose exact state (multiset of numbers) was already generated earlier in the search")
parser.add_argument("--temperature", type=float, default=1.0, help="DFS: temperature for the heuristic-weighted distribution (lower = sharper toward the best state)")
parser.add_argument("--max_nodes", type=int, default=None, help="DFS: cap on the number of explored successors; the search is cut short (rating 0) if it reaches this without finding the target. Bounds memory/trace size for stochastic policies. None = no limit")
parser.add_argument("--no_search_steps", action="store_true", help="Drop the per-step 'search_steps' records (eval-only, unused for training) from the large train/grow split to cut file size and memory. Keep them on the small val splits.")

# split for growth mode on or off 
parser.add_argument("--grow", action="store_true", help="grow mode on or off, only a new train set is created")
parser.add_argument("--offset", type=int, default=1, help="offset for random seed")


if __name__ == "__main__":
    args = parser.parse_args()
    # set random seed
    random.seed(args.seed)
    target_nums = [i for i in range(10, args.max_target+1)]

    # save 10% of target numbers for validation
    random.shuffle(target_nums)
    val_target_nums = target_nums[:len(target_nums)//10]
    print(val_target_nums)
    train_nums = target_nums[len(target_nums)//10:]

    if args.grow:
        splits = ["grow"]
        target_list = [train_nums]
        # to avoid reusing the same samples from the train set, change the seed
        random.seed(args.seed + args.offset)
    else:
        splits = ["train", "val_target", "val"]
        target_list = [train_nums, val_target_nums, train_nums]

    average_token_length = {3: [], 4: [], 5: []}
    average_reward = {3: [], 4: [], 5: []}
    average_zeros = {3: [], 4: [], 5: []}
    total_samples = {3: 0, 4: 0, 5: 0}

    os.makedirs(args.data_dir, exist_ok=True)
    train_solutions = set()

    # Encode which randomization flags are active in the filename so different
    # setups never overwrite each other. Deterministic runs keep the original
    # (suffix-less) name for backward compatibility. The order is fixed so the
    # tag is stable regardless of CLI flag order, and matches the dir/tag used
    # by scripts/{gen_data,train}_slurm.sh.
    rand_flags = [
        ("backtrack", args.randomize_backtrack),
        ("heuristic", args.randomize_heuristic),
        ("oporder", args.randomize_op_order),
        ("prune", args.prune_repeated_states),
    ]
    rand_tag = "_".join(name for name, on in rand_flags if on)
    rand_suffix = f"_{rand_tag}" if rand_tag else ""

    for split, target_nums in zip(splits, target_list):

        if split == "train" or split=="grow":
            num_samples = args.num_samples
        else:
            num_samples = 1000

        out_path = (
            f"{args.data_dir}/{split}{args.offset}_b{args.start_range}"
            f"_t{args.max_target}_n{args.num_samples}_{args.search}{rand_suffix}.json"
        )
        # Stream each sample straight to disk instead of accumulating the whole
        # split in RAM: the 500k-sample train split (each carrying a large
        # ``search_steps``) is tens of GB as live objects, which is what OOMs.
        out_f = open(out_path, "w")
        out_f.write("[")
        wrote_any = False
        # ``search_steps`` is only consumed by candidate eval, never by training,
        # and is ~80% of each sample. Drop it for the huge train/grow split when
        # requested to keep both memory and file size down.
        drop_steps = args.no_search_steps and split in ("train", "grow")

        zero_count = 0
        success_count = 0
        for t in tqdm.tqdm(range(num_samples)):
            start_size = random.randint(args.min_range, args.start_range)
            cd = CountDown(args.max_target, start_size)
            max_nodes = None
            if start_size == 2:
                # naive calculation of max nodes: 2c2 x 4 = 4
                max_rating = 4
            elif start_size == 3:
                # naive calculation of max nodes: 3c2 x 4 x 4 = 48
                max_rating = 3*4*4
            elif start_size == 4:
                # naive calculation of max nodes: 4c2 x 4 x 3c2 x 4 x 4 = 1152
                max_rating = 1152
            elif start_size == 5:
                # naive calculation of max nodes: 5c2 x 4 x 4c2 x 4 x 3c2 x 4 x 4 = 46080
                max_rating = 46080
            target = random.choice(target_nums)
            nums, solution = cd.generate(target)
            no_backtrack_trace = cd.convert_to_path(target, nums, solution)
            if split == "val":
                while repr(solution) in train_solutions:
                    target = random.choice(target_nums)
                    nums, solution = cd.generate(target)
            search_steps = None
            if args.search == "astar":
                # astar not adapted to new format
                raise NotImplementedError
            elif args.search == "dfs":
                # focus: dfs with the mult_heuristic
                heuristic = mult_heuristic
                search = dfs
                search_path, search_steps = dfs(
                    target, nums, heuristic=heuristic, threshold=target,
                    randomize_op_order=args.randomize_op_order,
                    randomize_heuristic=args.randomize_heuristic,
                    randomize_backtrack=args.randomize_backtrack,
                    prune_repeated_states=args.prune_repeated_states,
                    temperature=args.temperature,
                    max_nodes=args.max_nodes)
            elif args.search == "bfs":
                heuristic = mult_heuristic
                search = bfs
                search_path = bfs(target, nums, 5, heuristic=heuristic)
            elif args.search == "random":
                heuristic = random.choice([sum_heuristic, mult_heuristic])
                search = random.choice([dfs, bfs])
                if search == dfs:
                    search_path, search_steps = dfs(
                        target, nums, heuristic=heuristic, threshold=target,
                        randomize_op_order=args.randomize_op_order,
                        randomize_heuristic=args.randomize_heuristic,
                        randomize_backtrack=args.randomize_backtrack,
                        prune_repeated_states=args.prune_repeated_states,
                        temperature=args.temperature,
                        max_nodes=args.max_nodes)
                elif search == bfs:
                    beam_size = random.choice([1, 2, 3, 4, 5])
                    search_path = bfs(target, nums, beam_size, heuristic=heuristic)

            else:
                raise ValueError(f"Search type {args.search} not supported")
            if "Goal Reached" in search_path:
                success_count += 1
                rating = 1. - simple_rating(search_path) / max_rating
                rating = max(0., rating)
            else:
                rating = 0.
            if rating == 0.:
                zero_count += 1

            search_type = search.__name__
            if search_type == "bfs":
                search_type += f"_{beam_size}"

            sample = {
                "nums": nums,
                "target": target,
                "solution": solution,
                "search_path": search_path,
                "search_steps": None if drop_steps else search_steps,
                "rating": rating,
                "search_type": search_type,
                "optimal_path": no_backtrack_trace,
                "heuristic": heuristic.__name__
            }
            # One compact JSON object per line, together forming a valid JSON
            # array, so downstream json.load() still works while peak memory
            # stays at a single sample.
            out_f.write(("," if wrote_any else "") + "\n")
            json.dump(sample, out_f)
            wrote_any = True
            if split in ("train", "grow"):
                train_solutions.add(repr(solution))
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(search_path)
            if rating == 0:
                average_zeros[start_size].append(1)
            average_token_length[start_size].append(len(tokens))
            average_reward[start_size].append(rating)
            total_samples[start_size] += 1

        print(f"Zero count: {zero_count}")
        print(f"Goal reached (target found): {success_count}/{num_samples} ({100. * success_count / num_samples:.1f}%)")
        print(f"Total samples: {total_samples}")
        print(f"average zeros: start size 3: {sum(average_zeros[3])}, start size 4: {sum(average_zeros[4])}, start size 5: {sum(average_zeros[5])}")
        print(f"average token length: start size 3: {(sum(average_token_length[3]) / total_samples[3]) if total_samples[3] else None if total_samples[3] else None}, start size 4: {(sum(average_token_length[4]) / total_samples[4]) if total_samples[4] else None}, start size 5: {(sum(average_token_length[5]) / total_samples[5]) if total_samples[5] else None}")
        print(f"average reward: start size 3: {(sum(average_reward[3]) / total_samples[3]) if total_samples[3] else None}, start size 4: {(sum(average_reward[4]) / total_samples[4]) if total_samples[4] else None}, start size 5: {(sum(average_reward[5]) / total_samples[5]) if total_samples[5] else None}")

        out_f.write("\n]" if wrote_any else "]")
        out_f.close()
