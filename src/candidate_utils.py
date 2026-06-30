"""
Candidate-set verification utilities for Stream-of-Search traces.

These functions are deliberately torch-free so they can be unit-tested on CPU
and reused by the model-driven verifier (``eval_candidates.py``).

The central idea: every randomized DFS procedure records, for each expanded
node, the exact set of successor states that *could* have been explored (see
``countdown_dfs.dfs`` -> ``steps`` -> ``{"type": "expand", ...}``). After
fine-tuning, we check whether the model's generated next step is a member of
that candidate set, by **exact string match** on the trace line
``Exploring Operation: {op}, Resulting Numbers: {nums}``.

Two views are supported:

* Teacher-forced / exact: use the candidate sets recorded in ``search_steps``.
  ``build_state_to_candidates`` maps each node's "Current State" line to its set
  of valid "Exploring Operation" lines.
* Free-generation / recomputed: ``recompute_candidate_lines`` rebuilds the valid
  set for an arbitrary state directly from ``combine_nums`` + the same pruning
  rules, so a trace the model produced on its own can be checked even at states
  that are absent from any recorded trace.
"""
import re
from itertools import combinations

from countdown_utils import combine_nums, mult_heuristic


# ---------------------------------------------------------------------------
# Canonical trace-line formatting (must byte-match countdown_dfs output)
# ---------------------------------------------------------------------------

def state_key(nums):
    """Canonical key for a state: the multiset of available numbers."""
    return tuple(sorted(nums))


def exploring_line(operation, nums):
    """The exact trace line emitted when a successor is explored."""
    return f"Exploring Operation: {operation}, Resulting Numbers: {nums}"


def current_state_line(target, nums, operations):
    """The exact trace line emitted when a node becomes the current state.

    The operations list is the full path to the node, so this line uniquely
    identifies a node within a single trace."""
    return f"Current State: {target}:{nums}, Operations: {operations}"


# ---------------------------------------------------------------------------
# Candidate sets from recorded steps (exact) and from scratch (recomputed)
# ---------------------------------------------------------------------------

def step_candidate_lines(step):
    """Set of valid "Exploring Operation" lines for one ``expand`` step."""
    return {exploring_line(c["operation"], c["nums"]) for c in step["candidates"]}


def build_state_to_candidates(search_steps):
    """Map each node's "Current State" line to its recorded candidate-line set.

    Keyed by the exact current-state line, which is unique per node because the
    operations list encodes the full path to that node."""
    mapping = {}
    for step in search_steps:
        if step.get("type") != "expand":
            continue
        line = current_state_line(step["target"], step["nums"], step["operations"])
        mapping[line] = step_candidate_lines(step)
    return mapping


def recompute_candidate_lines(target, nums, threshold=None,
                              prune_repeated_states=False, visited=None,
                              heuristic=mult_heuristic):
    """Rebuild the valid candidate-line set for ``nums`` from scratch, mirroring
    ``countdown_dfs.dfs``'s generation + pruning exactly:

    1. enumerate successors in the fixed ``combinations`` x ``combine_nums``
       order,
    2. drop any whose heuristic exceeds ``threshold``,
    3. if ``prune_repeated_states``, drop any whose state was already generated
       (``visited``) or already generated earlier in this same expansion.

    ``visited`` is a set of canonical state keys treated as already-seen. This
    is the source of truth for the *free-generation* check, where a recorded
    candidate set may not exist for a state the model wandered into."""
    local_visited = set(visited) if visited is not None else set()
    lines = set()
    for i, j in combinations(range(len(nums)), 2):
        for result, operation in combine_nums(nums[i], nums[j]):
            new_nums = [nums[k] for k in range(len(nums)) if k != i and k != j] + [result]
            if threshold is not None and heuristic(new_nums, target) > threshold:
                continue
            if prune_repeated_states:
                key = state_key(new_nums)
                if key in local_visited:
                    continue
                local_visited.add(key)
            lines.add(exploring_line(operation, new_nums))
    return lines


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

_STATE_RE = re.compile(r"Current State: (\d+):\[(.*?)\], Operations:")
_RESULT_RE = re.compile(r"Resulting Numbers: \[(.*?)\]")


def parse_nums(bracket_contents):
    """Parse "8, 2, 3" -> [8, 2, 3]; "" -> []."""
    bracket_contents = bracket_contents.strip()
    if not bracket_contents:
        return []
    return [int(x) for x in bracket_contents.split(",")]


def parse_state_line(line):
    """Return (target, nums) from a "Current State:" line, or None."""
    m = _STATE_RE.search(line)
    if not m:
        return None
    return int(m.group(1)), parse_nums(m.group(2))


def parse_result_nums(explore_line):
    """Return the resulting numbers list from an "Exploring Operation:" line."""
    m = _RESULT_RE.search(explore_line)
    if not m:
        return None
    return parse_nums(m.group(1))


def iter_decision_prefixes(trace):
    """Yield ``(prefix, current_state_line, gold_explore_line)`` for every
    decision in ``trace``. ``prefix`` is all text strictly before the exploring
    line (so it can be fed to a model as the context), ``current_state_line`` is
    the most recent "Current State" line, and ``gold_explore_line`` is the
    ground-truth next line. Used by the teacher-forced verifier."""
    current = None
    prefix = ""
    for raw in trace.splitlines(keepends=True):
        stripped = raw.rstrip("\n")
        if stripped.startswith("Exploring Operation:"):
            yield prefix, current, stripped
            prefix += raw
        else:
            if stripped.startswith("Current State:"):
                current = stripped
            prefix += raw


def parse_decisions(trace):
    """List of ``(current_state_line, explore_line)`` pairs in a trace."""
    decisions = []
    current = None
    for raw in trace.splitlines():
        stripped = raw.rstrip("\n")
        if stripped.startswith("Current State:"):
            current = stripped
        elif stripped.startswith("Exploring Operation:"):
            decisions.append((current, stripped))
    return decisions


# ---------------------------------------------------------------------------
# Whole-trace evaluation (free generation)
# ---------------------------------------------------------------------------

def evaluate_generated_trace(trace, target, threshold=None,
                             prune_repeated_states=False, heuristic=mult_heuristic):
    """Score a model-generated ``trace`` by how many of its explored steps fall
    inside the recomputed candidate set of the state they were taken from.

    Returns a dict with ``total`` decisions, ``in_set`` count, ``parse_fail``
    count (lines whose current-state could not be parsed), and ``in_set_frac``.

    Note: the ``visited`` set used for repeated-state pruning is approximated
    from the states the model itself emitted (data generation also counts
    states generated-but-never-explored, which a trace does not reveal), so for
    ``prune_repeated_states`` this is a slightly *lenient* check across
    branches; the within-expansion dedup it applies is exact."""
    seen = set()
    total = 0
    in_set = 0
    parse_fail = 0
    current_state = None  # (target, nums)
    for raw in trace.splitlines():
        stripped = raw.rstrip("\n")
        if stripped.startswith("Current State:"):
            parsed = parse_state_line(stripped)
            if parsed is not None:
                current_state = parsed
                if prune_repeated_states:
                    seen.add(state_key(parsed[1]))
        elif stripped.startswith("Exploring Operation:"):
            total += 1
            if current_state is None:
                parse_fail += 1
                continue
            _, nums = current_state
            valid = recompute_candidate_lines(
                target, nums, threshold=threshold,
                prune_repeated_states=prune_repeated_states, visited=seen,
                heuristic=heuristic)
            if stripped in valid:
                in_set += 1
            result_nums = parse_result_nums(stripped)
            if prune_repeated_states and result_nums is not None:
                seen.add(state_key(result_nums))
    return {
        "total": total,
        "in_set": in_set,
        "parse_fail": parse_fail,
        "in_set_frac": (in_set / total) if total else 0.0,
    }


if __name__ == "__main__":
    # CPU self-test: every step actually explored in a ground-truth trace must
    # be a member of (a) its recorded candidate set and (b) the recomputed
    # candidate set, across all flag combinations.
    import random
    from countdown_dfs import dfs

    combos = [
        ("det",          dict()),
        ("op",           dict(randomize_op_order=True)),
        ("heur",         dict(randomize_heuristic=True)),
        ("backtrack",    dict(randomize_backtrack=True)),
        ("prune",        dict(prune_repeated_states=True)),
        ("all",          dict(randomize_op_order=True, randomize_heuristic=True,
                              randomize_backtrack=True, prune_repeated_states=True)),
    ]
    instances = [(24, [8, 2, 3, 2, 1]), (27, [2, 5, 2, 8]), (36, [2, 3, 6, 5]),
                 (100, [7, 8, 9, 3]), (47, [42, 61, 66])]

    recorded_ok = recorded_tot = 0
    recompute_ok = recompute_tot = 0
    for name, kw in combos:
        for seed, (target, nums) in enumerate(instances):
            trace, steps = dfs(target, nums, threshold=target, rng=random.Random(seed), **kw)
            mapping = build_state_to_candidates(steps)
            prune = kw.get("prune_repeated_states", False)
            seen = set()
            seen.add(state_key(nums))
            for prefix, state, gold in iter_decision_prefixes(trace):
                # (a) recorded candidate set
                recorded_tot += 1
                if state in mapping and gold in mapping[state]:
                    recorded_ok += 1
                # (b) recomputed candidate set
                tgt, cur_nums = parse_state_line(state)
                recompute_tot += 1
                valid = recompute_candidate_lines(
                    tgt, cur_nums, threshold=target,
                    prune_repeated_states=prune, visited=seen)
                if gold in valid:
                    recompute_ok += 1
                if prune:
                    rn = parse_result_nums(gold)
                    if rn is not None:
                        seen.add(state_key(rn))

    print(f"recorded   candidate membership: {recorded_ok}/{recorded_tot}")
    print(f"recomputed candidate membership: {recompute_ok}/{recompute_tot}")
    assert recorded_ok == recorded_tot, "recorded candidate set is missing an explored step!"
    assert recompute_ok == recompute_tot, "recomputed candidate set is missing an explored step!"
    print("OK: every explored step is in-set for both views, across all flag combos.")
