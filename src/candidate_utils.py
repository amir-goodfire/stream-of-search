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
import math
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


def heuristic_probs(heuristics, temperature=1.0):
    """Heuristic-weighted first-draw selection distribution: ``p_i propto
    exp(-(h_i - h_min) / T)`` (lower heuristic == better == more mass). This is
    exactly the distribution the randomized DFS samples the successor order from
    when ``randomize_heuristic`` is set (see ``countdown_dfs._heuristic_probs``);
    duplicated here to keep this module torch-free and importable on its own."""
    if not heuristics:
        return []
    h_min = min(heuristics)
    weights = [math.exp(-(h - h_min) / temperature) for h in heuristics]
    total = sum(weights)
    if total == 0:
        n = len(heuristics)
        return [1.0 / n] * n
    return [w / total for w in weights]


def recompute_candidate_dist(target, nums, threshold=None, temperature=1.0,
                             prune_repeated_states=False, visited=None,
                             heuristic=mult_heuristic):
    """Like ``recompute_candidate_lines`` but also returns the heuristic-weighted
    probability the randomized DFS assigns to each kept candidate.

    Returns a list of ``(line, heuristic_value, prob)`` tuples in generation
    order, where ``prob`` is the first-draw selection probability
    (``heuristic_probs`` over the kept candidates' heuristics). This is the
    "expected next-step distribution" the search would follow from ``nums`` under
    ``randomize_heuristic`` at ``temperature`` (the policy used to generate the
    ``heuristic`` / ``backtrack_heuristic`` datasets, with ``threshold=target``,
    ``temperature=1.0``). Candidate generation + pruning mirror
    ``recompute_candidate_lines`` / ``countdown_dfs.dfs`` exactly.

    Distinct index pairs can yield the *same* trace line when the input has
    repeated numbers (e.g. two ways to form ``3-2=1``); the DFS treats these as
    separate candidates, but as a distribution over distinct next-step *strings*
    their probabilities are summed here (mirroring ``recompute_candidate_lines``,
    which likewise dedupes to a set). Result is ordered by first appearance."""
    local_visited = set(visited) if visited is not None else set()
    kept = []  # (line, heuristic_value)
    for i, j in combinations(range(len(nums)), 2):
        for result, operation in combine_nums(nums[i], nums[j]):
            new_nums = [nums[k] for k in range(len(nums)) if k != i and k != j] + [result]
            h = heuristic(new_nums, target)
            if threshold is not None and h > threshold:
                continue
            if prune_repeated_states:
                key = state_key(new_nums)
                if key in local_visited:
                    continue
                local_visited.add(key)
            kept.append((exploring_line(operation, new_nums), h))
    probs = heuristic_probs([h for (_line, h) in kept], temperature)

    # Merge duplicate lines, summing their selection probability.
    merged = {}  # line -> [heuristic, summed_prob]
    for (line, h), p in zip(kept, probs):
        if line in merged:
            merged[line][1] += p
        else:
            merged[line] = [h, p]
    return [(line, h, p) for line, (h, p) in merged.items()]


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


_MOVING_RE = re.compile(r"Moving to Node #([\d,]+)")


def _is_desc_or_self(node_idx, other_idx):
    """True iff ``other_idx`` is ``node_idx`` or one of its descendants (node
    indices are comma-joined paths, e.g. "0,1,0"). The comma guard keeps "0,10"
    from matching "0,1"."""
    return other_idx == node_idx or other_idx.startswith(node_idx + ",")


def trace_on_track_labels(trace):
    """Label every "Current State:" occurrence in ``trace`` with whether the
    search goes on to reach the goal, and how directly.

    The DFS trace makes backtracking explicit: every visit is announced with a
    "Moving to Node #idx" line (a *forward* move always goes to a child of the
    node just left; anything else is a backtrack). For each occurrence of a
    "Current State" line (one decision point, at node ``N``) we report:

      * ``solved``                - "Goal Reached" appears later in the trace.
      * ``solved_within_subtree`` - the goal is reached before the search ever
        moves to a node outside ``N``'s subtree (moves to ``N`` itself or a
        descendant are fine, i.e. retrying siblings below ``N`` still counts as
        staying "on track" within the subtree).
      * ``solved_no_backtrack``   - the goal is reached by straight descent:
        every subsequent move goes to a child of the node just left (implies
        ``solved_within_subtree``).

    Returns a list of dicts in occurrence order, each carrying ``line_index``,
    ``current_state`` (the stripped line, for alignment checks), ``node_idx``,
    and the three labels. Works on model-generated traces too (unparseable
    moves are simply never matched); a trace with no goal labels everything
    False."""
    lines = trace.splitlines()
    goal_i = None
    for i, line in enumerate(lines):
        if "Goal Reached" in line:
            goal_i = i
            break

    # Current node at each "Current State" occurrence: the root ("0") until the
    # first move; every "Moving to Node #idx" line re-anchors it.
    occurrences = []
    cur = "0"
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = _MOVING_RE.match(stripped)
        if m:
            cur = m.group(1)
            continue
        if stripped.startswith("Current State:"):
            occurrences.append((i, cur, stripped))

    out = []
    for i, node, state_line in occurrences:
        solved = goal_i is not None and goal_i > i
        within = solved
        straight = solved
        prev = node
        if solved:
            for j in range(i + 1, goal_i):
                m = _MOVING_RE.match(lines[j].strip())
                if not m:
                    continue
                mv = m.group(1)
                if not _is_desc_or_self(node, mv):
                    within = False
                if not mv.startswith(prev + ","):  # forward move = child of prev
                    straight = False
                prev = mv
        out.append({
            "line_index": i,
            "current_state": state_line,
            "node_idx": node,
            "solved": solved,
            "solved_within_subtree": within,
            "solved_no_backtrack": straight and within,
        })
    return out


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
