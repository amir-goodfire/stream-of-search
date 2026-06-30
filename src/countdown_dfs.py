import itertools
import math
import random

import tiktoken

from countdown_utils import combine_nums, CountdownNode, mult_heuristic, metric_fn


# ---------------------------------------------------------------------------
# Ordering / sampling helpers
# ---------------------------------------------------------------------------

def _state_key(nums):
    """Canonical key for a game state: the multiset of available numbers. Two
    states are "the same exact state" iff they have the same numbers, regardless
    of the order or which operations produced them."""
    return tuple(sorted(nums))

def _heuristic_probs(heuristics, temperature):
    """Turn a list of heuristic values (lower == closer to a factor of the
    target == better) into a probability distribution where better states get
    more mass: ``p_i propto exp(-h_i / T)``.

    A small ``temperature`` sharpens the distribution toward the best state; a
    large ``temperature`` flattens it toward uniform. Returns a list of
    normalized probabilities aligned with ``heuristics``.
    """
    if not heuristics:
        return []
    # subtract the min for numerical stability (does not change the distribution)
    h_min = min(heuristics)
    weights = [math.exp(-(h - h_min) / temperature) for h in heuristics]
    total = sum(weights)
    if total == 0:
        n = len(heuristics)
        return [1.0 / n] * n
    return [w / total for w in weights]


def _sample_order(indices, probs, rng):
    """Sample an ordering of ``indices`` without replacement, where each draw is
    proportional to ``probs``. Returns the ordered list of indices. ``probs`` is
    indexed the same way as ``indices``.
    """
    remaining = list(indices)
    rem_probs = [probs[i] for i in indices]
    order = []
    while remaining:
        total = sum(rem_probs)
        if total <= 0:
            # numerical fallback: take whatever is left in current order
            order.extend(remaining)
            break
        r = rng.random() * total
        acc = 0.0
        chosen = len(remaining) - 1
        for k in range(len(remaining)):
            acc += rem_probs[k]
            if r <= acc:
                chosen = k
                break
        order.append(remaining.pop(chosen))
        rem_probs.pop(chosen)
    return order


def _order_candidates(kept, temperature, randomize_op_order,
                      randomize_heuristic, rng):
    """Decide the order in which the kept (non-pruned) successors are explored.

    ``kept`` is a list of ``(heuristic, CountdownNode)`` tuples. Returns
    ``(ordered, probs, mode)`` where ``ordered`` is the reordered list of the
    same tuples, ``probs[i]`` is the first-draw selection probability of the
    candidate now at position ``i`` (``None`` when deterministic), and ``mode``
    is a human-readable label of the policy that was applied.

    The two flags compose with **heuristic precedence**:
      * ``randomize_heuristic``        -> sample order weighted by the heuristic
      * ``randomize_op_order`` (only)  -> uniform random shuffle
      * neither                        -> deterministic best-first sort
    """
    if randomize_heuristic:
        mode = "heuristic"
    elif randomize_op_order:
        mode = "uniform"
    else:
        mode = "deterministic"

    n = len(kept)
    if n == 0:
        return [], [], mode

    heuristics = [h for (h, _node) in kept]

    if randomize_heuristic:
        probs = _heuristic_probs(heuristics, temperature)
        order_idx = _sample_order(list(range(n)), probs, rng)
        ordered = [kept[i] for i in order_idx]
        ordered_probs = [probs[i] for i in order_idx]
        return ordered, ordered_probs, "heuristic"

    if randomize_op_order:
        order_idx = list(range(n))
        rng.shuffle(order_idx)
        ordered = [kept[i] for i in order_idx]
        # uniform first-draw probability over the kept candidates
        ordered_probs = [1.0 / n] * n
        return ordered, ordered_probs, "uniform"

    # deterministic: best-first (lowest heuristic), stable on generation order
    ordered = sorted(kept, key=lambda x: x[0])
    return ordered, [None] * n, "deterministic"


def _select_open_node(open_nodes, randomize_backtrack, rng):
    """Pick which open node to (re)visit next. Returns ``(index, probs, mode)``.

    ``open_nodes`` is the current set of open states: nodes that have been
    moved to and still have untried successors (the active DFS path).
    Deterministic = the last open state (the deepest, i.e. ordinary
    depth-first). Stochastic = a uniformly sampled open state, i.e. backtrack to
    *any* node still on the path. ``probs`` is the distribution over the open set
    (``None`` when deterministic).
    """
    n = len(open_nodes)
    if randomize_backtrack:
        idx = rng.randrange(n)
        return idx, [1.0 / n] * n, "uniform"
    return n - 1, None, "lifo"


# ---------------------------------------------------------------------------
# Iterative DFS
# ---------------------------------------------------------------------------

def dfs(target, nums, heuristic=mult_heuristic, threshold=None,
        randomize_op_order=False, randomize_heuristic=False,
        randomize_backtrack=False, prune_repeated_states=False,
        temperature=1.0, max_nodes=None, rng=None):
    """Depth-first search over the Countdown game.

    The search is emitted lazily, exactly like the original recursive DFS: a
    node's successors are announced one at a time, right before the search
    descends into each, so the trace stays depth-first and does *not* enumerate
    the open set. Internally an explicit set of open nodes (the active DFS path:
    nodes that have been moved to and still have untried successors) is kept so
    that backtracking can be controlled. Three independent sources of
    determinism can be relaxed:

    1. ``randomize_op_order``  - uniformly shuffle the order in which a node's
       successors are explored (instead of the deterministic best-first sort).
    2. ``randomize_heuristic`` - keep pruning deterministic, but sample the
       successor exploration order from a distribution weighted by the
       heuristic (closer to a factor of the target => more likely to be
       explored first). Takes precedence over ``randomize_op_order``.
    3. ``randomize_backtrack`` - applies *only when the search actively
       backtracks*, i.e. after a terminal/dead-end (a leaf with no solution, or
       a node with no valid successors). At that point it samples a random open
       state to revisit instead of always taking the last (deepest) open state.
       A newly generated node is always visited immediately (a deterministic
       forward move), so randomization never applies to forward descent.

    A fourth, orthogonal flag controls repeated states:

    4. ``prune_repeated_states`` - when set, a generated successor is pruned if
       its exact state (the multiset of available numbers, see ``_state_key``)
       has already been generated anywhere earlier in the search. This avoids
       re-expanding states reached by a different order of operations, both
       within a single expansion (e.g. ``8*1`` and ``8/1`` both give the same
       numbers) and across branches. It can never prune the active path, since a
       successor always has one fewer number than its parent and so can only
       collide with same-depth states in other branches. Off by default.

    With all four flags off the trace is identical to the original deterministic
    best-first DFS.

    ``temperature`` controls the sharpness of the heuristic-weighted
    distribution. ``rng`` is an optional ``random.Random`` instance (defaults to
    the global ``random`` module) so callers retain full control / seeding.

    ``max_nodes`` caps the search length: once that many successors have been
    explored without reaching the goal, the search stops and the trace is cut
    short with a ``"Max search length reached"`` line (no goal => rating 0).
    This bounds memory/trace size for stochastic policies (e.g.
    ``randomize_backtrack``) that can otherwise wander most of the tree before
    finding a solution. ``None`` means no limit (original behaviour).

    Returns ``(search_trace, steps)``. ``steps`` is a chronological list of the
    decisions the algorithm made, so downstream training/eval can check that a
    model reproduced *exactly* this search. Each element is one of:

      * ``{"type": "expand", ...}`` - a node was expanded; lists every kept
        candidate successor (in explored order, with selection probabilities)
        and every deterministically pruned successor (each tagged with a
        ``reason`` of ``"threshold"`` or ``"repeated_state"``). This is the full
        "list of states that could have been explored" at that node.
      * ``{"type": "select", ...}`` - a branch point where more than one open
        state could be revisited; lists the open set, the distribution over it,
        and which node was chosen. (Forced single-option moves are not logged.)
    """
    if rng is None:
        rng = random

    search_trace = ""
    steps = []

    # Canonical states already generated anywhere in the search (only consulted
    # when prune_repeated_states is set).
    visited = set()
    if prune_repeated_states:
        visited.add(_state_key(nums))

    def expand(node):
        """Generate, prune, and order ``node``'s successors; attach them as
        ``node.children`` (ordered list of ``(heuristic, CountdownNode)``) with a
        ``node.cursor`` pointing at the next untried successor. Records the
        expansion (full candidate list + distribution) in ``steps``."""
        generated_nodes = []
        for i, j in itertools.combinations(range(len(node.nums)), 2):
            for result, operation in combine_nums(node.nums[i], node.nums[j]):
                new_nums = [node.nums[k] for k in range(len(node.nums)) if k != i and k != j] + [result]
                new_operations = node.operations + [operation]
                new_heuristic = heuristic(new_nums, target)
                child = CountdownNode(0, node, new_nums, new_operations, new_heuristic)
                generated_nodes.append((new_heuristic, child))

        # Deterministic pruning: first by the threshold, then (optionally) by
        # whether this exact state has already been generated elsewhere.
        kept_nodes = []
        pruned_records = []
        for h, child in generated_nodes:
            if threshold is not None and h > threshold:
                pruned_records.append({
                    "operation": child.operations[-1],
                    "nums": child.nums,
                    "heuristic": h,
                    "reason": "threshold",
                })
            elif prune_repeated_states and _state_key(child.nums) in visited:
                pruned_records.append({
                    "operation": child.operations[-1],
                    "nums": child.nums,
                    "heuristic": h,
                    "reason": "repeated_state",
                })
            else:
                if prune_repeated_states:
                    visited.add(_state_key(child.nums))
                kept_nodes.append((h, child))

        ordered, ordered_probs, ordering_mode = _order_candidates(
            kept_nodes, temperature, randomize_op_order, randomize_heuristic, rng)

        candidate_records = []
        for position, (h, child) in enumerate(ordered):
            child.idx = f"{node.idx},{position}"
            if len(child.nums) == 1 and child.nums[0] == target:
                terminal = "goal"
            elif len(child.nums) == 1:
                terminal = "no_solution"
            else:
                terminal = None
            candidate_records.append({
                "idx": child.idx,
                "operation": child.operations[-1],
                "nums": child.nums,
                "heuristic": h,
                "prob": ordered_probs[position],
                "terminal": terminal,
            })

        steps.append({
            "type": "expand",
            "node_idx": node.idx,
            "target": target,
            "nums": node.nums,
            "operations": node.operations,
            "ordering_mode": ordering_mode,
            "candidates": candidate_records,
            "pruned": pruned_records,
        })

        node.children = ordered
        node.cursor = 0

    root = CountdownNode(0, None, nums, [], heuristic(nums, target))
    root.idx = "0"
    expand(root)
    open_nodes = [root]
    search_trace += f"Current State: {target}:{root.nums}, Operations: {root.operations}\n"

    # ``anchor`` is the node whose "Current State" is currently in effect in the
    # trace; ``anchor_fresh`` is True iff that "Current State" line is the most
    # recent line (so the next successor can be explored without re-announcing).
    anchor = root
    anchor_fresh = True

    def backtrack():
        """Actively backtrack: choose an open node (one with an untried
        successor) to revisit. ``randomize_backtrack`` only takes effect here.
        Returns the chosen node, or ``None`` if the search is exhausted. A
        genuine branch point (more than one open state) is logged as a
        "select" decision."""
        candidates = [n for n in open_nodes if n.cursor < len(n.children)]
        if not candidates:
            return None
        sel_index, sel_probs, sel_mode = _select_open_node(
            candidates, randomize_backtrack, rng)
        chosen = candidates[sel_index]
        if len(candidates) > 1:
            steps.append({
                "type": "select",
                "selection_mode": sel_mode,
                "chosen_idx": chosen.idx,
                "num_open": len(candidates),
                "open_nodes": [
                    {"idx": n.idx, "nums": n.nums, "operations": n.operations,
                     "heuristic": n.heuristic}
                    for n in candidates
                ],
                "probs": sel_probs,
            })
        return chosen

    nodes_explored = 0
    node = root
    while node is not None:
        # Re-anchor the trace to ``node`` if we are not already there (this is
        # the "visit").
        if not (anchor is node and anchor_fresh):
            search_trace += f"Moving to Node #{node.idx}\n"
            search_trace += f"Current State: {target}:{node.nums}, Operations: {node.operations}\n"
            anchor = node
            anchor_fresh = True

        # No (more) successors to try from here -> this is a dead end, backtrack.
        if node.cursor >= len(node.children):
            node = backtrack()
            continue

        # Length limit: stop the search if we've already explored the maximum
        # number of successors without reaching the goal.
        if max_nodes is not None and nodes_explored >= max_nodes:
            search_trace += "Max search length reached\n"
            return search_trace, steps

        # Explore this node's next successor.
        _, child = node.children[node.cursor]
        node.cursor += 1
        nodes_explored += 1
        search_trace += f"Exploring Operation: {child.operations[-1]}, Resulting Numbers: {child.nums}\n"
        anchor_fresh = False

        if len(child.nums) == 1 and child.nums[0] == target:
            search_trace += f"{child.nums[0]},{target} equal: Goal Reached\n"
            return search_trace, steps
        elif len(child.nums) == 1:
            # Terminal (no solution) -> the model actively backtracks here.
            search_trace += f"{child.nums[0]},{target} unequal: No Solution\n"
            node = backtrack()
        else:
            # A new node was generated -> visit it immediately (deterministic
            # forward move; randomization never applies to this step).
            search_trace += f"Generated Node #{child.idx}: {target}:{child.nums} Operation: {child.operations[-1]}\n"
            expand(child)
            open_nodes.append(child)
            node = child

    return search_trace, steps


if __name__ == "__main__":
    # Example usage
    target = 24
    nums = [8, 2, 3, 2, 1]

    print("=== deterministic (identical to the original best-first DFS) ===")
    search_path, steps = dfs(target, nums, heuristic=mult_heuristic, threshold=target)
    print(search_path)
    print(f"num steps: {len(steps)}")
    print(metric_fn(search_path))

    print("\n=== stochastic heuristic ordering + random backtracking ===")
    rng = random.Random(0)
    search_path, steps = dfs(
        target, nums, heuristic=mult_heuristic, threshold=target,
        randomize_heuristic=True, randomize_backtrack=True,
        temperature=1.0, rng=rng)
    print(metric_fn(search_path))
    # show the recorded distribution at the first expansion
    first_expand = next(s for s in steps if s["type"] == "expand")
    print("first expansion candidates (op, heuristic, prob):")
    for c in first_expand["candidates"]:
        print(f"  {c['operation']:>12}  h={c['heuristic']:<4} p={c['prob']}")

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(search_path)
    print(f"token length: {len(tokens)}")
