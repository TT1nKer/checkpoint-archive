"""Plan a weight archive with a hard bound on distinct frames read per cold restore.

Each child chooses one representation with a measured byte cost. A bridge
needs both parents, whose own dependencies may overlap.
"""

from math import isfinite
from operator import index


def _masks(plan):
    """Return one inclusive ancestor bit mask per frame in a DAG."""
    n = len(plan)
    masks = [0] * n
    visiting = set()

    def visit(node):
        if masks[node]:
            return masks[node]
        if node in visiting:
            raise ValueError("cyclic archive dependencies")
        visiting.add(node)
        mask = 1 << node
        for parent in plan[node]:
            if not 0 <= parent < n:
                raise ValueError("parent index outside the archive")
            mask |= visit(parent)
        visiting.remove(node)
        masks[node] = mask
        return mask

    for node in range(n):
        visit(node)
    return masks


def closures(plan):
    """Return distinct frames required by each cold restore, including itself."""
    masks = _masks(plan)
    return tuple(frozenset(i for i in range(len(plan)) if mask & (1 << i))
                 for mask in masks)


def _fill_order(n, anchors):
    """Visit the remaining nodes from the widest gaps first."""
    order = list(anchors)
    remaining = set(range(n)) - set(anchors)
    while remaining:
        next_node = max(remaining, key=lambda v: (min(abs(v - a) for a in order), -v))
        order.append(next_node)
        remaining.remove(next_node)
    return order


def _starts(n, cap):
    indexes = tuple(range(n))
    yield indexes, ()
    yield indexes[::-1], ()
    ends = tuple(dict.fromkeys((0, n - 1)))
    yield _fill_order(n, ends), ()
    # Force a row of standalone anchors. Try scanning the intermediates
    # in time order and subdividing the gaps recursively.
    for gap in range(2, min(n, 2 * cap) + 1):
        for phase in (0, gap // 2):
            anchors = tuple(range(phase, n, gap))
            if not anchors:
                continue
            middle = tuple(i for i in indexes if i not in anchors)
            yield anchors + middle, anchors
            yield anchors + middle[::-1], anchors
            yield _fill_order(n, anchors), anchors


def _greedy(order, forced, options, cap):
    n = len(order)
    chosen = [()] * n
    masks = [0] * n
    for node in order:
        own = 1 << node
        masks[node] = own
        if node in forced:
            continue
        for _, parents in options[node]:
            mask = own
            for p in parents:
                if not masks[p]:
                    break
                mask |= masks[p]
            else:
                if mask.bit_count() <= cap:
                    chosen[node], masks[node] = parents, mask
                    break
    return tuple(chosen)


def _improve(plan, options, costs, cap, locked=None):
    """Reparent one node at a time; recheck all descendants after each trial."""
    plan = list(plan)
    n = len(plan)
    while True:
        masks = _masks(plan)
        reads = sum(m.bit_count() for m in masks)
        best_gain, best_move = (0, 0), None
        for node in range(n):
            if node == locked:
                continue
            previous = plan[node]
            old_cost = costs[node][previous]
            for price, parents in options[node]:
                saved = old_cost - price
                if saved < best_gain[0]:
                    break
                if parents == previous:
                    continue
                # A parent whose closure includes this node would form a cycle.
                dependencies = 0
                for p in parents:
                    dependencies |= masks[p]
                own = 1 << node
                if dependencies & own or (dependencies | own).bit_count() > cap:
                    continue
                plan[node] = parents
                trial = _masks(plan)
                plan[node] = previous
                counts = [m.bit_count() for m in trial]
                if max(counts) <= cap:
                    gain = (saved, reads - sum(counts))
                    if gain > best_gain:
                        best_gain, best_move = gain, (node, parents)
        if best_move is None:
            return tuple(plan)
        node, parents = best_move
        plan[node] = parents


def choose_plan(base_plan, anchors, singles, bridges, cap):
    """Return a feasible plan at least as small as the supplied feasible plan.

    anchors[i], singles[i,j] and bridges[i,left,right] are inclusive byte
    costs. Omit unavailable bridges and use infinity for unavailable singles.
    The comparison budget counts unique frames, with the requested frame
    counting once. Search is deterministic and heuristic.
    """
    n = len(anchors)
    base = tuple(tuple(index(p) for p in row) for row in base_plan)
    if len(base) != n or cap < 1:
        raise ValueError("inconsistent archive size or cold-read cap")
    if not n:
        return ()
    if max(m.bit_count() for m in _masks(base)) > cap:
        raise ValueError("supplied baseline violates the cold-read cap")

    costs = []
    for node in range(n):
        anchor = anchors[node]
        if not (isfinite(anchor) and anchor >= 0):
            raise ValueError("a standalone cost is required for every version")
        row = {(): anchor}
        for parent in range(n):
            if parent != node:
                price = singles[node, parent]
                if isfinite(price) and price >= 0:
                    row[(parent,)] = price
        costs.append(row)
    for (node, left, right), price in bridges.items():
        if not (0 <= left < node < right < n):
            raise ValueError("bridge parents must bracket the target")
        if isfinite(price) and price >= 0:
            costs[node][(left, right)] = price
    if any(row not in costs[i] for i, row in enumerate(base)):
        raise ValueError("baseline references an unavailable representation")
    options = [sorted((price, ps) for ps, price in row.items()) for row in costs]

    def score(plan):
        size = sum(row[parents] for row, parents in zip(costs, plan))
        return size, sum(m.bit_count() for m in _masks(plan))

    starts = {_greedy(order, forced, options, cap) for order, forced in _starts(n, cap)}
    starts.add(base)
    # Try the cheapest distinct seeds and explicitly improve the baseline too.
    finalists = [base, *sorted(starts, key=lambda p: (score(p), p))[:12]]
    best = min((_improve(p, options, costs, cap) for p in dict.fromkeys(finalists)),
               key=score)
    # Changing one frame into an anchor can allow multiple cheaper edges.
    # Keep that anchor during the first descent, then release it.
    for node in range(n):
        if not best[node]:
            continue
        new = best[:node] + ((),) + best[node + 1:]
        new = _improve(new, options, costs, cap, locked=node)
        new = _improve(new, options, costs, cap)
        if score(new) < score(best):
            best = new
    if score(best) > score(base) or max(m.bit_count() for m in _masks(best)) > cap:
        raise AssertionError("archive planner violated its baseline or cold-read bound")
    return best
