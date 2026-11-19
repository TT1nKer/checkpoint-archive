"""MILP storage bounds for mixed one- and two-parent cold-restore plans."""

from math import ceil, isfinite
from time import monotonic
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from plans import choose_plan, closures


def _bytes(value):
    integer = int(value)
    if integer != value or integer < 0:
        raise ValueError("costs must be nonnegative integer byte counts")
    return integer


def bound(anchors, singles, bridges, cap, seconds=30, incumbent=None):
    """Return an integer lower bound and an independently checked feasible plan.

    x[i,parents] selects a record. y[i,j] describes the inclusive ancestor
    relation. Selected edges force all the parent's ancestors into the child's
    row. Antisymmetry forbids cycles. Fractional y values suffice because a
    path of selected edges from i to j forces y[i,j] to one.
    """
    started = monotonic()
    n = len(anchors)
    if n < 1 or cap < 1 or seconds <= 0:
        raise ValueError("positive record count, read cap and solver budget required")
    k = min(cap, n)
    options = [{(): _bytes(a)} for a in anchors]
    # A standalone dominates an equally large dependent representation.
    if k >= 2:
        for i, row in enumerate(options):
            for p in range(n):
                val = singles[i, p]
                if i != p and isfinite(val) and 0 <= val < row[()]:
                    row[(p,)] = _bytes(val)
    if k >= 3:
        for (i, left, right), val in bridges.items():
            if not (0 <= left < i < right < n):
                raise ValueError("bridge inputs must bracket the target")
            row = options[i]
            # Either parent's single-reference representation has a subset
            # of this bridge's transitive dependencies.
            alternative = min(row[()], row.get((left,), row[()]), row.get((right,), row[()]))
            if isfinite(val) and 0 <= val < alternative:
                row[(left, right)] = _bytes(val)

    def checked(plan):
        if max(map(len, closures(plan))) > k:
            raise ValueError("incumbent violates the cold-restore cap")
        return sum(row[ps] for row, ps in zip(options, plan, strict=True))

    if incumbent is None:
        incumbent = choose_plan([()] * n, anchors, singles, bridges, cap)
    incumbent = tuple(tuple(map(int, ps)) for ps in incumbent)
    if len(incumbent) != n or max(map(len, closures(incumbent))) > k:
        raise ValueError("invalid incumbent")
    # If a selected option was dominated, replacing it by its cheaper subset
    # preserves feasibility, including for all its descendants.
    incumbent = tuple(min((ps for ps in row if set(ps).issubset(original)),
                          key=lambda ps: (row[ps], len(ps), ps))
                      for original, row in zip(incumbent, options))
    upper = checked(incumbent)
    minima = [min(row.values()) for row in options]
    offset = sum(minima)
    gap = upper - offset
    if gap == 0:
        return dict(lower_bound=upper, feasible_cost=upper, optimal=True, parents=incumbent,
                    solver_status="row minima attain incumbent", elapsed_seconds=monotonic()-started)

    # No representation in a solution as good as the incumbent can spend
    # more than this entire gap at a single target.
    x = []
    objective = []
    for i, row in enumerate(options):
        by_parents = {}
        for ps, cost in row.items():
            excess = cost - minima[i]
            if excess <= gap:
                by_parents[ps] = len(objective)
                objective.append(excess)
        x.append(by_parents)
    nx = len(objective)
    y = np.arange(nx, nx + n*n).reshape((n,n))
    nvar = nx + n*n
    lower, upper_rows = [], []
    rows, cols, vals = [], [], []

    def constraint(terms, lo=-np.inf, hi=np.inf):
        row_id = len(lower)
        for col, val in terms:
            rows.append(row_id)
            cols.append(col)
            vals.append(val)
        lower.append(lo)
        upper_rows.append(hi)

    constraint([(j, c) for j, c in enumerate(objective) if c], hi=gap)
    # Every nonempty DAG has a root; the first two topological positions
    # each have at most one parent.
    constraint([(row[()], 1) for row in x if () in row], lo=1)
    constraint([(j, 1) for row in x for ps, j in row.items() if len(ps) < 2], lo=min(n, 2))
    for i, row in enumerate(x):
        constraint([(j, 1) for j in row.values()], 1, 1)
        constraint([(j, 1) for j in y[i]], hi=k)
        for p in range(n):
            if p == i:
                continue
            if p < i:
                constraint([(y[i,p], 1), (y[p,i], 1)], hi=1)
            if () in row:
                constraint([(y[i,p], 1), (row[()], 1)], hi=1)
            using = [(j, 1) for ps, j in row.items() if p in ps]
            if not using:
                continue
            constraint([*using, (y[i,p], -1)], hi=0)
            for j in range(n):
                if j not in (i, p):
                    constraint([*using, (y[p,j], 1), (y[i,j], -1)], hi=1)
            # On an active edge the child's inclusive closure is strictly larger.
            constraint([*((j, k) for j, _ in using), *((j, 1) for j in y[p]),
                        *((j, -1) for j in y[i])], hi=k-1)

    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), nvar)).tocsc()
    integrality = np.zeros(nvar, dtype=np.uint8)
    integrality[:nx] = 1
    lb = np.zeros(nvar)
    lb[y.diagonal()] = 1
    # SciPy passes native HiGHS options through after warning. Limit HiGHS
    # to one worker so certificates can run alongside the codec measurements.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads")
        result = milp(c=np.r_[objective, np.zeros(n*n)], integrality=integrality,
                      bounds=Bounds(lb, np.ones(nvar)),
                      constraints=LinearConstraint(matrix, lower, upper_rows),
                      options={"time_limit": seconds, "mip_rel_gap": 0.0, "threads": 1})
    if result.status not in (0, 1):
        raise RuntimeError(f"MILP failed despite a checked incumbent: {result.message}")
    if result.x is not None:
        proposal = tuple(max(row, key=lambda ps: result.x[row[ps]]) for row in x)
        price = checked(proposal)
        if price < upper:
            incumbent, upper = proposal, price
    # The integer optimum is at least ceil(dual_bound) in exact arithmetic.
    # Subtract an absolute and relative numerical allowance before rounding.
    dual = getattr(result, "mip_dual_bound", None)
    certified = offset
    if dual is not None and isfinite(dual):
        allowance = 1e-3 + abs(dual)*1e-9
        certified += max(0, ceil(dual - allowance))
    if certified > upper:
        raise ArithmeticError("MILP dual bound exceeds the checked feasible cost")
    return dict(lower_bound=int(certified), feasible_cost=int(upper), optimal=certified==upper,
                parents=incumbent, solver_status=str(result.message),
                solver_dual_excess=float(dual) if dual is not None and isfinite(dual) else None,
                row_minima_bytes=int(offset), elapsed_seconds=monotonic()-started)
