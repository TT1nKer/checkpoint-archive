"""Compare bounded mixed dependency plans against all choices on a small graph."""
from itertools import product
import unittest

import numpy as np

from certify import bound
from plans import choose_plan, closures


def reads(plan, i, active=()):
    if i in active:
        raise ValueError("cycle")
    records = {i}
    for parent in plan[i]:
        records.update(reads(plan, parent, active + (i,)))
    return records


class MixedArchiveTest(unittest.TestCase):
    def test_shared_ancestor_and_cycles(self):
        diamond = ((), (0,), (1, 3), (0,), (2,))
        self.assertEqual(closures(diamond)[2], frozenset({0, 1, 2, 3}))
        self.assertEqual(closures(diamond)[4], frozenset(range(5)))
        with self.assertRaises(ValueError):
            closures(((1,), (0,)))
        with self.assertRaises(ValueError):
            closures(((9,),))

    def test_enumeration_and_checked_incumbents(self):
        anchors = [20, 50, 50, 50, 50]
        singles = np.full((5,5), -1)
        for i, p, c in [(1,0,3), (3,0,3), (4,2,2), (2,4,11), (0,4,7), (2,1,32)]:
            singles[i,p] = c
        bridges = {(2,1,3):1, (3,2,4):9, (2,0,4):40}
        choices = [[(), *((p,) for p in range(5) if singles[i,p] >= 0),
                    *((a,b) for j,a,b in bridges if i == j)] for i in range(5)]

        def cost(plan):
            return sum(anchors[i] if not ps else singles[i,ps[0]] if len(ps) == 1
                       else bridges[i,*ps] for i,ps in enumerate(plan))

        all_dags = []
        for plan in product(*choices):
            try:
                max_reads = max(len(reads(plan, i)) for i in range(5))
            except ValueError:
                continue
            all_dags.append((cost(plan), max_reads))
        for cap in range(1,6):
            with self.subTest(cap=cap):
                exact = min(price for price, cold in all_dags if cold <= cap)
                baseline = [()] * 5
                trial = choose_plan(baseline, anchors, singles, bridges, cap)
                self.assertLessEqual(cost(trial), cost(baseline))
                self.assertGreaterEqual(cost(trial), exact)
                self.assertLessEqual(max(len(reads(trial, i)) for i in range(5)), cap)
                certificate = bound(anchors, singles, bridges, cap, seconds=5, incumbent=trial)
                self.assertTrue(certificate["optimal"])
                self.assertEqual(certificate["lower_bound"], exact)
                self.assertEqual(cost(certificate["parents"]), exact)


if __name__ == "__main__":
    unittest.main()
