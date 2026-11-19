"""Compile the forest optimizer and compare it against exhaustive parent assignments."""
import itertools
from pathlib import Path
import random
import subprocess
import tempfile
import unittest


def evaluate(anchors, edges, parents):
    """Return bytes and height, or None for a cycle or forbidden parent."""
    height = 0
    cost = 0
    for i, parent in enumerate(parents):
        if parent != -1 and (not 0 <= parent < len(parents) or edges[i][parent] < 0):
            return None
        cost += anchors[i] if parent == -1 else edges[i][parent]
        chain = set()
        current = i
        while current != -1:
            if current in chain:
                return None
            chain.add(current)
            current = parents[current]
        height = max(height, len(chain))
    return cost, height


class ForestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.executable = Path(cls.directory.name) / "forest"
        source = Path(__file__).with_name("forest.cpp")
        subprocess.run(["c++", "-std=c++17", "-O2", str(source), "-o", str(cls.executable)],
                       check=True)

    def check_example(self, anchors, edges):
        n = len(anchors)
        originals = [[-1] + [j for j in range(n) if j != i and edges[i][j] >= 0]
                     for i in range(n)]
        forests = []
        for p in itertools.product(*originals):
            plan = evaluate(anchors, edges, p)
            if plan is not None:
                forests.append(plan)
        expected = [min(score for score, height in forests if height <= k) for k in range(1, n + 1)]
        data = [n, n, *anchors, *(value for row in edges for value in row)]
        text = subprocess.check_output([self.executable, "--all"], input=" ".join(map(str, data)),
                                       text=True).splitlines()
        self.assertEqual(len(text), 2 * n)
        for k, predicted in enumerate(expected, 1):
            score = int(text[2 * k - 2])
            parents = list(map(int, text[2 * k - 1].split()))
            self.assertEqual(len(parents), n)
            self.assertEqual(score, predicted)
            measured = evaluate(anchors, edges, parents)
            self.assertIsNotNone(measured)
            self.assertEqual(measured[0], score)
            self.assertLessEqual(measured[1], k)

    def test_future_parents_and_forbidden_arcs(self):
        self.check_example([8, 8, 8, 4], [[-1, 1, 6, -1], [-1, -1, 1, 7],
                                          [-1, -1, -1, 1], [-1, -1, -1, -1]])

    def test_all_anchors_and_zero_cost_ties(self):
        self.check_example([6], [[-1]])
        self.check_example([0, 3, 2], [[-1, -1, -1], [0, -1, 0], [0, 0, -1]])

    def test_small_directed_graphs(self):
        rng = random.Random(432)
        for n in range(2, 6):
            for _ in range(3):
                anchors = [rng.randrange(12, 45) for i in range(n)]
                edges = [[max(-1, rng.randrange(-8, 35)) for j in range(n)] for i in range(n)]
                self.check_example(anchors, edges)


if __name__ == "__main__":
    unittest.main()
