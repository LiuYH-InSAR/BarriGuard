"""Dataset export regressions: balance images without splitting recording hours."""
import unittest

import barrier_labels as labels


class SplitTest(unittest.TestCase):
    def test_weighted_groups(self):
        rows = []
        for hour, count in enumerate([100, 100, 100, 100, 100, 100, 100, 100, 1, 1]):
            rows.extend(dict(source=f'1749_20261013_{hour:02d}0000.mp4') for _ in range(count))
        splits = labels.assign_splits(rows, [0.8, 0.1, 0.1], 42)
        counts = {s: sum(splits[labels.session(r)] == s for r in rows) for s in ['train','val','test']}
        self.assertGreaterEqual(counts['val'], 80)
        self.assertGreaterEqual(counts['test'], 80)


if __name__ == '__main__':
    unittest.main()
