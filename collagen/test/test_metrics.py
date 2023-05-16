
import unittest

import torch

from collagen import metrics


class TestMetrics(unittest.TestCase):

    """TestMetrics is a class for testing."""
    
    def test_topk(self):
        labels = [
            [0,0,0,0,1],
            [0,0,0,1,1],
            [0,0,1,0,0],
            [0,1,0,0,0],
            [1,0,0,0,0],
            [1,1,0,0,0]
        ]

        predictions = [
            [0,0,0,0,0.5], # rank=0
            [0,0,0,0.5,0.1], # rank=1
            [1,0.6,0,0,0.1], # rank=3
            [0,0.4,0,0,0.1], # rank=2
        ]

        targets = [
            [0,0,0,0,1],
            [0,0,0,0,1],
            [0,0,0,0,1],
            [0,0,0,0,1],
        ]

        topk = metrics.top_k(
            predictions=torch.tensor(predictions),
            correct_predicton_targets=torch.tensor(targets),
            label_set_fingerprints=torch.tensor(labels),
            k=[1,2,3,4,5]
        )

        self.assertEqual(topk[1], 0.25)
        self.assertEqual(topk[2], 0.5)
        self.assertEqual(topk[3], 0.75)
        self.assertEqual(topk[4], 1)
        self.assertEqual(topk[5], 1)


if __name__=='__main__':
    unittest.main()
