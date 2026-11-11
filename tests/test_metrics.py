import unittest
import torch
from src.metrics import compute_metrics

class TestMetrics(unittest.TestCase):
    def test_metrics_calculation(self):
        logits = torch.tensor([[[[100.0, -100.0], [100.0, 100.0]]]])
        targets = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
        
        metrics = compute_metrics(logits, targets, threshold=0.5)
        self.assertAlmostEqual(metrics['dice'], 1.0, places=4)
        self.assertAlmostEqual(metrics['iou'], 1.0, places=4)
        self.assertAlmostEqual(metrics['precision'], 1.0, places=4)
        self.assertAlmostEqual(metrics['recall'], 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
