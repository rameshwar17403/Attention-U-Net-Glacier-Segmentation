import unittest
import torch
from src.losses import HybridBCEDiceLoss, DiceLoss, TverskyLoss, FocalTverskyLoss

class TestLosses(unittest.TestCase):
    def test_hybrid_loss(self):
        criterion = HybridBCEDiceLoss(alpha=0.5)
        logits = torch.randn(2, 1, 256, 256, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
        loss = criterion(logits, targets)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_dice_loss(self):
        criterion = DiceLoss()
        logits = torch.randn(2, 1, 256, 256)
        targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
        loss = criterion(logits, targets)
        self.assertFalse(torch.isnan(loss))

    def test_tversky_loss(self):
        criterion = TverskyLoss(alpha=0.3, beta=0.7)
        logits = torch.randn(2, 1, 256, 256)
        targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
        loss = criterion(logits, targets)
        self.assertFalse(torch.isnan(loss))

    def test_focal_tversky_loss(self):
        criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
        logits = torch.randn(2, 1, 256, 256, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
        loss = criterion(logits, targets)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

if __name__ == "__main__":
    unittest.main()
