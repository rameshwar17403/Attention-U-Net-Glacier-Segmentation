# src/losses.py
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
            (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class HybridBCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-6):
        super(HybridBCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced binary segmentation.
    TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    Loss = 1 - TI
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets_flat = targets.view(-1)

        tp = (probs * targets_flat).sum()
        fp = (probs * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss to address high class imbalance and hard negative mining.
    FTL = (1 - TI)^gamma
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives (catches debris-covered ice)
        self.gamma = gamma  # focal parameter
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma)
