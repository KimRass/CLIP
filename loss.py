# Source: https://github.com/omihub777/ViT-CIFAR/blob/main/criterions.py

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0):
        super().__init__()

        assert 0 <= smoothing <= 1, "The argument `smoothing` must be between 0 and 1!"

        self.n_classes = n_classes
        self.smoothing = smoothing

        self.ce = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, pred, gt):
        if gt.ndim == 1:
            new_gt = torch.full_like(pred, fill_value=self.smoothing / (self.n_classes - 1))
            new_gt.scatter_(1, gt.unsqueeze(1), 1 - self.smoothing)
        elif gt.ndim == 2:
            new_gt = gt.clone()
            new_gt.sum(dim=1)
            new_gt *= (1 - self.smoothing)
            is_zero = (gt == 0)
            likelihood = self.smoothing / (gt.shape[1] - (~is_zero).sum(dim=1))
            new_gt += is_zero * likelihood.unsqueeze(1).repeat(1, self.n_classes)
        loss = self.ce(pred, new_gt)
        return loss
