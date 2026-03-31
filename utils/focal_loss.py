"""
Logits-based focal loss utilities for binary classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary focal loss on logits.

    Args:
        alpha: Positive-class weighting factor. Set to None to disable alpha weighting.
        gamma: Focusing parameter. gamma=0 reduces to BCE on logits when alpha=None.
        reduction: "mean", "sum", or "none".
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        logits = logits.float()

        if logits.shape != targets.shape:
            targets = targets.view_as(logits)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is None:
            alpha_factor = 1.0
        else:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_factor * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedFocalLoss(nn.Module):
    """Focal loss with optional per-class weights."""

    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, logits, targets):
        focal_loss = FocalLoss(self.alpha, self.gamma, reduction='none')
        loss = focal_loss(logits, targets)

        if self.class_weights is not None:
            if isinstance(self.class_weights, torch.Tensor):
                weights = self.class_weights[targets.long()]
            else:
                weights = torch.tensor(
                    [self.class_weights[int(target.item())] for target in targets.view(-1)],
                    device=targets.device,
                    dtype=loss.dtype
                ).view_as(loss)
            loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def calculate_class_weights(labels, method='balanced'):
    """Compute class weights from binary labels."""
    from collections import Counter

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    counter = Counter(labels)
    total = len(labels)
    weights = {}

    if method == 'balanced':
        n_classes = len(counter)
        for label, count in counter.items():
            weights[label] = total / (n_classes * count)
    else:
        for label, count in counter.items():
            weights[label] = total / count

    n_classes = len(weights)
    total_weight = sum(weights.values())
    return {key: value * n_classes / total_weight for key, value in weights.items()}
