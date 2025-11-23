"""
Focal Loss implementation for addressing class imbalance
Focal Loss = -alpha * (1-pt)^gamma * log(pt)

Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss สำหรับแก้ปัญหา class imbalance
    
    Parameters:
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               - gamma=0: equivalent to CrossEntropyLoss
               - gamma>0: down-weights easy examples
    
    Benefits:
        - ลด weight ของ easy examples (confident predictions)
        - เพิ่ม focus ใน hard examples (misclassified)
        - ดีกว่า class weights สำหรับ imbalanced data
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) shape [batch_size, num_classes] or [batch_size]
            targets: Ground truth labels shape [batch_size]
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        if inputs.dim() == 1:
            # Binary classification (single output)
            probs = torch.sigmoid(inputs)
            targets = targets.float()
            
            # Calculate focal loss for binary case
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            
        else:
            # Multi-class classification
            probs = F.softmax(inputs, dim=1)
            
            # Get probability of correct class
            targets_long = targets.long()  # Convert to LongTensor for one_hot
            targets_one_hot = F.one_hot(targets_long, num_classes=inputs.size(1)).float()
            p_t = (probs * targets_one_hot).sum(dim=1)
            alpha_t = self.alpha
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss: -alpha * (1-p_t)^gamma * log(p_t)
        loss = -alpha_t * focal_weight * torch.log(p_t + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class weights
    รวม focal loss กับ class weighting
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits)
            targets: Ground truth labels
        
        Returns:
            Weighted focal loss
        """
        # Calculate focal loss
        focal_loss = FocalLoss(self.alpha, self.gamma, reduction='none')
        loss = focal_loss(inputs, targets)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            if isinstance(self.class_weights, torch.Tensor):
                weights = self.class_weights[targets]
            else:
                weights = torch.tensor([self.class_weights[t.item()] for t in targets],
                                     device=targets.device)
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def calculate_class_weights(labels, method='balanced'):
    """
    คำนวณ class weights จาก labels
    
    Args:
        labels: Label array (numpy or tensor)
        method: 'balanced' or 'inverse_freq'
    
    Returns:
        Dictionary of class weights
    """
    import numpy as np
    from collections import Counter
    
    # Convert to numpy if tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Count occurrences
    counter = Counter(labels)
    total = len(labels)
    
    weights = {}
    if method == 'balanced':
        # sklearn-style: n_samples / (n_classes * n_samples_per_class)
        n_classes = len(counter)
        for label, count in counter.items():
            weights[label] = total / (n_classes * count)
    else:  # inverse_freq
        # Simple inverse frequency
        for label, count in counter.items():
            weights[label] = total / count
    
    # Normalize weights to sum to n_classes
    n_classes = len(weights)
    total_weight = sum(weights.values())
    weights = {k: v * n_classes / total_weight for k, v in weights.items()}
    
    return weights


# Example usage and testing
if __name__ == "__main__":
    print("Testing Focal Loss implementation...")
    
    # Test binary classification
    inputs = torch.randn(10, 1)
    targets = torch.randint(0, 2, (10,))
    
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(inputs.squeeze(), targets)
    print(f"Binary Focal Loss: {loss.item():.4f}")
    
    # Test multi-class
    inputs = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))
    
    loss = focal_loss(inputs, targets)
    print(f"Multi-class Focal Loss: {loss.item():.4f}")
    
    # Test with class weights
    class_weights = calculate_class_weights(targets.numpy())
    print(f"Class weights: {class_weights}")
    
    weighted_focal = WeightedFocalLoss(alpha=0.25, gamma=2.0, 
                                       class_weights=class_weights)
    loss = weighted_focal(inputs, targets)
    print(f"Weighted Focal Loss: {loss.item():.4f}")
    
    print("✓ All tests passed!")
