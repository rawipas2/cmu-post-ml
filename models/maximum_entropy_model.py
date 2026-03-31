"""
Maximum Entropy Model (logistic regression with PyTorch).
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.focal_loss import FocalLoss
import config


class MaximumEntropyClassifier(nn.Module):
    """Maximum Entropy (logistic regression) classifier."""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class MaximumEntropyModel:
    """Maximum Entropy wrapper with logits-based BCE/Focal training."""

    def __init__(
        self,
        input_dim,
        learning_rate=None,
        epochs=None,
        l2_reg=0.01,
        use_class_weight=False,
        use_focal_loss=False
    ):
        self.model = MaximumEntropyClassifier(input_dim).to(config.DEVICE)
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.epochs = epochs or config.EPOCHS
        self.l2_reg = l2_reg
        self.use_class_weight = use_class_weight
        self.use_focal_loss = use_focal_loss
        self.model_name = 'Maximum_Entropy'
        self.pos_weight_value = None

        if self.use_focal_loss:
            print("   โ“ Using Focal Loss")
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif self.use_class_weight:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def _compute_loss(self, logits, targets):
        if self.use_focal_loss:
            return self.criterion(logits, targets)

        if self.use_class_weight and self.pos_weight_value is not None:
            weights = torch.where(
                targets == 1,
                torch.tensor(self.pos_weight_value, device=config.DEVICE),
                torch.tensor(1.0, device=config.DEVICE)
            )
            return (self.criterion(logits, targets) * weights).mean()

        return self.criterion(logits, targets)

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the Maximum Entropy model."""
        print(f"\n{'='*60}")
        print(f"๐€ Training {self.model_name} on GPU...")
        print(f"{'='*60}")

        X_train_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(config.DEVICE)

        if self.use_class_weight and not self.use_focal_loss:
            self.pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()
            print(f"   Using class weighting: {self.pos_weight_value:.2f}")

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self._compute_loss(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if X_valid is not None and y_valid is not None:
                val_loss = self._validate(X_valid, y_valid)
                self.scheduler.step(val_loss)

                if epoch % 10 == 0:
                    print(
                        f"   Epoch [{epoch+1}/{self.epochs}] - "
                        f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 10:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break
            elif epoch % 10 == 0:
                print(f"   Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_loss:.4f}")

        print(f"โ… {self.model_name} training completed!")

    def _validate(self, X_valid, y_valid):
        self.model.eval()
        with torch.no_grad():
            X_valid_tensor = torch.FloatTensor(X_valid).to(config.DEVICE)
            y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1).to(config.DEVICE)
            logits = self.model(X_valid_tensor)
            loss = self._compute_loss(logits, y_valid_tensor)
        return loss.item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(config.DEVICE)
            logits = self.model(X_tensor)
            predictions = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int).flatten()
        return predictions

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(config.DEVICE)
            logits = self.model(X_tensor)
            probas = torch.sigmoid(logits).cpu().numpy().flatten()
        return probas

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            filepath
        )
        print(f"๐’พ {self.model_name} saved to: {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(config.DEVICE)
        print(f"๐“ {self.model_name} loaded from: {filepath}")


def create_model(input_dim, **kwargs):
    """Factory function to create Maximum Entropy model."""
    return MaximumEntropyModel(input_dim=input_dim, **kwargs)
