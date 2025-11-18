"""
Maximum Entropy Model (Logistic Regression with GPU acceleration)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import config


class MaximumEntropyClassifier(nn.Module):
    """Maximum Entropy (Logistic Regression) classifier"""
    
    def __init__(self, input_dim):
        super(MaximumEntropyClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))


class MaximumEntropyModel:
    """Maximum Entropy wrapper with training utilities"""
    
    def __init__(self, input_dim, learning_rate=None, epochs=None, 
                 l2_reg=0.01):
        """
        Initialize Maximum Entropy model
        
        Args:
            input_dim: Input feature dimension
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            l2_reg: L2 regularization strength
        """
        self.model = MaximumEntropyClassifier(input_dim).to(config.DEVICE)
        
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.epochs = epochs or config.EPOCHS
        self.l2_reg = l2_reg
        self.model_name = 'Maximum_Entropy'
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.l2_reg
        )
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the Maximum Entropy model"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.model_name} on GPU...")
        print(f"{'='*60}")
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(config.DEVICE)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                 shuffle=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if X_valid is not None and y_valid is not None and epoch % 10 == 0:
                val_loss = self._validate(X_valid, y_valid)
                print(f"   Epoch [{epoch+1}/{self.epochs}] - "
                      f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            elif epoch % 10 == 0:
                print(f"   Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_loss:.4f}")
        
        print(f"✅ {self.model_name} training completed!")
    
    def _validate(self, X_valid, y_valid):
        """Validate model"""
        self.model.eval()
        with torch.no_grad():
            X_valid_tensor = torch.FloatTensor(X_valid).to(config.DEVICE)
            y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1).to(config.DEVICE)
            outputs = self.model(X_valid_tensor)
            loss = self.criterion(outputs, y_valid_tensor)
        return loss.item()
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(config.DEVICE)
            outputs = self.model(X_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(config.DEVICE)
            outputs = self.model(X_tensor)
            probas = outputs.cpu().numpy().flatten()
        return probas
    
    def save(self, filepath):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"💾 {self.model_name} saved to: {filepath}")
    
    def load(self, filepath):
        """Load model from disk"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(config.DEVICE)
        print(f"📂 {self.model_name} loaded from: {filepath}")


def create_model(input_dim, **kwargs):
    """Factory function to create Maximum Entropy model"""
    return MaximumEntropyModel(input_dim=input_dim, **kwargs)
