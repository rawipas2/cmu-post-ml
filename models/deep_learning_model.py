"""
Deep Learning Model with Multiple Hidden Layers (GPU-accelerated)
v1.2.2: Added Focal Loss support
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.focal_loss import FocalLoss
import config


class DeepLearningClassifier(nn.Module):
    """Deep neural network with multiple layers"""
    
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128, 64], dropout=0.4):
        super(DeepLearningClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout/2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DeepLearningModel:
    """Deep Learning wrapper with training utilities"""
    
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128, 64],
                 learning_rate=None, epochs=None, dropout=0.4):
        """
        Initialize Deep Learning model
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
        """
        self.model = DeepLearningClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(config.DEVICE)
        
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.epochs = epochs or config.EPOCHS
        self.model_name = 'Deep_Learning'
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the deep learning model"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.model_name} on GPU...")
        print(f"{'='*60}")
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(config.DEVICE)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                 shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if X_valid is not None and y_valid is not None:
                val_loss = self._validate(X_valid, y_valid)
                self.scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    print(f"   Epoch [{epoch+1}/{self.epochs}] - "
                          f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
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
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filepath)
        print(f"💾 {self.model_name} saved to: {filepath}")
    
    def load(self, filepath):
        """Load model from disk"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.model.to(config.DEVICE)
        print(f"📂 {self.model_name} loaded from: {filepath}")


def create_model(input_dim, **kwargs):
    """Factory function to create Deep Learning model"""
    return DeepLearningModel(input_dim=input_dim, **kwargs)
