"""
Ensemble Stacking Model - Combines predictions from all 6 base models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import config


class StackingMetaLearner(nn.Module):
    """Meta-learner for stacking ensemble"""
    
    def __init__(self, n_models=6, hidden_dim=128):
        """
        Initialize meta-learner
        
        Args:
            n_models: Number of base models (input features)
            hidden_dim: Hidden layer dimension
        """
        super(StackingMetaLearner, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class EnsembleStackingModel:
    """Stacking ensemble combining predictions from multiple models"""
    
    def __init__(self, base_models, learning_rate=None, epochs=None, hidden_dim=None):
        """
        Initialize Ensemble Stacking model
        
        Args:
            base_models: List of trained base models
            learning_rate: Learning rate for meta-learner
            epochs: Number of epochs for meta-learner
            hidden_dim: Hidden dimension for meta-learner
        """
        self.base_models = base_models
        self.n_models = len(base_models)
        
        meta_hidden = hidden_dim or getattr(config, 'META_HIDDEN_DIM', 128)
        
        self.meta_learner = StackingMetaLearner(
            n_models=self.n_models,
            hidden_dim=meta_hidden
        ).to(config.DEVICE)
        
        self.learning_rate = learning_rate or config.META_MODEL_LR
        self.epochs = epochs or config.META_MODEL_EPOCHS
        self.model_name = 'Ensemble_Stacking'
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.meta_learner.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
    def _get_base_predictions(self, X, use_proba=True):
        """Get predictions from all base models"""
        predictions = []
        
        for model in self.base_models:
            if use_proba and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        # Stack predictions as features
        stacked = np.column_stack(predictions)
        return stacked
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the meta-learner on base model predictions"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.model_name} on GPU...")
        print(f"{'='*60}")
        
        # Get base model predictions as meta-features
        print("   Getting predictions from base models...")
        meta_features_train = self._get_base_predictions(X_train, use_proba=True)
        
        # Prepare data
        X_meta_tensor = torch.FloatTensor(meta_features_train).to(config.DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(config.DEVICE)
        
        train_dataset = TensorDataset(X_meta_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                 shuffle=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            self.meta_learner.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.meta_learner(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if X_valid is not None and y_valid is not None:
                val_loss = self._validate(X_valid, y_valid)
                self.scheduler.step(val_loss)
                
                if epoch % 20 == 0:
                    print(f"   Epoch [{epoch+1}/{self.epochs}] - "
                          f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            elif epoch % 20 == 0:
                print(f"   Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_loss:.4f}")
        
        print(f"✅ {self.model_name} training completed!")
    
    def _validate(self, X_valid, y_valid):
        """Validate meta-learner"""
        self.meta_learner.eval()
        
        # Get base model predictions
        meta_features_valid = self._get_base_predictions(X_valid, use_proba=True)
        
        with torch.no_grad():
            X_meta_tensor = torch.FloatTensor(meta_features_valid).to(config.DEVICE)
            y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1).to(config.DEVICE)
            outputs = self.meta_learner(X_meta_tensor)
            loss = self.criterion(outputs, y_valid_tensor)
        return loss.item()
    
    def predict(self, X):
        """Make ensemble predictions"""
        self.meta_learner.eval()
        
        # Get base model predictions
        meta_features = self._get_base_predictions(X, use_proba=True)
        
        with torch.no_grad():
            X_meta_tensor = torch.FloatTensor(meta_features).to(config.DEVICE)
            outputs = self.meta_learner(X_meta_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X):
        """Predict ensemble probabilities"""
        self.meta_learner.eval()
        
        # Get base model predictions
        meta_features = self._get_base_predictions(X, use_proba=True)
        
        with torch.no_grad():
            X_meta_tensor = torch.FloatTensor(meta_features).to(config.DEVICE)
            outputs = self.meta_learner(X_meta_tensor)
            probas = outputs.cpu().numpy().flatten()
        
        return probas
    
    def save(self, filepath):
        """Save ensemble model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'meta_learner_state_dict': self.meta_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'n_models': self.n_models
        }, filepath)
        print(f"💾 {self.model_name} saved to: {filepath}")
    
    def load(self, filepath):
        """Load ensemble model from disk"""
        checkpoint = torch.load(filepath)
        self.meta_learner.load_state_dict(checkpoint['meta_learner_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.meta_learner.to(config.DEVICE)
        print(f"📂 {self.model_name} loaded from: {filepath}")


def create_ensemble(base_models, **kwargs):
    """Factory function to create Ensemble Stacking model"""
    return EnsembleStackingModel(base_models=base_models, **kwargs)
