"""
Bayesian Network Model (Neural Network with Bayesian approach on GPU)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import config


class BayesianLayer(nn.Module):
    """Bayesian Linear Layer with variational inference"""
    
    def __init__(self, in_features, out_features):
        super(BayesianLayer, self).__init__()
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))
        
    def forward(self, x, sample=True):
        if self.training or sample:
            # Sample weights
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight_epsilon = torch.randn_like(weight_sigma)
            weight = self.weight_mu + weight_sigma * weight_epsilon
            
            # Sample biases
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_epsilon = torch.randn_like(bias_sigma)
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return nn.functional.linear(x, weight, bias)


class BayesianNetworkClassifier(nn.Module):
    """Bayesian Neural Network"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(BayesianNetworkClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLayer(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(BayesianLayer(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BayesianNetworkModel:
    """Bayesian Network wrapper with training utilities"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64],
                 learning_rate=None, epochs=None):
        """
        Initialize Bayesian Network model
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
        """
        self.model = BayesianNetworkClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims
        ).to(config.DEVICE)
        
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.epochs = epochs or config.EPOCHS
        self.model_name = 'Bayesian_Network'
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the Bayesian network"""
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
    
    def predict(self, X, n_samples=10):
        """Make predictions with uncertainty estimation"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(config.DEVICE)
        
        # Multiple forward passes for uncertainty
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions.append(outputs.cpu().numpy())
        
        # Average predictions
        mean_pred = np.mean(predictions, axis=0)
        final_pred = (mean_pred > 0.5).astype(int).flatten()
        return final_pred
    
    def predict_proba(self, X, n_samples=10):
        """Predict probabilities with uncertainty estimation"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(config.DEVICE)
        
        # Multiple forward passes
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions.append(outputs.cpu().numpy())
        
        # Return mean probability
        mean_prob = np.mean(predictions, axis=0).flatten()
        return mean_prob
    
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
    """Factory function to create Bayesian Network model"""
    return BayesianNetworkModel(input_dim=input_dim, **kwargs)
