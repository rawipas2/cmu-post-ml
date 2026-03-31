"""
Sanity checks for the factorial experiment pipeline.
"""
import numpy as np
import torch

import config
from models import bayesian_network_model, deep_learning_model, maximum_entropy_model, neural_network_model
from utils.focal_loss import FocalLoss


def assert_focal_matches_bce_when_gamma_zero():
    logits = torch.randn(16, 1, device=config.DEVICE)
    targets = torch.randint(0, 2, (16, 1), device=config.DEVICE).float()

    focal = FocalLoss(alpha=None, gamma=0.0)
    bce = torch.nn.BCEWithLogitsLoss()

    focal_value = focal(logits, targets).item()
    bce_value = bce(logits, targets).item()
    assert abs(focal_value - bce_value) < 1e-6, (focal_value, bce_value)


def run_model_toggle_checks():
    np.random.seed(42)
    X_train = np.random.randn(64, 32).astype(np.float32)
    y_train = np.random.randint(0, 2, 64)
    X_valid = np.random.randn(16, 32).astype(np.float32)
    y_valid = np.random.randint(0, 2, 16)
    X_test = np.random.randn(8, 32).astype(np.float32)

    factories = [
        lambda use_focal_loss: neural_network_model.create_model(
            input_dim=32,
            hidden_dims=[32, 16],
            epochs=1,
            dropout=0.1,
            use_focal_loss=use_focal_loss
        ),
        lambda use_focal_loss: deep_learning_model.create_model(
            input_dim=32,
            hidden_dims=[32, 16, 8],
            epochs=1,
            dropout=0.1,
            use_focal_loss=use_focal_loss
        ),
        lambda use_focal_loss: bayesian_network_model.create_model(
            input_dim=32,
            hidden_dims=[16, 8],
            epochs=1,
            use_focal_loss=use_focal_loss
        ),
        lambda use_focal_loss: maximum_entropy_model.create_model(
            input_dim=32,
            epochs=1,
            use_class_weight=False,
            use_focal_loss=use_focal_loss
        ),
    ]

    for factory in factories:
        for use_focal_loss in (False, True):
            model = factory(use_focal_loss)
            model.train(X_train, y_train, X_valid, y_valid)
            probas = model.predict_proba(X_test)
            assert probas.shape[0] == X_test.shape[0]
            assert np.all((probas >= 0.0) & (probas <= 1.0))


def main():
    assert_focal_matches_bce_when_gamma_zero()
    run_model_toggle_checks()
    print("Factorial sanity checks passed.")


if __name__ == '__main__':
    main()
