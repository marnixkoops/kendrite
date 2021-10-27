"""Define, fit, and predict neural tabular models."""

import logging
import sys
from typing import Any, Dict, List, Union

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from kedro.utils import load_obj

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO,)
logger = logging.getLogger(" 🧠 kendrite")

def neural_model(params:dict) -> Union[TabNetRegressor, TabNetClassifier]:
    """
    Loads a regressor or classifier object based on given parameters.
    Args:
        params: dictionary of parameters
    Returns:
        compatible model
    """
    params["kwargs"]["optimizer_fn"]= load_obj(params.get("kwargs", {}).get("optimizer_fn", "torch.optim.Adam"))
    params["kwargs"]["scheduler_fn"] = load_obj(params.get("kwargs", {}).get("scheduler_fn", "torch.optim.lr_scheduler.StepLR"))
    task = params.get("task", "regression")
    if task == "regression":
        model = TabNetRegressor(**params.get("kwargs", {}))
    elif task == "classification":
        model = TabNetClassifier(**params.get("kwargs", {}))

    return model


def fit(
    model: Union[TabNetRegressor, TabNetClassifier],
    X_train: np.array,
    y_train: np.array,
    X_valid: np.array = None,
    y_valid: np.array = None,
    eval_set: List[tuple] = None,
    eval_name: List[str] = None,
    eval_metric: List[str] = None,
    loss_fn=None,
    weights: bool = 0,
    max_epochs: int = 64,
    patience: int = 64,
    batch_size: int = 256,
) -> Union[TabNetRegressor, TabNetClassifier]:
    """Train a neural tabular regression or classification model.

    Args:
        model: Neural tabular regression or classification model
        X_train: Training feature data.
        y_train: Training target data.
        X_valid: Validation feature data.
        y_valid: Validation target data.
        eval_set: List of eval tuple set (X, y). The last one is used for early stopping
        eval_name: List of eval set names.
        eval_metric: List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn: Loss function for training (default to mse for regression and
            cross-entropy for classification
        weights: Only for classifier. 0: no sampling 1: automated sampling with inverse
            class occurrences.
        max_epochs: Maximum number of epochs for trainng.
        patience: Number of consecutive epochs without improvement before performing
            early stopping. If patience is set to 0 then no early stopping will be
            performed. Note that if patience is enabled, best weights from best epoch
            will automatically be loaded at the end of fit.
        batch_size: Number of examples per batch, large batch sizes are recommended.

    Returns:
        model: Trained neural tabular regression or classification model.
    """
    if type(model) == TabNetRegressor:
        y_train = y_train.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)

    logger.info(f"Training {type(model)} model for max {max_epochs} epochs.")
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=eval_metric,
        weights=weights,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
    )
    return model


def predict(
    model: Union[TabNetRegressor, TabNetClassifier], X_test: np.array
) -> np.array:
    """Predict on a test set using a neural tabular regression or classification model.

    Args:
        model: Trained neural tabular regression or classification model.
        X_test: Test feature data.

    Returns:
        y_pred: Predictions on test set.
    """
    logger.info(f"Predicting on {len(X_test)} samples in test set.")
    y_pred = model.predict(X_test)
    return y_pred
