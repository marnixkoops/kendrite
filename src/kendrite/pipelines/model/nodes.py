"""Define, fit, and predict neural tabular models."""

import logging
import sys
from typing import Any, Dict, List, Union

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO,)
logger = logging.getLogger(" 🧠 kendrite")


def neural_regressor(
    n_decision: int = 8,
    n_attention: int = 8,
    n_steps: int = 3,
    gamma: float = 1.3,
    cat_idxs: List[int] = None,
    cat_dims: List[int] = None,
    cat_emb_dim: int = None,
    n_independent: int = 2,
    n_shared: int = 2,
    epsilon: float = 1e-15,
    momentum: float = 0.02,
    lambda_sparse: float = 1e-3,
    seed: int = 0,
    clip_value: int = 1,
    verbose: int = 1,
    optimizer_fn: Any = torch.optim.Adam,
    optimizer_params: Dict = dict(lr=2e-2),
    scheduler_fn: Any = torch.optim.lr_scheduler.StepLR,
    scheduler_params: Dict = {"gamma": 0.95, "step_size": 20},
    mask_type: str = "sparsemax",
    device_name: str = "auto",
) -> TabNetRegressor:
    """Define a neural tabular regression model.

    PyTorch implementation of "TabNet: Attentive Interpretable Tabular Learning"
    by Sercan O. Arik & Tomas Pfister. https://arxiv.org/abs/1908.07442

    Args:
        n_decision: Width of the decision prediction layer. Bigger values gives more
            capacity to the model with the risk to overfit. Usually ranges from 8 to 64.
        n_attention: Width of the attention embedding for each mask. According to the
            paper n_decision=n_attentionttention is usually a good choice.
        n_steps: Number of steps in the architecture. Usually between 3 and 10.
        gamma: This is the coefficient for feature reusage in the masks. A value close
            to 1 will make mask selection least correlated between layers.
            Values range from 1.0 to 2.0.
        cat_idxs: List of categorical features indices.
        cat_dims: List of categorical features number of modalities. (Number of unique
            values for a categorical feature).
        cat_emb_dim: List of embeddings size for each categorical features.
        n_independent: Number of independent Gated Linear Units layers at each step.
            Usual values range from 1 to 5.
        n_shared: Number of shared Gated Linear Units at each step.
            Usual values range from 1 to 5.
        epsilon: Learning rate. Should be left untouched.
        momentum: Momentum for batch normalization. Typically ranges from 0.01 to 0.4.
        lambda_sparse: This is the extra sparsity loss coefficient as proposed in the
            original paper. The bigger this coefficient is, the sparser your model will
            be in terms of feature selection. Depending on the difficulty of your
            problem, reducing this value could help.
        seed: Random seed for reproducibility.
        clip_value: If a float is given this will clip the gradient at clip_value.
        verbose: Verbosity, set to 1 to see every epoch, 0 to get None.
        optimizer_fn: Pytorch optimizer function.
        optimizer_params: Parameters compatible with optimizer_fn used initialize the
            optimizer. Since we have Adam as our default optimizer, we use this to
            define the initial learning rate used for training. As mentionned in the
            original paper, a large initial learning of 0.02 with decay is a good start.
        scheduler_fn: Pytorch Scheduler to change learning rates during training.
        scheduler_params: Dictionnary of parameters to apply to the scheduler_fn.
            For example: {"gamma": 0.95, "step_size": 10}
        mask_type: Either "sparsemax" or "entmax" : this is the masking function to use
            for selecting features
        device_name: "cpu" for cpu training, "gpu" for gpu training or "auto".

    Returns:
        TabNetRegressor: Neural tabular regression model.
    """
    logger.info("Defining neural regressor.")
    model = TabNetRegressor(
        n_d=n_decision,
        n_a=n_attention,
        n_steps=n_steps,
        gamma=gamma,
        n_independent=n_independent,
        n_shared=n_shared,
        epsilon=epsilon,
        momentum=momentum,
        lambda_sparse=lambda_sparse,
        seed=seed,
        clip_value=clip_value,
        verbose=verbose,
        optimizer_fn=optimizer_fn,
        optimizer_params=optimizer_params,
        scheduler_fn=scheduler_fn,
        scheduler_params=scheduler_params,
        mask_type=mask_type,
        device_name=device_name,
    )
    return model


def neural_classifier(
    n_decision: int = 8,
    n_attention: int = 8,
    n_steps: int = 3,
    gamma: float = 1.3,
    n_independent: int = 2,
    n_shared: int = 2,
    epsilon: float = 1e-15,
    momentum: float = 0.02,
    lambda_sparse: float = 1e-3,
    seed: int = 0,
    clip_value: int = 1,
    verbose: int = 1,
    optimizer_fn: Any = torch.optim.Adam,
    optimizer_params: Dict = dict(lr=2e-2),
    mask_type: str = "sparsemax",
    device_name: str = "auto",
) -> TabNetClassifier:
    """Define a neural tabular classification model.

    PyTorch implementation of "TabNet: Attentive Interpretable Tabular Learning"
    by Sercan O. Arik & Tomas Pfister. https://arxiv.org/abs/1908.07442

    Args:
        n_decision: Width of the decision prediction layer. Bigger values gives more
            capacity to the model with the risk to overfit. Usually ranges from 8 to 64.
        n_attention: Width of the attention embedding for each mask. According to the
            paper n_decision=n_attentionttention is usually a good choice.
        n_steps: Number of steps in the architecture. Usually between 3 and 10.
        gamma: This is the coefficient for feature reusage in the masks. A value close
            to 1 will make mask selection least correlated between layers.
            Values range from 1.0 to 2.0.
        cat_idxs: List of categorical features indices.
        cat_dims: List of categorical features number of modalities. (Number of unique
            values for a categorical feature).
        cat_emb_dim: List of embeddings size for each categorical features.
        n_independent: Number of independent Gated Linear Units layers at each step.
            Usual values range from 1 to 5.
        n_shared: Number of shared Gated Linear Units at each step.
            Usual values range from 1 to 5.
        epsilon: Learning rate. Should be left untouched.
        momentum: Momentum for batch normalization. Typically ranges from 0.01 to 0.4.
        lambda_sparse: This is the extra sparsity loss coefficient as proposed in the
            original paper. The bigger this coefficient is, the sparser your model will
            be in terms of feature selection. Depending on the difficulty of your
            problem, reducing this value could help.
        seed: Random seed for reproducibility.
        clip_value: If a float is given this will clip the gradient at clip_value.
        verbose: Verbosity, set to 1 to see every epoch, 0 to get None.
        optimizer_fn: Pytorch optimizer function.
        optimizer_params: Parameters compatible with optimizer_fn used initialize the
            optimizer. Since we have Adam as our default optimizer, we use this to
            define the initial learning rate used for training. As mentionned in the
            original paper, a large initial learning of 0.02 with decay is a good start.
        scheduler_fn: Pytorch Scheduler to change learning rates during training.
        scheduler_params: Dictionnary of parameters to apply to the scheduler_fn.
            For example: {"gamma": 0.95, "step_size": 10}
        mask_type: Either "sparsemax" or "entmax" : this is the masking function to use
            for selecting features
        device_name: "cpu" for cpu training, "gpu" for gpu training or "auto".

    Returns:
        TabNetClassifier: Neural tabular classification model.
    """
    logger.info("Defining neural classifier.")
    model = TabNetClassifier(
        n_d=48,
        n_a=48,
        n_steps=6,
        gamma=1.5,
        n_independent=n_independent,
        n_shared=n_shared,
        epsilon=epsilon,
        momentum=0.3,
        lambda_sparse=lambda_sparse,
        seed=seed,
        clip_value=clip_value,
        verbose=verbose,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=optimizer_params,
        mask_type=mask_type,
        device_name=device_name,
    )
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