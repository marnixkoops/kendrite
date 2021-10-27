"""Evaluate neural tabular models."""

import logging
import sys
from typing import List, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

sns.set_style("darkgrid")
sns.set_palette("crest")
sns.set(rc={"figure.dpi": 200, "savefig.dpi": 200, "figure.figsize": (12, 8)})
sns.set_context("notebook", font_scale=0.7)


logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(" 🧠 kendrite")


def report_metrics(
    model: Union[TabNetRegressor, TabNetClassifier],
    y_pred: np.ndarray,
    y_test: np.ndarray,
) -> float:
    logger.info(f"Best validation score: {model.best_cost}")
    logger.info("Computing error on test set.")
    if type(model) == TabNetRegressor:
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Mean squared error on test set: {mse}.")
        return mse
    elif type(model) == TabNetClassifier:
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy on test set: {accuracy}.")
        return accuracy


def plot_model_history(
    model: Union[TabNetRegressor, TabNetClassifier], show: bool = False
):
    fig, axs = plt.subplots(3)
    fig.suptitle("Model History")
    sns.lineplot(
        data=model.history["train_accuracy"], label="Train accuracy", ax=axs[0]
    )
    sns.lineplot(
        data=model.history["valid_accuracy"], label="Valid accuracy", ax=axs[0]
    )
    sns.lineplot(data=model.history["loss"], label="Loss", ax=axs[1])
    sns.lineplot(data=model.history["lr"], label="Learning Rate", ax=axs[2])
    plt.xlabel("Epoch")
    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_feature_importances(
    model: Union[TabNetRegressor, TabNetClassifier], features: List, show: bool = False
):
    feature_importances = pd.DataFrame(
        zip(features, model.feature_importances_), columns=["feature", "importance"]
    ).sort_values(by="importance", ascending=False)

    fig, axs = plt.subplots(1)
    sns.barplot(
        x="importance",
        y="feature",
        hue="importance",
        data=feature_importances,
        palette="crest",
        dodge=False,
    )
    plt.title("Global Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.legend("")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_feature_masks(
    model: Union[TabNetRegressor, TabNetClassifier],
    X_test: np.ndarray,
    features: List,
    show: bool = False,
):
    explain_matrix, masks = model.explain(X_test)
    n_masks = model.n_steps

    fig, axs = plt.subplots(1, n_masks, figsize=(12, 8))
    axs[0].set_ylabel("Instance")
    for i in range(model.n_steps):
        axs[i].imshow(masks[i][:35], cmap="crest")
        axs[i].set_title(f"Feature Mask {i+1}")
        axs[i].set_yticks([])
        axs[i].set_xticks(np.arange(len(features)))
        axs[i].set_xticklabels(features, rotation=90)
        axs[i].grid(False)

    if show:
        plt.show()

    return fig
