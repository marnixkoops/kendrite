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
sns.set_palette("flare")
sns.set(rc={"figure.dpi": 200, "savefig.dpi": 200, "figure.figsize": (12, 8)})


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
    model: Union[TabNetRegressor, TabNetClassifier], show: bool = True
) -> None:
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


def plot_feature_importances(
    model: Union[TabNetRegressor, TabNetClassifier], features: List, show: bool = True
) -> None:
    feature_importances = pd.DataFrame(
        zip(features, model.feature_importances_), columns=["feature", "importance"]
    ).sort_values(by="importance", ascending=False)

    fig, axs = plt.subplots(1)
    sns.barplot(
        x="importance",
        y="feature",
        hue="importance",
        data=feature_importances,
        palette="flare",
        dodge=False,
    )
    plt.title("Global Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.legend("")
    plt.tight_layout()
    if show:
        plt.show()
