"""Data processing and engineering."""

import logging
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO,)
logger = logging.getLogger(" 🧠 kendrite")


def select_features(data: pd.DataFrame, target: str, exclude_cols: List = None) -> List:
    features = [col for col in data.columns if col not in target]
    if exclude_cols:
        features = [col for col in features if col not in exclude_cols]
    logger.info(f"Selected features: {features}.")
    return features


def split_data(
    data: pd.DataFrame,
    target: str,
    features: List,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 0,
) -> np.array:
    """Splits data into train, validation, and test set."""
    logger.info(f"Splitting data with {test_size} test and {val_size} val ratio.")
    train_val_indices, test_indices = train_test_split(
        range(len(data)), test_size=test_size, random_state=seed
    )

    train_indices, valid_indices = train_test_split(
        train_val_indices, test_size=val_size, random_state=seed
    )

    X_train = data[features].values[train_indices]
    y_train = data[target].values[train_indices]

    X_valid = data[features].values[valid_indices]
    y_valid = data[target].values[valid_indices]

    X_test = data[features].values[test_indices]
    y_test = data[target].values[test_indices]

    return X_train, y_train, X_valid, y_valid, X_test, y_test
