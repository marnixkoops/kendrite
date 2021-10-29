# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""
Nodes for tuning tabnet models
"""
# pylint: disable=no-name-in-module, too-many-locals
import logging
import sys
from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd
import ray
from kedro.utils import load_obj
from ray import tune

from ..model.nodes import fit, neural_model

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO,)
logger = logging.getLogger(" 🧠 kendrite")


def _replace_value(obj, key, value):
    if key in obj:
        obj[key] = value
        return True
    for k, v in obj.items():
        if isinstance(v, dict):
            if _replace_value(v, key, value):
                return True


def map_config_params(params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map config settings to params dictionary.

    This updated parameter set allows us to potentially
    search over a complete parameter set of neural net architecture parameters,
    and optimizer parameters.

    Args:
        params: Initial parameter dictionary defining how to construct our DL model.
        config: The worker's configuration parameters, which we want to test
        performance of. This is a dictionary created from the call to `ray.tune.run`

    Returns: updated params dictionary.

    """
    tune_params = params.pop("tune_params", {})
    for key, val in config.items():
        _replace_value(params, key, val)
    params["tune_params"] = tune_params
    return params


def create_tabnet_trainer(
    X_train: np.array,
    y_train: np.array,
    X_valid: np.array = None,
    y_valid: np.array = None,
    params: Dict[str, Any] = None,
    config_mapper: Callable = map_config_params,
) -> Callable:
    """Creates a function that maps a config into a trained model

    Args:
        X_train (np.array): Training feature data.
        y_train (np.array): Training target data.
        X_valid (np.array, optional): Validation feature data. Defaults to None.
        y_valid (np.array, optional): Validation target data. Defaults to None.
        params (Dict[str, Any], optional): params for tabnet. Defaults to None.
        config_mapper (Callable, optional): maps tune config to tabnet config.
            Defaults to map_config_params.

    Returns:
        Callable: trainer function
    """

    def tabnet_trainer(config):
        mapped_params = config_mapper(params, config)
        model = neural_model(mapped_params["estimator"])
        fit(
            model, X_train, y_train, X_valid, y_valid, mapped_params["fit"], tuning=True
        )

    return tabnet_trainer


def tune_tabnet(
    tune_params: Dict[str, Any], tabnet_trainer: Callable
) -> Dict[str, Union[Dict[str, Any], pd.DataFrame]]:
    """
    Leveraging `ray.tune` for distributed hyper-parameter tuning.

    This function creates a tuning function that creates and begins
    training a configurable neural network model. Once the best configuration
    of hyper-parameters is found, it builds a model with those parameters, and
    returns it along with the best configuration and a dataframe log of the
    details behind all of the trials.

    Some schedulers and search algorithms available within `tune` require
    additional keyword arguments which are not straight-forward expressed
    within a parameters.yaml config file. Concretely, when using
    population based training,, the scheduler requires the specification
    of hyperparam_mutations, which expects a dict of param: Callable
    key:value pairs. The `ray` package is a convenient interface to many
    other packages for hyper-parameter tuning, the user is encouraged to
    check out https://docs.ray.io/en/master/tune/api_docs/suggestion.html
    and https://docs.ray.io/en/master/tune/api_docs/schedulers.html for
    more information.

    By default, if no arguments for a search_alg or scheduler are present
    within config, default to random/grid search.

    Args:
        tune_params: A dictionary of parameters for training and fitting our DL model
        tabnet_trainer: Function to be passed to `ray.tune`. Likely output of
        config_mapper: dictionary mapping params to config param name.

    Returns: dictionary with keys:
        - best_config: Dict/JSON of best configurations.
        - tune_results: Dataframe of tuning configurations and performance results.
    """

    analysis = tune.run(tabnet_trainer, **tune_params)
    best_config = analysis.get_best_config(
        metric=tune_params.get("metric"), mode=tune_params.get("mode")
    )
    return dict(best_config=best_config, tune_results=analysis.dataframe())


def prepare_tune_params(tune_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare dictionary of params for usage with ray[tune].

    Args:
        tune_params: Input dictionary of params

    Returns: Params with objects replaced by loaded objects where
    required.

    """
    # Add documentation for where this could be replaced by teams.
    try:
        if tune_params.get("scheduler") is not None:
            scheduler = load_obj(tune_params["scheduler"]["class"])(
                **tune_params["scheduler"].get("kwargs", {})
            )
            tune_params["scheduler"] = scheduler
    except ray.tune.error.TuneError:
        raise Exception(
            "Please double check that the scheduler you "
            "are using has all of the appropriate "
            "arguments, either defined in "
            "`tune_keras_estimator` or in parameters.yml"
        )
    try:
        if tune_params.get("search_alg") is not None:
            search_alg = load_obj(tune_params["search_alg"]["class"])(
                **tune_params["search_alg"].get("kwargs", {})
            )
            tune_params["search_alg"] = search_alg

    except (ray.tune.error.TuneError, ImportError):
        raise Exception(
            "Please double check that the search algorithm you "
            "would like to use is a) installed, b) has the "
            "appropriate arguments, either defined in code at  "
            "`tune_keras_estimator` or in parameters.yml"
        )
    # callbacks list
    if tune_params.get("callbacks", []):
        for idx, callback_config in enumerate(tune_params.get("callbacks")):
            tune_params["callbacks"][idx] = load_obj(callback_config["class"])(
                **callback_config.get("kwargs", {})
            )
    # create search config space
    if tune_params.get("config", {}):
        tune_params["config"] = _create_search_space(tune_params.get("config", {}))
    return tune_params


def _create_search_space(search_space_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Within search_space, we've defined the parameter grid we'll be optimizing.
    Let's load the appropriate samplers.
    Args:
        search_space_config:

    Returns:

    """
    for k, v in search_space_config.items():
        search_space_config[k] = load_obj(v["class"])(**v["kwargs"])
    return search_space_config


def get_estimator_config(
    params: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get estimator config from config.

    Args:
        config:

    Returns:

    """
    params = map_config_params(params, config)
    return params["estimator"]
