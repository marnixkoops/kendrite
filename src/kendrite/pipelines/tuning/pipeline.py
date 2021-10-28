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
Model Tuning Pipeline
"""
from functools import partial
from typing import Callable

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_tabnet_trainer,
    get_estimator_config,
    map_config_params,
    prepare_tune_params,
    tune_tabnet,
)


def create_pipeline(config_mapper: Callable = map_config_params) -> Pipeline:
    """
    This pipeline expects a params block with the following schema ::

        train_tabnet:
            fit:
                eval_metric: ["accuracy", "balanced_accuracy", "logloss"]
                weights: 1
                max_epochs: 64
                patience: 64
                batch_size: 256
                callbacks:
                    - class: kendrite.pipelines.tuning.TuneReportCallback
                      kwargs:
                          metrics: "val_logloss"
                          "on": 'epoch_end'
            tune_params:
                metric: val_logloss
                mode: min
                local_dir: data/07_model_output/ray_tune_results/ # store tune results
                fail_fast: True
                config:
                  n_d:
                    class: ray.tune.randint
                    kwargs:
                        lower: 4
                        upper: 64
                  n_a:
                    class: ray.tune.randint
                    kwargs:
                        lower: 4
                        upper: 64
                  n_steps:
                    class: ray.tune.randint
                    kwargs:
                        lower: 3
                        upper: 10
                  gamma:
                    class: ray.tune.uniform
                    kwargs:
                        lower: !!float 1.0
                        upper: !!float 2.0
                scheduler:
                    class: ray.tune.schedulers.AsyncHyperBandScheduler
                    kwargs:
                        time_attr: training_iteration
                        max_t: 40
                        grace_period: 20
                search_alg:
                    class: ray.tune.suggest.optuna.OptunaSearch
                    kwargs: {}
                # callbacks: - List[Dict]. Each dict-> class and kwarg: Dict[str,str]
                num_samples: 8
                verbose: 1 # level of messages to print
                stop:
                    training_iteration: 10
                resources_per_trial:
                    cpu: 1
                    gpu: 0

    Returns:

    """
    return pipeline(
        Pipeline(
            [
                node(
                    partial(create_tabnet_trainer, config_mapper=config_mapper),
                    inputs=dict(
                        X_train="X_train",
                        y_train="y_train",
                        X_valid="X_valid",
                        y_valid="y_valid",
                        params="params:train_tabnet",
                    ),
                    outputs="tabnet_trainer",
                    name="create_tabnet_trainable",
                ),
                node(
                    prepare_tune_params,
                    inputs="params:train_tabnet.tune_params",
                    outputs="tune_params",
                    name="prepare_params_for_ray_tune",
                ),
                node(
                    tune_tabnet,
                    dict(tabnet_trainer="tabnet_trainer", tune_params="tune_params"),
                    dict(best_config="best_config", tune_results="tune_results"),
                    name="tune_keras_estimator",
                ),
                node(
                    get_estimator_config,
                    inputs=dict(params="params:train_tabnet", config="best_config"),
                    outputs="estimator_config",
                    name="get_estimator_config",
                ),
            ]
        )
    )
