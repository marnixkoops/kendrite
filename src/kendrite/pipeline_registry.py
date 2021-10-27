"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from kendrite.pipelines import engineering, evaluation, model, tune_model


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    engineering_pipeline = engineering.create_data_pipeline()
    model_pipeline = model.create_model_pipeline()
    evaluation_pipeline = evaluation.create_evaluation_pipeline()
    tune_pipeline = tune_model.create_pipeline()
    model_pipeline_from_tune = model.create_model_pipeline(
        from_hyperparameter_tuning=True
    )

    return {
        "engineering": engineering_pipeline,
        "model": model_pipeline,
        "evaluation": evaluation_pipeline,
        "tune": tune_pipeline,
        "tune_and_train": engineering_pipeline
        + tune_pipeline
        + model_pipeline_from_tune
        + evaluation_pipeline,
        "__default__": engineering_pipeline + model_pipeline + evaluation_pipeline,
    }
