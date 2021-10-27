"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from kendrite.pipelines import engineering, evaluation, model


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    engineering_pipeline = engineering.create_data_pipeline()
    model_pipeline = model.create_model_pipeline()
    evaluation_pipeline = evaluation.create_evaluation_pipeline()

    return {
        "engineering": engineering_pipeline,
        "model": model_pipeline,
        "evaluation": evaluation_pipeline,
        "__default__": engineering_pipeline + model_pipeline + evaluation_pipeline,
    }
