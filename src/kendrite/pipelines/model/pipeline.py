from kedro.pipeline import Pipeline, node

from .nodes import fit, predict, neural_model


def create_model_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=neural_model,
                inputs=[
                    "params:estimator"
                ],
                outputs="model",
                name="regressor",
            ),
            node(
                func=fit,
                inputs=["model", "X_train", "y_train", "X_valid", "y_valid"],
                outputs="trained_model",
                name="fit",
            ),
            node(
                func=predict,
                inputs=["trained_model", "X_test"],
                outputs="y_pred",
                name="predict",
            ),
        ]
    )
