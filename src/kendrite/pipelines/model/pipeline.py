from kedro.pipeline import Pipeline, node

from .nodes import fit, neural_model, predict


def create_model_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=neural_model,
                inputs=["params:train_tabnet.estimator"],
                outputs="model",
                name="neural_model",
            ),
            node(
                func=fit,
                inputs=[
                    "model",
                    "X_train",
                    "y_train",
                    "X_valid",
                    "y_valid",
                    "params:train_tabnet.fit",
                ],
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
