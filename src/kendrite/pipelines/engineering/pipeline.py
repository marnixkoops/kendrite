from kedro.pipeline import Pipeline, node

from .nodes import select_features, split_data


def create_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=select_features,
                inputs=["wine", "params:target", "params:exclude_cols"],
                outputs="features",
                name="select_features",
            ),
            node(
                func=split_data,
                inputs=[
                    "wine",
                    "params:target",
                    "features",
                    "params:test_size",
                    "params:val_size",
                    "params:seed",
                ],
                outputs=[
                    "X_train",
                    "y_train",
                    "X_valid",
                    "y_valid",
                    "X_test",
                    "y_test",
                ],
                name="split_data",
            ),
        ]
    )
