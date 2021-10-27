from kedro.pipeline import Pipeline, node

from .nodes import fit, neural_classifier, neural_regressor, predict


def create_model_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=neural_classifier,
                inputs=None,
                # inputs=[
                #     "params:n_decision",
                #     "params:n_attention",
                #     "params:n_steps",
                #     "params:gamma",
                #     #     "params:cat_idxs",
                #     #     "params:cat_dims",
                #     #     "params:cat_emb_dim",
                #     "params:n_independent",
                #     "params:n_shared",
                #     "params:epsilon",
                #     "params:momentum",
                #     "params:lambda_sparse",
                #     "params:seed",
                #     "params:clip_value",
                #     "params:verbose",
                #     # "params:optimizer_fn",
                #     # "params:optimizer_params",
                #     # "params:scheduler_fn",
                #     # "params:scheduler_params",
                #     "params:mask_type",
                #     "params:device_name",
                # ],
                outputs="model",
                name="classifier",
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
