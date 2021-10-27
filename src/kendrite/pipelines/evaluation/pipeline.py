from kedro.pipeline import Pipeline, node

from .nodes import (
    plot_feature_importances,
    plot_feature_masks,
    plot_model_history,
    report_metrics,
)


def create_evaluation_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=report_metrics,
                inputs=["trained_model", "y_pred", "y_test"],
                outputs="test_metric",
                name="report_metrics",
            ),
            node(
                func=plot_model_history,
                inputs="trained_model",
                outputs=None,
                name="plot_model_history",
            ),
            node(
                func=plot_feature_importances,
                inputs=["trained_model", "features"],
                outputs=None,
                name="plot_feature_importances",
            ),
            node(
                func=plot_feature_masks,
                inputs=["trained_model", "X_test", "features"],
                outputs=None,
                name="plot_feature_masks",
            ),
        ]
    )
