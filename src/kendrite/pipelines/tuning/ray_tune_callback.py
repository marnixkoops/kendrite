import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Union

from pytorch_tabnet.callbacks import Callback
from ray import tune

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO,)
logger = logging.getLogger(" 🧠 kendrite")


class TuneCallback(Callback):
    """Base class for Tune's Tabnet callbacks."""

    _allowed = [
        "epoch_begin",
        "epoch_end",
        "batch_begin",
        "batch_end",
        "train_begin",
        "train_end",
    ]

    def __init__(self, on: Union[str, List[str]] = "epoch_end"):
        if not isinstance(on, list):
            on = [on]
        if any(w not in self._allowed for w in on):
            raise ValueError(
                "Invalid trigger time selected: {}. Must be one of {}".format(
                    on, self._allowed
                )
            )
        self._on = on

    def on_epoch_begin(self, epoch, logs=None):
        if "epoch_begin" in self._on:
            self._handle()

    def on_epoch_end(self, epoch, logs=None):
        if "epoch_end" in self._on:
            self._handle()

    def on_batch_begin(self, batch, logs=None):
        if "batch_begin" in self._on:
            self._handle()

    def on_batch_end(self, batch, logs=None):
        if "batch_end" in self._on:
            self._handle()

    def on_train_begin(self, logs=None):
        if "train_begin" in self._on:
            self._handle()

    def on_train_end(self, logs=None):
        if "train_end" in self._on:
            self._handle()


class TuneReportCallback(TuneCallback):
    """Tabnet to Ray Tune reporting callback

    Reports metrics to Ray Tune.

    Args:
        metrics (str|list|dict): Metrics to report to Tune. If this is a list,
            each item describes the metric key reported to ray,
            and it will reported under the same name to Tune. If this is a
            dict, each key will be the name reported to Tune and the respective
            value will be the metric key reported to ray.
        on (str|list): When to trigger checkpoint creations. Must be one of
            the tabnet event hooks (less the ``on_``), e.g.
            "batch_begin", or "train_end". Defaults to "epoch_end".

    Example:

    .. code-block:: python

        from kendrite.pipelines.tuning import TuneReportCallback


    """

    def __init__(
        self,
        metrics: Union[None, str, List[str], Dict[str, str]] = None,
        on: Union[str, List[str]] = "epoch_end",
    ):
        super(TuneReportCallback, self).__init__(on)
        if isinstance(metrics, str):
            metrics = [metrics]
        self._metrics = metrics

    def _get_report_dict(self):
        if not self._metrics:
            report_dict = {
                k: v[-1]
                for k, v in self._callback_container.callbacks[0].history.items()
            }
        else:
            report_dict = {}
            for key in self._metrics:
                if isinstance(self._metrics, dict):
                    metric = self._metrics[key]
                else:
                    metric = key
                if metric in self.trainer._metrics_names:
                    report_dict[key] = self._callback_container.callbacks[0].history[
                        metric
                    ][-1]
                else:
                    logger.warning(
                        f"Metric {metric} does not exist in " "`trainer._metric_names."
                    )

        return report_dict

    def _handle(self):
        report_dict = self._get_report_dict()
        if report_dict is not None:
            tune.report(**report_dict)
