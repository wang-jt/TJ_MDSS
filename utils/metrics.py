import torchmetrics

__DEBUG__ = False


class F1ScoreMetric(torchmetrics.F1Score):
    def __init__(self, average, num_classes, multiclass, threshold, **metric_args):

        metrics_args = {"average":average, "num_classes":num_classes, "multiclass":multiclass, "threshold":threshold}

        super().__init__(**metrics_args)
