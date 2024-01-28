import time

import pandas as pd


class EvenflowMetricsTracker:
    def __init__(self, name: str, **kwargs):
        self.name: str = name
        self.cur_round: int = 0
        self.metrics_root: str = kwargs["metrics_root"]
        self.metrics_path: str = f"{self.metrics_root}/{self.name}.csv"
        self.metrics_per_round: list[dict[str, any]] = []

    def _make_entry(self, metric_name: str, metric_value: any) -> dict[str, any]:
        metric_item: dict[str, any] = {
            "ts": time.time(),
            "round": self.cur_round,
            "node_name": self.name,
            "metric_name": metric_name,
            "metrics_value": metric_value,
        }
        self.metrics_per_round.append(metric_item)

    def log_round(self, cur_round: int) -> None:
        self.cur_round = cur_round
        self._make_entry("round_change", self.cur_round)

    def log_training_loss(self, loss_val: float) -> None:
        self._make_entry("training_loss", loss_val)

    def log_test_loss(self, loss_val: float) -> None:
        self._make_entry("test_loss", loss_val)

    def log_epoch(self, epoch: int) -> None:
        self._make_entry("epoch", epoch)

    def log_pushed_model_size(self, model_size: int) -> None:
        self._make_entry("pushed_model_size", model_size)

    def log_batch(self, batch_idx: int) -> None:
        self._make_entry("batch_idx", batch_idx)

    def log_processed_items(self, processed_items: int) -> None:
        self._make_entry("processed_items", processed_items)

    def log_model_was_pushed(self, model_was_pushed: bool) -> None:
        self._make_entry("model_was_pushed", model_was_pushed)

    def log_round_was_stopped(self, round_was_stopped: bool) -> None:
        self._make_entry("round_was_stopped", round_was_stopped)

    def log_local_violation(self, val: any) -> None:
        self._make_entry("local_violation", val)

    def log_global_violation(self, val: any) -> None:
        self._make_entry("global_violation", val)

    def log_training_duration(self, duration: float) -> None:
        self._make_entry("training_duration", duration)

    def metrics_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics_per_round)

    def dump_metrics_to_file(self) -> None:
        df: pd.DataFrame = self.metrics_to_df()
        df.to_csv(self.metrics_path, index=False)
