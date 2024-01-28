from typing import Any

import torch.distributed.rpc as rpc
import torch.nn as nn
from client.evenflow_worker import EvenflowWorker
from data.evenflow_dataset import EvenflowDataset


class AsyncWorker(EvenflowWorker):
    """PS Worker following the 'async' strategy."""

    def __init__(
        self,
        ps_rref: rpc.RRef["AsyncServer"],  # noqa: F821
        dataset: EvenflowDataset,
        batch_size: int = 32,
        epoch_batches: int = 8,
        loss_fn: Any = nn.MSELoss,
        **kwargs,
    ):
        super().__init__(ps_rref, dataset, batch_size, epoch_batches, loss_fn, **kwargs)

        self.should_stop_round: bool = False

    def decide_push(self):
        return True

    def notify_to_stop_round(self) -> None:
        self.should_stop_round = True
        self.debug("Notified to stop round.")

    def decide_stop_round(self):
        return self.should_stop_round
