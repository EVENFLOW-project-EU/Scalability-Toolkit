import threading
import time
from abc import ABC
from typing import Any, Optional

import torch
import torch.distributed.rpc as rpc
from data.evenflow_dataset import EvenflowDataset
from evenflow_logger import EvenflowLogger
from evenflow_metrics_tracker import EvenflowMetricsTracker
from evenflow_model import EvenflowModel
from torch import optim
from torch.utils.data import DataLoader


class EvenflowServer(EvenflowLogger, EvenflowMetricsTracker, ABC):
    def __init__(
        self,
        world_size: int,
        lr: float,
        dataset: EvenflowDataset,
        batch_size: int,
        num_test_batches: int,
        loss_fn: Any,
        **kwargs,
    ):
        # Logger init
        EvenflowLogger.__init__(self, "ps", **kwargs)
        EvenflowMetricsTracker.__init__(self, "ps", **kwargs)

        self.world_size: int = world_size
        self.model: EvenflowModel = EvenflowModel()
        self.optimizer: optim.Adam = optim.Adam(self.model.parameters(), lr=lr)
        self.enter_barrier: threading.Barrier = threading.Barrier(world_size - 1, action=self.enter_barrier_action)
        self.exit_barrier: threading.Barrier = threading.Barrier(world_size - 1, action=self.exit_barrier_action)
        self.worker_data: dict[str, tuple[Optional[torch.Tensor], int]] = {}
        self.lock: threading.Lock = threading.Lock()
        self.test_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.loss_fn = loss_fn
        self.num_test_batches: int = num_test_batches

        self.start_time: float

        # Init model grads
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

        self.debug(f"Server context: {str(self)}")

    def enter_barrier_action(self) -> None:
        pass

    def exit_barrier_action(self) -> None:
        pass

    def complete_round(self) -> None:
        pass

    def get_model(self):
        self.debug("Transmitting model to worker (deep copy).")
        return self.model.clone()

    def evaluate_model(self):
        self.debug("Evaluating model..")
        self.model.eval()

        test_loss: float = 0.0
        test_size: int = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = x.cpu(), y.cpu()
                y_view = y.view(-1, 1)
                y_hat = self.model(x)
                test_loss += self.loss_fn(y_hat, y_view).item()
                test_size += 1
                if i >= self.num_test_batches:
                    break

        self.model.train()
        test_loss: float = test_loss / test_size if test_size > 0 else float("+inf")

        self.debug(f"Evaluating model using {test_size} batches")
        self.log_processed_items(processed_items=test_size)

        self.log_test_loss(test_loss)
        self.debug(f"Test loss: {test_loss}")

    def provide_round_context_to_worker(self) -> dict[str, any]:
        """Information to be broadcast to all workers."""
        cloned_model: EvenflowModel = self.model.clone()
        round_context: dict[str, any] = {"messages_this_round": 0, "model": cloned_model}
        return round_context

    def make_return_future(self, val: any = "") -> torch.futures.Future:
        res: torch.futures.Future = torch.futures.Future()
        res.set_result(val)
        return res

    def setup(self) -> None:
        self.debug("PS setup.")
        self.start_time = time.time()

    def teardown(self) -> None:
        """
        PS Teardown tasks:
        * Dump ps metrics to file.
        """
        self.log_training_duration(time.time() - self.start_time)
        self.dump_metrics_to_file()
        self.debug("PS teardown.")

    def extract_data_from_worker(
        self, worker_name: str, grads: list[Optional[torch.Tensor]], message_cnt: int, current_round: int
    ) -> None:
        """
        Thread safe.
        Retrieve worker fields and save them to a PS shared structure.
        """
        self.debug(f"Worker {worker_name} sent grads")
        self.worker_data[worker_name] = (grads, message_cnt)

    def calc_worker_weights(self) -> dict[str, float]:
        """
        Weights for the weighted averaging process.
        Weights are between [0.0, 1.0].
        """
        if len(self.worker_data) == 0:
            raise RuntimeError()

        all_workers_sum: int = sum([message_cnt for (_, message_cnt) in self.worker_data.values()])
        worker_weights: dict[str, float] = (
            {name: message_cnt / all_workers_sum for name, (_, message_cnt) in self.worker_data.items()}
            if all_workers_sum > 0
            else {name: 0 for name in self.worker_data.keys()}
        )
        return worker_weights

    def on_before_grad_agg(self, worker_name: str, current_round: int) -> None:
        """Executed before the grad aggregation"""
        pass

    def on_barrier_exit(
        self, worker_name: str, worker_rref: rpc.RRef["EvenflowWorker"], current_round: int  # noqa: F821
    ) -> None:
        """Executed on thread-workers before the exit barrier is reset"""
        pass

    def aggregate_gradients(self, worker_weights: dict[str, float]) -> None:
        """"""
        for iter_worker_name, (grads, _) in self.worker_data.items():
            worker_weight: float = worker_weights[iter_worker_name]
            for p, g in zip(self.model.parameters(), grads):
                p.grad += g / worker_weight

    @staticmethod
    @rpc.functions.async_execution
    def accept_worker_model(
        ps_rref: rpc.RRef["EvenflowServer"],  # noqa: F821
        worker_rref: rpc.RRef["EvenflowWorker"],  # noqa: F821
        worker_name: str,
        grads: list[Optional[torch.Tensor]],
        message_cnt: int,
        current_round: int,
    ):
        raise NotImplementedError("Each server subclass should override this method")
