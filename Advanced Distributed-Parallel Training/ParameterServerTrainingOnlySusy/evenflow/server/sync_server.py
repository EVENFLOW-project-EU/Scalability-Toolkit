import threading
from typing import Any, Optional

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from client.sync_worker import SyncWorker
from data.evenflow_dataset import EvenflowDataset
from server.evenflow_server import EvenflowServer


class SyncServer(EvenflowServer):
    def __init__(
        self,
        world_size: int,
        lr: float,
        dataset: EvenflowDataset,
        batch_size: int,
        num_test_batches: int,
        loss_fn: Any = nn.MSELoss,
        **kwargs,
    ):
        super().__init__(world_size, lr, dataset, batch_size, num_test_batches, loss_fn, **kwargs)

    @staticmethod
    @rpc.functions.async_execution
    def accept_worker_model(
        ps_rref: rpc.RRef["SyncServer"],  # noqa: F821
        worker_rref: rpc.RRef[SyncWorker],
        worker_name: str,
        grads: list[Optional[torch.Tensor]],
        message_cnt: int,
        current_round: int,
    ) -> torch.futures.Future:
        self: SyncServer = ps_rref.local_value()
        self.log_round(current_round)

        # Worker details dict is accessed (and modified) concurrently
        with self.lock:
            self.extract_data_from_worker(worker_name, grads, message_cnt, current_round)
            self.debug(f"PS accepted data from {worker_name} for round {current_round}.")

        # Allow for a single thread to aggregate gradients
        self.debug(f"Worker {worker_name} about to hit the enter barrier for round {current_round}.")
        try:
            if self.enter_barrier.wait() == 0:
                self.debug(f"Worker {worker_name} entered the critical section for round {current_round}.")
                self.on_before_grad_agg(worker_name, current_round)

                # Worker weights for the following weighted averaging
                worker_weights: dict[str, float] = self.calc_worker_weights()
                self.debug(f"Weights for round {current_round}: {str(worker_weights)}")

                # Accumulate worker gradients and train the global model
                self.aggregate_gradients(worker_weights)

                # Optimizer step
                self.optimizer.step()

                # Ensure the gradients are set to zero so that they can be averaged
                # properly the next round.
                #
                # <rant>
                # For some unholy reason the `set_to_none=False` needs to be set explicitly
                # because a function called 'ZERO'_grad turns every grad into a None and not actually a zero.
                # </rant>
                self.optimizer.zero_grad(set_to_none=False)
                self.debug("Optimizer step.")

                # Evaluate model, run a test loader
                self.debug(f"Worker {worker_name} is evaluating the server model for round {current_round}.")

                # Underlying dataloader implementation is Kafka-based and therefore prone to
                # `ValueError: generator already executing` errors. Lock for now but consider
                # implementing a lock-free solution.
                with self.lock:
                    self.evaluate_model()

                # Wait for the worker to ack their round completion
                self.debug(f"Worker {worker_name} exitted the critical section for round {current_round}.")

                # Reset the barrier
                self.enter_barrier.reset()
                self.debug(f"Worker {worker_name} reset the enter barrier for round {current_round}.")
        except threading.BrokenBarrierError as _:  # noqa: F841
            # A BrokenBarrierError is raised when the barrier gets reset
            self.debug(f"Worker {worker_name} left the enter barrier due to a reset.")

        self.debug(f"Worker {worker_name} about to hit the exit barrier for round {current_round}.")
        try:
            if self.exit_barrier.wait() == 0:
                self.on_barrier_exit(worker_name, worker_rref, current_round)
                self.debug(f"Worker {worker_name} reset the exit barrier for round {current_round}.")
                self.exit_barrier.reset()
        except threading.BrokenBarrierError as _:  # noqa: F841
            # A BrokenBarrierError is raised when the barrier gets reset
            self.debug(f"Worker {worker_name} left the exit barrier due to a reset.")

        self.debug(f"Worker {worker_name} hit the exit barrier for round {current_round}.")

        return self.make_return_future()
