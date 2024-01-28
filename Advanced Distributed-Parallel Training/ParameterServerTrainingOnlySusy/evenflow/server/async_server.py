import queue
import threading
from typing import Any, Optional

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from client.async_worker import AsyncWorker
from data.evenflow_dataset import EvenflowDataset
from server.evenflow_server import EvenflowServer


class AsyncServer(EvenflowServer):
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

        # Don't accept models created before this round, i.e. late model arrivals
        self.active_round: int = 0

        # Define a shared message queue
        self.message_queue: queue.Queue = queue.Queue()  # Size is infinite

        # Latest resources for each worker
        self.worker_rrefs: dict[str, rpc.RRef["AsyncWorker"]] = {}

        # Thread constantly polling worker gradient queue
        self._init_consumer_thread()

    def _init_consumer_thread(self) -> None:
        """Initialize the consumer thread."""
        self.consumer_thread = threading.Thread(target=self._consume_messages)
        self.consumer_thread.daemon = True  # Dont wait for the thread to join in order to halt
        self.consumer_thread.start()

    def complete_round(self) -> None:
        super().complete_round()
        self.debug("Waiting for message queue to be empty.")
        self.message_queue.join()
        self.debug("Message queue empty.")

    def _consume_messages(self) -> None:
        """Consume messages from the message queue."""
        while True:
            # Blocking call, wait for new messages
            self.debug("Waiting for new messages/models.")
            message = self.message_queue.get(block=True, timeout=None)

            # Poll from queue
            worker_name, grads, message_cnt, current_round, worker_rref = message

            # Discard late arrivals
            if current_round < self.active_round:
                self.debug(f"Discarding late message from {worker_name} for round {current_round}.")
                self.message_queue.task_done()
                continue

            # Mark the worker as seen
            self.debug(f"Processing grads from {worker_name}")

            # Update the latest resources for the worker
            self.worker_rrefs[worker_name] = worker_rref
            self.extract_data_from_worker(worker_name, grads, message_cnt, current_round)

            # If all workers have sent gradients at least once
            if len(self.worker_rrefs.keys()) == self.world_size - 1:
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

                # Evaluate the model
                self.evaluate_model()

                # Notify all workers to stop sending models and complete this round
                for worker_name, worker_rref in self.worker_rrefs.items():
                    self.debug(f"Worker {worker_name} about to be notified to stop round {current_round}.")
                    worker_rref.rpc_sync().notify_to_stop_round()
                    self.debug(f"Worker {worker_name} notified to stop round {current_round}.")

                # Increment the active round
                self.active_round += 1
                self.debug(f"Server active round set to {self.active_round}.")

                # Clear the seen workers
                self.worker_rrefs.clear()
                self.debug(f"Seen workers cleared for round {current_round}.")
            else:
                self.debug("Will wait for at least one message/model from all workers.")

            # Mark the message as completed
            self.message_queue.task_done()

    @staticmethod
    @rpc.functions.async_execution
    def accept_worker_model(
        ps_rref: rpc.RRef["AsyncServer"],  # noqa: F821
        worker_rref: rpc.RRef[AsyncWorker],
        worker_name: str,
        grads: list[Optional[torch.Tensor]],
        message_cnt: int,
        current_round: int,
    ) -> torch.futures.Future:
        self: AsyncServer = ps_rref.local_value()
        self.log_round(current_round)
        self.debug(f"Message from {worker_name} for round {current_round}.")
        self.message_queue.put((worker_name, grads, message_cnt, current_round, worker_rref), block=True, timeout=None)
        return self.make_return_future()
