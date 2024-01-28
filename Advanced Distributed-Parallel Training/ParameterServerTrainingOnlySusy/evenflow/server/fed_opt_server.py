import threading
from typing import Any, Optional

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from client.fed_opt_worker import FedOptWorker
from data.evenflow_dataset import EvenflowDataset
from server.evenflow_server import EvenflowServer
from sphere_exploration.gm_utils import GMUtils


class FedOptServer(EvenflowServer):
    def __init__(
        self,
        world_size: int,
        lr: float,
        dataset: EvenflowDataset,
        batch_size: int,
        num_test_batches: int,
        fft_coeffs: int,
        loss_fn: Any = nn.MSELoss,
        **kwargs,
    ):
        super().__init__(world_size, lr, dataset, batch_size, num_test_batches, loss_fn, **kwargs)

        # Last observed estimate vector of this round.
        self.current_estimate: Optional[torch.Tensor] = None

        # FFT for local vector
        self.fft_coeffs: int = fft_coeffs

        # Worker rrefs
        self.worker_rrefs: dict[str, rpc.RRef[FedOptWorker]] = {}

    def extract_data_from_worker(
        self, worker_name: str, grads: list[Optional[torch.Tensor]], message_cnt: int, current_round: int
    ) -> None:
        """Extracts the local vector v_i from the worker's gradients and stores it in the worker data dict."""
        v_i: torch.Tensor = GMUtils.compute_local_vector(grads, self.fft_coeffs)
        self.worker_data[worker_name] = (v_i, message_cnt)
        self.debug(f"v_i extracted from worker {worker_name} : {v_i}")

    def provide_round_context_to_worker(self) -> dict[str, any]:
        """"""
        super_context: dict[str, any] = super().provide_round_context_to_worker()
        super_context["estimate"] = self.current_estimate.clone() if self.current_estimate is not None else None
        ctx_to_display: dict[str, any] = {k: v for k, v in super_context.items() if k not in {"model"}}
        self.debug(f"Server providing workers with context: {ctx_to_display}")
        return super_context

    def make_return_future(self) -> torch.futures.Future:
        """"""
        ret: dict[str, any] = {}
        res: torch.futures.Future = torch.futures.Future()
        res.set_result(ret)
        self.debug(f"Return future: {ret}")
        return res

    def calc_worker_weights(self) -> dict[str, float]:
        """The super method is executed but the global vector estimate is also produced."""
        worker_weights: dict[str, float] = super().calc_worker_weights()
        self.generate_estimate_vec(worker_weights)
        return worker_weights

    def generate_estimate_vec(self, worker_weights: dict[str, float]) -> None:
        """"""
        v_i_per_worker: dict[str, torch.Tensor] = {w_name: grads for w_name, (grads, _) in self.worker_data.items()}
        self.current_estimate = GMUtils.compute_estimate(v_i_per_worker, worker_weights)
        self.debug(f"New server estimate: {self.current_estimate}")

    def register_worker(self, worker_name: str, worker_rref: rpc.RRef[FedOptWorker]) -> None:
        """"""
        self.worker_rrefs[worker_name] = worker_rref
        self.debug(f"Server registered worker {worker_name}.")

    def notify_workers(self) -> None:
        """Call method 'notify_to_stop_round' via RPC on all workers using their names."""

        for worker_name, worker_rref in self.worker_rrefs.items():
            self.debug(f"Server sending notification to worker {worker_name}.")
            worker_rref.rpc_sync().notify_to_stop_round()
            self.debug(f"Server notified worker {worker_name}.")

        self.debug("Server notified all workers.")

    @staticmethod
    @rpc.functions.async_execution
    def accept_worker_model(
        ps_rref: rpc.RRef["FedOptServer"],  # noqa: F821
        worker_rref: rpc.RRef[FedOptWorker],
        worker_name: str,
        grads: list[Optional[torch.Tensor]],
        message_cnt: int,
        current_round: int,
    ) -> torch.futures.Future:
        self: FedOptServer = ps_rref.local_value()
        self.log_round(current_round)

        # Notify all other workers to send their weights
        self.notify_workers()

        # Ran out of Kafka messages
        if len(grads) == 0:
            self.debug(f"Worker {worker_name} sent an empty gradient list for round {current_round}.")
            return self.make_return_future()

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
