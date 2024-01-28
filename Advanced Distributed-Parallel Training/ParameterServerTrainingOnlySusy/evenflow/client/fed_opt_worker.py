from typing import Any, Optional

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from client.evenflow_worker import EvenflowWorker
from data.evenflow_dataset import EvenflowDataset
from sphere_exploration.factory import SphereExplorationFactory
from sphere_exploration.gm_utils import GMUtils
from sphere_exploration.max_min_on_sphere import ParameterSpaceExplorer


class FedOptWorker(EvenflowWorker):
    """PS Worker following the 'FedOpt' strategy."""

    def __init__(
        self,
        ps_rref: rpc.RRef["FedOptServer"],  # noqa: F821
        dataset: EvenflowDataset,
        fft_coeffs: int,
        batch_size: int = 32,
        epoch_batches: int = 8,
        loss_fn: Any = nn.MSELoss,
        **kwargs,
    ):
        super().__init__(ps_rref, dataset, batch_size, epoch_batches, loss_fn, **kwargs)
        self.old_grads: Optional[list[torch.Tensor]] = None

        # Distance measure of 2 n-dim torch vectors
        # Currently set to L-inf norm.
        self.vector_distance = lambda x, y: torch.norm(x - y, p=float("inf"))

        # Flag for stopping current round when a threshold has been crossed
        self.need_to_stop_round: bool = False

        # Threshold value used during the threshold crossing check
        self.gm_threshold: float = kwargs["gm_threshold"]

        # Server estimate (e), sent by the PS at the start of the round
        self.server_estimate: Optional[torch.Tensor] = None

        # Local vector v_i sent to the PS last round
        self.last_transmitted_v_i: Optional[torch.Tensor] = None

        # MaxMinOnSphere implementation
        self.fft_coeffs: int = fft_coeffs
        self.max_min_on_sphere: callable = self.get_max_min_on_sphere_impl(kwargs["max_min_on_sphere_impl"])

    def get_max_min_on_sphere_impl(self, impl_name: str) -> callable:
        """Return the MaxMinOnSphere implementation to use."""
        impl: ParameterSpaceExplorer = SphereExplorationFactory.create(impl_name, logger=self._local_logger)
        self.debug(f"Selecting MaxMinOnSphere implementation: {impl_name}")
        return impl.compute

    def decide_push(self):
        return self.need_to_stop_round or self.cur_round == 1

    def decide_stop_round(self):
        return self.need_to_stop_round or self.cur_round == 1

    def on_round_start(self, round_num: int) -> None:
        if round_num == 1:
            self.debug("First round, registering on PS.")
            self.ps_rref.rpc_sync().register_worker(self.worker_name, self.worker_rref)

    def on_round_completed(self, round_num: int) -> None:
        super().on_round_completed(round_num)
        self.need_to_stop_round = False

    def notify_to_stop_round(self) -> None:
        self.need_to_stop_round = True
        self.debug("Notified to stop round.")

    def accept_round_countext(self, round_context: dict[str, any]) -> None:
        super().accept_round_countext(round_context)
        self.server_estimate = round_context["estimate"]

    def transform_weights(
        self, cur_round: int, initial_grads: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        # Return the deltas instead of the initial grads
        if cur_round == 1:
            self.debug(f"Worker using initial grads for the first round {cur_round}: {initial_grads}")
        else:
            self.debug(f"Worker old grads: {self.old_grads}")
            self.debug(f"Worker current grads: {initial_grads}")
            initial_grads = [torch.sub(g_old, g_new) for g_old, g_new in zip(self.old_grads, initial_grads)]
            self.debug(f"Worker new grads: {initial_grads}")

        # Keep track of current gradients for next round
        self.old_grads = [g.clone() for g in initial_grads]

        # Local vector / Worker gradients, flattened and stacked
        v_i: torch.Tensor = GMUtils.compute_local_vector(initial_grads, self.fft_coeffs)
        self.debug(f"v_i: {v_i}")

        # First round results to a sync
        if self.last_transmitted_v_i is None or self.server_estimate is None:
            self.debug(
                "Last transimitted v_i or estimate are None,"
                f"the threshold has been crossed and v_i is assigned the value of: {v_i}"
            )
            self.last_transmitted_v_i = v_i
            self.need_to_stop_round = True
        else:
            self.debug(f"Last transimitted v_i {self.last_transmitted_v_i}")

            # Difference between the current and the last transmitted local vectors
            delta_v_i: torch.Tensor = GMUtils.compute_delta_local_vectors(v_i, self.last_transmitted_v_i)
            self.debug(f"Delta v_i {delta_v_i}")

            # Drift vector
            u_i: torch.Tensor = self.server_estimate + delta_v_i
            self.debug(f"u_i {delta_v_i}")

            # Check if the GM threshold has been crossed
            center: torch.Tensor = GMUtils.compute_center(self.server_estimate, u_i)
            self.debug(f"Center: {center}")

            radius: int = GMUtils.compute_radius(self.server_estimate, u_i, self.vector_distance)
            self.debug(f"Radius: {radius}")

            tmp_min, tmp_max = self.max_min_on_sphere(center, self.server_estimate, radius)
            self.debug(f"Min/Max on sphere: {tmp_min, tmp_max}")

            self.debug(f">> {tmp_min} < {self.gm_threshold} < {tmp_max}")
            threshold_was_crossed: bool = tmp_min < self.gm_threshold < tmp_max
            self.log_local_violation(threshold_was_crossed)
            self.debug(f"Threshold was crossed: {threshold_was_crossed}")

            # Local violation, round needs to end
            if threshold_was_crossed:
                self.last_transmitted_v_i = v_i
                self.need_to_stop_round = True

        return initial_grads
