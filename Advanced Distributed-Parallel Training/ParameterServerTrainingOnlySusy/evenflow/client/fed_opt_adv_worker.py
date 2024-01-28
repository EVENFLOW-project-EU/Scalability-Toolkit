from typing import Any, Optional

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from client.evenflow_worker import EvenflowWorker
from data.evenflow_dataset import EvenflowDataset
from sphere_exploration.gm_utils import GMUtils


class FedOptAdvancedWorker(EvenflowWorker):
    """PS Worker following the 'FedOptAdv' strategy."""

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
        self.l2_norm = lambda x, y: torch.norm(x - y, p=2)

        # Flag for stopping current round when a threshold has been crossed
        self.need_to_stop_round: bool = False

        # Threshold value, used in the FedOpt algorithm
        self.gm_threshold: float = kwargs["gm_threshold"] ** 2 / (kwargs["world_size"] - 1)

        # Server estimate (e), sent by the PS at the start of the round
        self.server_estimate: Optional[torch.Tensor] = None

        # Local vector v_i sent to the PS last round
        self.last_transmitted_v_i: Optional[torch.Tensor] = None

        # MaxMinOnSphere implementation
        self.fft_coeffs: int = fft_coeffs

    def decide_push(self):
        return self.need_to_stop_round

    def decide_stop_round(self):
        return self.need_to_stop_round

    def on_round_completed(self, round_num: int) -> None:
        super().on_round_completed(round_num)
        self.need_to_stop_round = False

    def accept_round_countext(self, round_context: dict[str, any]) -> None:
        super().accept_round_countext(round_context)
        self.server_estimate = round_context["estimate"]

    def transform_weights(
        self, cur_round: int, initial_grads: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        local_vec: torch.Tensor = GMUtils.compute_local_vector(initial_grads, self.fft_coeffs)

        # Keep track of current gradients for next round
        self.old_grads = [g.clone() for g in initial_grads]

        # Local violation is a given in the first round
        if cur_round == 1:
            self.debug(f"Worker using initial grads for the first round {cur_round}: {initial_grads}")

        # Gradients that will be transmitted to the server
        grads_to_return: list[Optional[torch.Tensor]] = (
            initial_grads
            if cur_round == 1
            else [torch.sub(g_old, g_new) for g_old, g_new in zip(self.old_grads, initial_grads)]
        )

        # Check if we need to stop the current round
        local_violation: bool = cur_round == 1 or self.check_threshold_crossing(local_vec, self.server_estimate)
        if local_violation:
            self.last_transmitted_v_i = local_vec
            self.need_to_stop_round = True

        self.log_local_violation(self.need_to_stop_round)
        self.debug(f"Local_vector: {local_vec}")
        self.debug(f"Initial_grads {initial_grads}")
        self.debug(f"Old_grads {self.old_grads}")
        self.debug(f"Last transmitted v_i {self.last_transmitted_v_i}")

        return grads_to_return

    def check_threshold_crossing(self, x_i: torch.Tensor, e: torch.Tensor) -> bool:
        self.debug(f"Using threshold: {self.gm_threshold}")
        sigma: torch.Tensor = self.l2_norm(x_i, e)
        self.debug(f"Using sigma: {sigma}")
        res: bool = torch.le(sigma, self.gm_threshold)
        return res
