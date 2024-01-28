import random
from abc import ABC, abstractmethod
from typing import Any, Generator, Iterator, Optional

import numpy as np
import torch
import torch.distributed.rpc as rpc
from data.evenflow_dataset import EvenflowDataset
from evenflow_logger import EvenflowLogger
from evenflow_metrics_tracker import EvenflowMetricsTracker
from evenflow_model import EvenflowModel
from server.evenflow_server import EvenflowServer
from torch import nn
from torch.utils.data.dataloader import DataLoader


class EvenflowWorker(EvenflowLogger, EvenflowMetricsTracker, ABC):
    def __init__(
        self,
        ps_rref: rpc.RRef[EvenflowServer],
        dataset: EvenflowDataset,
        batch_size: int = 32,
        epoch_batches: int = 8,
        loss_fn: Any = nn.MSELoss,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.ps_rref: rpc.RRef[EvenflowServer] = ps_rref

        self.worker_rref: rpc.RRef["EvenflowWorker"] = rpc.RRef(self)
        self.worker_name: str = rpc.get_worker_info().name
        self.rank: int = int(self.worker_name[-1]) if self.worker_name[-1].isdigit() else None

        # Logger init
        EvenflowLogger.__init__(self, self.worker_name, **kwargs)
        EvenflowMetricsTracker.__init__(self, self.worker_name, **kwargs)

        self.batch_size: int = batch_size
        self.num_training_batches: int = epoch_batches
        self.loss_fn: Any = loss_fn
        self.seed: int = int(abs(seed if seed is not None else hash(self.worker_name))) % (1 << 32 - 1)

        self._set_deterministic()

        self.dataset: EvenflowDataset = dataset
        self.dataloader_iter: Iterator

        # Stats
        self.round_context: dict[str, Any] = {}
        self.consumed_messages: int = 0

        self.debug(f"Worker context: {str(self.__dict__)}")

    def _set_deterministic(self) -> None:
        # Ensure determinism
        torch.set_deterministic_debug_mode(1)
        torch.backends.cudnn.benchmark = False

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def epoch_batch_gen(self) -> Generator[Optional[tuple[torch.Tensor, torch.Tensor]], None, None]:
        """
        Yields epoch batches.
        Returns None if the dataloader ran out of data.
        """

        for _ in range(self.num_training_batches):
            inputs, labels = next(self.dataloader_iter, (None, None))  # [batch_size, 18], [batch_size, 1]

            if inputs is None and labels is None:
                yield None
                return

            ret_payload: tuple[torch.Tensor, torch.Tensor] = (inputs.cpu(), labels.cpu())
            self.consumed_messages += 1
            yield ret_payload

    @abstractmethod
    def decide_push(self):
        """Condition for pushing the model to the server."""
        raise NotImplementedError()

    @abstractmethod
    def decide_stop_round(self):
        """
        Condition for completing the current round and exitting training.
        Method should evaluate fast, as it could be called often.
        """
        raise NotImplementedError()

    def accept_round_countext(self, round_context: dict[str, any]) -> None:
        """Called before each round start, contains information broadcast by the PS."""
        self.round_context = round_context
        ctx_to_display: dict[str, any] = {k: v for k, v in round_context.items() if k not in {"model"}}
        self.debug(f"Accepting new context from PS: {str(ctx_to_display)}")

    def transform_weights(
        self, cur_round: int, initial_grads: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Hook for transforming the produced gradients."""
        return initial_grads

    def train_round(self, current_round: int) -> None:
        """
        Main training loop for each round.
        Consider overriding individuals segments and not the entire method.
        """
        # Keep track if a model was pushed this epoch
        server: EvenflowServer = self.ps_rref.rpc_sync()

        # Model version of this round
        model: EvenflowModel = self.round_context["model"]
        self.debug(f"Processing model ID: {id(model)}")

        # A final sync needs to take place when we run out of messages
        batch_idx: int = 0
        need_to_stop_round: bool = False

        # Iterate over the incoming Kafka messages
        while not need_to_stop_round:

            # Pull and iterate over a batch of Kafka messages
            for (
                dataloader_payload
            ) in self.epoch_batch_gen():  # num_training_batches * ([batch_size, 18], [batch_size, 1])

                # Check if we ran out of messages
                if dataloader_payload is None:
                    self.debug(f"Worker ran out of messages on round {current_round}")
                    need_to_stop_round = True
                    break

                batch_idx += 1

                # Unpack the dataloader payload
                x, y = dataloader_payload
                self.debug(f"Processing batch {batch_idx}")
                self.log_batch(batch_idx)

                # Forward pass
                y_hat: torch.Tensor = model(x)
                local_loss = self.loss_fn(y_hat, y)
                local_loss.backward()
                loss_val: float = local_loss.item()
                self.log_training_loss(loss_val)
                self.debug(f"Worker loss {loss_val}")
                self.round_context["messages_this_round"] += x.shape[0]

            # Transform weights
            if batch_idx > 0:
                # Module parameters and model weights
                weights_to_push: list[torch.Tensor] = [p.grad for p in model.cpu().parameters()]

                # Considering transforming the weights before pushing, useful for FedOpt
                transformed_weights_to_push: list[Optional[torch.Tensor]] = self.transform_weights(
                    current_round, weights_to_push
                )
            else:
                transformed_weights_to_push: list[Optional[torch.Tensor]] = []

            # Consider pushing local model
            if need_to_stop_round or self.decide_push():
                self.debug("Worker pushing its model...")

                # Bytes size of weights that will be sent 'over the wire'
                model_weights_size: int = sum([w.numel() * w.element_size() for w in transformed_weights_to_push])
                self.log_pushed_model_size(model_weights_size)

                # Perform a call but don't block
                res_fut: torch.futures.Future = rpc.rpc_async(
                    self.ps_rref.owner(),
                    server.accept_worker_model,
                    args=(
                        self.ps_rref,
                        self.worker_rref,
                        self.worker_name,
                        transformed_weights_to_push,
                        self.round_context["messages_this_round"],
                        current_round,
                    ),
                )
                self.debug("Waiting for PS sync.")
                _: any = res_fut.wait()
                self.debug("PS synced.")

                # Block if necessary
                self.debug(f"Worker received PS future for round {current_round}.")
                self.log_model_was_pushed(True)
                need_to_stop_round = True
            else:
                self.log_model_was_pushed(False)
                self.debug("Worker did not push its model")

            if self.decide_stop_round():
                self.debug("Worker decided to stop training")
                need_to_stop_round = True

        # Increment the number of messages this round by 'batch_size' messages
        self.debug(f'Messages consumed in round {current_round}: {self.round_context["messages_this_round"]}')
        self.log_processed_items(self.round_context["messages_this_round"])

    def on_round_start(self, round_num: int) -> None:
        """Runs at the start of the round"""
        self.log_round(round_num)
        self.debug(f"Started round {round_num}")

    def on_round_completed(self, round_num: int) -> None:
        """Runs at the end of the round"""
        self.debug(f"Completed round {round_num}")

    def setup(self):
        """Called before the first round."""
        self.dataloader_iter = iter(DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0))
        self.debug("Completed setup.")

    def teardown(self):
        """
        Called after all rounds have been completed.
        Teardown tasks inclued:
        * Ship metrics to the PS.
        """
        self.dump_metrics_to_file()
        self.debug("Completed teardown.")
