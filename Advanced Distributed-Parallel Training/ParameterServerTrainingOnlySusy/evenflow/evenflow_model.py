from typing import Optional

import torch
from torch import nn


class EvenflowModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(18, 2), nn.Linear(2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net1(x)

    def clone(self) -> "EvenflowModel":
        """Produce a deep copy of the current model."""
        cur_state_dict: dict[str, any] = self.state_dict()
        _clone: EvenflowModel = EvenflowModel()
        _clone.load_state_dict(cur_state_dict)
        return _clone

    def get_gradients(self) -> list[Optional[torch.Tensor]]:
        return [p.grad for p in self.parameters()]
