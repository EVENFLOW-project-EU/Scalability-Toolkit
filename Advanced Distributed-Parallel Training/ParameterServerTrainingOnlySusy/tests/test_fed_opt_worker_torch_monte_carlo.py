import math

import numpy as np
import pytest
import torch
from sphere_exploration.gm_utils import GMUtils
from torch import dtype


@pytest.fixture()
def vector_len() -> int:
    return 100


@pytest.fixture()
def center(vector_len: int, device: str, dt: dtype) -> torch.Tensor:
    return torch.tensor(np.concatenate((np.zeros(vector_len - 1), [1.0])), dtype=dt, device=device)


@pytest.fixture()
def server_estimate(vector_len: int, device: str, dt: dtype) -> torch.Tensor:
    return torch.tensor(np.zeros(vector_len), dtype=dt, device=device)


@pytest.fixture()
def radius() -> int:
    return 1


@pytest.fixture()
def expected_min_max() -> tuple[float, float]:
    return 0.0, 10.0


@pytest.fixture()
def device() -> str:
    dev: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")
    return dev


@pytest.fixture()
def dt() -> dtype:
    return torch.float16


@pytest.fixture()
def batch_size() -> int:
    return 1000


@pytest.fixture()
def num_samples() -> int:
    return 10_000_000


class TestFedOptWorker:
    def test_compute_max_min_on_sphere_torch_monte_carlo(
        self,
        center: torch.Tensor,
        radius: int,
        server_estimate: torch.Tensor,
        expected_min_max: tuple[float, float],
        device: str,
        batch_size: int,
        num_samples: int,
    ):
        test_method: callable = GMUtils.compute_min_max_on_sphere_monte_carlo
        test_args: tuple = (center, server_estimate, radius)
        kwargs: dict = {"num_samples": num_samples, "batch_size": batch_size}

        result: tuple = (torch.tensor(np.nan), torch.tensor(np.nan))
        for i in range(100):
            result: tuple = test_method(*test_args, **kwargs)
            if math.isfinite(result[0]) and math.isfinite(result[0]):
                print(f"Took {i + 1} rounds to get a finite result")
                break
            else:
                print(f"Round {i + 1} gave {result}")
        print(result)

        torch.testing.assert_close(result[0], expected_min_max[0], rtol=1, atol=1)
        torch.testing.assert_close(result[1], expected_min_max[1], rtol=1, atol=1)
