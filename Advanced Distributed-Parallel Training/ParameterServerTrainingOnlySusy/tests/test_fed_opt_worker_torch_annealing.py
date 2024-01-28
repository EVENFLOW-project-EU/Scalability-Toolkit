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
    return 10


@pytest.fixture()
def expected_min_max() -> tuple[float, float]:
    return 0.0, 10.0


@pytest.fixture()
def device() -> str:
    dev: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")
    return dev


@pytest.fixture()
def num_iterations() -> int:
    return 1


@pytest.fixture()
def T_init() -> float:
    return 10


@pytest.fixture()
def T_final() -> float:
    return 0.0


@pytest.fixture()
def cooling_rate() -> float:
    return 0.1


@pytest.fixture()
def dt() -> dtype:
    return torch.float16


class TestFedOptWorker:
    def test_compute_max_min_on_sphere_sim_annealing(
        self,
        center: torch.Tensor,
        radius: int,
        server_estimate: torch.Tensor,
        expected_min_max: tuple[float, float],
        device: str,
        num_iterations: int,
        T_init: float,
        T_final: float,
        cooling_rate: float,
    ):
        test_method: callable = GMUtils.compute_min_max_on_sphere_annealing
        test_args: tuple = (center, server_estimate, radius)
        test_kwargs: dict = {
            "num_iterations": num_iterations,
            "T_init": T_init,
            "T_final": T_final,
            "cooling_rate": cooling_rate,
        }

        result: tuple = test_method(*test_args, **test_kwargs)
        print(result)

        torch.testing.assert_close(result[0], expected_min_max[0], rtol=1, atol=1)
        torch.testing.assert_close(result[1], expected_min_max[1], rtol=1, atol=1)
