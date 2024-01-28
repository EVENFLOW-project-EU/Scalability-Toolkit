import numpy as np
import pytest
from sphere_exploration.gm_utils import GMUtils


@pytest.fixture()
def vector_len() -> int:
    return 5


@pytest.fixture()
def center(vector_len: int) -> np.ndarray:
    return np.concatenate((np.zeros(vector_len - 1), [1.0]), dtype=np.float16)


@pytest.fixture()
def server_estimate(vector_len: int) -> np.ndarray:
    return np.zeros(vector_len, dtype=np.float16)


@pytest.fixture()
def radius() -> int:
    return 10


@pytest.fixture()
def points() -> int:
    return 5


@pytest.fixture()
def expected_min_max() -> tuple[float, float]:
    return 0.0, 10.0


@pytest.mark.timeout(5)
class TestFedOptWorker:
    def test_compute_max_min_on_sphere_np_implementations(
        self,
        center: np.ndarray,
        radius: int,
        server_estimate: np.ndarray,
        points: int,
        expected_min_max: tuple[float, float],
    ):
        test_method: callable = GMUtils.compute_min_max_on_sphere
        test_args: tuple = (center, server_estimate, radius, points)
        test_kwargs: dict = {}

        result: tuple = test_method(*test_args, **test_kwargs)
        print(result)

        assert expected_min_max == pytest.approx(result, abs=1)
