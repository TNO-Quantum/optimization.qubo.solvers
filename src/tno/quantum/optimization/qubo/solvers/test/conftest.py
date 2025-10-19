from typing import Any

import numpy as np
import pytest
from tno.quantum.optimization.qubo.components import QUBO
from tno.quantum.utils import BitVector


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate QUBO test cases."""
    params: list[Any]

    if all(
        arg in metafunc.fixturenames
        for arg in ["qubo", "expected_bitvector", "expected_value"]
    ):
        params = [
            (QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]], 13), BitVector("010"), -37),
            (QUBO([[1]], 123), BitVector("0"), 123),
            (QUBO(np.zeros((0, 0)), -123), BitVector(""), -123),
        ]
        metafunc.parametrize(("qubo", "expected_bitvector", "expected_value"), params)
    elif "qubo" in metafunc.fixturenames:
        params = [
            QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]], 13),
            QUBO([[1]], 123),
            QUBO(np.zeros((0, 0)), -123),
        ]
        metafunc.parametrize("qubo", params)
