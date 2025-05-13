"""This module contains tests for the utility functions of D-Wave solvers."""

from tno.quantum.optimization.qubo.solvers._dwave._utils import get_singleton


class Dummy:
    def __init__(self, x: int = 0) -> None:
        self.x = x


def test_get_singleton_default() -> None:
    """Test `get_singleton` returns the same object the second time."""
    dummy_1 = get_singleton(Dummy)
    dummy_2 = get_singleton(Dummy)
    assert dummy_1 is dummy_2


def test_get_singleton_with_args() -> None:
    """Test `get_singleton` returns the same object only when arguments are same."""
    dummy_1 = get_singleton(Dummy, 1)
    dummy_2 = get_singleton(Dummy, 1)
    dummy_3 = get_singleton(Dummy, 2)
    assert dummy_2 is dummy_1
    assert dummy_3 is not dummy_1
