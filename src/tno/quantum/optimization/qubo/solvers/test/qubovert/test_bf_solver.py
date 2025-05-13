"""Test module for the ``BFSolver`` class."""

from __future__ import annotations

import pytest

from tno.quantum.optimization.qubo.components import QUBO, BasicResult
from tno.quantum.optimization.qubo.solvers import BFSolver
from tno.quantum.utils import BitVector


@pytest.fixture
def qubo() -> QUBO:
    """Test Qubo, Solution: '010', Value = -37"""
    return QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]], 13)


@pytest.fixture
def expected_bitvector() -> BitVector:
    return BitVector("010")


@pytest.fixture
def expected_value() -> float:
    """Test Qubo, Solution: '010', Value = -37"""
    return -37


def test_bf_solver(
    qubo: QUBO,
    expected_value: float,
    expected_bitvector: BitVector,
) -> None:
    """Test that BFSolver produces a basic result object with correct values."""
    solver = BFSolver()
    result = solver.solve(qubo)

    assert isinstance(result, BasicResult)
    assert result.best_value == expected_value
    assert result.best_bitvector == expected_bitvector
