"""Test module for the ``SA2Solver`` class."""

from __future__ import annotations

import numpy as np
import pytest

from tno.quantum.optimization.qubo.components import QUBO
from tno.quantum.optimization.qubo.solvers import (
    QubovertAnnealResult,
    SA2Solver,
)
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


def test_sa2_solver(
    qubo: QUBO,
    expected_value: float,
    expected_bitvector: BitVector,
) -> None:
    """Test solvers that produce an annealing results object."""
    solver = SA2Solver()
    result = solver.solve(qubo)

    assert isinstance(result, QubovertAnnealResult)
    assert result.best_bitvector == expected_bitvector
    assert result.best_value == expected_value


def test_sa2_solver_random_state() -> None:
    """Test consistency for solvers with random seed."""
    # Generate random qubo dim 250x250
    rng = np.random.default_rng(0)
    qubo = QUBO(rng.normal(0, 10, size=(250, 250)), offset=rng.normal(0, 10))

    solver1 = SA2Solver(random_state=42, num_reads=100)
    results1 = solver1.solve(qubo)

    solver2 = SA2Solver(random_state=42, num_reads=100)
    results2 = solver2.solve(qubo)

    solver3 = SA2Solver(random_state=43, num_reads=100)
    results3 = solver3.solve(qubo)

    assert results1.freq == results2.freq
    assert results1.freq != results3.freq
