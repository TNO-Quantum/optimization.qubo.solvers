"""Test module for ``NeighborhoodSolver``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tno.quantum.optimization.qubo.components import QUBO, ResultInterface, Solver
from tno.quantum.optimization.qubo.solvers import (
    IterativeResult,
    NeighborhoodSolver,
    RSSolver,
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
    return -37


@pytest.mark.parametrize("solver_type", ["local_descent", "local2_descent"])
def test_neighborhood_solver(
    solver_type: str, qubo: QUBO, expected_bitvector: BitVector
) -> None:
    """Test neighborhood specific solver."""
    solver = NeighborhoodSolver(solver_type=solver_type)
    result = solver.solve(qubo)
    assert isinstance(result, IterativeResult)
    assert hasattr(result, "num_iterations")

    # Test initial vector is optimal solution
    solver = NeighborhoodSolver(
        solver_type=solver_type, initial_bitvector=expected_bitvector
    )
    result = solver.solve(qubo)
    assert result.num_iterations == 0

    # Test initial vector is not optimal solution
    solver = NeighborhoodSolver(solver_type=solver_type, initial_bitvector=[0, 1, 1])
    result = solver.solve(qubo)
    assert result.num_iterations > 0


@pytest.mark.parametrize(
    ("solver_class", "solver_args"),
    [
        (NeighborhoodSolver, {"solver_type": "local_descent"}),
        (NeighborhoodSolver, {"solver_type": "local2_descent"}),
        (RSSolver, {}),
    ],
)
def test_random_seed(
    solver_class: type[Solver[Any]], solver_args: dict[str, Any]
) -> None:
    """Test consistency for solvers with random seed."""
    # Generate random qubo dim 250x250
    rng = np.random.default_rng(0)
    qubo = QUBO(rng.normal(0, 10, size=(250, 250)), offset=rng.normal(0, 10))

    solver1 = solver_class(random_state=42, **solver_args)  # type: ignore[call-arg]
    results1 = solver1.solve(qubo)

    solver2 = solver_class(random_state=42, **solver_args)  # type: ignore[call-arg]
    results2 = solver2.solve(qubo)

    solver3 = solver_class(random_state=43, **solver_args)  # type: ignore[call-arg]
    results3 = solver3.solve(qubo)

    assert results1.freq == results2.freq
    assert results1.freq != results3.freq


@pytest.mark.parametrize(
    ("solver_class", "solver_args"),
    [
        (NeighborhoodSolver, {"solver_type": "local_descent"}),
        (NeighborhoodSolver, {"solver_type": "local2_descent"}),
        (RSSolver, {}),
    ],
)
def test_solver_result(
    solver_class: type[Solver[Any]],
    solver_args: dict[str, Any],
    qubo: QUBO,
    expected_bitvector: BitVector,
    expected_value: float,
) -> None:
    """Test result of solvers."""
    solver = solver_class(random_state=42, **solver_args)  # type: ignore[call-arg]
    result: ResultInterface = solver.solve(qubo)

    assert result.best_bitvector == expected_bitvector
    assert result.best_value == expected_value


def test_adjust_distribution() -> None:
    """Test apply minimum entropy of RSSolver."""
    input_distribution = np.linspace(0, 1, 10).astype(np.float64)

    output_distribution = RSSolver._adjust_distribution(input_distribution, alpha=0.0)
    np.testing.assert_array_equal(output_distribution, input_distribution)

    output_distribution = RSSolver._adjust_distribution(input_distribution, alpha=1.0)
    np.testing.assert_array_equal(output_distribution, 0.5 * np.ones(10))

    output_distribution = RSSolver._adjust_distribution(input_distribution, alpha=0.5)
    np.testing.assert_array_equal(output_distribution, 0.5 * input_distribution + 0.25)
