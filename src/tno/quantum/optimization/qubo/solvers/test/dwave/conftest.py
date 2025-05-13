"""Pytest fixtures."""

from __future__ import annotations

from typing import Any, cast

import pytest

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers import (
    DimodSampleSetResult,
    DWaveCliqueSamplerSolver,
    DWaveSamplerSolver,
    ExactSamplerSolver,
    KerberosSamplerSolver,
    LeapHybridSolver,
    RandomSamplerSolver,
    SimulatedAnnealingSolver,
    SteepestDescentSolver,
    TabuSolver,
    TreeDecompositionSolver,
)


@pytest.fixture
def qubo() -> QUBO:
    """Test Qubo, Solution: '010', Value = -37"""
    return QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]], 13)


@pytest.fixture
def expected_variables() -> list[int]:
    """Test Qubo, Solution: '010', Value = -37"""
    return [0, 1, 0]


@pytest.fixture
def expected_value() -> float:
    """Test Qubo, Solution: '010', Value = -37"""
    return -37


@pytest.fixture(
    params=[
        RandomSamplerSolver,
        SimulatedAnnealingSolver,
        SteepestDescentSolver,
        TabuSolver,
    ]
)
def solver_with_seed(request: Any) -> Solver[DimodSampleSetResult]:
    """Fixture for solvers that use random seed."""
    return cast("Solver[DimodSampleSetResult]", request.param())


@pytest.fixture(
    params=[
        KerberosSamplerSolver,
        LeapHybridSolver,
        DWaveCliqueSamplerSolver,
        DWaveSamplerSolver,
    ]
)
def solver_with_api(request: Any) -> Solver[DimodSampleSetResult]:
    """Fixture for solvers that use dwave api."""
    return cast("Solver[DimodSampleSetResult]", request.param())


@pytest.fixture(
    params=[
        KerberosSamplerSolver,
        LeapHybridSolver,
        ExactSamplerSolver,
        DWaveCliqueSamplerSolver,
        DWaveSamplerSolver,
        TreeDecompositionSolver,
    ]
)
def solver_without_seed(request: Any) -> Solver[DimodSampleSetResult]:
    """Fixture for solvers that don't use random seed."""
    return cast("Solver[DimodSampleSetResult]", request.param())


@pytest.fixture(
    params=[
        ExactSamplerSolver,
        TreeDecompositionSolver,
        RandomSamplerSolver,
        SimulatedAnnealingSolver,
        SteepestDescentSolver,
        TabuSolver,
    ]
)
def solver_no_api(request: Any) -> Solver[DimodSampleSetResult]:
    """Fixture for solvers that don't require dwave api."""
    return cast("Solver[DimodSampleSetResult]", request.param())
