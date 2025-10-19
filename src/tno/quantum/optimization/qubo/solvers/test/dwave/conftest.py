"""Pytest fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from dwave.cloud import Client

from tno.quantum.optimization.qubo.solvers import (
    DimodSampleSetResult,
    DWaveCliqueEmbeddedSimulatedAnnealingSolver,
    DWaveCliqueSamplerSolver,
    DWaveEmbeddedSimulatedAnnealingSolver,
    DWaveParallelEmbeddingSolver,
    DWaveSamplerSolver,
    DWaveTilingSolver,
    ExactSamplerSolver,
    KerberosSamplerSolver,
    LeapHybridSolver,
    RandomSamplerSolver,
    SimulatedAnnealingSolver,
    SteepestDescentSolver,
    TabuSolver,
    TreeDecompositionSolver,
)

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components import Solver

try:
    CLIENT = Client.from_config()
    DWAVE_HAS_API_TOKEN = True
    QPU_BACKEND_ID = [solver.id for solver in CLIENT.get_solvers(qpu=True)]
    CLIENT.close()
except ValueError:  # API token not defined
    DWAVE_HAS_API_TOKEN = False
    QPU_BACKEND_ID = []

dwave_api = pytest.mark.skipif(
    not DWAVE_HAS_API_TOKEN, reason="Skipping test, No D-Wave API token was defined."
)

DWAVE_SOLVERS_WITH_SEED = [
    RandomSamplerSolver,
    SimulatedAnnealingSolver,
    SteepestDescentSolver,
    TabuSolver,
    DWaveEmbeddedSimulatedAnnealingSolver,
    DWaveCliqueEmbeddedSimulatedAnnealingSolver,
]


@pytest.fixture(params=DWAVE_SOLVERS_WITH_SEED)
def solver_with_seed(request: Any) -> Solver[DimodSampleSetResult]:
    """Fixture for solvers that use random seed."""
    return cast("Solver[DimodSampleSetResult]", request.param())


DWAVE_SOLVERS_WITHOUT_SEED = [
    KerberosSamplerSolver,
    LeapHybridSolver,
    ExactSamplerSolver,
    DWaveCliqueSamplerSolver,
    DWaveSamplerSolver,
    DWaveTilingSolver,
    TreeDecompositionSolver,
    DWaveParallelEmbeddingSolver,
]


@pytest.fixture(params=DWAVE_SOLVERS_WITHOUT_SEED)
def solver_without_seed(request: Any) -> Solver[DimodSampleSetResult]:
    """Fixture for solvers that don't use random seed."""
    return cast("Solver[DimodSampleSetResult]", request.param())


DWAVE_SOLVERS_WITH_API: list[type[Solver[DimodSampleSetResult]]] = [
    KerberosSamplerSolver,
    LeapHybridSolver,
    DWaveCliqueSamplerSolver,
    DWaveSamplerSolver,
    DWaveTilingSolver,
    DWaveEmbeddedSimulatedAnnealingSolver,
    DWaveCliqueEmbeddedSimulatedAnnealingSolver,
    DWaveParallelEmbeddingSolver,
]


@pytest.fixture(params=DWAVE_SOLVERS_WITH_API)
def solver_with_api(request: Any) -> Solver[DimodSampleSetResult]:
    """Fixture for solvers that use dwave api."""
    return cast("Solver[DimodSampleSetResult]", request.param())


DWAVE_SOLVERS_WITHOUT_API: list[type[Solver[DimodSampleSetResult]]] = [
    ExactSamplerSolver,
    TreeDecompositionSolver,
    RandomSamplerSolver,
    SimulatedAnnealingSolver,
    SteepestDescentSolver,
    TabuSolver,
]


@pytest.fixture(params=DWAVE_SOLVERS_WITHOUT_API)
def solver_no_api(request: Any) -> Solver[DimodSampleSetResult]:
    """Fixture for solvers that don't require dwave api."""
    return cast("Solver[DimodSampleSetResult]", request.param())


DWAVE_AVAILABLE_SOLVERS: list[type[Solver[DimodSampleSetResult]]] = (
    DWAVE_SOLVERS_WITH_API + DWAVE_SOLVERS_WITHOUT_API
    if DWAVE_HAS_API_TOKEN
    else DWAVE_SOLVERS_WITHOUT_API
)
