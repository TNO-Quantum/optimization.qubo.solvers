"""This module contains tests for all D-Wave solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hybrid.composers
import hybrid.decomposers
import hybrid.samplers
import numpy as np
import pytest
from dwave.cloud import Client

from tno.quantum.optimization.qubo.components import QUBO
from tno.quantum.optimization.qubo.solvers import (
    CustomSolver,
    DWaveSamplerSolver,
    RandomSamplerSolver,
    SimulatedAnnealingSolver,
    SteepestDescentSolver,
    TabuSolver,
)

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components import Solver
    from tno.quantum.optimization.qubo.solvers import (
        DimodSampleSetResult,
    )

try:
    CLIENT = Client.from_config()
    DWAVE_CONFIG = True
    QPU_BACKEND_ID = [solver.id for solver in CLIENT.get_solvers(qpu=True)]
    CLIENT.close()
except ValueError:  # API token not defined
    DWAVE_CONFIG = False
    QPU_BACKEND_ID = []

dwave_api = pytest.mark.skipif(
    not DWAVE_CONFIG, reason="Skipping test, No D-Wave API token was defined."
)


class TestSolvers:
    """Test that solvers are able to solve dummy qubo."""

    @dwave_api
    def test_with_api(
        self,
        solver_with_api: Solver[DimodSampleSetResult],
        qubo: QUBO,
        expected_value: float,
        expected_variables: list[int],
    ) -> None:
        """Test solvers that use the dwave api"""
        results = solver_with_api.solve(qubo)
        assert results.best_value == expected_value
        np.testing.assert_array_equal(results.best_bitvector, expected_variables)

        assert results.execution_time is not None

    def test_no_api(
        self,
        solver_no_api: Solver[DimodSampleSetResult],
        qubo: QUBO,
        expected_value: float,
        expected_variables: list[int],
    ) -> None:
        """Test solvers that don't use the dwave api"""
        results = solver_no_api.solve(qubo)
        execution_time = results.execution_time
        assert execution_time is not None

        if isinstance(solver_no_api, RandomSamplerSolver):
            return
        assert results.best_value == expected_value
        assert np.array_equal(results.best_bitvector, expected_variables)


@dwave_api
class TestDWaveSamplerSolver:
    """Tests specific for the DWaveSamplerSolver."""

    @pytest.mark.parametrize("backend", [None, *QPU_BACKEND_ID])
    def test_backend(self, backend: str | None, qubo: QUBO) -> None:
        """Test get different qpu backends"""
        solver = DWaveSamplerSolver(backend_id=backend)
        solver.solve(qubo)

    def test_fixed_embedding_solve(self, qubo: QUBO) -> None:
        """Test use fixed embedding"""
        embedding = {0: (699,), 1: (714,), 2: (4668,)}
        solver = DWaveSamplerSolver(embedding=embedding)
        results = solver.solve(qubo)
        assert results.sampleset.info["embedding_context"]["embedding"] == embedding

    @pytest.mark.parametrize("return_embedding", [True, False])
    def test_reuse_embedding(self, qubo: QUBO, *, return_embedding: bool) -> None:
        """Test reuse embedding"""
        solver = DWaveSamplerSolver(
            reuse_embedding=True, return_embedding=return_embedding
        )
        assert solver.embedding is None

        results1 = solver.solve(qubo)
        embedding1 = solver.embedding
        if return_embedding:
            assert (
                embedding1 == results1.sampleset.info["embedding_context"]["embedding"]
            )

        results2 = solver.solve(qubo)
        embedding2 = solver.embedding
        if return_embedding:
            assert (
                embedding2 == results2.sampleset.info["embedding_context"]["embedding"]
            )

        assert embedding1 is embedding2

    def test_embedding_seed(self, qubo: QUBO) -> None:
        """Test seed for embedding"""
        results1 = DWaveSamplerSolver(embedding_seed=42).solve(qubo)
        results2 = DWaveSamplerSolver(embedding_seed=42).solve(qubo)
        results3 = DWaveSamplerSolver(embedding_seed=43).solve(qubo)

        assert (
            results1.sampleset.info["embedding_context"]["embedding"]
            == results2.sampleset.info["embedding_context"]["embedding"]
        )
        assert (
            results1.sampleset.info["embedding_context"]["embedding"]
            != results3.sampleset.info["embedding_context"]["embedding"]
        )


def test_random_solver(qubo: QUBO) -> None:
    """Test random solver returning valid outputs"""
    solver = RandomSamplerSolver()
    for _ in range(100):
        results = solver.solve(qubo)
        assert qubo.evaluate(results.best_bitvector) == results.best_value


@pytest.mark.parametrize(
    "solver_with_seed",
    [
        RandomSamplerSolver(),
        SimulatedAnnealingSolver(),
        SteepestDescentSolver(),
        TabuSolver(),
    ],
)
def test_random_seed(solver_with_seed: Solver[DimodSampleSetResult]) -> None:
    """Test random seed."""
    if isinstance(solver_with_seed, TabuSolver):
        pytest.skip("TabuSolver seed of D-Wave is bugged.")

    if hasattr(solver_with_seed, "num_reads") and hasattr(
        solver_with_seed, "random_state"
    ):
        # Generate random qubo dim 100x100
        rng = np.random.default_rng(0)
        qubo = QUBO(rng.normal(0, 10, size=(100, 100)))

        solver_with_seed.num_reads = 100
        solver_with_seed.random_state = np.random.RandomState(42)
        results1 = solver_with_seed.solve(qubo)
        solver_with_seed.random_state = np.random.RandomState(42)
        results2 = solver_with_seed.solve(qubo)
        solver_with_seed.random_state = np.random.RandomState(43)
        results3 = solver_with_seed.solve(qubo)
        assert results1.freq == results2.freq
        assert results1.freq != results3.freq


def test_custom_solver(
    qubo: QUBO, expected_value: float, expected_variables: list[int]
) -> None:
    """Test a simple custom solver branch."""
    # Simple branch
    branch = (
        hybrid.decomposers.IdentityDecomposer()
        | hybrid.samplers.SimulatedAnnealingSubproblemSampler()
        | hybrid.composers.IdentityComposer()
    )

    solver = CustomSolver(branch=branch)
    results = solver.solve(qubo)
    assert results.best_value == expected_value
    np.testing.assert_array_equal(results.best_bitvector, expected_variables)
