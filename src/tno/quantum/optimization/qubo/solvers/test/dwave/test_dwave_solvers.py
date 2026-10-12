"""This module contains tests for all D-Wave solvers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import dwave.graphs
import numpy as np
import pytest
from dwave.system import DWaveSampler
from tno.quantum.optimization.qubo.components import QUBO

from tno.quantum.optimization.qubo.solvers import (
    DWaveCliqueEmbeddedSimulatedAnnealingSolver,
    DWaveCliqueSamplerSolver,
    DWaveEmbeddedSimulatedAnnealingSolver,
    DWaveParallelEmbeddingSolver,
    DWaveSamplerSolver,
    DWaveTilingSolver,
    RandomSamplerSolver,
    SimulatedAnnealingSolver,
    SteepestDescentSolver,
    TabuSolver,
)
from tno.quantum.optimization.qubo.solvers._dwave import (
    _dwave_clique_embedded_simulated_annealing_solver as clique_solver_module,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import get_singleton
from tno.quantum.optimization.qubo.solvers.test.dwave.conftest import (
    QPU_BACKEND_ID,
    dwave_api,
)

if TYPE_CHECKING:
    from networkx import Graph
    from tno.quantum.optimization.qubo.components import Solver

    from tno.quantum.optimization.qubo.solvers import DimodSampleSetResult


@pytest.mark.parametrize(
    ("topology_type", "shape", "topology_graph"),
    [
        ("pegasus", (2,), dwave.graphs.pegasus_graph(2)),
        ("chimera", (2, 2, 4), dwave.graphs.chimera_graph(2, 2, 4)),
        ("zephyr", (2,), dwave.graphs.zephyr_graph(2)),
    ],
)
def test_clique_embedded_solver_supports_dwave_topologies(
    monkeypatch: pytest.MonkeyPatch,
    topology_type: str,
    shape: tuple[int, ...],
    topology_graph: Graph,
) -> None:
    """Test topology graph construction and clique embedding compatibility."""
    sampler_backend = SimpleNamespace(
        nodelist=list(topology_graph.nodes),
        edgelist=list(topology_graph.edges),
        properties={"topology": {"type": topology_type, "shape": shape}},
    )

    def get_mock_sampler(sampler_class: type, **_: str) -> object:
        return sampler_backend if sampler_class is DWaveSampler else sampler_class()

    monkeypatch.setattr(clique_solver_module, "get_singleton", get_mock_sampler)

    qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]], 13)
    result = DWaveCliqueEmbeddedSimulatedAnnealingSolver().solve(qubo)

    assert result.embedding is not None


@dwave_api
class TestDWaveSamplerSolver:
    """Tests specific for the DWaveSamplerSolver."""

    @pytest.mark.parametrize("backend", [None, *QPU_BACKEND_ID])
    def test_backend(self, backend: str | None, qubo: QUBO) -> None:
        """Test get different qpu backends"""
        solver = DWaveSamplerSolver(backend_id=backend)
        solver.solve(qubo)

    def test_fixed_embedding_solve(self) -> None:
        """Test use fixed embedding"""
        qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]], 13)
        embedding = {0: (699,), 1: (714,), 2: (4668,)}
        solver = DWaveSamplerSolver(embedding=embedding)
        results = solver.solve(qubo)
        assert results.embedding == embedding

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
            assert embedding1 == results1.embedding

        results2 = solver.solve(qubo)
        embedding2 = solver.embedding
        if return_embedding:
            assert embedding2 == results2.embedding

        assert embedding1 is embedding2

    def test_embedding_seed(self, qubo: QUBO) -> None:
        """Test seed for embedding"""
        results1 = DWaveSamplerSolver(embedding_seed=42).solve(qubo)
        results2 = DWaveSamplerSolver(embedding_seed=42).solve(qubo)
        results3 = DWaveSamplerSolver(embedding_seed=43).solve(qubo)

        assert results1.embedding == results2.embedding
        assert qubo.size == 0 or results1.embedding != results3.embedding

    @pytest.mark.parametrize("num_spin_reversal_transforms", [0, 1, 2])
    def test_spin_reversal_transform(
        self,
        num_spin_reversal_transforms: int,
        qubo: QUBO,
    ) -> None:
        """Test different values for num_spin_reversal_transforms."""
        num_reads: int = 10
        solver = DWaveSamplerSolver(
            num_reads=num_reads,
            num_spin_reversal_transforms=num_spin_reversal_transforms,
        )
        result = solver.solve(qubo)

        # should behave like without using the feature if == 0
        if num_spin_reversal_transforms == 0:
            exp = int(num_reads)
        else:
            exp = int(num_reads * num_spin_reversal_transforms)

        # for null-qubos no solver is called;
        # sampleset with a single sample is manually constructed
        if qubo.size == 0:
            exp = 1

        occur = int(result.sampleset.record["num_occurrences"].sum())
        assert exp == occur


@dwave_api
class TestDWaveCliqueSamplerSolver:
    """Tests specific for the DWaveCliqueSamplerSolver."""

    @pytest.mark.parametrize("num_spin_reversal_transforms", [0, 1, 2])
    def test_spin_reversal_transform(
        self,
        num_spin_reversal_transforms: int,
        qubo: QUBO,
    ) -> None:
        """Test different values for num_spin_reversal_transforms."""
        num_reads: int = 10
        solver = DWaveCliqueSamplerSolver(
            num_reads=num_reads,
            num_spin_reversal_transforms=num_spin_reversal_transforms,
        )
        result = solver.solve(qubo)

        # should behave like without using the feature if == 0
        if num_spin_reversal_transforms == 0:
            exp = int(num_reads)
        else:
            exp = int(num_reads * num_spin_reversal_transforms)

        # for null-qubos no solver is called;
        # sampleset with a single sample is manually constructed
        if qubo.size == 0:
            exp = 1

        occur = int(result.sampleset.record["num_occurrences"].sum())
        assert exp == occur


@dwave_api
class TestDWaveTilingSolver:
    """Tests specific for the DWaveTilingSolver."""

    @pytest.mark.parametrize("backend", [None, *QPU_BACKEND_ID])
    def test_backend(self, backend: str | None) -> None:
        """Test different qpu backends."""
        sampler = get_singleton(DWaveSampler, solver=backend)
        solver = DWaveTilingSolver(backend_id=backend, sub_m=1, sub_n=1, t=4)
        qubo = QUBO([[1, -1], [-1, 1]])

        if sampler.properties["topology"]["type"] == "zephyr":
            warn_msg = "TilingComposite on Zephyr devices is not supported"
            error_msg = "topology shape is not of length 1"

            with (
                pytest.raises(ValueError, match=error_msg),
                pytest.warns(UserWarning, match=warn_msg),
            ):
                solver.solve(qubo)

        else:
            solver.solve(qubo)

    def test_num_samples_greater_than_num_reads(self) -> None:
        """Test that num_reads=1 results in multiple samples in the raw output."""
        solver = DWaveTilingSolver(sub_m=1, sub_n=1, t=4, num_reads=1)
        qubo = QUBO([[1, -1], [-1, 1]])
        result = solver.solve(qubo)
        assert result.sampleset.record.num_occurrences.sum() > solver.num_reads


def test_random_solver(qubo: QUBO) -> None:
    """Test random solver returning valid outputs"""
    solver = RandomSamplerSolver()
    for _ in range(100):
        results = solver.solve(qubo)
        assert qubo.evaluate(results.best_bitvector) == results.best_value


@dwave_api
class TestDWaveParallelEmbeddingSolver:
    """Tests specific for the DWaveParallelEmbeddingSolver."""

    @pytest.mark.parametrize("backend", [None, *QPU_BACKEND_ID])
    def test_backend(self, backend: str | None, qubo: QUBO) -> None:
        """Test get different qpu backends"""

        solver = DWaveParallelEmbeddingSolver(backend_id=backend)
        solver.solve(qubo)

    def test_sampleset(self, qubo: QUBO) -> None:
        num_reads = 100
        max_num_emb = 5
        solver = DWaveParallelEmbeddingSolver(
            num_reads=num_reads, max_num_emb=max_num_emb
        )
        qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        result = solver.solve(qubo)
        assert sum(result.freq.num_occurrences) == max_num_emb * num_reads

    def test_return_embedding(self) -> None:
        """Test that embeddings are stored in the result object."""
        qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        max_num_emb = 3
        solver = DWaveParallelEmbeddingSolver(
            reuse_embedding=True, max_num_emb=max_num_emb
        )

        result = solver.solve(qubo)

        assert isinstance(result.embedding, list)
        assert len(result.embedding) == 3

        for embedding in result.embedding:
            assert isinstance(embedding, dict)

    def test_small_qubo_warning(self) -> None:
        """Test that warning is raised for QUBOs too small for parallel embedding."""
        qubo = QUBO([[1]])
        solver = DWaveParallelEmbeddingSolver()

        expected_warning = (
            "QUBO size \\(1\\) is too small for this solver\\. "
            "Consider using another solver\\. "
            "Defaulting to AutoEmbeddingComposite instead\\."
        )

        with pytest.warns(
            UserWarning,
            match=expected_warning,
        ):
            solver.solve(qubo)

    @pytest.mark.parametrize("qubo_size", [5, 10, 20])
    def test_various_size_qubos(self, qubo_size: int) -> None:
        """Test that the DWaveParallelEmbeddingSolver can solve various size QUBOs."""
        num_embs = [1, 5, 10]
        for max_num_emb in num_embs:
            num_reads = 3
            solver = DWaveParallelEmbeddingSolver(
                embedder_kwargs={"max_num_emb": max_num_emb}, num_reads=num_reads
            )
            n = qubo_size
            rng = np.random.default_rng(42)
            matrix = rng.random((n, n))
            qubo = QUBO(matrix)

            result = solver.solve(qubo)
            sampleset = result.sampleset

            assert len(sampleset.variables) == n
            assert set(sampleset.variables) == set(qubo.to_bqm().variables)

            sample_array = sampleset.record.sample
            assert sample_array.shape[1] == n

            total_occ = int(sampleset.record["num_occurrences"].sum())
            assert total_occ % num_reads == 0

            embeddings_used = total_occ // num_reads
            assert 1 <= embeddings_used <= max_num_emb

            if qubo_size <= 10:
                assert embeddings_used == max_num_emb

            for sample_row in sample_array:
                assert all(val in [0, 1] for val in sample_row)

            for _, (sample_dict, energy) in enumerate(
                zip(sampleset.samples(), sampleset.record.energy, strict=False)  # type: ignore[no-untyped-call]
            ):
                calculated_energy = qubo.evaluate([sample_dict[j] for j in range(n)])
                assert abs(calculated_energy - energy) < 1e-10  #  noqa: PLR2004

    @pytest.mark.parametrize("num_emb", [3, 5, 9])
    def test_num_embeddings(self, num_emb: int) -> None:
        """Test that the number of returned embeddings does not exceed max_num_emb."""
        n = 15

        rng = np.random.default_rng(42)
        matrix = rng.random((n, n))
        qubo = QUBO(matrix)
        solver = DWaveParallelEmbeddingSolver(
            embedder_kwargs={"max_num_emb": num_emb},
        )

        result = solver.solve(qubo)

        assert len(result.sampleset.samples()) <= num_emb  # type: ignore[no-untyped-call]

    def test_multiple_param_declarations(self) -> None:
        """Test that multiple declarations of `max_num_emb` raise a warning."""

        expected_warning = (
            "Multiple declarations of parameter 'max_num_emb': "
            "provided both as function argument and inside 'embedder_kwargs'. "
            "Priority will be given to the value set inside 'embedder_kwargs'."
        )

        with pytest.warns(UserWarning, match=expected_warning):
            DWaveParallelEmbeddingSolver(
                embedder_kwargs={"max_num_emb": 3}, max_num_emb=5
            )

    def test_one_to_iterable_warning(self) -> None:
        """Test that warning is raised when one_to_iterable=False and embedder=None."""

        expected_warning = (
            "`one_to_iterable` was provided as `False` and `embedder` as `None`"
            " Therefore it will be set to `True` instead\\."
        )

        with pytest.warns(UserWarning, match=expected_warning):
            DWaveParallelEmbeddingSolver(one_to_iterable=False)

    @pytest.mark.parametrize("num_spin_reversal_transforms", [0, 1, 2])
    def test_spin_reversal_transform(
        self,
        num_spin_reversal_transforms: int,
        qubo: QUBO,
    ) -> None:
        """Test different values for num_spin_reversal_transforms."""
        num_reads: int = 10
        max_num_emb: int = 5
        solver = DWaveParallelEmbeddingSolver(
            num_reads=num_reads,
            max_num_emb=5,
            num_spin_reversal_transforms=num_spin_reversal_transforms,
        )
        result = solver.solve(qubo)

        # should behave like without using the SRT feature if == 0
        srt_factor = max(1, num_spin_reversal_transforms)
        # qubo size too small for solver
        num_emb_factor = 1 if qubo.size < 2 else max_num_emb

        # for null-qubos no solver is called;
        # sampleset with a single sample is manually constructed
        exp = 1 if qubo.size == 0 else int(num_reads * num_emb_factor * srt_factor)

        occur = int(result.sampleset.record["num_occurrences"].sum())
        assert exp == occur


@dwave_api
class TestEmbeddedVsStandardSimulatedAnnealing:
    """Tests specific for the DWaveEmbeddedSimulatedAnnealingSolver."""

    @pytest.mark.parametrize("backend", [None, *QPU_BACKEND_ID])
    def test_embedded_sa_solver_is_worse_than_standard(
        self, backend: str | None
    ) -> None:
        """Test that the DWaveEmbeddedSimulatedAnnealingSolver finds the best solution
        less frequently than the standard SimulatedAnnealingSolver.
        """
        # Use large random QUBO to test that the embedded solver is worse
        # On the hardcoded small QUBOs, the embedded solver works similarly to the
        # standard solver
        size = 50
        rng = np.random.default_rng(0)
        qubo = QUBO(rng.normal(0, 10, size=(size, size)), offset=rng.normal(0, 10))

        sa_solver = SimulatedAnnealingSolver(num_reads=100, random_state=42)
        sa_result = sa_solver.solve(qubo)
        best_value = sa_result.best_value

        embedded_solver = DWaveEmbeddedSimulatedAnnealingSolver(
            backend_id=backend, num_reads=100, random_state=42
        )
        embedded_result = embedded_solver.solve(qubo)

        freq_standard = np.sum(sa_result.sampleset.record.energy == best_value)
        freq_embedded = np.sum(embedded_result.sampleset.record.energy == best_value)
        assert freq_embedded < freq_standard


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
