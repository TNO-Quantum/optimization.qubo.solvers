"""This module contains the ``DWaveEmbeddedSimulatedAnnealingSolver`` class."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, SupportsInt

from dimod import StructureComposite
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import AutoEmbeddingComposite, DWaveSampler, FixedEmbeddingComposite
from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.utils.validation import (
    check_bool,
    check_int,
    check_random_state,
    check_string,
)

from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import (
    get_singleton,
    retry_on_network_errors,
)


class DWaveEmbeddedSimulatedAnnealingSolver(Solver[DimodSampleSetResult]):
    """D-Wave embedded simulated annealing solver.

    The :py:class:`DWaveEmbeddedSimulatedAnnealingSolver` class solves QUBOs using the
    :py:class:`~dimod.StructureComposite` with a simulated annealing sampler, using a
    topology retrieved from a D-Wave backend. This allows for classical analysis of
    embedding-related parameters, such as chain strength, without requiring QPU time.
    The solver mimics the connectivity constraints of a quantum annealer by embedding
    the problem onto the hardware graph, enabling fair comparison between quantum and
    classical approaches under the same connectivity limitations.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import DWaveEmbeddedSimulatedAnnealingSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])  # doctest: +SKIP
        >>> solver = DWaveEmbeddedSimulatedAnnealingSolver(
        ...     backend_id="Advantage_system4.1", num_reads=100, beta_range=(0.1, 4.2)  # doctest: +SKIP
        ... )  # doctest: +SKIP
        >>> result = solver.solve(qubo)  # doctest: +SKIP
        >>> result.best_bitvector  # doctest: +SKIP
        BitVector(010)
    """  # noqa: E501

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_reads: SupportsInt = 1,
        num_sweeps: SupportsInt = 1000,
        embedding: Mapping[int, tuple[int, ...]] | None = None,
        embedding_seed: SupportsInt | None = None,
        backend_id: str | None = None,
        return_embedding: bool = True,
        reuse_embedding: bool = False,
        random_state: int | None = None,
        **sample_kwargs: Any,
    ) -> None:
        """Init :py:class:`DWaveEmbeddedSimulatedAnnealingSolver`.

        Args:
            num_reads: Maximum number of random samples to be drawn. Default is 1.
            num_sweeps: Number of sweeps used in annealing. Default is 1000.
            embedding: (Optional) embedding to be used with FixedEmbeddingComposite.
            embedding_seed: Seed used for finding the embedding.
            return_embedding: If ``True``, the embedding is included in the result.
            reuse_embedding: If ``True``, the embedding is reused in subsequent calls to
                the solver. If ``False``, a new embedding is created for each call to
                the solver. Default is ``False``. Note that this option locks the solver
                to solve just one QUBO 'shape'.
            backend_id: Id of the sampler to use (e.g. ``'Advantage_system4.1'``). If
                ``None`` is given, the D-Wave default sampler is used. Default is
                ``None``.
            random_state: Seed or random state for reproducibility.
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler. See the D-Wave documentation of
                :py:meth:`~dwave.samplers.SimulatedAnnealingSampler.parameters` for
                possible additional keyword definitions.

        Raises:
            ValueError: If `random_state` has invalid value, or if `num_reads` or
                `num_sweeps` is less than 1.
            TypeError: If `num_reads` or `num_sweeps` is not an integer.
        """
        self.num_reads = check_int(num_reads, "num_reads", l_bound=1)
        self.num_sweeps = check_int(num_sweeps, "num_sweeps", l_bound=1)
        self.backend_id = (
            check_string(backend_id, "backend_id") if backend_id is not None else None
        )
        self.embedding = embedding
        self.embedding_seed = embedding_seed
        self.return_embedding = check_bool(
            return_embedding, "return_embedding", safe=True
        )
        self.reuse_embedding = check_bool(reuse_embedding, "reuse_embedding", safe=True)
        self.random_state = check_random_state(random_state, "random_state")
        self.sample_kwargs = sample_kwargs

        self._get_sampler()

    def _get_sampler(self) -> None:
        sampler_backend = get_singleton(DWaveSampler, solver=self.backend_id)
        nodes = sampler_backend.nodelist
        edges = sampler_backend.edgelist

        base_sampler = get_singleton(SimulatedAnnealingSampler)
        sampler = StructureComposite(base_sampler, nodes, edges)  # type: ignore[no-untyped-call]

        if self.embedding is not None:
            sampler = FixedEmbeddingComposite(sampler, self.embedding)
        else:
            sampler = AutoEmbeddingComposite(
                sampler,
                embedding_parameters={"random_seed": self.embedding_seed}
                if self.embedding_seed is not None
                else None,
            )
        self.sampler = sampler

    @retry_on_network_errors
    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        seed = self.sample_kwargs.get("seed", self.random_state.randint(2**31))

        kwargs = dict(self.sample_kwargs)
        kwargs.pop("seed", None)
        kwargs.pop("num_reads", None)
        kwargs.pop("num_sweeps", None)

        return_embedding = (
            True
            if self.reuse_embedding and self.embedding is None
            else self.return_embedding
        )

        response = self.sampler.sample(
            qubo.to_bqm(),
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            return_embedding=return_embedding,
            seed=seed,
            **kwargs,
        )

        if self.reuse_embedding and self.embedding is None:
            if self.return_embedding:
                self.embedding = deepcopy(
                    response.info["embedding_context"]["embedding"]
                )
            else:
                embedding_context = response.info.pop("embedding_context")
                self.embedding = embedding_context["embedding"]

        return DimodSampleSetResult.from_result(qubo, response)
