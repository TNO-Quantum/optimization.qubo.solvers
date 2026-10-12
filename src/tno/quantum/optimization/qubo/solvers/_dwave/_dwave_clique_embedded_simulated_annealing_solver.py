"""This module contains the ``DWaveCliqueEmbeddedSimulatedAnnealingSolver`` class."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, SupportsInt

import dwave.graphs
from dimod import SampleSet, StructureComposite
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from minorminer.busclique import find_clique_embedding
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


class DWaveCliqueEmbeddedSimulatedAnnealingSolver(Solver[DimodSampleSetResult]):
    """D-Wave clique-embedded simulated annealing solver.

    The :py:class:`DWaveCliqueEmbeddedSimulatedAnnealingSolver` class solves QUBOs using the
    :py:class:`~dimod.StructureComposite` and a simulated annealing sampler, with a
    clique embedding found via :py:func:`~minorminer.busclique.find_clique_embedding`
    on a topology retrieved from a D-Wave backend. This enables classical analysis of
    minor-embedding parameters, such as chain strength, for clique-embeddable problems,
    without requiring QPU time.

    This solver allows fair comparison between quantum and classical approaches under
    the same connectivity constraints, helping to isolate the impact of minor-embedding
    from hardware effects and enabling efficient exploration of embedding parameters.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import DWaveCliqueEmbeddedSimulatedAnnealingSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])  # doctest: +SKIP
        >>> solver = DWaveCliqueEmbeddedSimulatedAnnealingSolver(  # doctest: +SKIP
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
        """Init :py:class:`DWaveCliqueEmbeddedSimulatedAnnealingSolver`.

        Args:
            num_reads: Number of annealing runs (samples) to perform. Default is 1.
            num_sweeps: Number of sweeps per annealing run. Default is 1000.
            embedding: (Optional) embedding to be used with FixedEmbeddingComposite.
                If ``None``, a clique embedding will be used instead.
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
            ValueError: For invalid values in ``num_reads`` or ``num_sweeps``.
            TypeError: For incorrect types in ``num_reads``, ``num_sweeps``
                or ``embedding_seed``.
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
            topology_type = sampler_backend.properties["topology"]["type"]
            if topology_type == "pegasus":
                graph = dwave.graphs.pegasus_graph(
                    sampler_backend.properties["topology"]["shape"][0]
                )
            elif topology_type == "chimera":
                graph = dwave.graphs.chimera_graph(
                    *sampler_backend.properties["topology"]["shape"]
                )
            elif topology_type == "zephyr":
                graph = dwave.graphs.zephyr_graph(
                    sampler_backend.properties["topology"]["shape"][0]
                )
            else:
                error_msg = f"Unknown topology type: {topology_type}"
                raise ValueError(error_msg)

            self._graph = graph
            self._clique_embedding = None

        self.sampler = sampler

    @retry_on_network_errors
    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        if qubo.size == 0:  # NOTE: sampler.sample fails on BQM of size zero
            sampleset = SampleSet.from_samples([[]], "BINARY", qubo.offset)  # type: ignore[no-untyped-call]
            return DimodSampleSetResult.from_result(qubo, sampleset)

        sampler = self.sampler

        # If no embedding is provided, find a clique embedding for this QUBO
        if self.embedding is None:
            variables = qubo.to_bqm().variables
            clique_embedding = find_clique_embedding(
                variables, self._graph, use_cache=True
            )
            if not clique_embedding:
                error_msg = "No clique embedding found for the given QUBO."
                raise ValueError(error_msg)
            sampler = FixedEmbeddingComposite(sampler, clique_embedding)

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

        response = sampler.sample(
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
