"""This module contains the ``DWaveSamplerSolver`` class."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, SupportsInt

import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import (
    get_singleton,
    retry_on_network_errors,
)
from tno.quantum.utils.validation import (
    check_bool,
    check_int,
    check_string,
)


class DWaveSamplerSolver(Solver[DimodSampleSetResult]):
    """D-Wave sampler solver.

    The :py:class:`DWaveSamplerSolver` class solves QUBOs using the
    :py:class:`~dwave.system.samplers.DWaveSampler`. This class allows for more easy
    reproducibility with the optional embedding parameters.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import DWaveSamplerSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = DWaveSamplerSolver()
        >>> result = solver.solve(qubo)  # doctest: +SKIP
        >>> result.best_bitvector # doctest: +SKIP
        BitVector(010)
    """

    non_deterministic = True

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_attempts: int = 1,
        num_reads: SupportsInt = 1,
        backend_id: str | None = None,
        embedding_seed: SupportsInt | None = None,
        embedding: Mapping[int, tuple[int, ...]] | None = None,
        return_embedding: bool = True,
        reuse_embedding: bool = False,
        **sample_kwargs: Any,
    ) -> None:
        """Init :py:class:`DWaveSamplerSolver`.

        Args:
            num_attempts: Number of solve attempts whenever a
                :py:exc:`~http.client.RemoteDisconnected`, or
                :py:exc:`~urllib3.exceptions.ProtocolError` or
                :py:exc:`~requests.ConnectionError` errors has occurred. Waits 15
                seconds in between attempts.
            num_reads: Number reads to sample. Default is 1.
            backend_id: Id of the sampler to use (e.g. ``'Advantage_system4.1'``). If
                ``None`` is given, the D-Wave default sampler is used. Default is
                ``None``.
            embedding_seed: Seed used for finding the embedding. For more information.
            embedding: Embedding of the QUBO onto the sampler topology. If ``None`` is
                given a new embedding will made using the minorminer from D-Wave.
                Default is ``None``.  See :py:func:`minorminer.find_embedding`.
            return_embedding: If ``True``, the embedding is included in the result.
            reuse_embedding: If ``True``, the embedding is reused in subsequent calls to
                the solver. If ``False``, a new embedding is created for each call to
                the solver. Default is ``False``. Note that this option locks the solver
                to solve just one QUBO 'shape'.
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler. See the D-Wave documentation of
                :py:attr:`~dwave.system.samplers.DWaveSampler.parameters` for possible
                additional keyword definitions.

        Raises:
            ValueError: For invalid values in `embedding_seed` or `num_reads`.
            TypeError: For incorrect types in `embedding_seed` or `num_reads`.
        """
        self.num_attempts = check_int(num_attempts, "num_attempts", l_bound=1)
        self.num_reads = check_int(num_reads, "num_reads", l_bound=1)
        self.backend_id = (
            check_string(backend_id, "backend_id") if backend_id is not None else None
        )
        self.embedding_seed = (
            check_int(embedding_seed, "embedding_seed")
            if embedding_seed is not None
            else np.random.default_rng().integers(2**31)
        )
        if embedding_seed and embedding is not None:
            warnings.warn(
                "Fixed embedding was given, `embedding_seed` will not be used.",
                stacklevel=2,
            )
        self.embedding = embedding
        self.return_embedding = check_bool(
            return_embedding, "return_embedding", safe=True
        )
        self.reuse_embedding = check_bool(reuse_embedding, "reuse_embedding", safe=True)
        self.sample_kwargs = sample_kwargs

    @retry_on_network_errors
    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        """Solves the given QUBO using sample_qubo functionality of the DWaveSampler."""
        sampler = get_singleton(DWaveSampler, solver=self.backend_id)

        # Wrap the raw sampler in the (Fixed)EmbeddingComposite
        if self.embedding is not None:
            sampler = FixedEmbeddingComposite(sampler, self.embedding)
        else:
            if np.count_nonzero(qubo.matrix + qubo.matrix.T) == len(qubo) ** 2:
                warnings.warn(
                    "The QUBO is fully connected, it is recommended to use the "
                    "DWaveCliqueSamplerSolver instead.",
                    stacklevel=2,
                )
            sampler = EmbeddingComposite(
                sampler,
                embedding_parameters={"random_seed": self.embedding_seed},
            )

        return_embedding = (
            True
            if self.reuse_embedding and self.embedding is None
            else self.return_embedding
        )

        response = sampler.sample(
            qubo.to_bqm(),
            num_reads=self.num_reads,
            return_embedding=return_embedding,
            **self.sample_kwargs,
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
