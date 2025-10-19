"""This module contains the ``DWaveTilingSolver`` class."""

from __future__ import annotations

import warnings
from typing import Any, SupportsInt

from dimod import SampleSet
from dwave.system import (
    AutoEmbeddingComposite,
    DWaveSampler,
    TilingComposite,
)
from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.utils.validation import (
    check_int,
    check_string,
)

from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import (
    get_singleton,
    retry_on_network_errors,
)


class DWaveTilingSolver(Solver[DimodSampleSetResult]):
    """D-Wave tiling composite solver.

    .. deprecated:: 1.4.0
        Use :py:class:`DWaveParallelEmbeddingSolver` instead. This solver will be
        removed in 2.0.0.

    The :py:class:`DWaveTilingSolver` class solves QUBOs using the
    :py:class:`~dwave.system.composites.TilingComposite`. This class allows for easier
    tiling sampling.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import DWaveTilingSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = DWaveTilingSolver()
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
        sub_m: int = 1,
        sub_n: int = 1,
        t: int = 4,
        **sample_kwargs: Any,
    ) -> None:
        """Init :py:class:`DWaveTilingSolver`.

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
            sub_m: Minimum number of Chimera unit cell rows for a subproblem.
            sub_n: Minimum number of Chimera unit cell columns for a subproblem.
            t: Size of the shore within each Chimera unit cell (default 4).
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler. See the D-Wave documentation of
                :py:attr:`~dwave.system.samplers.DWaveSampler.parameters` for possible
                additional keyword definitions.

        Raises:
            ValueError: For invalid values in `num_reads`.
            TypeError: For incorrect types in `num_reads`.
        """
        warnings.warn(
            "DWaveTilingSolver is deprecated since 1.4.0 and will be removed in 2.0.0; "
            "use `DWaveParallelEmbeddingSolver` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )

        self.num_attempts = check_int(num_attempts, "num_attempts", l_bound=1)
        self.num_reads = check_int(num_reads, "num_reads", l_bound=1)
        self.backend_id = (
            check_string(backend_id, "backend_id") if backend_id is not None else None
        )
        self.sub_m = check_int(sub_m, "sub_m", l_bound=1)
        self.sub_n = check_int(sub_n, "sub_n", l_bound=1)
        self.t = check_int(t, "t", l_bound=1)
        self.sample_kwargs = sample_kwargs

    @retry_on_network_errors
    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        """Solves the given QUBO using TilingComposite."""
        if qubo.size == 0:  # NOTE: sampler.sample fails on BQM of size zero
            sampleset = SampleSet.from_samples([[]], "BINARY", qubo.offset)  # type: ignore[no-untyped-call]
            return DimodSampleSetResult.from_result(qubo, sampleset)

        sampler = get_singleton(DWaveSampler, solver=self.backend_id)

        if sampler.properties["topology"]["type"] == "zephyr":
            warn_msg = (
                "TilingComposite on Zephyr devices is not supported, "
                "despite DWave error pointing towards incompatibility with Pegasus"
            )
            warnings.warn(warn_msg, stacklevel=2)

        sampler = TilingComposite(sampler, sub_m=self.sub_m, sub_n=self.sub_n, t=self.t)
        sampler = AutoEmbeddingComposite(sampler)

        response = sampler.sample(
            qubo.to_bqm(),
            num_reads=self.num_reads,
            **self.sample_kwargs,
        )

        return DimodSampleSetResult.from_result(qubo, response)
