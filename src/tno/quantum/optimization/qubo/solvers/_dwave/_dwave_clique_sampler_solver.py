"""This module contains the ``DWaveCliqueSamplerSolver`` class."""

from __future__ import annotations

from typing import Any

from dimod import SampleSet
from dwave.preprocessing.composites import SpinReversalTransformComposite
from dwave.system import DWaveCliqueSampler
from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.utils.validation import check_int

from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import (
    get_singleton,
    retry_on_network_errors,
)


class DWaveCliqueSamplerSolver(Solver[DimodSampleSetResult]):
    """D-Wave clique sampler.

    The :py:class:`DWaveCliqueSamplerSolver` class solves QUBOs using the
    :py:class:`~dwave.system.samplers.DWaveCliqueSampler`. The sampler finds embeddings
    for dense QUBO with chains of equal length, which should improve the performance.
    This solver should not be used on sparse QUBOs.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import DWaveCliqueSamplerSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = DWaveCliqueSamplerSolver()  # doctest: +SKIP
        >>> result = solver.solve(qubo)          # doctest: +SKIP
        >>> result.best_bitvector                # doctest: +SKIP
        BitVector(010)
    """

    non_deterministic = True

    def __init__(
        self,
        *,
        num_attempts: int = 1,
        num_spin_reversal_transforms: int = 0,
        **sample_kwargs: Any,
    ) -> None:
        """Init :py:class:`DWaveCliqueSamplerSolver`.

        Args:
            num_attempts: Number of solve attempts whenever a
                :py:exc:`~http.client.RemoteDisconnected`, or
                :py:exc:`~urllib3.exceptions.ProtocolError` or
                :py:exc:`~requests.ConnectionError` errors has occurred. Waits 15
                seconds in between attempts.
            num_spin_reversal_transforms: Number of spin reversal transform runs. A
                value of ``0`` will not transform the problem. If you specify a
                nonzero value, each spin reversal transform will result in an
                independent run of the child sampler, with all results aggregated
                into a single sample set containing
                ``num_reads*num_spin_reversal_transforms`` samples. Default is ``0``.
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler. See the D-Wave documentation of
                :py:attr:`~dwave.system.samplers.DWaveCliqueSampler.parameters` for
                possible additional keyword definitions.
        """
        self.sampler = get_singleton(DWaveCliqueSampler)
        self.num_attempts = check_int(num_attempts, "num_attempts", l_bound=1)
        self.num_spin_reversal_transforms = check_int(
            num_spin_reversal_transforms, "num_spin_reversal_transforms", l_bound=0
        )

        self.sampler = SpinReversalTransformComposite(self.sampler)
        self.sample_kwargs = sample_kwargs

    @retry_on_network_errors
    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        """Solve QUBO using ``DWaveSampler``."""
        if qubo.size == 0:  # NOTE: DWaveCliqueSampler.sample fails on BQM of size zero
            sampleset = SampleSet.from_samples([[]], "BINARY", qubo.offset)  # type: ignore[no-untyped-call]
            return DimodSampleSetResult.from_result(qubo, sampleset)

        response = self.sampler.sample(
            qubo.to_bqm(),
            num_spin_reversal_transforms=self.num_spin_reversal_transforms,
            **self.sample_kwargs,
        )

        return DimodSampleSetResult.from_result(qubo, response)
