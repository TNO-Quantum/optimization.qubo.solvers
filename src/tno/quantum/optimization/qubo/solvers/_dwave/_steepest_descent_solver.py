"""This module contains the ``SteepestDescentSolver`` class."""

from __future__ import annotations

from typing import Any, SupportsInt

from dwave.samplers import SteepestDescentSolver as DWAVESteepestDescentSolver
from numpy.random import RandomState

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import get_singleton
from tno.quantum.utils.validation import check_int, check_random_state


class SteepestDescentSolver(Solver[DimodSampleSetResult]):
    """D-Wave steepest decent solver.

    The :py:class:`SteepestDescentSolver` class solves QUBOs using the D-Wave
    implementation (:py:class:`~dwave.samplers.SteepestDescentSolver`) of steepest
    descent.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import SteepestDescentSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>>
        >>> solver = SteepestDescentSolver()
        >>> result = solver.solve(qubo)
        >>> result.best_bitvector
        BitVector(010)
    """

    def __init__(
        self,
        *,
        random_state: int | RandomState | None = None,
        num_reads: SupportsInt = 1,
        **sample_kwargs: Any,
    ) -> None:
        """Init :py:class:`SteepestDescentSolver`.

        Args:
            random_state: Random state for reproducibility. Overrides the D-Wave
                ``'seed'`` argument. Default is ``None``.
            num_reads: Maximum number of random samples to be drawn. Default is 1.
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler. See the D-Wave documentation of
                :py:meth:`~dwave.samplers.SteepestDescentSolver.sample` for possible
                additional keyword definitions.

        Raises:
            ValueError: If `random_state` has invalid value, or `num_reads` is less
                than 1.
            TypeError: If `num_reads` is not an integer.
        """
        self.sampler = get_singleton(DWAVESteepestDescentSolver)
        self.random_state = check_random_state(random_state, "random_state")
        self.num_reads = check_int(num_reads, "num_reads", l_bound=1)
        self.sample_kwargs = sample_kwargs

    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        """Solve QUBO using ``SteepestDescentSampler``."""
        # Use seed from sample_kwargs, or generate one using random_state
        seed = self.sample_kwargs.get("seed", self.random_state.randint(2**31))

        kwargs = dict(self.sample_kwargs)
        kwargs.pop("seed", None)
        kwargs.pop("num_reads", None)

        response = self.sampler.sample(
            qubo.to_bqm(), num_reads=self.num_reads, seed=seed, **kwargs
        )

        return DimodSampleSetResult.from_result(qubo, response)
