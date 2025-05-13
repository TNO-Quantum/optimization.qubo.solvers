"""This module contains the ``ExactSamplerSolver`` class."""

from __future__ import annotations

from typing import Any

from dimod import ExactSolver

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import (
    get_singleton,
)


class ExactSamplerSolver(Solver[DimodSampleSetResult]):
    """D-Wave exact sampler solver.

    The :py:class:`ExactSamplerSolver` class solves QUBOs exactly using a brute force
    search by the :py:class:`~dimod.reference.samplers.ExactSolver`.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import ExactSamplerSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = ExactSamplerSolver()
        >>> result = solver.solve(qubo)
        >>> result.best_bitvector
        BitVector(010)
    """

    def __init__(self, **sample_kwargs: Any) -> None:
        """Init :py:class:`ExactSamplerSolver`.

        Args:
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler. The :py:class:`~dimod.reference.samplers.ExactSolver` does not
                take additional parameters.
        """
        self.sampler = get_singleton(ExactSolver)
        self.sample_kwargs = sample_kwargs

    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        """Solves the given QUBO using sample_qubo functionality of the ExactSolver."""
        response = self.sampler.sample(qubo.to_bqm(), **self.sample_kwargs)

        return DimodSampleSetResult.from_result(qubo, response)
