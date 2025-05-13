"""This module contains the ``TreeDecompositionSolver`` class."""

from __future__ import annotations

from typing import Any

from dwave.samplers import TreeDecompositionSolver as DWAVETreeDecompositionSolver

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import get_singleton


class TreeDecompositionSolver(Solver[DimodSampleSetResult]):
    """D-Wave tree decomposition solver.

    The :py:class:`TreeDecompositionSolver` solves QUBOs exactly using a tree
    decomposition algorithm implemented by D-Wave
    (:py:class:`~dwave.samplers.TreeDecompositionSolver`).

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import TreeDecompositionSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = TreeDecompositionSolver()
        >>> result = solver.solve(qubo)
        >>> result.best_bitvector
        BitVector(010)
    """

    def __init__(self, **sample_kwargs: Any) -> None:
        """Init :py:class:`TreeDecompositionSolver`.

        Args:
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler. See the D-Wave documentation of
                :py:meth:`~dwave.samplers.TreeDecompositionSolver.sample` for possible
                additional keyword definitions.

        .. __: https://docs.ocean.dwavesys.com/en/stable/docs_samplers/generated/dwave.samplers.TreeDecompositionSolver.sample.html
        """
        self.sampler = get_singleton(DWAVETreeDecompositionSolver)
        self.sample_kwargs = sample_kwargs

    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        """Solve QUBO using ``TreeDecompositionSolver``."""
        response = self.sampler.sample(qubo.to_bqm(), **self.sample_kwargs)

        return DimodSampleSetResult.from_result(qubo, response)
