"""This module contains the ``BFSolver`` class."""

from __future__ import annotations

from tno.quantum.optimization.qubo.components import QUBO, BasicResult, Solver
from tno.quantum.optimization.qubo.solvers._qubovert._utils import _get_qubovert_model


class BFSolver(Solver[BasicResult]):
    """Brute force solver using a qubovert implementation.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import BFSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = BFSolver()
        >>> result = solver.solve(qubo)
        >>> result.best_bitvector
        BitVector(010)
    """

    def _solve(self, qubo: QUBO) -> BasicResult:
        """Solve QUBO using qubovert implementation of a brute force algorithm."""
        model = _get_qubovert_model(qubo)
        model_solution = model.solve_bruteforce()
        return BasicResult.from_result(model_solution, model.value(model_solution))
