"""This module contains the ``NeighborhoodSolver`` class.

The ``NeighborhoodSolver`` addresses QUBOs through a local neighborhood search.
"""

from __future__ import annotations

import numpy as np
from numpy.random import RandomState

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers._classical._iterative_result import (
    IterativeResult,
)
from tno.quantum.utils import BitVector, BitVectorLike
from tno.quantum.utils.validation import (
    check_int,
    check_random_state,
    check_string,
)


class NeighborhoodSolver(Solver[IterativeResult]):
    r"""Neighborhood solver.

    The :py:class:`NeighborhoodSolver` addresses QUBOs through a local neighborhood
    search algorithm. By default, it employs the ``local2_descent`` method, which
    performs two bit-flips per iteration. This approach often yields more optimal
    solutions compared to the ``local_descent`` method, which is limited to flipping one
    bit at a time. The ``local_descent`` method can get stuck in local minima more
    quickly because it explores fewer neighboring states per iteration, making it less
    effective at escaping suboptimal solutions. However, the advantage of the
    ``local_descent`` method is that it requires fewer computations and hence quicker
    to perform. However, in the current implementation both have a complexity of
    $\mathcal{O}(N^2)$, where $N$ is the size of the QUBO.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import NeighborhoodSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = NeighborhoodSolver()
        >>> result = solver.solve(qubo)
        >>> result.best_bitvector
        BitVector(010)
    """

    def __init__(
        self,
        random_state: int | RandomState | None = None,
        initial_bitvector: BitVectorLike | None = None,
        solver_type: str = "local2_descent",
        max_iterations: int = 1000,
    ) -> None:
        """Init :py:class:`NeighborhoodSolver`.

        Args:
            random_state: Random state for reproducibility. Defaults to ``None``.
            initial_bitvector: Initial bitvector. Defaults to ``None``.
            solver_type: Type of solver to use ('local_descent' or 'local2_descent').
                Defaults to 'local2_descent'.
            max_iterations: Maximum number of iterations during local descent.

        Raises:
            ValueError: If `random_state` has invalid value, or if `max_iterations`  is
                less than 1, or if `solver_type` is not a valid option.
            TypeError: If `max_iterations` is not an integer or `solver_type` is not a
                string.
        """
        self.random_state = check_random_state(random_state, "random_state")
        self.initial_bitvector = initial_bitvector
        solver_type = check_string(solver_type, "solver_type")
        if solver_type not in ["local_descent", "local2_descent"]:
            error_msg = f"Unknown solver type: {solver_type}"
            raise ValueError(error_msg)
        self.solver_type = solver_type
        self.max_iterations = check_int(max_iterations, "max_iterations", l_bound=1)

    def _solve_via_local_descent(
        self,
        qubo: QUBO,
        initial_bitvector: BitVector,
        max_iterations: int,
    ) -> IterativeResult:
        """Solve QUBO using local descent.

        Args:
            qubo: QUBO instance.
            initial_bitvector: Initial bitvector.
            max_iterations: Maximum number of iterations.

        Returns:
            Result of the optimization as a IterativeResult.
        """
        x = initial_bitvector
        for _iteration in range(max_iterations):
            delta_x = qubo.delta_x(x)
            idx_minimal_change_in_energy = int(np.argmin(delta_x))
            if delta_x[idx_minimal_change_in_energy] >= 0:
                break
            x.flip_indices(idx_minimal_change_in_energy, inplace=True)

        return IterativeResult.from_result(x, qubo.evaluate(x), _iteration)

    def _solve_via_local2_descent(
        self,
        qubo: QUBO,
        initial_bitvector: BitVector,
        max_iterations: int,
    ) -> IterativeResult:
        """Solve QUBO using local2 descent.

        Args:
            qubo: QUBO instance.
            initial_bitvector: Initial bitvector.
            max_iterations: Maximum number of iterations.

        Returns:
            Result of the optimization as a IterativeResult.
        """
        x = initial_bitvector
        for _iteration in range(max_iterations):
            delta_x2 = qubo.delta_x2(x)
            i, j = map(int, np.unravel_index(np.argmin(delta_x2), delta_x2.shape))
            if delta_x2[i, j] >= 0:
                break
            x.flip_indices(i, j, inplace=True)

        return IterativeResult.from_result(x, qubo.evaluate(x), _iteration)

    def _solve(self, qubo: QUBO) -> IterativeResult:
        """Use the neighborhood solver to solve the QUBO."""
        if self.initial_bitvector is None:
            cutoff = 0.5
            initial_bitvector = BitVector(
                (self.random_state.random(qubo.size) < cutoff).astype(int)
            )
        else:
            initial_bitvector = BitVector(self.initial_bitvector)

        if self.solver_type == "local_descent":
            return self._solve_via_local_descent(
                qubo, initial_bitvector, self.max_iterations
            )
        return self._solve_via_local2_descent(
            qubo, initial_bitvector, self.max_iterations
        )
