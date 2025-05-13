"""This module contains the ``SA2Solver`` class."""

from __future__ import annotations

from typing import Any, SupportsInt

import qubovert
from numpy.random import RandomState

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers._qubovert._qubovert_anneal_result import (
    QubovertAnnealResult,
)
from tno.quantum.optimization.qubo.solvers._qubovert._utils import _get_qubovert_model
from tno.quantum.utils.validation import check_int, check_random_state


class SA2Solver(Solver[QubovertAnnealResult]):
    """Simulated annealing solver using a qubovert implementation.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import BFSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = SA2Solver()
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
        """Init :py:class:`SA2Solver`.

        Args:
            random_state: Random state for reproducibility. Default is ``None``.
            num_reads: Number reads to sample. Default is 10.
            sample_kwargs: Additional kwargs for qubovert sampler.

        Raises:
            ValueError: If `random_state` has invalid value, or if `num_reads` is less
                than 1.
            TypeError: If `num_reads` is not an integer.
        """
        self.random_state = check_random_state(random_state, "random_state")
        self.num_reads = check_int(num_reads, "num_reads", l_bound=1)
        self.sample_kwargs = sample_kwargs

    def _solve(self, qubo: QUBO) -> QubovertAnnealResult:
        """Solve QUBO using qubovert implementation of simulated annealing."""
        # Use seed from sample_kwargs, or generate one using random_state
        seed = self.sample_kwargs.get("seed", self.random_state.randint(2**31))

        kwargs = dict(self.sample_kwargs)
        kwargs.pop("seed", None)
        kwargs.pop("num_reads", None)

        model = _get_qubovert_model(qubo)
        anneal_results = qubovert.sim.anneal_qubo(
            model,
            seed=seed,
            num_anneals=self.num_reads,
            **kwargs,
        )
        return QubovertAnnealResult.from_result(anneal_results)
