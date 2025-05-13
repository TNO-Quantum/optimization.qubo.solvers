"""This module contains the ``CustomSolver`` class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hybrid.core import State
from hybrid.utils import min_sample

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import retry_on_network_errors
from tno.quantum.utils.validation import check_int

if TYPE_CHECKING:
    from hybrid.flow import Branch


class CustomSolver(Solver[DimodSampleSetResult]):
    """D-Wave custom created solver.

    With the :py:class:`CustomSolver` class, custom quantum-classical hybrid D-Wave
    solvers can be made.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import CustomSolver
        >>> import hybrid
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> # Simple branch
        >>> branch = (
        ...     hybrid.decomposers.IdentityDecomposer()
        ...     | hybrid.samplers.SimulatedAnnealingSubproblemSampler()
        ...     | hybrid.composers.IdentityComposer()
        ... )
        >>>
        >>> solver = CustomSolver(branch=branch)
        >>> result = solver.solve(qubo)
        >>> result.best_bitvector
        BitVector(010)
    """

    non_deterministic = True

    def __init__(
        self, branch: Branch, *, num_attempts: int = 1, **sample_kwargs: Any
    ) -> None:
        """Init :py:class:`CustomSolver`.

        Args:
            branch: Branch.
            num_attempts: Number of solve attempts whenever a
                :py:exc:`~http.client.RemoteDisconnected`, or
                :py:exc:`~urllib3.exceptions.ProtocolError` or
                :py:exc:`~requests.ConnectionError` errors has occurred. Waits 15
                seconds in between attempts.
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler.
        """
        self.branch = branch
        self.num_attempts = check_int(num_attempts, "num_attempts", l_bound=1)
        self.sample_kwargs = sample_kwargs

    @retry_on_network_errors
    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        """Solves the given QUBO using custom created solver."""
        bqm = qubo.to_bqm()
        state = State.from_sample(min_sample(bqm), bqm)
        response = self.branch.next(state)["samples"]

        return DimodSampleSetResult.from_result(qubo, response)
