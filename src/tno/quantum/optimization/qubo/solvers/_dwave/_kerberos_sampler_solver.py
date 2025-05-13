"""This module contains the ``KerberosSamplerSolver`` class."""

from __future__ import annotations

from typing import Any, SupportsInt

from hybrid.reference.kerberos import KerberosSampler

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import (
    get_singleton,
    retry_on_network_errors,
)
from tno.quantum.utils.validation import check_int


class KerberosSamplerSolver(Solver[DimodSampleSetResult]):
    """Kerberos sampler solver.

    The :py:class:`KerberosSamplerSolver` class solves QUBOs using the quantum-classical
    hybrid :py:class:`~hybrid.reference.kerberos.KerberosSampler`.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import KerberosSamplerSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = KerberosSamplerSolver()  # doctest: +SKIP
        >>> result = solver.solve(qubo)       # doctest: +SKIP
        >>> result.best_bitvector             # doctest: +SKIP
        BitVector(010)
    """

    non_deterministic = True

    def __init__(
        self,
        *,
        num_attempts: int = 1,
        max_iter: SupportsInt = 5,
        max_time: SupportsInt = 60,
        **sample_kwargs: Any,
    ) -> None:
        """Init :py:class:`KerberosSamplerSolver`.

        Args:
            num_attempts: Number of solve attempts whenever a
                :py:exc:`~http.client.RemoteDisconnected`, or
                :py:exc:`~urllib3.exceptions.ProtocolError` or
                :py:exc:`~requests.ConnectionError` errors has occurred. Waits 15
                seconds in between attempts.
            max_iter: Maximum number of iterations. Default is 5. See the D-Wave
                documentation of
                :py:meth:`~hybrid.reference.kerberos.KerberosSampler.sample` for more
                information.
            max_time: Wall clock runtime termination criterion. Default is 60.
                See the D-Wave documentation of
                :py:meth:`~hybrid.reference.kerberos.KerberosSampler.sample` for more
                information.
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler. See the D-Wave documentation of
                :py:meth:`~hybrid.reference.kerberos.KerberosSampler.sample` for more
                information.

        Raises:
            ValueError: If `max_iter` is less than 1, or `max_time` is less than 0.
            TypeError: If `max_iter` or `max_time` is not an integer.
        """
        self.sampler = get_singleton(KerberosSampler)
        self.num_attempts = check_int(num_attempts, "num_attempts", l_bound=1)
        self.max_iter = check_int(max_iter, "max_iter", l_bound=1)
        self.max_time = check_int(max_time, "max_time", l_bound=0)
        self.sample_kwargs = sample_kwargs

    @retry_on_network_errors
    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        """Solve QUBO using the quantum-classical hybrid ``KerberosSampler``."""
        response = self.sampler.sample(
            qubo.to_bqm(),
            max_iter=self.max_iter,
            max_time=self.max_time,
            **self.sample_kwargs,
        )

        return DimodSampleSetResult.from_result(qubo, response)
