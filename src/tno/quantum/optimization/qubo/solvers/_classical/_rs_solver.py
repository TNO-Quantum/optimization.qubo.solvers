"""This module contains the ``RSSolver`` class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray

from tno.quantum.optimization.qubo.components import BasicResult, Freq, Solver
from tno.quantum.utils import BitVector
from tno.quantum.utils.validation import (
    check_int,
    check_random_state,
    check_real,
)

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components import QUBO


class RSSolver(Solver[BasicResult]):
    r"""Relaxation sampler solver.

    The Relaxation Sampler Solver solves QUBOs by randomly sampling
    bitvectors from the distribution given by the (fractional) solution to the
    relaxed QUBO.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import RSSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = RSSolver()
        >>> result = solver.solve(qubo)
        >>> result.best_bitvector
        BitVector(010)
    """

    def __init__(
        self,
        random_state: int | RandomState | None = None,
        num_samples: int = 100,
        alpha: float = 0.0,
    ) -> None:
        """Init :py:class:`RSSolver`.

        Args:
            random_state: Random state for reproducibility. Defaults to ``None``.
            num_samples: Number of bitvectors to sample.
            alpha: Parameter controlling minimum randomness in the distribution.
                If `alpha` is ``0.0``, the distribution remains unchanged. If `alpha` is
                greater than ``0.0``, the distribution is a convex combination of the
                original distribution and a uniform distribution. Must be in the range
                $[0, 1]$.

        Raises:
            ValueError: If `alpha` is not in $[0, 1]$.
        """
        self.random_state = check_random_state(random_state, "random_state")
        self.num_samples = check_int(num_samples, "num_samples", l_bound=1)
        self.alpha = check_real(alpha, "alpha", l_bound=0, u_bound=1.0)

    @staticmethod
    def _adjust_distribution(
        distribution: NDArray[np.float64],
        alpha: float = 0.0,
    ) -> NDArray[np.float64]:
        """Adjust distribution.

        The new distribution is a convex combination of the original
        distribution and a uniform, with combination parameter `alpha`.
        If `alpha` is zero, the distribution stays unchanged.

        Args:
            distribution: original distribution.
            alpha: convex combination parameter.

        Returns:
            The adjusted distribution.
        """
        return (1 - alpha) * distribution + alpha * 0.5

    def _solve(self, qubo: QUBO) -> BasicResult:
        """Use the Relaxation Sampler solver to solve the QUBO.

        The solution to the QUBO relaxation is used as a probability
        distribution.
        """
        distribution = qubo.compute_bounds()
        distribution = np.clip(distribution, 0.0, 1.0)

        distribution = self._adjust_distribution(distribution, self.alpha)

        rand_values = self.random_state.random_sample((self.num_samples, qubo.size))
        bit_vector_matrix = (rand_values < distribution).T.astype(np.uint8)
        energies = (
            np.einsum(
                "ij,jk,ki->i", bit_vector_matrix.T, qubo.matrix, bit_vector_matrix
            )
            + qubo.offset
        )

        min_idx = np.argmin(energies)
        best_value = energies[min_idx]
        best_bitvector = bit_vector_matrix.T[min_idx]

        unique, unique_indices, unique_counts = np.unique(
            bit_vector_matrix.T, axis=0, return_index=True, return_counts=True
        )
        frequencies = Freq(
            [BitVector(v) for v in unique], energies[unique_indices], unique_counts
        )

        return BasicResult.from_result(
            BitVector(best_bitvector), best_value, frequencies
        )
