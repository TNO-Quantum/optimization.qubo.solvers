"""This module contains the ``IterativeResult`` class."""

from __future__ import annotations

from typing import TYPE_CHECKING, SupportsFloat, SupportsInt

from tno.quantum.optimization.qubo.components import Freq, ResultInterface
from tno.quantum.utils.validation import check_int

if TYPE_CHECKING:
    from typing import Self

    from tno.quantum.utils import BitVectorLike


class IterativeResult(ResultInterface):
    """Implementation of `ResultInterface` for the iterative solvers.

    The :py:class:`IterativeResult` is similar to the
    :py:class:`~tno.quantum.optimization.qubo.components.BasicResult`, but contains an
    additional property for the number of iterations performed.
    """

    def __init__(
        self,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq,
        num_iterations: SupportsInt,
    ) -> None:
        super().__init__(best_bitvector, best_value, freq)
        self._num_iterations = check_int(num_iterations, "num_iterations", l_bound=0)

    @property
    def num_iterations(self) -> int:
        """Number of iterations performed when solving the QUBO."""
        return self._num_iterations

    @classmethod
    def from_result(
        cls,
        bitvector: BitVectorLike,
        best_value: SupportsFloat,
        num_iterations: SupportsInt,
        freq: Freq | None = None,
    ) -> Self:
        """Parse the result of the iterative solver.

         This includes the additional information `num_iterations`.

        Args:
            bitvector: Bitvector-like object containing the found solution of the solve
                method.
            best_value: The outcome when the QUBO is solved using the variables.
            num_iterations: The number of iterations performed when solving the QUBO.
            freq: How often the results occure if none is given the best value is
                given returned once.
        """
        if freq is None:
            freq = Freq([bitvector], [best_value], [1])
        return cls(bitvector, best_value, freq, num_iterations)
