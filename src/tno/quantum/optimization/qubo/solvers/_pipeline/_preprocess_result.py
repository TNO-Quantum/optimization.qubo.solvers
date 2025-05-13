"""This module contains the ``PreprocessResult`` class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsFloat

from tno.quantum.optimization.qubo.components import (
    Freq,
    PartialSolution,
    ResultInterface,
)

if TYPE_CHECKING:
    from typing import Self

    from tno.quantum.utils import BitVectorLike


class PreprocessResult(ResultInterface):
    """Class containing information about a preprocessing step.

    When solving a QUBO with preprocessing enabled, the QUBO is partially solved and
    the reduced QUBO is given to the actual solver. This class contains the partial
    solution, and contains the result returned by the solver.
    """

    def __init__(
        self,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq,
        partial_solution: PartialSolution,
    ) -> None:
        """Init :py:class:`PreprocessResult`.

        Args:
            best_bitvector: Bitvector corresponding to the best result.
            best_value: Objective value of the best result.
            freq: Frequency object containing the frequency of found bitvectors and
                energies.
            partial_solution: Partial solution to QUBO problem obtained by
                preprocessing.
        """
        super().__init__(best_bitvector, best_value, freq)
        self._partial_solution = partial_solution

    @property
    def partial_solution(self) -> PartialSolution:
        """Partial solution to QUBO problem obtained by preprocessing."""
        return self._partial_solution

    @classmethod
    def from_result(
        cls, partial_solution: PartialSolution, result: ResultInterface
    ) -> Self:
        """Create :py:class:`PreprocessResult` from partial solution and result object.

        Args:
            partial_solution: Partial solution to QUBO problem obtained by
                preprocessing.
            result: Result obtained by solver after solving the preprocessed QUBO.

        Returns:
            A :py:class:`PreprocessResult` instance.
        """
        best_bitvector = partial_solution.expand(result.best_bitvector)
        best_value = result.best_value
        freq = Freq(
            [partial_solution.expand(b) for b in result.freq.bitvectors],
            result.freq.energies,
            result.freq.num_occurrences,
        )
        return cls(best_bitvector, best_value, freq, partial_solution)

    def __eq__(self, other: Any) -> bool:
        """Check if two instances of :py:class:`PreprocessResult` are equal."""
        if not isinstance(other, PreprocessResult):
            return False

        return (
            self.best_bitvector == other.best_bitvector
            and self.best_value == other.best_value
            and self.freq == other.freq
            and self.partial_solution == other.partial_solution
        )
