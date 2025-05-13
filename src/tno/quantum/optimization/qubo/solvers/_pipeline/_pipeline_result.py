"""This module contains the ``PipelineResult`` class."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, SupportsFloat, SupportsInt

from tno.quantum.optimization.qubo.components import (
    Freq,
    ResultInterface,
)

if TYPE_CHECKING:
    from datetime import timedelta
    from typing import Self

    from tno.quantum.optimization.qubo.solvers._pipeline._preprocess_result import (
        PreprocessResult,
    )
    from tno.quantum.utils import BitVectorLike


@dataclass(init=False)
class PipelineResult(ResultInterface):
    """Implementation of `ResultInterface` for :py:class:`PipelineSolver`."""

    def __init__(  # noqa: PLR0913
        self,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq,
        *,
        solver_result: ResultInterface,
        preprocess_results: Iterable[PreprocessResult] | None = None,
        postprocess_results: Iterable[ResultInterface] | None = None,
        execution_time: timedelta | SupportsInt = 0,
    ) -> None:
        """Init :py:class:`PipelineResult`.

        Args:
            best_bitvector: Bitvector corresponding to the best result.
            best_value: Objective value of the best result.
            freq: Frequency object containing the frequency of found bitvectors and
                energies.
            solver_result: Result returned by the main solver of the pipeline.
            preprocess_results: List of results obtained from the preprocessing steps.
            postprocess_results: List of results returned by the postprocessors.
            execution_time: Time to execute the pipeline.
        """
        super().__init__(
            best_bitvector,
            best_value,
            freq,
            execution_time=execution_time,
        )
        self.solver_result = solver_result
        self.preprocess_results = list(preprocess_results or [])
        self.postprocess_results = list(postprocess_results or [])

    @classmethod
    def from_result(cls, *args: Any, **kwargs: Any) -> Self:
        """Creates a :py:class:`PipelineResult` instance using the default constructor.

        For arguments, see :py:meth:`__init__`.
        """
        return cls(*args, **kwargs)
