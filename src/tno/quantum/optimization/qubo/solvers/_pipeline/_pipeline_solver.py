"""This module contains the ``PipelineSolver`` class."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from tno.quantum.optimization.qubo.components import (
    QUBO,
    PostprocessorConfig,
    PreprocessorConfig,
    ResultInterface,
    Solver,
    SolverConfig,
)
from tno.quantum.optimization.qubo.solvers._pipeline._pipeline_result import (
    PipelineResult,
)
from tno.quantum.optimization.qubo.solvers._pipeline._preprocess_result import (
    PreprocessResult,
)


class PipelineSolver(Solver[PipelineResult]):
    """Solver class that represents a pipeline of a solver with pre- and postprocessors.

    A pipeline solver solves a QUBO problem in three stages:
        1. Preprocess the QUBO using the provided preprocessors.
        2. Solve the preprocessed QUBO using the provided main solver.
        3. Postprocess the obtained results using the provided postprocessors.

    To make use of a pipeline, create a :py:class:`PipelineSolver` and provide it the
    desired main solver, preprocessing and postprocessing. Then, a
    :py:class:`PipelineSolver` can be treated like any other
    :py:class:`~tno.quantum.optimization.qubo.components.Solver`, as shown in the
    following example.

    Example:
        >>> from tno.quantum.optimization.qubo.solvers import PipelineSolver
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>>
        >>> qubo = QUBO([
        ...     [ 7,  6, -2,  9,  5],
        ...     [ 8, -5,  9, -7, -5],
        ...     [ 7,  9, -6, -3, -4],
        ...     [ 7, -8,  1,  8,  6],
        ...     [ 1,  1,  7,  8, -2]
        ... ])
        >>>
        >>> # Construct PipelineSolver from configuration objects
        >>> solver_config = { "name": "bf_solver" }
        >>> preprocess_config = { "name": "q_pro_plus_preprocessor" }
        >>> postprocess_config = { "name": "steepest_descent_postprocessor" }
        >>>
        >>> pipeline = PipelineSolver(
        ...     solver_config,
        ...     preprocess = [ preprocess_config ],
        ...     postprocess = [ postprocess_config ]
        ... )  # doctest: +SKIP
        >>>
        >>> # Solve QUBO using pipline like any other solver
        >>> result = pipeline.solve(qubo)  # doctest: +SKIP
        >>> result.best_bitvector  # doctest: +SKIP
        BitVector(01010)

    The `result` is a :py:class:`PipelineResult`. The the intermediate results can be
    accessed as follows:
        >>> # The result of the preprocessor
        >>> result.preprocess_results[0]  # doctest: +SKIP
        <...>
        >>> # The result of the main solver
        >>> result.solver_result  # doctest: +SKIP
        <...>
        >>> # The result of the postprocessor
        >>> result.postprocess_results[0]  # doctest: +SKIP
        <...>
    """

    def __init__(
        self,
        solver_config: SolverConfig | Mapping[str, Any],
        *,
        preprocess: (
            PreprocessorConfig
            | Mapping[str, Any]
            | list[PreprocessorConfig | Mapping[str, Any]]
            | None
        ) = None,
        postprocess: (
            PostprocessorConfig
            | Mapping[str, Any]
            | list[PostprocessorConfig | Mapping[str, Any]]
            | None
        ) = None,
    ) -> None:
        """Init :py:class:`PipelineSolver`.

        Args:
            solver_config: Configuration of the main solver.
            preprocess: Configuration(s) of the preprocessors.
            postprocess: Configuration(s) of the postprocessors.
        """
        # Instantiate solver
        solver_config = SolverConfig.from_mapping(solver_config)
        self._solver = solver_config.get_instance()

        # Instantiate preprocessors
        if isinstance(preprocess, Mapping):
            preprocess = [preprocess]
        if preprocess is None:
            preprocess = []
        self._preprocessors = [
            PreprocessorConfig.from_mapping(config).get_instance()
            for config in preprocess
        ]

        # Instantiate postprocessors
        if isinstance(postprocess, Mapping):
            postprocess = [postprocess]
        if postprocess is None:
            postprocess = []
        self._postprocessors = [
            PostprocessorConfig.from_mapping(config).get_instance()
            for config in postprocess
        ]

    def _solve(self, qubo: QUBO) -> PipelineResult:
        """Solve QUBO using a pipeline of a solver with pre- and postprocessors.

        Args:
            qubo: QUBO to solve.

        Returns:
            A ``PipelineResult`` instance.
        """
        # Apply preprocessing steps in order, storing the intermediate
        # partial solutions and storing the execution time for the preprocess results
        preprocess_partial_solutions = []
        preprocess_execution_times = []
        for preprocessor in self._preprocessors:
            start_time = time.perf_counter()
            partial_solution, qubo = preprocessor.preprocess(qubo)
            execution_time = time.perf_counter() - start_time
            preprocess_partial_solutions.append(partial_solution)
            preprocess_execution_times.append(execution_time)

        # Solve preprocessed QUBO
        solver_result = self._solver.solve(qubo)

        result: ResultInterface = solver_result

        # Apply postprocessing steps
        postprocess_results: list[ResultInterface] = []
        for postprocessor in self._postprocessors:
            result = postprocessor.postprocess(qubo, result)
            postprocess_results.append(result)

        # Construct `PreprocessResult`s (they are created in reverse order)
        preprocess_results: list[PreprocessResult] = []
        for partial_solution, execution_time in zip(
            reversed(preprocess_partial_solutions), reversed(preprocess_execution_times)
        ):
            result = PreprocessResult.from_result(partial_solution, result)
            result.execution_time = timedelta(seconds=execution_time)
            preprocess_results.append(result)
        preprocess_results.reverse()

        # Obtain `best_bitvector`, `best_value` and `freq` from the final result
        best_bitvector = result.best_bitvector
        best_value = result.best_value
        freq = result.freq

        # Construct and return PipelineResult
        return PipelineResult.from_result(
            best_bitvector,
            best_value,
            freq,
            solver_result=solver_result,
            preprocess_results=preprocess_results,
            postprocess_results=postprocess_results,
        )
