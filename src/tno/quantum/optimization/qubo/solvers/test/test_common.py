import hybrid
import pytest
from tno.quantum.optimization.qubo.components import (
    QUBO,
    BasicResult,
    ResultInterface,
    Solver,
    SolverConfig,
)
from tno.quantum.utils import BitVector

from tno.quantum.optimization.qubo.solvers import (
    BFSolver,
    CustomSolver,
    DAQOResult,
    DAQOSolver,
    DimodSampleSetResult,
    IterativeResult,
    NeighborhoodSolver,
    PipelineResult,
    PipelineSolver,
    QAOAResult,
    QAOASolver,
    QubovertAnnealResult,
    RSSolver,
    SA2Solver,
)
from tno.quantum.optimization.qubo.solvers.test.dwave.conftest import (
    DWAVE_AVAILABLE_SOLVERS,
    DWAVE_HAS_API_TOKEN,
    DWAVE_SOLVERS_WITH_API,
    DWAVE_SOLVERS_WITH_SEED,
)

SOLVERS_WITH_RESULT_TYPES = [
    (NeighborhoodSolver(random_state=42), IterativeResult),
    (RSSolver(random_state=42), BasicResult),
    (PipelineSolver({"name": "bf_solver"}), PipelineResult),
    (QAOASolver(random_state=42), QAOAResult),
    (DAQOSolver(1), DAQOResult),
    (BFSolver(), BasicResult),
    (SA2Solver(random_state=42), QubovertAnnealResult),
    *[
        (
            solver(random_state=42) if solver in DWAVE_SOLVERS_WITH_SEED else solver(),  # type: ignore[call-arg]
            DimodSampleSetResult,
        )
        for solver in DWAVE_AVAILABLE_SOLVERS
    ],
    (
        CustomSolver(
            branch=(
                hybrid.decomposers.IdentityDecomposer()
                | hybrid.samplers.SimulatedAnnealingSubproblemSampler()
                | hybrid.composers.IdentityComposer()
            )
        ),
        DimodSampleSetResult,
    ),
]


@pytest.mark.parametrize(("solver", "expected_result_type"), SOLVERS_WITH_RESULT_TYPES)
def test_solvers(
    solver: Solver[ResultInterface],
    qubo: QUBO,
    expected_result_type: type[ResultInterface],
    expected_bitvector: BitVector,
    expected_value: float,
) -> None:
    """Test basic functionality of solver.

    This test checks if the solver produces a result of the expected type, containing
    the expected best bitvector and best value. Also, this test checks if the values in
    the frequency object correspond to the evaluation of the QUBO.
    """
    result = solver.solve(qubo)

    # Check type of result
    if not isinstance(result, expected_result_type):
        msg = (
            f"{type(solver).__name__} expected to produce a "
            f"{expected_result_type.__name__}, but produced a {type(result).__name__}"
        )
        raise TypeError(msg)

    # Check best bitvector
    if result.best_bitvector != expected_bitvector:
        msg = (
            f"{type(solver).__name__} did not produce the expected best bitvector "
            f"{expected_bitvector}, instead produced {result.best_bitvector}"
        )
        raise AssertionError(msg)

    # Check best value
    if result.best_value != expected_value:
        msg = (
            f"{type(solver).__name__} did not produce the expected best value "
            f"{expected_value}, instead produced {result.best_value}"
        )
        raise AssertionError(msg)

    # Check frequency object
    for bitvector, value, _ in result.freq:
        expected_value = qubo.evaluate(bitvector)
        if value != pytest.approx(expected_value):
            msg = (
                f"{type(solver).__name__} produced incorrect `Freq` object: "
                f"bitvector {bitvector} has corresponding value {value}, but expected "
                f"{expected_value}"
            )
            raise AssertionError(msg)


def test_solvers_coverage() -> None:
    """Test if the `test_solvers` covers all supported solvers."""
    for solver in SolverConfig.supported_items().values():
        if not DWAVE_HAS_API_TOKEN and solver in DWAVE_SOLVERS_WITH_API:
            continue
        assert any(isinstance(s, solver) for s, _ in SOLVERS_WITH_RESULT_TYPES), (
            f"Solver `{solver.__name__}` is not covered by `test_solvers`. "
            "Please add it to `SOLVERS_WITH_RESULT_TYPES`."
        )
