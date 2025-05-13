"""Test module for the ``QAOAResult`` class."""

import pytest

from tno.quantum.optimization.qubo.components import Freq
from tno.quantum.optimization.qubo.solvers._qaoa._qaoa_result import QAOAResult
from tno.quantum.utils import BackendConfig, OptimizerConfig


@pytest.fixture(name="result")
def result_fixture() -> QAOAResult:
    return QAOAResult(
        "0",
        0,
        Freq(["0"], [0], [1]),
        init_beta=[1.0, 2.0],
        init_gamma=[3.0, 4.0],
        final_beta=[1.0, 2.0],
        final_gamma=[3.0, 4.0],
        expval_history=[-1.0, -2.0],
        training_backend=BackendConfig(name="default.qubit"),
        evaluation_backend=BackendConfig(name="default.qubit"),
        optimizer=OptimizerConfig(name="adagrad"),
    )


def test_draw_expval_history(result: QAOAResult) -> None:
    result.plot_expval_history()


def test_draw_parameters(result: QAOAResult) -> None:
    result.plot_parameters()


def test_draw_shots_histogram(result: QAOAResult) -> None:
    result.plot_shots_histogram()
