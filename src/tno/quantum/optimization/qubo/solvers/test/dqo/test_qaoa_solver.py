"""Test module for the ``QAOASolver`` class."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from tno.quantum.devices import DeviceConfig
from tno.quantum.optimization.qubo.components import QUBO, Freq
from tno.quantum.utils import BitVector, OptimizerConfig

from tno.quantum.optimization.qubo.solvers import QAOAResult, QAOASolver
from tno.quantum.optimization.qubo.solvers._dqo._qaoa_solver import (
    get_default_evaluation_device_if_none,
    get_default_optimizer_if_none,
    get_default_training_device_if_none,
)

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components import QUBO, Freq


@pytest.fixture
def solver() -> QAOASolver:
    return QAOASolver(
        random_state=42,
        num_layers=4,
        num_iter=50,
        training_device={"name": "default.qubit", "options": {"wires": 3}},
        evaluation_device={
            "name": "default.qubit",
            "options": {"wires": 3, "shots": 1000, "seed": 42},
        },
    )


def test_qaoa_solver(
    solver: QAOASolver, qubo: QUBO, expected_value: float, expected_bitvector: BitVector
) -> None:
    result = solver.solve(qubo)

    assert isinstance(result, QAOAResult)
    assert result.best_value == expected_value
    assert result.best_bitvector == expected_bitvector
    assert _get_mode(result.freq) == expected_bitvector


def _get_mode(freq: Freq) -> BitVector:
    mode = BitVector("")
    mode_count = 0
    for bitvector, _, count in freq:
        if count > mode_count:
            mode_count = count
            mode = bitvector
    return mode


def test_init_parameters(qubo: QUBO, solver: QAOASolver) -> None:
    solver.init_beta = [1.0, 2.0, 3.0, 4.0]
    solver.init_gamma = [3.0, 4.0, 5.0, 6.0]
    solver.num_iter = 1

    result = solver.solve(qubo)

    np.testing.assert_array_equal(result.init_beta, solver._init_beta)
    np.testing.assert_array_equal(result.init_gamma, solver._init_gamma)


@pytest.mark.parametrize(
    ("device", "expected_device"),
    [
        (None, {"name": "default.qubit", "options": {}}),
        (
            {"name": "default.qubit", "options": {"seed": 42, "shots": 10}},
            {"name": "default.qubit", "options": {"seed": 42, "shots": 10}},
        ),
    ],
)
def test_training_device(
    qubo: QUBO,
    solver: QAOASolver,
    device: Mapping[str, str | Mapping[str, Any]] | None,
    expected_device: Mapping[str, str | Mapping[str, Any]],
) -> None:
    solver.num_iter = 1
    solver.training_device = device
    result = solver.solve(qubo)

    assert result.training_device == DeviceConfig.from_mapping(expected_device)


@pytest.mark.parametrize(
    ("device", "expected_device"),
    [
        (None, {"name": "default.qubit", "options": {"shots": 100}}),
        (
            {"name": "default.qubit", "options": {"seed": 42, "shots": 10}},
            {"name": "default.qubit", "options": {"seed": 42, "shots": 10}},
        ),
    ],
)
def test_evaluation_device(
    qubo: QUBO,
    solver: QAOASolver,
    device: Mapping[str, str | Mapping[str, Any]] | None,
    expected_device: Mapping[str, str | Mapping[str, Any]],
) -> None:
    solver.num_iter = 1
    solver.evaluation_device = device
    result = solver.solve(qubo)
    assert result.evaluation_device == DeviceConfig.from_mapping(expected_device)


@pytest.mark.parametrize(
    ("optimizer", "expected_optimizer"),
    [
        (None, {"name": "adagrad", "options": {}}),
        (
            {"name": "adam", "options": {"lr": 1.0}},
            {"name": "adam", "options": {"lr": 1.0}},
        ),
    ],
)
def test_optimizer(
    qubo: QUBO,
    solver: QAOASolver,
    optimizer: Mapping[str, str | Mapping[str, Any]] | None,
    expected_optimizer: Mapping[str, str | Mapping[str, Any]],
) -> None:
    solver.num_iter = 1
    solver.optimizer = optimizer
    result = solver.solve(qubo)
    assert result.optimizer == OptimizerConfig.from_mapping(expected_optimizer)


def test_init_qaoa_solver() -> None:
    """Test initialisation of QAOASolver."""
    solver = QAOASolver(
        num_layers=2,
        num_iter=3,
        init_beta=np.array([1.0, 1.0]),
        init_gamma=np.array([1.0, 1.0]),
        training_device={"name": "default.qubit", "options": {"shots": 123}},
        evaluation_device={"name": "default.qubit", "options": {"shots": 456}},
        optimizer={"name": "adagrad"},
        verbose=True,
    )

    assert solver.num_layers == 2
    assert solver.num_iter == 3
    assert isinstance(solver._init_beta, np.ndarray)
    assert isinstance(solver._init_gamma, np.ndarray)
    assert solver.training_device == DeviceConfig(
        name="default.qubit", options={"shots": 123}
    )
    assert solver.evaluation_device == DeviceConfig(
        name="default.qubit", options={"shots": 456}
    )
    assert solver.optimizer == OptimizerConfig(name="adagrad")
    assert solver.verbose


def test_init_qaoa_solver_default_arguments() -> None:
    """Test default values for QAOASolver."""
    solver = QAOASolver()
    assert solver.num_layers == 1
    assert solver.num_iter == 100
    assert isinstance(solver._init_beta, np.ndarray)
    assert isinstance(solver._init_gamma, np.ndarray)
    assert solver.training_device == get_default_training_device_if_none()
    assert solver.evaluation_device == get_default_evaluation_device_if_none()
    assert solver.optimizer == get_default_optimizer_if_none()
    assert solver.diff_method == "best"
    assert not solver.verbose
