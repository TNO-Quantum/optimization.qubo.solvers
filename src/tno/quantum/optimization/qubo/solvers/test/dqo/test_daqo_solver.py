"""Test module for the ``DAQOSolver`` class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import ArrayLike

from tno.quantum.optimization.qubo.solvers import DAQOSolver

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components import QUBO
    from tno.quantum.utils import BitVector


@pytest.fixture(name="solver")
def solver_fixture() -> DAQOSolver:
    backend = {
        "name": "default.qubit",
        "options": {"shots": 1000},
    }
    return DAQOSolver(n_layers=100, backend=backend)


@pytest.mark.parametrize(
    "schedule",
    [
        "sinusoidal",
        "linear",
        np.linspace(0.0, 1.0, 100),
        (np.linspace(1.0, 0.0, 100), np.linspace(0.0, 1.0, 100)),
    ],
)
def test_daqo_schedule(
    qubo: QUBO,
    solver: DAQOSolver,
    expected_value: float,
    expected_bitvector: BitVector,
    schedule: str | ArrayLike,
) -> None:
    solver.schedule = schedule
    result = solver.solve(qubo)

    assert result.best_value == expected_value
    assert result.best_bitvector == expected_bitvector
    expected_bitvector_idx = result.freq.bitvectors.index(expected_bitvector)
    if qubo.size > 0:
        assert result.freq.num_occurrences[expected_bitvector_idx] > 975


@pytest.mark.parametrize(
    ("schedule", "error"),
    [
        (None, "Invalid schedule value: must either be"),
        # String schedules
        ("non-existent", "Unsupported schedule"),
        # ArrayLike schedules
        (np.zeros((2, 2, 2)), "Invalid schedule value"),
        # 1D schedules
        (np.zeros(50), "Schedule must have length"),
        (np.linspace(0.0, 2.0, 100), "must be in the range"),
        (np.linspace(-1.0, 1.0, 100), "must be in the range"),
        (np.linspace(1.0, 0.0, 100), "much smaller"),
        # 2D schedules
        (np.zeros((2, 50)), "Schedule must have shape"),
        ((np.linspace(0.25, 0.75, 100), np.linspace(0.0, 1.0, 100)), "much larger"),
        ((np.linspace(1.0, 0.0, 100), np.linspace(0.75, 0.25, 100)), "much smaller"),
        ((np.linspace(0.25, 0.0, 100), np.linspace(0.75, 1.0, 100)), "much larger"),
        ((np.linspace(1.0, 0.75, 100), np.linspace(0.0, 0.25, 100)), "much smaller"),
    ],
)
def test_daqo_invalid_schedule(
    solver: DAQOSolver, schedule: str | ArrayLike, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        solver.schedule = schedule
