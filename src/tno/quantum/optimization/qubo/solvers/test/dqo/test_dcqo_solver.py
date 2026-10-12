"""Test module for the ``DCQOSolver`` class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from numpy.typing import ArrayLike

from tno.quantum.optimization.qubo.solvers import DCQOSolver

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components import QUBO
    from tno.quantum.utils import BitVector


@pytest.fixture(name="solver")
def solver_fixture() -> DCQOSolver:
    device = {
        "name": "default.qubit",
        "options": {"shots": 1000},
    }
    return DCQOSolver(n_layers=90, device=device, end_time=100)


@pytest.mark.parametrize(
    "schedule",
    [
        "sinusoidal",
        "linear",
        np.linspace(0.0, 1.0, 90),
        (np.linspace(1.0, 0.0, 90), np.linspace(0.0, 1.0, 90), np.full(90, 0.01)),
    ],
)
def test_dcqo_schedule(
    qubo: QUBO,
    solver: DCQOSolver,
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


@pytest.mark.parametrize(("cd_layer", "n_layers"), [("Y", 65), ("YZY", 60)])
def test_dcqo_cd_layer(
    qubo: QUBO,
    cd_layer: Literal["Y", "YZY"],
    n_layers: int,
    expected_value: float,
    expected_bitvector: BitVector,
) -> None:
    solver = DCQOSolver(n_layers, cd_layer=cd_layer, end_time=100)
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
        (np.linspace(0.0, 2.0, 90), "must be in the range"),
        (np.linspace(-1.0, 1.0, 90), "must be in the range"),
        (np.linspace(1.0, 0.0, 90), "smaller"),
        # 2D schedules
        (
            (np.linspace(0.25, 0.75, 90), np.linspace(0.0, 1.0, 90), np.zeros(90)),
            "much larger",
        ),
        (
            (np.linspace(1.0, 0.0, 90), np.linspace(0.75, 0.25, 90), np.zeros(90)),
            "much smaller",
        ),
        (
            (np.linspace(0.25, 0.0, 90), np.linspace(0.75, 1.0, 90), np.zeros(90)),
            "much larger",
        ),
        (
            (np.linspace(1.0, 0.75, 90), np.linspace(0.0, 0.25, 90), np.zeros(90)),
            "much smaller",
        ),
    ],
)
def test_dcqo_invalid_schedule(
    solver: DCQOSolver, schedule: str | ArrayLike, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        solver.schedule = schedule
