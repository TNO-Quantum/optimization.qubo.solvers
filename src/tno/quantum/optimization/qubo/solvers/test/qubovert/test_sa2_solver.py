"""Test module for the ``SA2Solver`` class."""

from __future__ import annotations

import numpy as np
from tno.quantum.optimization.qubo.components import QUBO

from tno.quantum.optimization.qubo.solvers import (
    SA2Solver,
)


def test_sa2_solver_random_state() -> None:
    """Test consistency for solvers with random seed."""
    # Generate random QUBO of size 250 x 250
    rng = np.random.default_rng(0)
    qubo = QUBO(rng.normal(0, 10, size=(250, 250)), offset=rng.normal(0, 10))

    solver1 = SA2Solver(random_state=42, num_reads=100)
    results1 = solver1.solve(qubo)

    solver2 = SA2Solver(random_state=42, num_reads=100)
    results2 = solver2.solve(qubo)

    solver3 = SA2Solver(random_state=43, num_reads=100)
    results3 = solver3.solve(qubo)

    assert results1.freq == results2.freq
    assert results1.freq != results3.freq
